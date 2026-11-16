"""Loopback-only dashboard with bounded proxying and local receiver discovery."""
import argparse
import functools
import hashlib
import http.client
import http.server
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
MAX_BODY = 1024 * 1024


def session_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def receiver_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('receiver must be an HTTP loopback origin')
    try:
        port = parsed.port if parsed.port is not None else 80
        if not 1 <= port <= 65535:
            raise ValueError('invalid port')
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return port


class ReceiverLocator:
    """Only explicit loopback targets or records under one trusted local directory."""
    def __init__(self, port=None, session_root=None):
        self.port = port
        self.root = Path(session_root).resolve() if session_root else None
        self.lock = threading.Lock()
        self.checked = 0
        self.cached = None

    def target(self):
        if self.port:
            return self.port, f'explicit-{self.port}'
        with self.lock:
            if time.monotonic() - self.checked >= 1:
                self.cached = self.discover()
                self.checked = time.monotonic()
            return self.cached

    def discover(self):
        if not self.root or not self.root.is_dir():
            return None
        # newest directories first; bounded scan, no network port probing
        candidates = sorted(self.root.glob('*/session.json'), key=session_mtime, reverse=True)[:128]
        found = []
        for path in candidates:
            try:
                if not path.resolve().is_relative_to(self.root) or path.stat().st_size > 65536:
                    continue
                session = json.loads(path.read_text())
                started = float(session['started_unix'])
                if session.get('state') != 'running' or not 0 <= time.time() - started <= 600:
                    continue
                port = receiver_url(session['panel_url'])
                identity = hashlib.sha256(f'{path.parent.name}:{started}:{port}'.encode()).hexdigest()[:20]
                found.append((started, port, identity))
            except (OSError, ValueError, KeyError, TypeError, argparse.ArgumentTypeError):
                continue  # partially-written session files are retried on the next scan
        if not found:
            return None
        _, port, identity = max(found)
        return port, identity


def static_allowed(path):
    asset = (ROOT / path.lstrip('/')).resolve()
    entrypoints = ('/', '/index.html', '/app.js', '/recovery.js', '/menu.js', '/style.css')
    return path in entrypoints or (path.startswith('/assets/') and asset.is_relative_to(ROOT/'assets') and asset.is_file())


def response_bytes(handler, body, content_type):
    handler.send_response(200)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.reply_started = True
    handler.wfile.write(body)


def recovery_status(locator, target):
    state = 'not-supervised'
    if locator and locator.root:
        path = locator.root/'summit-status.json'
        try:
            if path.stat().st_size <= 65536 and path.resolve().is_relative_to(locator.root):
                report = json.loads(path.read_text())
                state = str(report.get('state', state))[:100]
        except FileNotFoundError:
            state = 'not-supervised'
        except (OSError, ValueError, AttributeError):
            state = 'supervisor-status-unavailable'
    return {'supervisor_state': state, 'receiver_available': target is not None}


def receiver_body(response, path, identity):
    body = response.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise ValueError('Receiver response exceeds limit')
    if path.endswith('.json'):
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError('Expected JSON object')
        payload['receiver_id'] = identity
        return json.dumps(payload, allow_nan=False).encode(), 'application/json'
    if not response.getheader('Content-Type', '').startswith('image/jpeg'):
        raise ValueError('Expected JPEG frame')
    return body, 'image/jpeg'


def stream_video(handler, response, target):
    content_type = response.getheader('Content-Type', '')
    if not content_type.startswith('multipart/x-mixed-replace'):
        raise ValueError('Receiver returned invalid video')
    handler.send_response(200)
    handler.send_header('Content-Type', content_type)
    handler.end_headers()
    handler.reply_started = True
    while handler.target() == target:
        chunk = response.read1(65536)
        if not chunk:
            break
        handler.wfile.write(chunk)
        handler.wfile.flush()


def proxy_receiver(handler, path, target):
    connection = http.client.HTTPConnection('127.0.0.1', target[0], timeout=2)
    handler.reply_started = False
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        if response.status != 200:
            handler.send_error(response.status if response.status in (404, 409, 503) else 502)
            return
        if path == '/live.mjpg':
            # After streaming headers, connection errors must only close the stream.
            stream_video(handler, response, target)
        else:
            body, content_type = receiver_body(response, path, target[1])
            response_bytes(handler, body, content_type)
    except (OSError, http.client.HTTPException, ValueError, TypeError) as error:
        if not handler.reply_started:
            handler.send_error(502, 'Receiver unavailable or invalid; retrying automatically')
        else:
            handler.log_message('Receiver stream ended: %s', type(error).__name__)
    finally:
        connection.close()


class DashboardServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, locator):
        self.locator = locator
        self.proxy_slots = threading.BoundedSemaphore(12)
        self.stream_slots = threading.BoundedSemaphore(2)
        super().__init__(address, functools.partial(Handler, directory=str(ROOT)))


class Handler(http.server.SimpleHTTPRequestHandler):
    # Compatibility for direct handler use in existing tests.
    receiver_port = None

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        super().end_headers()

    def do_HEAD(self):
        self.send_error(405)

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as error:
            self.log_message('Client connection ended: %s', type(error).__name__)

    def route(self):
        host = urlsplit('//'+self.headers.get('Host', '')).hostname
        if host not in ('127.0.0.1', 'localhost'):
            self.send_error(403, 'Loopback Host required')
            return
        path = urlsplit(self.path).path
        if path.startswith('/bridge/'):
            self.bridge(path.removeprefix('/bridge'))
            return
        if not static_allowed(path):
            self.send_error(404)
            return
        super().do_GET()

    def target(self):
        locator = getattr(self.server, 'locator', None)
        return locator.target() if locator else ((self.receiver_port, f'explicit-{self.receiver_port}') if self.receiver_port else None)

    def bridge(self, path):
        if path == '/status.json':
            status = recovery_status(getattr(self.server, 'locator', None), self.target())
            response_bytes(self, json.dumps(status).encode(), 'application/json')
            return
        if path not in ('/inspection.json', '/stats.json', '/live.mjpg') and not re.fullmatch(r'/frames/[a-f0-9]{64}\.jpg', path):
            self.send_error(404)
            return
        target = self.target()
        if not target:
            self.send_error(503, 'Waiting for a running receiver session; retrying automatically')
            return
        requested = parse_qs(urlsplit(self.path).query).get('receiver_id', [None])[0]
        if requested and requested != target[1]:
            self.send_error(409, 'Receiver session changed; request a fresh inspection')
            return
        slots = getattr(self.server, 'stream_slots' if path == '/live.mjpg' else 'proxy_slots', None)
        if slots and not slots.acquire(blocking=False):
            self.send_error(503, 'Receiver connection limit; retry shortly')
            return
        try:
            proxy_receiver(self, path, target)
        finally:
            if slots:
                slots.release()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8088)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--receiver', type=receiver_url, help='fixed loopback receiver origin')
    group.add_argument('--session-root', type=Path, default=ROOT.parent/'local/runs', help='discover receiver records here (default local/runs)')
    args = parser.parse_args()
    locator = ReceiverLocator(args.receiver, args.session_root)
    server = DashboardServer(('127.0.0.1', args.port), locator)
    print(f'Dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    print('Receiver:', f'fixed port {args.receiver}' if args.receiver else f'auto-discovery under {args.session_root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
