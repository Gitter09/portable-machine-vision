"""Serve the offline dashboard and optionally proxy a loopback inspection receiver."""
import argparse
import http.client
import http.server
import functools
from pathlib import Path
from urllib.parse import urlsplit


def receiver_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('receiver must be an HTTP loopback origin')
    try:
        port = parsed.port or 80
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return port


class Handler(http.server.SimpleHTTPRequestHandler):
    receiver_port = None

    def do_GET(self):
        if not self.path.startswith('/bridge/'):
            return super().do_GET()
        path = urlsplit(self.path).path.removeprefix('/bridge')
        import re
        if path not in ('/inspection.json', '/stats.json', '/live.mjpg') and not re.fullmatch(r'/frames/[a-f0-9]+\.jpg', path):
            self.send_error(404)
            return
        if self.receiver_port is None:
            self.send_error(503, 'No receiver configured; restart with --receiver http://127.0.0.1:PORT')
            return
        connection = http.client.HTTPConnection('127.0.0.1', self.receiver_port, timeout=3)
        headers_sent = False
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/octet-stream'))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            headers_sent = True
            if path == '/live.mjpg':
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
            else:
                self.wfile.write(response.read(8 * 1024 * 1024))
        except (OSError, http.client.HTTPException):
            if not headers_sent:
                self.send_error(502, 'Receiver unavailable')
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--receiver', type=receiver_url, help='existing receiver origin, e.g. http://127.0.0.1:58078')
    args = parser.parse_args()
    Handler.receiver_port = args.receiver
    handler = functools.partial(Handler, directory=str(Path(__file__).parent))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(f'Dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
