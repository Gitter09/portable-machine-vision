"""Fault-injection checks against real loopback HTTP fixtures; no board access."""
from contextlib import contextmanager
import http.client
import http.server
import importlib.util
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

spec = importlib.util.spec_from_file_location('dashboard_recovery_server', Path(__file__).parents[1]/'frontend/serve.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


class Receiver(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        mode = getattr(self.server, 'mode', 'healthy')
        if mode == 'unavailable':
            self.send_error(503); return
        if mode == 'malformed':
            body = b'not JSON'
        elif mode == 'oversized':
            body = b'x'*(dashboard.MAX_BODY + 1)
        elif self.path == '/live.mjpg' and mode == 'stream':
            body = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\nJPEG\r\n'
        elif self.path.startswith('/frames/'):
            body = b'jpeg-fixture'
        else:
            body = json.dumps({'state':'matched', 'sequence':1, 'decoded_frames':60}).encode()
        self.send_response(200)
        content_type = 'image/jpeg' if self.path.startswith('/frames/') else 'application/json'
        if self.path == '/live.mjpg' and mode == 'stream':
            content_type = 'multipart/x-mixed-replace; boundary=frame'
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def record(root, name, port, **changes):
    directory = root/name
    directory.mkdir(exist_ok=True)
    values = {'state':'running', 'started_unix':time.time(), 'panel_url':f'http://127.0.0.1:{port}/'}
    values.update(changes)
    (directory/'session.json').write_text(json.dumps(values))
    return directory/'session.json'


def code(url, expected):
    with pytest.raises(HTTPError) as error:
        urlopen(url, timeout=4)
    assert error.value.code == expected


def test_proxy_survives_receiver_failure_and_invalid_payloads():
    upstream = http.server.ThreadingHTTPServer(('127.0.0.1',0), Receiver)
    with running(upstream), running(dashboard.DashboardServer(('127.0.0.1',0),dashboard.ReceiverLocator(upstream.server_port))) as base:
        for mode, expected in [('unavailable',503),('malformed',502),('oversized',502)] * 5:
            upstream.mode = mode
            code(base+'/bridge/inspection.json', expected)
        upstream.mode = 'healthy'
        with urlopen(base+'/bridge/inspection.json') as response:
            assert json.load(response)['receiver_id'] == f'explicit-{upstream.server_port}'
        with urlopen(base+'/bridge/frames/'+ 'a'*64 + '.jpg') as response:
            assert response.read() == b'jpeg-fixture'


def test_discovery_follows_new_session_and_rejects_old_frame_requests(tmp_path):
    first = http.server.ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    second = http.server.ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    locator = dashboard.ReceiverLocator(session_root=tmp_path)
    with running(first), running(second), running(dashboard.DashboardServer(('127.0.0.1',0),locator)) as base:
        code(base+'/bridge/inspection.json',503)
        record(tmp_path,'a',first.server_port)
        locator.checked = 0
        with urlopen(base+'/bridge/inspection.json') as response:
            old_id = json.load(response)['receiver_id']
        record(tmp_path,'b',second.server_port)
        locator.checked = 0
        with urlopen(base+'/bridge/inspection.json') as response:
            new_id = json.load(response)['receiver_id']
        assert old_id != new_id
        code(base+'/bridge/frames/'+ 'a'*64 + '.jpg?receiver_id='+old_id,409)
        with urlopen(base+'/bridge/frames/'+ 'a'*64 + '.jpg?receiver_id='+new_id) as response:
            assert response.status == 200


def test_discovery_ignores_stale_untrusted_and_partial_records(tmp_path):
    record(tmp_path,'old',1234,started_unix=time.time()-601)
    record(tmp_path,'completed',1234,state='completed')
    record(tmp_path,'remote',1234,panel_url='http://example.com')
    record(tmp_path,'future',1234,started_unix=time.time()+100)
    partial = record(tmp_path,'partial',1234)
    partial.write_text('{')
    assert dashboard.ReceiverLocator(session_root=tmp_path).discover() is None
    partial.write_text(json.dumps({'state':'running','started_unix':time.time(),'panel_url':'http://127.0.0.1:4321'}))
    assert dashboard.ReceiverLocator(session_root=tmp_path).discover()[0] == 4321


def test_static_server_does_not_expose_python_or_directory_listings():
    with running(dashboard.DashboardServer(('127.0.0.1',0),dashboard.ReceiverLocator())) as base:
        for path in ('/serve.py','/assets/','/../PRODUCT.md','/bridge/frames/bad.jpg'):
            code(base+path,404)
        with urlopen(base+'/recovery.js') as response:
            assert response.headers['Cache-Control'] == 'no-store'
            assert response.headers['X-Content-Type-Options'] == 'nosniff'


def test_external_host_and_proxy_overload_are_rejected():
    upstream = http.server.ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    server = dashboard.DashboardServer(('127.0.0.1',0),dashboard.ReceiverLocator(upstream.server_port))
    with running(upstream), running(server) as base:
        conn = http.client.HTTPConnection('127.0.0.1',server.server_port)
        conn.request('GET','/',headers={'Host':'evil.example'})
        assert conn.getresponse().status == 403
        conn.close()
        for _ in range(12):
            assert server.proxy_slots.acquire(blocking=False)
        code(base+'/bridge/stats.json',503)
        for _ in range(12):
            server.proxy_slots.release()
        assert urlopen(base+'/bridge/stats.json').status == 200


def test_ended_video_stream_releases_its_slot_for_reconnection():
    upstream = http.server.ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    server = dashboard.DashboardServer(('127.0.0.1',0),dashboard.ReceiverLocator(upstream.server_port))
    with running(upstream), running(server) as base:
        code(base+'/bridge/live.mjpg',502)  # JSON is not an MJPEG stream
        upstream.mode = 'stream'
        for _ in range(5):
            with urlopen(base+'/bridge/live.mjpg',timeout=3) as response:
                assert b'JPEG' in response.read()
        assert server.stream_slots.acquire(blocking=False)
        assert server.stream_slots.acquire(blocking=False)
        server.stream_slots.release();server.stream_slots.release()


def test_status_reports_a_manual_safety_stop_without_exposing_secrets(tmp_path):
    (tmp_path/'summit-status.json').write_text(json.dumps({'state':'cleanup-unconfirmed-manual-check-required','secret':'not-for-client'}))
    locator=dashboard.ReceiverLocator(session_root=tmp_path)
    with running(dashboard.DashboardServer(('127.0.0.1',0),locator)) as base:
        with urlopen(base+'/bridge/status.json') as response:
            status=json.load(response)
            assert status['supervisor_state']=='cleanup-unconfirmed-manual-check-required'
            assert 'secret' not in status


def test_dead_receiver_returns_bounded_failure():
    # Reserve a port, then close it: deterministic connection refusal, no board.
    upstream = http.server.ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    port = upstream.server_port
    upstream.server_close()
    with running(dashboard.DashboardServer(('127.0.0.1',0),dashboard.ReceiverLocator(port))) as base:
        started = time.monotonic()
        code(base+'/bridge/inspection.json',502)
        assert time.monotonic()-started < 3
