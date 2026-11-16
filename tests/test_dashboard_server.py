"""Offline checks for the summit dashboard's read-only receiver bridge."""
import argparse
import functools
import importlib.util
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen
import http.server
import pytest

spec = importlib.util.spec_from_file_location('dashboard_server', Path(__file__).parents[1] / 'frontend/serve.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@pytest.mark.parametrize('url', ['https://127.0.0.1:80', 'http://example.com', 'http://127.0.0.1/path', 'http://user:secret@localhost', 'http://localhost?x=1'])
def test_receiver_rejects_non_loopback_origins(url):
    with pytest.raises(argparse.ArgumentTypeError):
        dashboard.receiver_url(url)


def test_receiver_accepts_loopback():
    assert dashboard.receiver_url('http://127.0.0.1:58078/') == 58078


def test_offline_dashboard_and_bridge():
    root = Path(__file__).parents[1] / 'frontend'
    handler = functools.partial(dashboard.Handler, directory=str(root))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base) as response:
            assert b'See what' in response.read()
        for path, code in [('/bridge/inspection.json', 503), ('/bridge/other', 404)]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
