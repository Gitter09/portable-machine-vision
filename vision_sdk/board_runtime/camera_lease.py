"""Self-expiring camera lease for the standalone SiMa worker."""

from __future__ import annotations

import socket
import threading
import time


LEASE_SECONDS = 30.0


class CameraLease:
    def __init__(self, bind: str, admin_ip: str, release_id: str, port: int):
        self.bind = bind
        self.admin_ip = admin_ip
        self.release_id = release_id
        self.port = port
        self.deadline = time.monotonic() + 120.0  # activation grace before the first heartbeat
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        channel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            channel.bind((self.bind, self.port))
            channel.settimeout(1.0)
        except Exception:
            channel.close()
            raise
        self.port = channel.getsockname()[1]
        self._socket = channel
        self._thread = threading.Thread(target=self._listen, daemon=True, name="sdk-camera-lease")
        self._thread.start()

    def _listen(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                payload, address = self._socket.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            if address[0] == self.admin_ip and payload == self.release_id.encode("ascii"):
                self.deadline = time.monotonic() + LEASE_SECONDS

    def expired(self) -> bool:
        return time.monotonic() > self.deadline

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
