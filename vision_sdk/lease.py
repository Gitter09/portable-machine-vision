"""PC-side heartbeat for the SDK's camera-owning SiMa worker."""

from __future__ import annotations

import logging
import socket
import threading

from .config import VisionStore


LOGGER = logging.getLogger(__name__)


class LeaseController:
    def __init__(self, store: VisionStore, interval: float = 5.0):
        self.store = store
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sdk-board-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
            self._thread = None

    def tick(self) -> None:
        config = self.store.load()
        if not config.active_release:
            return
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
            channel.sendto(config.active_release.encode("ascii"), (config.board_host, config.board_port + 1))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except (OSError, ValueError, UnicodeEncodeError) as error:
                LOGGER.warning("Could not renew SDK board camera lease: %s", error)
            self._stop.wait(self.interval)
