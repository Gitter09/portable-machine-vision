"""Small client for the versioned SiMa worker protocol."""

from __future__ import annotations

import json
import socket
import struct
import uuid
from dataclasses import dataclass


class BoardError(RuntimeError):
    pass


@dataclass
class BoardResult:
    header: dict
    image_jpeg: bytes


class BoardConnection:
    def __init__(self, host: str, port: int = 5005, timeout: float = 10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.channel: socket.socket | None = None
        self.ready: dict = {}

    def __enter__(self):
        try:
            self.channel = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.channel.settimeout(self.timeout)
            header, _ = self._receive()
            if header.get("type") != "ready":
                raise BoardError(f"Board worker is not ready: {header.get('type')}")
            self.ready = dict(header.get("body") or {})
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_exc):
        if self.channel:
            self.channel.close()
            self.channel = None

    def _read(self, count: int) -> bytes:
        if not self.channel:
            raise BoardError("Board connection is closed")
        data = bytearray()
        while len(data) < count:
            part = self.channel.recv(count - len(data))
            if not part:
                raise BoardError("Board closed the connection")
            data.extend(part)
        return bytes(data)

    def _receive(self) -> tuple[dict, bytes]:
        size = struct.unpack("!I", self._read(4))[0]
        if not 0 < size <= 1024 * 1024:
            raise BoardError("Board sent an invalid message")
        header = json.loads(self._read(size))
        payload_size = int(header.get("payload_length", 0))
        if not 0 <= payload_size <= 16 * 1024 * 1024:
            raise BoardError("Board sent an oversized image")
        return header, self._read(payload_size)

    def _send(self, kind: str, body: dict | None = None) -> str:
        if not self.channel:
            raise BoardError("Board connection is closed")
        request_id = uuid.uuid4().hex
        message = json.dumps({"type": kind, "request_id": request_id,
                              "payload_length": 0, "body": body or {}}, separators=(",", ":")).encode()
        self.channel.sendall(struct.pack("!I", len(message)) + message)
        return request_id

    def capture(self) -> BoardResult:
        request_id = self._send("capture", {"parallel_display": True})
        while True:
            header, payload = self._receive()
            if header.get("request_id") != request_id:
                raise BoardError("Board response ID did not match the capture")
            kind = header.get("type")
            if kind == "captured":
                continue
            if kind == "error":
                body = header.get("body") or {}
                raise BoardError(str(body.get("message") or body.get("code") or "Board inspection failed"))
            if kind == "result":
                body = dict(header.get("body") or {})
                if body.get("status") != "ok":
                    raise BoardError("Board did not return a valid inspection")
                return BoardResult(body, payload)
            raise BoardError(f"Unexpected board response: {kind}")

    def shutdown(self) -> None:
        request_id = self._send("shutdown")
        header, _ = self._receive()
        body = header.get("body") or {}
        if header.get("request_id") != request_id or header.get("type") != "stopped" or not body.get("camera_released"):
            raise BoardError("Board did not confirm that the camera was released")
