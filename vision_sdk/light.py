"""The supported two-channel serial inspection light."""

from __future__ import annotations

import time


class LightError(RuntimeError):
    pass


def port_present(port: str) -> bool:
    try:
        from serial.tools import list_ports
    except ImportError:
        return False
    return any(item.device.casefold() == port.casefold() for item in list_ports.comports())


def _frame(command: int, channel: int, value: int = 0) -> bytes:
    digits = [command, channel, (value >> 8) & 15, (value >> 4) & 15, value & 15]
    checksum = 0x14
    for digit in digits:
        checksum ^= digit
    return ("$" + "".join(f"{digit:x}" for digit in digits) + f"{checksum:02x}").encode("ascii")


class LightSession:
    def __init__(self, port: str, intensity: int = 0x032):
        self.port = port
        self.intensity = intensity
        self.serial = None

    def _send(self, command: int, value: int = 0) -> None:
        for channel in (1, 2):
            self.serial.reset_input_buffer()
            self.serial.write(_frame(command, channel, value))
            time.sleep(0.04)
            waiting = self.serial.in_waiting
            reply = self.serial.read(waiting or 1)
            if b"$" not in reply:
                raise LightError(f"Light controller on {self.port} did not acknowledge channel {channel}")

    def __enter__(self):
        try:
            import serial
            self.serial = serial.Serial(self.port, baudrate=9600, bytesize=8, parity="N", stopbits=1,
                                        timeout=0.5, write_timeout=0.5)
            self.serial.dtr = True
            self.serial.rts = True
            time.sleep(0.05)
            self._send(1)
            self._send(3, self.intensity)
            time.sleep(0.25)
            return self
        except Exception as error:
            self.__exit__(None, None, None)
            raise LightError(f"Could not turn on the inspection light: {error}") from error

    def __exit__(self, *_exc):
        if self.serial is None:
            return
        try:
            self._send(2)
        finally:
            self.serial.close()
            self.serial = None
