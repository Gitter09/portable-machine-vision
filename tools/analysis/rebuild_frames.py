"""Rebuild the captured frames from the diagnostic packet log.

Each record is a big-endian wall clock, a length, then the buffer. The final
pipeline stage encodes to H.264, so the concatenated payload is a playable
elementary stream.
"""

import struct
from pathlib import Path
from analysis_paths import DATA

D = DATA / "verify-r24"
OUT = D / "frames.h264"


def main() -> None:
    data = (D / "packets.bin").read_bytes()
    offset = 0
    payloads = []
    while offset + 12 <= len(data):
        _, length = struct.unpack_from(">QI", data, offset)
        offset += 12
        if length > len(data) - offset:
            break
        payloads.append(data[offset:offset + length])
        offset += length

    # keep only access units that start with a start code
    body = b"".join(p for p in payloads if p[:3] == b"\x00\x00\x01" or p[:4] == b"\x00\x00\x00\x01")
    OUT.write_bytes(body)
    kept = sum(1 for p in payloads if p[:1] == b"\x00")
    print(f"records {len(payloads)}  kept {kept}  bytes {len(body):,}")


if __name__ == "__main__":
    main()
