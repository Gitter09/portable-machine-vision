#!/usr/bin/env python3
"""Warm, isolated board worker for the persistent Admin inspection dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from live_camera_benchmark import HardwareDetector
from triggered_camera import TriggeredCamera as Camera
from camera_lease import CameraLease
from fault_evidence import inspection_error
from preprocessing.state_counts import summarise


MAX_HEADER = 1024 * 1024


def event(root: Path, name: str, **fields) -> None:
    record = {"monotonic_ns": time.monotonic_ns(), "event": name, **fields}
    with (root / "persistent_worker_events.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, separators=(",", ":")) + "\n")


def recv_exact(channel: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = channel.recv(size - len(data))
        if not part:
            raise EOFError("control connection closed")
        data.extend(part)
    return bytes(data)


def recv_message(channel: socket.socket) -> tuple[dict, bytes]:
    header_size = struct.unpack("!I", recv_exact(channel, 4))[0]
    if header_size <= 0 or header_size > MAX_HEADER:
        raise ValueError("invalid header length")
    header = json.loads(recv_exact(channel, header_size).decode("utf-8"))
    payload_size = int(header.get("payload_length", 0))
    if payload_size < 0 or payload_size > 16 * 1024 * 1024:
        raise ValueError("invalid payload length")
    return header, recv_exact(channel, payload_size)


def send_message(channel: socket.socket, kind: str, request_id: str, body=None, payload=b"") -> None:
    header = {
        "type": kind,
        "request_id": request_id,
        "payload_length": len(payload),
        "body": body or {},
    }
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    channel.sendall(struct.pack("!I", len(encoded)) + encoded + payload)


def encode_display(frame: np.ndarray) -> tuple[bytes, dict]:
    started = time.perf_counter()
    display = cv2.resize(frame, (1600, 1128), interpolation=cv2.INTER_AREA)
    resized = time.perf_counter()
    ok, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
    finished = time.perf_counter()
    if not ok:
        raise RuntimeError("display JPEG encoding failed")
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    # Fixed product region: this is the screw area after the 2:1 display resize.
    roi = gray[625:825, 350:1000]
    return encoded.tobytes(), {
        "display_resize": (resized - started) * 1000,
        "jpeg_encoding": (finished - resized) * 1000,
        "display_mean_luminance": float(np.mean(gray)),
        "display_saturated_percent": float(np.mean(gray >= 250) * 100.0),
        "roi_mean_luminance": float(np.mean(roi)),
        "roi_saturated_percent": float(np.mean(roi >= 250) * 100.0),
    }


def board_health() -> dict:
    temperatures = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temperatures.append(float(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return {
        "pid": os.getpid(),
        "temperature_c_max": max(temperatures) if temperatures else None,
    }


def serve(connection: socket.socket, detector: HardwareDetector, camera: Camera) -> bool:
    send_message(connection, "ready", "", {"camera": True, "camera_ip": camera.camera_ip, "model": True, "camera_mode": camera.mode,
                                            "release_id": detector.release_id, "classes": detector.labels,
                                            "alignment_fault_evidence_version": 1,
                 "capture_freshness_version": getattr(camera, "freshness_version", 0), **board_health()})
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="display-encoder") as encoder:
        while True:
            command, _ = recv_message(connection)
            kind = command.get("type")
            request_id = str(command.get("request_id", ""))
            if kind == "shutdown":
                camera.close()
                send_message(connection, "stopped", request_id,
                             {"release_id": detector.release_id, "camera_released": True})
                return False
            if kind == "health":
                send_message(connection, "ready", request_id, {"release_id": detector.release_id, **board_health()})
                continue
            if kind != "capture" or not request_id:
                send_message(connection, "error", request_id, {"code": "BAD_COMMAND"})
                continue
            frame = None
            camera_timings = {}
            try:
                parallel_display = bool(command.get("body", {}).get("parallel_display", True))
                capture_started = time.perf_counter()
                frame, camera_timings = camera.capture_timed()
                captured = time.perf_counter()
                send_message(connection, "captured", request_id, {
                    "capture_ms": (captured - capture_started) * 1000,
                    **camera_timings,
                })

                if parallel_display:
                    encode_future = encoder.submit(encode_display, frame)
                    detections, metadata, details = detector.infer(frame)
                    inferred = time.perf_counter()
                    jpeg, display_details = encode_future.result()
                else:
                    detections, metadata, details = detector.infer(frame)
                    inferred = time.perf_counter()
                    jpeg, display_details = encode_display(frame)
                finished = time.perf_counter()
                aligned = metadata.get("aligned")
                if aligned is None:
                    aligned = metadata.get("window") in {"located", "fixed", "fallback"}
                presence = metadata.get("presence_similarity")
                result = {
                    "status": "ok",
                    "screw_count": len(detections),
                    "detection_count": len(detections),
                    "detections": detections,
                    "summary": summarise(detections),
                    "instance_count": int(metadata.get("instance_count", 1)),
                    "possibly_merged": int(metadata.get("possibly_merged", 0)),
                    "preprocessor": {
                        "name": metadata.get("preprocessor"),
                        "profile": metadata.get("profile"),
                        "window": metadata.get("window"),
                        "presence_similarity": (None if presence is None else float(presence)),
                        "presence_threshold": (None if metadata.get("presence_threshold") is None
                                               else float(metadata["presence_threshold"])),
                    },
                    "alignment": {
                        "aligned": bool(aligned),
                        "orb_matches": int(metadata.get("orb_matches", 0)),
                        "orb_inliers": int(metadata.get("orb_inliers", 0)),
                    },
                    "source_width": 3200,
                    "source_height": 2256,
                    "display_width": 1600,
                    "display_height": 1128,
                    "image_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "timings_ms": {
                        **camera_timings,
                        **details["timings"],
                        **display_details,
                        "parallel_result_wait": (finished - inferred) * 1000,
                        "capture_to_result": (finished - capture_started) * 1000,
                        "display_encoding_mode": "parallel" if parallel_display else "sequential",
                    },
                    "acquisition": dict(getattr(camera,"last_evidence",{})),
                    "health": board_health(),
                }
                send_message(connection, "result", request_id, result, jpeg)
            except Exception as error:
                body, payload = inspection_error(error, frame)
                body["acquisition"] = dict(getattr(camera,"last_evidence",{}))
                body["camera_timings_ms"] = camera_timings
                if error.__class__.__name__ == "CameraFreshnessError":
                    body.update(code="CAMERA_FRESHNESS_FAILED",retryable=False)
                send_message(connection, "error", request_id, body, payload)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/sima/Desktop/test_model"))
    parser.add_argument("--bind", default="192.168.1.20")
    parser.add_argument("--admin-ip", default="192.168.1.17")
    parser.add_argument("--camera", default="192.168.1.10")
    parser.add_argument("--camera-mode", choices=("5fps", "10fps-jumbo", "10fps-jumbo-trigger"), default="5fps")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--lease-port", type=int, default=5006)
    args = parser.parse_args()

    detector = HardwareDetector(args.root)
    camera = Camera(args.camera, args.camera_mode)
    if not detector.release_id:
        camera.close()
        raise RuntimeError("SDK release ID is required for a leased board worker")
    lease = CameraLease(args.bind, args.admin_ip, detector.release_id, args.lease_port)
    try:
        lease.start()
        # Warm-up only: no part is guaranteed to be in view yet, so a rejection
        # from the part locator must not abort startup. Prime with synthetic pixels.
        detector.warmup()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.port))
            server.listen(1)
            server.settimeout(1.0)
            print("PERSISTENT_BOARD_READY", flush=True)
            event(args.root, "listening", bind=args.bind, admin_ip=args.admin_ip)
            while True:
                if lease.expired():
                    event(args.root, "lease_expired")
                    break
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(15.0)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    event(args.root, "connection", peer=address[0])
                    if address[0] != args.admin_ip:
                        event(args.root, "address_denied", peer=address[0])
                        send_message(connection, "error", "", {"code": "ADDRESS_DENIED", "peer": address[0]})
                        continue
                    try:
                        if not serve(connection, detector, camera):
                            event(args.root, "shutdown")
                            break
                    except (EOFError, ConnectionError, OSError):
                        pass
    finally:
        lease.close()
        camera.close()


if __name__ == "__main__":
    main()
