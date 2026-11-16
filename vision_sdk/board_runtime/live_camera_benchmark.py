#!/usr/bin/env python3
"""Isolated live-camera benchmark for the Modalix screw detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from mpk_parser import Parser
from sima.operations import SimaDetessDequant, SimaModelRunner, SimaQuantTess
from preprocessing.base import Box
from preprocessing.factory import resolve_preprocessor


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class HardwareDetector:
    def __init__(self, root: Path, confidence: float = 0.30):
        labels_path = root / "labels.json"
        self.labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else ["target_screw"]
        if not isinstance(self.labels, list) or not self.labels or any(not isinstance(label, str) for label in self.labels):
            raise ValueError("labels.json must contain a non-empty list of class names")
        release_path = root / "vision_release.json"
        self.release_id = json.loads(release_path.read_text(encoding="utf-8"))["release_id"] if release_path.is_file() else None
        parser = Parser(str(root / "yolov8_mpk.json"))
        self.quant = SimaQuantTess(parser)
        self.quant.configure({})
        self.runner = SimaModelRunner(parser)
        self.runner.configure({"model_path": str(root / "yolov8_stage1_mla.elf")})
        self.dequant = SimaDetessDequant(parser)
        setup_path = root / "vision_setup.json"
        setup = json.loads(setup_path.read_text(encoding="utf-8")) if setup_path.is_file() else {}
        reference = root / "reference.png"
        if not reference.is_file():
            reference = root / "preprocessing/Preprocessing_Fixed_Reference/058e928d2154443e9aac380c7b11d3d5.png"
        # The release states which conditioning it was built with, and the board
        # builds the same one. Guessing here would be a train/serve skew.
        preprocessor_name = setup.get("preprocessor", "fixed_reference_realtime")
        profile_path = setup.get("preprocessing_profile") or None
        flatfield_lut = setup.get("flatfield_lut") or None
        if flatfield_lut and not Path(flatfield_lut).is_absolute():
            flatfield_lut = str(root / flatfield_lut)
        if profile_path and not Path(profile_path).is_absolute():
            profile_path = str(root / profile_path)
        self.preprocessor = resolve_preprocessor(
            preprocessor_name,
            reference_image=reference if reference.is_file() else None,
            roi=tuple(setup["roi"]) if "roi" in setup else None,
            profile_path=profile_path,
            flatfield_lut=flatfield_lut,
        )
        self.preprocessor_name = preprocessor_name
        self.confidence = confidence
        grids, strides = [], []
        for size, stride in ((80, 8), (40, 16), (20, 32)):
            y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            grids.append(np.stack((x.ravel() + 0.5, y.ravel() + 0.5), axis=0))
            strides.append(np.full(size * size, stride, dtype=np.float32))
        self.grid = np.concatenate(grids, axis=1)[None].astype(np.float32)
        self.strides = np.concatenate(strides)[None, None].astype(np.float32)

    def _execute(self, hw_input: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        q0 = time.perf_counter()
        quantized = self.quant.run(hw_input)
        q1 = time.perf_counter()
        accelerated = self.runner.run(quantized)
        q2 = time.perf_counter()
        outputs = self.dequant.run(accelerated)
        q3 = time.perf_counter()
        expected = ((80, 80, 4), (40, 40, 4), (20, 20, 4),
                    (80, 80, len(self.labels)), (40, 40, len(self.labels)), (20, 20, len(self.labels)))
        for index, (height, width, channels) in enumerate(expected):
            shape = tuple(outputs[index].shape)
            if shape != (1, height, width, channels):
                raise RuntimeError(
                    f"SiMa output {index} has shape {shape}, expected {(1, height, width, channels)}; "
                    "the compiled graph does not match this release's class count or input size"
                )
        timings = {
            "quantization": (q1 - q0) * 1000,
            "mla_execution": (q2 - q1) * 1000,
            "dequantization": (q3 - q2) * 1000,
        }
        return outputs, timings

    def warmup(self) -> None:
        """Prime the MLA without requiring a part in view.

        The product-agnostic preprocessor rejects a frame that differs from the
        stored empty fixture, so an empty warm-up frame must not be a failure.
        """
        try:
            self.infer(np.zeros((2256, 3200, 3), dtype=np.uint8), require_alignment=False)
            return
        except Exception:
            pass
        self._execute(np.zeros((640, 640, 3), dtype=np.float32))

    def infer(self, frame: np.ndarray, require_alignment: bool = True) -> tuple[list[dict], dict, dict]:
        t0 = time.perf_counter()
        tensor, model_image, metadata = self.preprocessor.preprocess(
            frame, require_alignment=require_alignment
        )
        t1 = time.perf_counter()
        hw_input = tensor[0].transpose(1, 2, 0).astype(np.float32)
        outputs, model_timings = self._execute(hw_input)
        t2 = time.perf_counter()

        bbox = np.concatenate(
            [np.transpose(outputs[i], (0, 3, 1, 2)).reshape(1, 4, outputs[i].shape[1] * outputs[i].shape[2])
             for i in (0, 1, 2)],
            axis=2,
        )
        class_count = len(self.labels)
        logits = np.concatenate(
            [np.transpose(outputs[i], (0, 3, 1, 2)).reshape(1, class_count, outputs[i].shape[1] * outputs[i].shape[2])
             for i in (3, 4, 5)],
            axis=2,
        )
        low, high = bbox[:, :2], bbox[:, 2:]
        center = ((self.grid - low) + (self.grid + high)) / 2.0
        size = (low + high)
        scores = 1.0 / (1.0 + np.exp(-logits))
        best_class = scores.argmax(axis=1)[0]
        best_score = scores.max(axis=1, keepdims=True)
        predictions = np.concatenate(
            (np.concatenate((center, size), axis=1) * self.strides, best_score),
            axis=1,
        )[0].T
        selected = predictions[:, 4] >= self.confidence
        best_class = best_class[selected]
        predictions = predictions[selected]
        detections: list[dict] = []
        if len(predictions):
            boxes = np.empty((len(predictions), 4), dtype=np.float32)
            boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
            boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
            boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
            boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2
            keep = []
            for class_id in range(class_count):
                indices = np.flatnonzero(best_class == class_id)
                keep.extend(int(indices[i]) for i in nms(boxes[indices], predictions[indices, 4]))
            for index in sorted(keep, key=lambda i: predictions[i, 4], reverse=True):
                x1, y1, x2, y2 = boxes[index]
                source = self.preprocessor.inverse_map(
                    Box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)), metadata
                )
                detections.append(
                    {"confidence": float(predictions[index, 4]), "class_id": int(best_class[index]),
                     "class_name": self.labels[int(best_class[index])], "box_source": source.as_dict()}
                )
        t3 = time.perf_counter()
        timings = dict(metadata.get("stage_timings_ms", {}))
        timings.update(model_timings)
        timings.update({
            "preprocessing": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "decoding_nms_inverse_mapping": (t3 - t2) * 1000,
            "postprocessing": (t3 - t2) * 1000,
        })
        return detections, metadata, {"timings": timings, "model_image": model_image}


class Camera:
    def __init__(self, camera_ip: str, mode: str = "5fps"):
        Gst.init(None)
        self.camera_ip = camera_ip
        if mode not in {"5fps", "10fps-jumbo", "10fps-jumbo-trigger"}:
            raise ValueError(f"unsupported camera mode {mode}")
        self.mode = mode
        self.software_trigger = mode == "10fps-jumbo-trigger"
        if mode == "5fps":
            transport = "auto-packet-size=true"
            features = "PixelFormat=BayerRG8"
        else:
            transport = "auto-packet-size=false packet-size=9000"
            trigger = " TriggerMode=On TriggerSource=Software" if self.software_trigger else " TriggerMode=Off"
            features = "PixelFormat=BayerRG8 AcquisitionFrameRateEnable=true AcquisitionFrameRate=10" + trigger
        description = (
            f'aravissrc name=source camera-name={camera_ip} {transport} do-timestamp=true '
            f'features="{features}" ! '
            "video/x-bayer,format=rggb,width=3200,height=2256 ! "
            "bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink max-buffers=1 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(description)
        self.sink = self.pipeline.get_by_name("sink")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("camera pipeline failed to enter PLAYING state")
        self.aravis_camera = self.pipeline.get_by_name("source").get_property("camera")

    def drain(self) -> None:
        while self.sink.emit("try-pull-sample", 0) is not None:
            pass

    def capture_timed(self, timeout_seconds: float = 2.0) -> tuple[np.ndarray, dict]:
        waited = time.perf_counter()
        trigger_started = time.perf_counter()
        if self.software_trigger:
            if self.aravis_camera is None or not hasattr(self.aravis_camera, "software_trigger"):
                raise RuntimeError("software trigger API is unavailable")
            self.aravis_camera.software_trigger()
        trigger_finished = time.perf_counter()
        sample = self.sink.emit("try-pull-sample", int(timeout_seconds * Gst.SECOND))
        received = time.perf_counter()
        if sample is None:
            message = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            raise RuntimeError("camera frame timeout" if message is None else str(message.parse_error()))
        buffer = sample.get_buffer()
        map_started = time.perf_counter()
        ok, mapping = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map camera buffer")
        try:
            expected = 3200 * 2256 * 3
            if mapping.size != expected:
                raise RuntimeError(f"unexpected camera buffer size {mapping.size}, expected {expected}")
            frame = np.frombuffer(mapping.data, dtype=np.uint8).reshape(2256, 3200, 3).copy()
            copied = time.perf_counter()
            return frame, {
                "software_trigger_transmission": (trigger_finished - trigger_started) * 1000,
                "exposure_frame_wait_and_camera_transfer": (received - waited) * 1000,
                "buffer_mapping_and_memory_copy": (copied - map_started) * 1000,
            }
        finally:
            buffer.unmap(mapping)

    def capture(self, timeout_seconds: float = 2.0) -> np.ndarray:
        return self.capture_timed(timeout_seconds)[0]

    def close(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


def draw_overlay(frame: np.ndarray, detections: list[dict], path: Path) -> None:
    overlay = frame.copy()
    for detection in detections:
        box = detection["box_source"]
        p1 = (round(box["x"]), round(box["y"]))
        p2 = (round(box["x"] + box["width"]), round(box["y"] + box["height"]))
        cv2.rectangle(overlay, p1, p2, (0, 255, 0), 3)
    cv2.imwrite(str(path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/sima/Desktop/test_model"))
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--expected", type=int, default=6)
    parser.add_argument("--camera", default="192.168.1.10")
    args = parser.parse_args()
    detector = HardwareDetector(args.root)
    camera = Camera(args.camera)
    rows: list[dict] = []
    try:
        for _ in range(3):
            camera.capture()
        warm_frame = camera.capture()
        try:
            detector.infer(warm_frame)
        except Exception:
            detector.warmup()
        for cycle in range(1, args.cycles + 1):
            camera.drain()
            started = time.perf_counter()
            frame = camera.capture()
            captured = time.perf_counter()
            detections, metadata, details = detector.infer(frame)
            finished = time.perf_counter()
            aligned = metadata.get("aligned")
            if aligned is None:
                aligned = metadata.get("window") in {"located", "fixed", "fallback"}
            row = {
                "cycle": cycle,
                "detections": len(detections),
                "correct_count": len(detections) == args.expected,
                "aligned": bool(aligned),
                "orb_matches": int(metadata.get("orb_matches", 0)),
                "orb_inliers": int(metadata.get("orb_inliers", 0)),
                "capture_wait_ms": (captured - started) * 1000,
                **details["timings"],
                "e2e_ms": (finished - started) * 1000,
                "confidences": [d["confidence"] for d in detections],
            }
            rows.append(row)
            if cycle == 1:
                cv2.imwrite(str(args.root / "live_benchmark_first.jpg"), frame)
                draw_overlay(frame, detections, args.root / "live_benchmark_first_overlay.jpg")
            if cycle % 10 == 0:
                print(json.dumps({k: row[k] for k in ("cycle", "detections", "aligned", "e2e_ms")}), flush=True)
    finally:
        camera.close()

    timing_names = ("capture_wait_ms", "preprocessing", "inference", "postprocessing", "e2e_ms")
    summary = {
        "cycles": args.cycles,
        "expected_screws": args.expected,
        "exact_count_cycles": sum(row["correct_count"] for row in rows),
        "exact_count_accuracy_percent": 100.0 * sum(row["correct_count"] for row in rows) / len(rows),
        "aligned_cycles": sum(row["aligned"] for row in rows),
        "detection_count_histogram": {
            str(count): sum(row["detections"] == count for row in rows)
            for count in sorted({row["detections"] for row in rows})
        },
        "latency_ms": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "p50": percentile([row[name] for row in rows], 50),
                "p95": percentile([row[name] for row in rows], 95),
                "p99": percentile([row[name] for row in rows], 99),
                "max": max(row[name] for row in rows),
            }
            for name in timing_names
        },
        "latency_gate_ms": 200.0,
        "serial_p99_under_gate": percentile([row["e2e_ms"] for row in rows], 99) < 200.0,
    }
    output = {"summary": summary, "cycles": rows}
    path = args.root / "live_camera_benchmark_100.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary), flush=True)
    print(f"RESULT {path}", flush=True)


if __name__ == "__main__":
    main()

