"""Verify the decoded six-head graph against Ultralytics' fused output."""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from analysis_paths import DATA

EXPORT = DATA / "export-1280"
DATA = DATA / "dataset" / "v4-1280"
SIZE = 1280


def main() -> None:
    fused = ort.InferenceSession(str(EXPORT / "model_1280.onnx"), providers=["CPUExecutionProvider"])
    decoded = ort.InferenceSession(str(EXPORT / "model_1280_decoded_6out.onnx"), providers=["CPUExecutionProvider"])
    images = sorted((DATA / "images" / "val").glob("*.png"))[:5]
    worst_box = worst_score = 0.0
    for path in images:
        image = cv2.imread(str(path))
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        reference = fused.run(None, {"images": tensor})[0][0]  # 4+C, 33600
        outputs = dict(zip([o.name for o in decoded.get_outputs()], decoded.run(None, {"images": tensor})))

        boxes = np.concatenate([outputs[f"decoded_bbox_{level}"][0].reshape(4, -1) for level in range(3)], axis=1)
        scores = np.concatenate([outputs[f"decoded_class_prob_{level}"][0].reshape(4, -1) for level in range(3)], axis=1)

        worst_box = max(worst_box, float(np.abs(boxes - reference[:4]).max()))
        worst_score = max(worst_score, float(np.abs(scores - reference[4:]).max()))
        conf = scores.max(axis=0)
        print(f"{path.name}: max conf {conf.max():.3f} | boxes at >0.5: {(conf > 0.5).sum()}")

    print()
    print(f"max |box| difference   : {worst_box:.4f} model pixels")
    print(f"max |score| difference : {worst_score:.6f}")
    print("result:", "PASS" if worst_box <= 1.0 and worst_score <= 0.001 else "FAIL")


if __name__ == "__main__":
    main()
