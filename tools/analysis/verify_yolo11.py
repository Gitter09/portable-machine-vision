"""Verify the Modalix six-head export against Ultralytics' fused output.

The decoder's ``yolov8`` path applies the DFL softmax itself, so the check has to
reproduce that maths: reshape each 64-channel box head into 4 x 16 bins, softmax
over the bins, expectation gives (l, t, r, b) in stride units, then the usual
anchor/stride decode to centre-based pixels.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from analysis_paths import DATA

TRAIN = DATA
EXPORT = TRAIN / "export-yolo11"
DATA = TRAIN / "dataset" / "v4-1280"
SIDE = 1280
STRIDES = (8, 16, 32)


def dfl_decode(bbox: np.ndarray, stride: int, grid: int) -> np.ndarray:
    """(1, 64, g, g) DFL logits -> (4, g*g) centre-based pixel boxes."""
    bins = bbox[0].reshape(4, 16, grid, grid)
    bins = bins - bins.max(axis=1, keepdims=True)
    probability = np.exp(bins)
    probability /= probability.sum(axis=1, keepdims=True)
    distances = (probability * np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)).sum(axis=1)

    columns = (np.arange(grid, dtype=np.float32) + 0.5) * stride
    anchor_x = np.repeat(columns.reshape(1, 1, grid), grid, axis=1)
    anchor_y = np.repeat(columns.reshape(1, grid, 1), grid, axis=2)

    left, top, right, bottom = distances[0], distances[1], distances[2], distances[3]
    x1 = anchor_x - left * stride
    y1 = anchor_y - top * stride
    x2 = anchor_x + right * stride
    y2 = anchor_y + bottom * stride
    cx = ((x1 + x2) / 2).reshape(-1)
    cy = ((y1 + y2) / 2).reshape(-1)
    width = (x2 - x1).reshape(-1)
    height = (y2 - y1).reshape(-1)
    return np.stack([cx, cy, width, height], axis=0)


def main() -> None:
    fused = ort.InferenceSession(str(EXPORT / "yolo11n_fused.onnx"), providers=["CPUExecutionProvider"])
    six = ort.InferenceSession(str(EXPORT / "yolo11n_modalix_6out.onnx"), providers=["CPUExecutionProvider"])
    images = sorted((DATA / "images" / "val").glob("*.png"))[:6]
    worst_box = worst_score = 0.0
    for path in images:
        image = cv2.imread(str(path))
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        reference = fused.run(None, {"images": tensor})[0][0]
        outputs = dict(zip([o.name for o in six.get_outputs()], six.run(None, {"images": tensor})))

        boxes = np.concatenate(
            [dfl_decode(outputs[f"bbox_{level}"], stride, SIDE // stride)
             for level, stride in enumerate(STRIDES)], axis=1)
        scores = np.concatenate(
            [outputs[f"class_sigmoid_{level}"][0].reshape(4, -1) for level in range(3)], axis=1)

        ref_boxes = np.stack([reference[0] - reference[2] / 2, reference[1] - reference[3] / 2,
                              reference[0] + reference[2] / 2, reference[1] + reference[3] / 2], axis=1).T
        ours = np.stack([boxes[0] - boxes[2] / 2, boxes[1] - boxes[3] / 2,
                         boxes[0] + boxes[2] / 2, boxes[1] + boxes[3] / 2], axis=1).T
        worst_box = max(worst_box, float(np.abs(ours - ref_boxes).max()))
        worst_score = max(worst_score, float(np.abs(scores - reference[4:]).max()))

    print(f"images {len(images)}")
    print(f"max box difference   : {worst_box:.4f} model pixels")
    print(f"max score difference : {worst_score:.6f}")
    print("result:", "PASS" if worst_box <= 1.0 and worst_score <= 0.001 else "FAIL")


if __name__ == "__main__":
    main()
