"""Prove the six-head ONNX is numerically the same model as the standard export.

The native decoder consumes three box and three class tensors, so the split graph
has to reproduce the fused output exactly -- not approximately, and not just in
tensor shape. This decodes both graphs on the real prepared validation images and
compares them.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from analysis_paths import DATA

EXPORT = DATA / "export"
DATA = DATA / "dataset" / "v3-prod"
STRIDES = (8, 16, 32)
GRIDS = (80, 40, 20)


def load(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def decode_six_heads(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Decode the native contract: (l,t,r,b) distances in stride units, sigmoid on classes."""
    boxes, scores = [], []
    for level, (stride, grid) in enumerate(zip(STRIDES, GRIDS)):
        raw = outputs[f"bbox_{level}"][0]  # 4, grid, grid
        classes = outputs[f"class_prob_{level}"][0]  # C, grid, grid
        columns, rows = np.meshgrid(np.arange(grid), np.arange(grid))
        anchor_x = (columns + 0.5) * stride
        anchor_y = (rows + 0.5) * stride
        left, top, right, bottom = raw[0], raw[1], raw[2], raw[3]
        x1 = anchor_x - left * stride
        y1 = anchor_y - top * stride
        x2 = anchor_x + right * stride
        y2 = anchor_y + bottom * stride
        boxes.append(np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4))
        scores.append(1.0 / (1.0 + np.exp(-classes)).reshape(classes.shape[0], -1).T)
    return np.concatenate(boxes), np.concatenate(scores)


def main() -> None:
    standard = load(EXPORT / "model.onnx")
    six = load(EXPORT / "model_sima_6out.onnx")
    images = sorted((DATA / "images" / "val").glob("*.png"))[:12]
    worst_box = 0.0
    worst_score = 0.0
    for path in images:
        image = cv2.imread(str(path))
        # The prepared images are already the exact model input; the graph wants RGB /255.
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        fused = standard.run(None, {"images": tensor})[0][0]  # 4+C, 8400
        fused_boxes = np.stack([fused[0] - fused[2] / 2, fused[1] - fused[3] / 2,
                                fused[0] + fused[2] / 2, fused[1] + fused[3] / 2], axis=1)
        fused_scores = fused[4:].T  # already sigmoided in the graph

        names = [o.name for o in six.get_outputs()]
        split = decode_six_heads(dict(zip(names, six.run(None, {"images": tensor}))))
        boxes, scores = split

        worst_box = max(worst_box, float(np.abs(boxes - fused_boxes).max()))
        worst_score = max(worst_score, float(np.abs(scores - fused_scores).max()))

    print(f"images compared      : {len(images)}")
    print(f"max box difference   : {worst_box:.4f} model pixels")
    print(f"max score difference : {worst_score:.6f}")
    print(f"gate                 : <= 1.0 px and <= 0.001")
    print("result               :", "PASS" if worst_box <= 1.0 and worst_score <= 0.001 else "FAIL")


if __name__ == "__main__":
    main()
