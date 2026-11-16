"""FP32 parity for the 1280 export: six-head decode vs the fused graph."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from analysis_paths import DATA

EXPORT = DATA / "export-1280"
DATA = DATA / "dataset" / "v4-1280"
SIDE = 1280


def decode(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    boxes, scores = [], []
    for level, stride in enumerate((8, 16, 32)):
        grid = SIDE // stride
        raw = outputs[f"bbox_{level}"][0]
        classes = outputs[f"class_prob_{level}"][0]
        columns, rows = np.meshgrid(np.arange(grid), np.arange(grid))
        ax, ay = (columns + 0.5) * stride, (rows + 0.5) * stride
        boxes.append(np.stack([ax - raw[0] * stride, ay - raw[1] * stride,
                               ax + raw[2] * stride, ay + raw[3] * stride], axis=-1).reshape(-1, 4))
        scores.append((1.0 / (1.0 + np.exp(-classes))).reshape(classes.shape[0], -1).T)
    return np.concatenate(boxes), np.concatenate(scores)


def main() -> None:
    fused = ort.InferenceSession(str(EXPORT / "model_1280.onnx"), providers=["CPUExecutionProvider"])
    six = ort.InferenceSession(str(EXPORT / "model_1280_sima_6out.onnx"), providers=["CPUExecutionProvider"])
    images = sorted((DATA / "images" / "val").glob("*.png"))[:8]
    worst_box = worst_score = 0.0
    for path in images:
        image = cv2.imread(str(path))
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = fused.run(None, {"images": tensor})[0][0]
        ref_boxes = np.stack([out[0] - out[2] / 2, out[1] - out[3] / 2,
                              out[0] + out[2] / 2, out[1] + out[3] / 2], axis=1)
        boxes, scores = decode(dict(zip([o.name for o in six.get_outputs()],
                                        six.run(None, {"images": tensor}))))
        worst_box = max(worst_box, float(np.abs(boxes - ref_boxes).max()))
        worst_score = max(worst_score, float(np.abs(scores - out[4:].T).max()))
    print(f"images {len(images)}  max box diff {worst_box:.4f} px  max score diff {worst_score:.6f}")
    print("result:", "PASS" if worst_box <= 1.0 and worst_score <= 0.001 else "FAIL")


if __name__ == "__main__":
    main()
