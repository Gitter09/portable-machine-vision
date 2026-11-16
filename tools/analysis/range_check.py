"""Measure the real value ranges of the decoded ONNX outputs."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from analysis_paths import DATA

EXPORT = DATA / "export-1280"
DATA = DATA / "dataset" / "v4-1280"


def main() -> None:
    session = ort.InferenceSession(str(EXPORT / "model_1280_decoded_6out.onnx"),
                                   providers=["CPUExecutionProvider"])
    image = cv2.imread(str(sorted((DATA / "images" / "val").glob("*.png"))[0]))
    tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"images": tensor})))
    for name in sorted(outputs):
        values = outputs[name]
        print(f"{name:<24} min {values.min():>10.3f}  max {values.max():>10.3f}  "
              f"abs-max {np.abs(values).max():>10.3f}  mean {values.mean():>8.3f}")


if __name__ == "__main__":
    main()
