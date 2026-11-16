"""Compare sharpness of the new camera position against the training frames."""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

TRAIN = DATA
NEW = TRAIN / "newset" / "frame.jpg"
OLD = TRAIN / "dataset" / "v4-1280" / "images" / "train"


def laplacian(path: Path, crop=None) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if crop:
        x, y, w, h = crop
        image = image[y:y + h, x:x + w]
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def main() -> None:
    print(f"new frame (full)        : {laplacian(NEW):8.1f}")
    print(f"new frame (board region): {laplacian(NEW, (960, 280, 140, 190)):8.1f}")
    print()

    olds = sorted(OLD.glob("*.png"))[:6]
    values = []
    for path in olds:
        # training images are 1280x1280 letterboxed; the frame occupies rows 280..1000
        full = laplacian(path, (0, 280, 1280, 720))
        values.append(full)
        print(f"{path.name}: full-frame sharpness {full:8.1f}")
    print()
    print(f"training frames: median {np.median(values):.1f}")
    ratio = laplacian(NEW) / max(1e-6, float(np.median(values)))
    print(f"new / training sharpness ratio: {ratio:.2f}")


if __name__ == "__main__":
    main()
