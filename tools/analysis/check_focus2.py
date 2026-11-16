"""Compare sharpness and scene movement for the latest focus adjustment."""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

D = DATA / "newset"


def laplacian(path: Path, crop=None) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    if crop:
        x, y, w, h = crop
        image = image[y:y + h, x:x + w]
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def main() -> None:
    now = laplacian(D / "focus2.jpg")
    print(f"focus now           : {now:6.1f}")
    print(f"previous reading    :   49.5")
    print(f"reading before that :   42.7")
    print(f"training baseline   :   57.1")
    print(f"ratio to training   : {now / 57.1:.2f}")

    a = cv2.imread(str(D / "focus2.jpg"))
    b = cv2.imread(str(D / "focus2b.jpg"))
    if a is not None and b is not None and a.shape == b.shape:
        diff = cv2.absdiff(a, b)
        changed = int((diff.max(axis=2) > 25).sum())
        print(f"\nscene movement over 3 s: {changed} px changed"
              f"{'  (static)' if changed == 0 else ''}")


if __name__ == "__main__":
    main()
