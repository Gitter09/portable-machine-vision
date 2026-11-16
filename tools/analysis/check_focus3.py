"""Sharpness reading for the latest focus adjustment, against the history."""

import sys
from pathlib import Path

import cv2
from analysis_paths import DATA

D = DATA / "newset"


def sharp(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "focus3.jpg"
    now = sharp(D / name)
    history = [("training baseline", 57.1), ("best so far", 49.5),
               ("earlier", 42.7), ("previous reading", 26.7)]
    print(f"this reading        : {now:6.1f}   ({name})")
    for label, value in history:
        print(f"{label:<20}: {value:6.1f}")
    print(f"ratio to training   : {now / 57.1:.2f}")

    a = cv2.imread(str(D / name))
    b = cv2.imread(str(D / name.replace(".jpg", "b.jpg")))
    if a is not None and b is not None and a.shape == b.shape:
        diff = cv2.absdiff(a, b)
        changed = int((diff.max(axis=2) > 25).sum())
        print(f"scene movement 3s   : {changed} px changed"
              f"{'  (static)' if changed == 0 else ''}")


if __name__ == "__main__":
    main()
