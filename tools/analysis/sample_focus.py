"""Sample the live feed several times and report a stable sharpness figure.

One reading on a rotating plate can be an outlier, so take several across
different board poses and report the median against the training baseline.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

URL = "http://172.21.30.120:8080/frame.jpg"
OUT = DATA / "newset" / "focus_samples"
SAMPLES = 6
GAP = 2.0


def grab(path: Path) -> np.ndarray | None:
    try:
        data = urllib.request.urlopen(URL, timeout=10).read()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  fetch failed: {exc}")
        return None
    path.write_bytes(data)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    values = []
    for index in range(SAMPLES):
        image = grab(OUT / f"s{index}.jpg")
        if image is None:
            continue
        value = float(cv2.Laplacian(image, cv2.CV_64F).var())
        values.append(value)
        print(f"  sample {index}: {value:6.1f}")
        if index != SAMPLES - 1:
            time.sleep(GAP)

    if not values:
        print("no samples captured")
        return

    values.sort()
    print(f"\nmedian   : {np.median(values):6.1f}")
    print(f"min..max : {values[0]:.1f} .. {values[-1]:.1f}")
    print(f"training :   57.1")
    print(f"ratio    : {np.median(values) / 57.1:.2f}")


if __name__ == "__main__":
    main()
