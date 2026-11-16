"""Measure what the new camera position actually gives us."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from analysis_paths import DATA

TRAIN = DATA
FRAME = TRAIN / "newset" / "frame.jpg"


def main() -> None:
    image = cv2.imread(str(FRAME))
    print("frame:", image.shape[1], "x", image.shape[0])

    model = YOLO(str(TRAIN / "runs" / "train-yolo11n-1280" / "weights" / "best.pt"))
    result = model.predict(str(FRAME), imgsz=1280, conf=0.25, verbose=False)[0]

    names = result.names
    print(f"\ndetections above 0.25: {len(result.boxes)}")
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        width, height = x2 - x1, y2 - y1
        edge = "  <-- touches frame edge" if (x1 <= 2 or y1 <= 2 or x2 >= image.shape[1] - 2 or y2 >= image.shape[0] - 2) else ""
        print(f"  {names[int(box.cls)]:<15} conf {float(box.conf):.3f}  "
              f"box ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  {width:.0f} x {height:.0f} px{edge}")

    # board silhouette radius about the plate axis, for framing margin
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 25)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)), reverse=True)
    print("\nlargest saturated components (board candidates):")
    for area, index in areas[:4]:
        x, y, w, h, _ = stats[index]
        print(f"  area {area:>6}  bbox ({x},{y}) {w}x{h}  "
              f"clipped={'yes' if x == 0 or y == 0 or x + w >= image.shape[1] or y + h >= image.shape[0] else 'no'}")


if __name__ == "__main__":
    main()
