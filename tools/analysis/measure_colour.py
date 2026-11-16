"""Measure board and terminal size at the new camera distance."""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

TRAIN = DATA
FRAME = TRAIN / "newset" / "frame.jpg"
OUT = TRAIN / "newset"


def main() -> None:
    image = cv2.imread(str(FRAME))
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(hsv, (95, 80, 20), (135, 255, 200))     # PCB body
    green = cv2.inRange(hsv, (35, 80, 20), (90, 255, 255))     # screw terminal

    def report(mask, label, min_area=400):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area < min_area:
                continue
            clipped = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
            found.append((area, x, y, w, h, clipped))
        found.sort(reverse=True)
        print(f"\n{label}: {len(found)} component(s) over {min_area} px")
        for area, x, y, w, h, clipped in found[:4]:
            print(f"  area {area:>6}  bbox ({x:>4},{y:>3})  {w:>3} x {h:>3} px"
                  f"{'  CLIPPED BY FRAME' if clipped else ''}")

    report(blue, "blue PCB")
    report(green, "green terminal")

    # board extent from the image centre, as a framing sanity check
    print(f"\nframe {width}x{height}")
    for area, x, y, w, h, clipped in sorted(
            [(s[4], *s[0:4], s[5]) for s in []] or [], reverse=True):
        pass

    cv2.imwrite(str(OUT / "frame_annotated.jpg"), image)
    for name, mask in (("blue", blue), ("green", green)):
        cv2.imwrite(str(OUT / f"mask_{name}.png"), mask)


if __name__ == "__main__":
    main()
