"""Measure the single-board sweep using the blue PCB as the anchor.

Earlier masks latched onto a static pale-green object in the bottom-right
corner, which corrupted the orbit fit. Anchoring on the blue PCB and attaching
the green terminal only when it sits adjacent to that anchor avoids it.
"""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

SRC = DATA / "sweep3"
FRAME_W, FRAME_H = 1280, 720


def board_box(image: np.ndarray):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 70, 20), (135, 255, 210))
    green = cv2.inRange(hsv, (35, 90, 20), (90, 255, 255))

    def parts(mask, min_area):
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        return [(stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_LEFT],
                 stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH],
                 stats[i, cv2.CC_STAT_HEIGHT], centres[i])
                for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    blues = parts(blue, 300)
    if not blues:
        return None
    anchor = max(blues, key=lambda p: p[0])
    ax, ay = anchor[5]
    keep = [p for p in blues if np.hypot(p[5][0] - ax, p[5][1] - ay) <= 260]
    for p in parts(green, 100):
        if np.hypot(p[5][0] - ax, p[5][1] - ay) <= 150:
            keep.append(p)
    x0 = min(p[1] for p in keep)
    y0 = min(p[2] for p in keep)
    x1 = max(p[1] + p[3] for p in keep)
    y1 = max(p[2] + p[4] for p in keep)
    return int(x0), int(y0), int(x1), int(y1)


def main() -> None:
    boxes = []
    for path in sorted(SRC.glob("b*.jpg")):
        image = cv2.imread(str(path))
        found = board_box(image)
        if found is None:
            print(f"{path.stem}: not found")
            continue
        x0, y0, x1, y1 = found
        boxes.append((x0, y0, x1, y1))
        touch = "  CLIPPED" if (x0 <= 1 or y0 <= 1 or x1 >= FRAME_W - 1 or y1 >= FRAME_H - 1) else ""
        print(f"{path.stem}: {x1 - x0:>3}x{y1 - y0:<3} at ({x0:>4},{y0:>3})"
              f"  gaps L{x0:>4} T{y0:>4} R{FRAME_W - x1:>4} B{FRAME_H - y1:>4}{touch}")

    arr = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = arr.T
    centres = np.stack([(x0 + x1) / 2, (y0 + y1) / 2], axis=1)
    sizes = np.stack([x1 - x0, y1 - y0], axis=1)

    span_x, span_y = x1.max() - x0.min(), y1.max() - y0.min()
    clipped = int(np.sum((x0 <= 1) | (y0 <= 1) | (x1 >= FRAME_W - 1) | (y1 >= FRAME_H - 1)))
    centre = centres.mean(axis=0)
    radii = np.hypot(centres[:, 0] - centre[0], centres[:, 1] - centre[1])

    print(f"\nsamples        : {len(boxes)}")
    print(f"board size     : median {np.median(sizes[:, 0]):.0f} x {np.median(sizes[:, 1]):.0f} px"
          f"   max {sizes.max(axis=0)[0]:.0f} x {sizes.max(axis=0)[1]:.0f} px")
    print(f"orbit centre   : ({centre[0]:.0f}, {centre[1]:.0f})  frame centre (640, 360)")
    print(f"orbit radius   : median {np.median(radii):.0f} px")
    print(f"swept extent   : {span_x:.0f} x {span_y:.0f} px")
    print(f"gaps           : left {x0.min():.0f}  top {y0.min():.0f}"
          f"  right {FRAME_W - x1.max():.0f}  bottom {FRAME_H - y1.max():.0f}")
    print(f"clipped frames : {clipped}/{len(boxes)}")

    need = max(span_x / (0.92 * FRAME_W), span_y / (0.92 * FRAME_H))
    print(f"\nrequired scale : x{need:.2f}")
    print(f"verdict        : " + ("distance is fine" if need <= 1.0
                                 else f"back off {100 * (need - 1):.0f}%"))


if __name__ == "__main__":
    main()
