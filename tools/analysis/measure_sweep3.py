"""Measure the single-board sweep at the current camera position.

Samples the live feed from the board's HTTP endpoint, so the capture pipeline
keeps running. Board is located by colour (PCB blue + terminal green) and
clustered around its largest component, then the union bbox is tracked across
a full revolution to measure the swept extent and the margin to each edge.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

URL = "http://172.21.30.120:8080/frame.jpg"
OUT = DATA / "sweep3"
FRAME_W, FRAME_H = 1280, 720
SAMPLES = 40
GAP = 0.6


def grab(path: Path) -> np.ndarray | None:
    try:
        data = urllib.request.urlopen(URL, timeout=10).read()
    except Exception as exc:
        print(f"  fetch failed: {exc}")
        return None
    path.write_bytes(data)
    return cv2.imread(str(path))


def board_box(image: np.ndarray, min_area: int = 300, reach: float = 200.0):
    """Union bbox of board-coloured components near the largest one."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, (95, 70, 20), (135, 255, 210)),
                          cv2.inRange(hsv, (35, 70, 20), (90, 255, 255)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pieces = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not pieces:
        return None
    anchor = max(pieces, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ax, ay = centroids[anchor]
    near = [i for i in pieces
            if np.hypot(centroids[i][0] - ax, centroids[i][1] - ay) <= reach]
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in near)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in near)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in near)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in near)
    return int(x0), int(y0), int(x1), int(y1), len(near)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    boxes = []
    for index in range(SAMPLES):
        image = grab(OUT / f"b{index:02d}.jpg")
        if image is None:
            continue
        found = board_box(image)
        if found is None:
            print(f"b{index:02d}: no board found")
            continue
        x0, y0, x1, y1, pieces = found
        boxes.append((x0, y0, x1, y1))
        print(f"b{index:02d}: {x1 - x0:>3}x{y1 - y0:<3} at ({x0:>4},{y0:>3})"
              f"  margins L{x0:>4} T{y0:>4} R{FRAME_W - x1:>4} B{FRAME_H - y1:>4}"
              f"  ({pieces} pieces)")
        if index != SAMPLES - 1:
            time.sleep(GAP)

    if len(boxes) < 4:
        print("\nnot enough samples")
        return

    arr = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = arr.T
    centres = np.stack([(x0 + x1) / 2, (y0 + y1) / 2], axis=1)
    sizes = np.stack([x1 - x0, y1 - y0], axis=1)

    span_x = x1.max() - x0.min()
    span_y = y1.max() - y0.min()
    min_margin = min(x0.min(), y0.min(), FRAME_W - x1.max(), FRAME_H - y1.max())
    clipped = int(np.sum((x0 <= 1) | (y0 <= 1) | (x1 >= FRAME_W - 1) | (y1 >= FRAME_H - 1)))

    centre = centres.mean(axis=0)
    radii = np.hypot(centres[:, 0] - centre[0], centres[:, 1] - centre[1])

    print(f"\nsamples: {len(boxes)}")
    print(f"board size      : median {np.median(sizes[:, 0]):.0f} x {np.median(sizes[:, 1]):.0f} px")
    print(f"orbit centre    : ({centre[0]:.0f}, {centre[1]:.0f})")
    print(f"orbit radius    : median {np.median(radii):.0f} px")
    print(f"swept extent    : {span_x:.0f} x {span_y:.0f} px")
    print(f"frame           : {FRAME_W} x {FRAME_H}")
    print(f"tightest margin : {min_margin:.0f} px")
    print(f"clipped frames  : {clipped}/{len(boxes)}")

    # how much to back off so the swept extent sits inside a 92% safe area
    need_x = span_x / (0.92 * FRAME_W)
    need_y = span_y / (0.92 * FRAME_H)
    need = max(need_x, need_y)
    print(f"\nrequired scale  : x{need:.2f}"
          f"  (vertical governs: {need_y:.2f}, horizontal: {need_x:.2f})")
    if need <= 1.0:
        print("verdict         : distance is fine, no backing off needed")
    else:
        print(f"verdict         : move camera {100 * (need - 1):.0f}% further away")


if __name__ == "__main__":
    main()
