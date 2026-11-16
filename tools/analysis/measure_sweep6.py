"""Re-measure the sweep after the camera re-aim.

Samples a fresh orbit from the live feed, tracks the board by its blue PCB,
reports the clearance to each frame edge, and writes a contact sheet for a
visual cross-check.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

URL = "http://172.21.30.120:8080/frame.jpg"
OUT = DATA / "sweep4"
FRAME_W, FRAME_H = 1280, 720
SAMPLES = 40
GAP = 0.6


def grab(path: Path):
    try:
        data = urllib.request.urlopen(URL, timeout=10).read()
    except Exception as exc:
        print(f"  fetch failed: {exc}")
        return None
    path.write_bytes(data)
    return cv2.imread(str(path))


def board_box(image):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 60, 15), (140, 255, 220))
    green = cv2.inRange(hsv, (35, 90, 20), (90, 255, 255))

    def parts(mask, min_area):
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        return [(int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_LEFT]),
                 int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
                 int(stats[i, cv2.CC_STAT_HEIGHT]), centres[i])
                for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    blues = parts(blue, 250)
    if not blues:
        return None
    anchor = max(blues, key=lambda p: p[0])
    ax, ay = anchor[5]
    keep = [p for p in blues if np.hypot(p[5][0] - ax, p[5][1] - ay) <= 260]
    for p in parts(green, 100):
        if np.hypot(p[5][0] - ax, p[5][1] - ay) <= 150:
            keep.append(p)
    return (min(p[1] for p in keep), min(p[2] for p in keep),
            max(p[1] + p[3] for p in keep), max(p[2] + p[4] for p in keep))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    shots = []
    for index in range(SAMPLES):
        image = grab(OUT / f"c{index:02d}.jpg")
        if image is None:
            continue
        shots.append(image)
        if index != SAMPLES - 1:
            time.sleep(GAP)

    boxes, missing = [], 0
    for index, image in enumerate(shots):
        found = board_box(image)
        if found is None:
            missing += 1
            continue
        x0, y0, x1, y1 = found
        boxes.append((x0, y0, x1, y1))
        touch = "  CUT" if (x0 <= 1 or y0 <= 1 or x1 >= FRAME_W - 1 or y1 >= FRAME_H - 1) else ""
        print(f"c{index:02d}: {x1 - x0:>3}x{y1 - y0:<3} at ({x0:>4},{y0:>3})"
              f"  gaps L{x0:>4} T{y0:>4} R{FRAME_W - x1:>4} B{FRAME_H - y1:>4}{touch}")

    tiles = []
    for path in sorted(OUT.glob("c*.jpg")):
        tile = cv2.resize(cv2.imread(str(path)), (320, 180))
        cv2.putText(tile, path.stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(OUT / "contact.jpg"), np.vstack(rows))

    if len(boxes) < 4:
        print("\nnot enough detections")
        return
    arr = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = arr.T
    cut = int(np.sum((x0 <= 1) | (y0 <= 1) | (x1 >= FRAME_W - 1) | (y1 >= FRAME_H - 1)))
    print(f"\nframes sampled : {len(shots)}")
    print(f"board detected : {len(boxes)}  (not found {missing})")
    print(f"gaps           : left {x0.min():.0f}  top {y0.min():.0f}"
          f"  right {FRAME_W - x1.max():.0f}  bottom {FRAME_H - y1.max():.0f}")
    print(f"swept extent   : {x1.max() - x0.min():.0f} x {y1.max() - y0.min():.0f} px")
    print(f"cut frames     : {cut}/{len(boxes)}")


if __name__ == "__main__":
    main()
