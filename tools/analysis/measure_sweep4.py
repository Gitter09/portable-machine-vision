"""Measure the board sweep by motion, not colour.

Colour masks latch onto static saturated objects (cables, labels), so instead
build a background from the temporal median of the sampled sequence and treat
whatever differs from it as the moving board. Reports the swept extent, the
margin to each frame edge, and whether more than one board is present.
"""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

SRC = DATA / "sweep3"
FRAME_W, FRAME_H = 1280, 720
DIFF_THRESHOLD = 32
MIN_AREA = 700


def moving_blobs(frame: np.ndarray, background: np.ndarray):
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < MIN_AREA:
            continue
        blobs.append((area, int(x), int(y), int(w), int(h)))
    blobs.sort(reverse=True)
    return blobs


def main() -> None:
    files = sorted(SRC.glob("b*.jpg"))
    frames = [cv2.imread(str(path)) for path in files]
    frames = [f for f in frames if f is not None]
    print(f"{len(frames)} frames")

    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(SRC / "background.jpg"), background)

    boxes = []
    multi = 0
    for path, frame in zip(files, frames):
        blobs = moving_blobs(frame, background)
        if not blobs:
            print(f"{path.name}: no moving blob")
            continue
        # board = the largest blob; keep any comparable second blob as a warning
        area, x, y, w, h = blobs[0]
        others = [b for b in blobs[1:] if b[0] >= 0.35 * area]
        if others:
            multi += 1
        x1, y1 = x + w, y + h
        boxes.append((x, y, x1, y1))
        warn = f"  SECOND BLOB {others[0][3]}x{others[0][4]}" if others else ""
        print(f"{path.name}: {w:>3}x{h:<3} at ({x:>4},{y:>3})"
              f"  margins L{x:>4} T{y:>4} R{FRAME_W - x1:>4} B{FRAME_H - y1:>4}{warn}")

    if len(boxes) < 4:
        print("\nnot enough samples")
        return

    arr = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = arr.T
    centres = np.stack([(x0 + x1) / 2, (y0 + y1) / 2], axis=1)
    sizes = np.stack([x1 - x0, y1 - y0], axis=1)

    span_x = x1.max() - x0.min()
    span_y = y1.max() - y0.min()
    clipped = int(np.sum((x0 <= 1) | (y0 <= 1) | (x1 >= FRAME_W - 1) | (y1 >= FRAME_H - 1)))
    (left, top), (right, bottom) = (x0.min(), y0.min()), (FRAME_W - x1.max(), FRAME_H - y1.max())

    centre = centres.mean(axis=0)
    radii = np.hypot(centres[:, 0] - centre[0], centres[:, 1] - centre[1])

    print(f"\nsamples          : {len(boxes)}")
    print(f"board size       : median {np.median(sizes[:, 0]):.0f} x {np.median(sizes[:, 1]):.0f} px")
    print(f"orbit centre     : ({centre[0]:.0f}, {centre[1]:.0f})")
    print(f"orbit radius     : median {np.median(radii):.0f} px")
    print(f"swept extent     : {span_x:.0f} x {span_y:.0f} px  (frame {FRAME_W}x{FRAME_H})")
    print(f"margins          : left {left:.0f}  top {top:.0f}  right {right:.0f}  bottom {bottom:.0f}")
    print(f"clipped frames   : {clipped}/{len(boxes)}")
    print(f"frames with a second moving blob: {multi}")

    need = max(span_x / (0.92 * FRAME_W), span_y / (0.92 * FRAME_H))
    print(f"\nrequired scale   : x{need:.2f}")
    if need <= 1.0:
        print("verdict          : distance is fine")
    else:
        print(f"verdict          : back off {100 * (need - 1):.0f}%")


if __name__ == "__main__":
    main()
