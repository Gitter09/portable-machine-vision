"""Check whether the board stays in frame through the rotation at the new distance."""

from pathlib import Path

import cv2
import numpy as np
from analysis_paths import DATA

D = DATA / "newset"


def main() -> None:
    files = sorted(D.glob("f*.jpg"))
    print(f"{len(files)} sampled frames\n")
    clipped_frames = 0
    centres = []
    for path in files:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (95, 80, 20), (135, 255, 200))
        green = cv2.inRange(hsv, (35, 80, 20), (90, 255, 255))
        board = cv2.bitwise_or(blue, green)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(board, 8)
        big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 1500]
        edge = False
        for index in big:
            x, y, w, h, _ = stats[index]
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                edge = True
        if edge:
            clipped_frames += 1

        if big:
            largest = max(big, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            centres.append(centroids[largest])
            x, y, w, h, _ = stats[largest]
            print(f"{path.name}: main board {w:>3}x{h:<3} at ({x:>4},{y:>3})"
                  f"{'  CLIPPED' if edge else ''}")
        else:
            print(f"{path.name}: no board found  CLIPPED" if edge else f"{path.name}: no board found")

    print(f"\nframes with a board touching the frame edge: {clipped_frames}/{len(files)}")
    if centres:
        pts = np.array(centres)
        print(f"main-board centre range: x {pts[:,0].min():.0f}..{pts[:,0].max():.0f}  "
              f"y {pts[:,1].min():.0f}..{pts[:,1].max():.0f}")
        centre = pts.mean(axis=0)
        radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
        print(f"orbit radius about its own mean: median {np.median(radii):.0f} px")


if __name__ == "__main__":
    main()
