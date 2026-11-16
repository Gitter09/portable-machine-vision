"""Measure the single-board sweep at the new camera position."""

import os
from pathlib import Path

import cv2
import numpy as np
import paramiko
from analysis_paths import DATA

HOST = "172.21.30.120"
USER = "sima"
PASSWORD = os.environ["BOARD_PASS"]
REMOTE = "/tmp/sweep2"
LOCAL = DATA / "sweep2"


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=20,
                   look_for_keys=False, allow_agent=False)
    sftp = client.open_sftp()
    names = sorted(sftp.listdir(REMOTE))
    picks = [names[int(i * (len(names) - 1) / 19)] for i in range(20)]
    for name in picks:
        sftp.get(f"{REMOTE}/{name}", str(LOCAL / name))
    sftp.close()
    client.close()

    print(f"{len(picks)} frames sampled from {len(names)} "
          f"(~{len(names)/25:.0f} fps over 25 s)\n")
    centres, sizes, clipped = [], [], 0
    for path in sorted(LOCAL.glob("s*.jpg")):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
        board = cv2.bitwise_or(cv2.inRange(hsv, (95, 80, 20), (135, 255, 200)),
                               cv2.inRange(hsv, (35, 80, 20), (90, 255, 255)))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(board, 8)
        big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 800]
        if not big:
            print(f"{path.name}: not found")
            continue
        # union bbox of all board components
        xs = [stats[i, cv2.CC_STAT_LEFT] for i in big]
        ys = [stats[i, cv2.CC_STAT_TOP] for i in big]
        x2 = [stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in big]
        y2 = [stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in big]
        x0, y0, x1, y1 = min(xs), min(ys), max(x2), max(y2)
        touch = x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1
        clipped += touch
        centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
        sizes.append((x1 - x0, y1 - y0))
        print(f"{path.name}: board {x1-x0:>3}x{y1-y0:<3} centre ({(x0+x1)/2:>6.0f},{(y0+y1)/2:>5.0f})"
              f"{'  CLIPPED' if touch else ''}")

    pts = np.array(centres)
    centre = pts.mean(axis=0)
    radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
    size = np.array(sizes)
    print(f"\nboard size: median {np.median(size[:,0]):.0f} x {np.median(size[:,1]):.0f} px")
    print(f"orbit centre ~ ({centre[0]:.0f}, {centre[1]:.0f})  median radius {np.median(radii):.0f} px")
    print(f"swept radius (orbit + half board) ~ {np.median(radii) + np.max(size)/2:.0f} px")
    print(f"frame half-height {720/2:.0f} px, half-width {1280/2:.0f} px")
    print(f"frames with board touching an edge: {clipped}/{len(pts)}")


if __name__ == "__main__":
    main()
