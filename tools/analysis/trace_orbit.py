"""Trace the board around the rotation using model detections.

The colour mask is too sensitive to shadows and highlights, so use the trained
detector's terminal box as a compact reference point on the board. Reports the
orbit geometry and whether the region we must label is ever cut by the frame.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from analysis_paths import DATA

TRAIN = DATA
SRC = TRAIN / "sweep3"
FRAME_W, FRAME_H = 1280, 720
TERMINALS = {"Terminal_pass", "Terminal_Fail"}


def main() -> None:
    model = YOLO(str(TRAIN / "runs" / "train-yolo11n-1280" / "weights" / "best.pt"))
    files = sorted(SRC.glob("b*.jpg"))
    results = model.predict([str(p) for p in files], imgsz=640, conf=0.25, verbose=False)

    centres, cut, missing = [], 0, 0
    for path, result in zip(files, results):
        best = None
        for box in result.boxes:
            name = result.names[int(box.cls)]
            if name not in TERMINALS:
                continue
            if best is None or float(box.conf) > float(best.conf):
                best = box
        if best is None:
            missing += 1
            print(f"{path.stem}: no terminal")
            continue

        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        centres.append(centre)
        edge = (x1 <= 2) or (y1 <= 2) or (x2 >= FRAME_W - 2) or (y2 >= FRAME_H - 2)
        cut += int(edge)
        print(f"{path.stem}: {result.names[int(best.cls)]:<14} {float(best.conf):.2f}"
              f"  {x2 - x1:>3.0f}x{y2 - y1:<3.0f} at ({centre[0]:>5.0f},{centre[1]:>5.0f})"
              f"{'  CUT' if edge else ''}")

    pts = np.array(centres)
    print(f"\nterminals found : {len(pts)}/{len(files)} (missing {missing})")
    print(f"terminals cut   : {cut}/{len(pts)}")
    print(f"centre range    : x {pts[:,0].min():.0f}..{pts[:,0].max():.0f}"
          f"   y {pts[:,1].min():.0f}..{pts[:,1].max():.0f}")
    print(f"mean centre     : ({pts[:,0].mean():.0f}, {pts[:,1].mean():.0f})"
          f"   frame centre (640, 360)")
    radii = np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean())
    print(f"orbit radius    : median {np.median(radii):.0f} px   max {radii.max():.0f} px")
    print(f"vertical span   : {pts[:,1].max() - pts[:,1].min():.0f} px of {FRAME_H}")
    print(f"horizontal span : {pts[:,0].max() - pts[:,0].min():.0f} px of {FRAME_W}")


if __name__ == "__main__":
    main()
