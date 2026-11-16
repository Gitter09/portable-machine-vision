"""Host reference at 1280 on a live frame, to compare against board output."""

import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO
from analysis_paths import DATA

TRAIN = DATA
OUT = TRAIN / "newset" / "compare1280.jpg"


def main() -> None:
    data = urllib.request.urlopen("http://172.21.30.120:8080/frame.jpg", timeout=10).read()
    OUT.write_bytes(data)
    image = cv2.imread(str(OUT))
    print("frame:", image.shape[1], "x", image.shape[0])

    model = YOLO(str(TRAIN / "runs" / "train-yolo11n-1280" / "weights" / "best.pt"))
    for conf in (0.25, 0.05, 0.01):
        result = model.predict(str(OUT), imgsz=1280, conf=conf, verbose=False)[0]
        rows = []
        for box in result.boxes[:6]:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            rows.append(f"{result.names[int(box.cls)]} {float(box.conf):.3f}"
                        f" {x2-x1:.0f}x{y2-y1:.0f}")
        print(f"\nconf {conf}: {len(result.boxes)} boxes")
        for row in rows:
            print("   ", row)


if __name__ == "__main__":
    main()
