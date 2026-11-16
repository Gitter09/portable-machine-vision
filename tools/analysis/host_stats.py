"""Host detection statistics over real frames, for comparison with the board run."""

from pathlib import Path

from ultralytics import YOLO
from analysis_paths import DATA

TRAIN = DATA
FRAMES = TRAIN / "sweep4"


def main() -> None:
    model = YOLO(str(TRAIN / "runs" / "train-yolo11n-1280" / "weights" / "best.pt"))
    files = sorted(FRAMES.glob("c[0-9][0-9].jpg"))
    results = model.predict([str(p) for p in files], imgsz=1280, conf=0.05, verbose=False)

    scores, widths, heights, per_frame = [], [], [], []
    for result in results:
        per_frame.append(len(result.boxes))
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            scores.append(float(box.conf))
            widths.append(x2 - x1)
            heights.append(y2 - y1)

    scores.sort()
    widths.sort()
    heights.sort()
    per_frame.sort()
    print(f"frames     : {len(files)}")
    print(f"boxes      : {len(scores)}  ({len(scores) / max(1, len(files)):.1f} per frame,"
          f" median {per_frame[len(per_frame)//2]})")
    if scores:
        print(f"score      : median {scores[len(scores)//2]:.3f}  max {scores[-1]:.3f}")
        print(f"width      : median {widths[len(widths)//2]:.0f}")
        print(f"height     : median {heights[len(heights)//2]:.0f}")


if __name__ == "__main__":
    main()
