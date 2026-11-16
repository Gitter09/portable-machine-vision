"""Judge whether the box decoder now produces sane boxes."""

import json
from collections import Counter
from pathlib import Path
from analysis_paths import DATA

D = DATA / "verify-r24"
W, H = 1280, 720


def main() -> None:
    records = []
    for line in (D / "detections.jsonl").read_text(errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    total = neg_h = neg_w = off_frame = 0
    classes: Counter[int] = Counter()
    scores = []
    widths, heights = [], []
    per_frame = []
    samples = []

    for record in records:
        boxes = record.get("detections") or []
        per_frame.append(len(boxes))
        for box in boxes:
            native = box["native_box"]
            x, y = native["x"], native["y"]
            w, h = native["width"], native["height"]
            total += 1
            neg_h += h < 0
            neg_w += w < 0
            classes[box["class_id"]] += 1
            scores.append(box["score"])
            if w >= 0 and h >= 0:
                widths.append(w)
                heights.append(h)
            if x < 0 or y < 0 or w < 0 or h < 0 or x + w > W or y + h > H:
                off_frame += 1
            if len(samples) < 6:
                samples.append((box["label"], round(box["score"], 3), x, y, w, h))

    print(f"frames          : {len(records)}")
    print(f"boxes           : {total}")
    print(f"boxes per frame : max {max(per_frame) if per_frame else 0}"
          f"  median {sorted(per_frame)[len(per_frame)//2] if per_frame else 0}")
    print(f"negative height : {neg_h}  ({100 * neg_h / max(1, total):.1f}%)")
    print(f"negative width  : {neg_w}")
    print(f"outside frame   : {off_frame}")
    print(f"classes         : {dict(classes)}")
    if scores:
        scores.sort()
        print(f"score           : median {scores[len(scores)//2]:.3f}"
              f"  min {scores[0]:.3f}  max {scores[-1]:.3f}")
    if widths:
        widths.sort()
        heights.sort()
        print(f"width           : median {widths[len(widths)//2]}  range {widths[0]}..{widths[-1]}")
        print(f"height          : median {heights[len(heights)//2]}  range {heights[0]}..{heights[-1]}")
    print("\nsample (label, score, x, y, w, h):")
    for row in samples:
        print("   ", row)


if __name__ == "__main__":
    main()
