"""Re-check the r24 detections with signed extents, and separate the groups."""

import json
import struct
from collections import Counter
from pathlib import Path
from analysis_paths import DATA

D = DATA / "verify-r24"
W, H = 1280, 720


def signed(value: int) -> int:
    return struct.unpack("<i", struct.pack("<I", value & 0xFFFFFFFF))[0]


def main() -> None:
    records = [json.loads(line) for line in
               (D / "detections.jsonl").read_text(errors="replace").splitlines() if line.strip()]

    total = 0
    negative = []
    positive = []
    classes = Counter()
    for record in records:
        for box in record.get("detections") or []:
            native = box["native_box"]
            item = (box["label"], round(box["score"], 3), native["x"], native["y"],
                    signed(native["width"]), signed(native["height"]))
            total += 1
            classes[box["class_id"]] += 1
            (negative if item[5] < 0 else positive).append(item)

    print(f"frames        : {len(records)}")
    print(f"boxes         : {total}")
    print(f"negative h    : {len(negative)}  ({100 * len(negative) / max(1, total):.1f}%)")
    print(f"classes       : {dict(classes)}")

    if positive:
        widths = sorted(item[4] for item in positive)
        heights = sorted(item[5] for item in positive)
        scores = sorted(item[1] for item in positive)
        print(f"\nsane boxes    : {len(positive)}")
        print(f"  width  median {widths[len(widths)//2]}  range {widths[0]}..{widths[-1]}")
        print(f"  height median {heights[len(heights)//2]}  range {heights[0]}..{heights[-1]}")
        print(f"  score  median {scores[len(scores)//2]:.3f}  max {scores[-1]:.3f}")
        print("  strongest:")
        for item in sorted(positive, key=lambda row: -row[1])[:10]:
            print("   ", item)

    if negative:
        print(f"\nnegative-height boxes: {len(negative)}")
        print("  y values :", Counter(item[3] for item in negative).most_common(4))
        print("  widths   :", Counter(item[4] for item in negative).most_common(4))
        print("  heights  :", Counter(item[5] for item in negative).most_common(4))


if __name__ == "__main__":
    main()
