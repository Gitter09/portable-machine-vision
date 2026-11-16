"""Produce an interleaved tensor-order variant of the board configs.

The decoder pairs tensors into heads as (0,1), (2,3), (4,5). If the unpack node
lists all three box tensors before all three class tensors, each "head" gets two
tensors of the same kind. Interleaving them pairs box/class per level.
"""

import json
from pathlib import Path
from analysis_paths import DATA

SRC = DATA / "candidate" / "etc"
OUT = DATA / "candidate-interleaved" / "etc"

# level order 160, 80, 40 with (box, class) per level
WIDTHS = [160, 160, 80, 80, 40, 40]
DEPTHS = [64, 4, 64, 4, 64, 4]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "0_preproc.json").write_text((SRC / "0_preproc.json").read_text())

    mla = json.loads((SRC / "0_process_mla.json").read_text())
    mla["input_width"] = WIDTHS
    mla["input_height"] = WIDTHS
    mla["input_depth"] = DEPTHS
    mla["slice_width"] = WIDTHS
    mla["slice_height"] = WIDTHS
    mla["slice_depth"] = DEPTHS
    # dequantisation travels with its tensor, so permute the scales to match
    print("mla slice_depth:", mla["slice_depth"])

    box = json.loads((SRC / "boxdecoder.json").read_text())
    original_scale = list(box["dq_scale"])
    original_zp = list(box["dq_zp"])
    # old order: bbox0,bbox1,bbox2,cls0,cls1,cls2 -> new: b0,c0,b1,c1,b2,c2
    order = [0, 3, 1, 4, 2, 5]
    box["input_width"] = WIDTHS
    box["input_height"] = WIDTHS
    box["input_depth"] = DEPTHS
    box["slice_width"] = WIDTHS
    box["slice_height"] = WIDTHS
    box["slice_depth"] = DEPTHS
    box["dq_scale"] = [original_scale[i] for i in order]
    box["dq_zp"] = [original_zp[i] for i in order]
    box["detection_threshold"] = 0.05

    (OUT / "0_process_mla.json").write_text(json.dumps(mla, indent=4) + "\n")
    (OUT / "boxdecoder.json").write_text(json.dumps(box, indent=4) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
