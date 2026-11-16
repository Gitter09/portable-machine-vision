"""Variant: negate the class dequantisation scales.

If the runtime applies the class scale with the opposite sign, every logit is
mirrored about zero, which caps confidence without disturbing the box geometry
(the DFL softmax is invariant to a constant shift). Box scales stay untouched.
"""

import json
from pathlib import Path
from analysis_paths import DATA

SRC = DATA / "candidate" / "etc"
OUT = DATA / "candidate-signed" / "etc"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "0_preproc.json").write_text((SRC / "0_preproc.json").read_text())
    (OUT / "0_process_mla.json").write_text((SRC / "0_process_mla.json").read_text())

    box = json.loads((SRC / "boxdecoder.json").read_text())
    box["dq_scale"] = [scale if index < 3 else -scale
                       for index, scale in enumerate(box["dq_scale"])]
    box["detection_threshold"] = 0.05
    (OUT / "boxdecoder.json").write_text(json.dumps(box, indent=4) + "\n")
    print("class scales:", [round(v, 3) for v in box["dq_scale"][3:]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
