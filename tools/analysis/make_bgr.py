"""Variant: ask the preprocessing node for BGR instead of RGB.

The detector was trained by Ultralytics on RGB images. If the board's
preprocessing emits RGB into a graph that was quantised expecting BGR (or the
reverse), geometry stays right while confidence collapses.
"""

import json
from pathlib import Path
from analysis_paths import DATA

SRC = DATA / "candidate" / "etc"
OUT = DATA / "candidate-bgr" / "etc"
TARGET = "BGR"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preproc = json.loads((SRC / "0_preproc.json").read_text())
    print("was:", preproc["output_img_type"])
    preproc["output_img_type"] = TARGET
    for pad in preproc["caps"]["src_pads"]:
        for param in pad["params"]:
            if param.get("json_field") == "output_img_type":
                param["values"] = "RGB, BGR"
    (OUT / "0_preproc.json").write_text(json.dumps(preproc, indent=4) + "\n")
    for name in ("0_process_mla.json", "boxdecoder.json"):
        (OUT / name).write_text((SRC / name).read_text())
    print("now:", TARGET)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
