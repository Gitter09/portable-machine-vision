"""Show the full plugin chain of the compiled MPK."""

import json
from pathlib import Path
from analysis_paths import DATA

MPK = DATA / "export-1280" / "yolov8_mpk.json"


def main() -> None:
    mpk = json.loads(MPK.read_text())
    for index, plugin in enumerate(mpk["plugins"]):
        params = (plugin.get("config_params") or {}).get("params", {})
        shapes = params.get("output_shapes")
        print(f"{index:>2} {plugin.get('processor'):<5} {plugin.get('name')[:70]:<70} {shapes if shapes else ''}")


if __name__ == "__main__":
    main()
