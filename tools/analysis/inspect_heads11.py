"""List each decoded head's shapes and quantisation from the MPK."""

import json
import re
from pathlib import Path
from analysis_paths import DATA

MPK = DATA / "export-yolo11" / "yolov8_mpk.json"


def main() -> None:
    mpk = json.loads(MPK.read_text())
    item = None
    for plugin in mpk["plugins"]:
        name = plugin.get("name", "")
        params = (plugin.get("config_params") or {}).get("params", {})
        match = re.search(r"tuple_get_item_(\d+)", name)
        if match:
            item = match.group(1)
            print(f"item {item}: {params.get('output_shapes')}")
            continue
        if "dequant" in name.lower():
            channel = params.get("channel_params")
            print(f"     {name:<14} scale/zp={channel} in={params.get('input_shapes')}")
    print()
    for plugin in mpk["plugins"]:
        if plugin.get("name") == "MLA_0_ofm_unpack_transform":
            params = plugin["config_params"]["params"]
            print("unpack input :", params.get("input_shapes"))
            print("unpack output:", params.get("output_shapes"))


if __name__ == "__main__":
    main()
