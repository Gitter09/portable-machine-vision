"""Report the compiler's own view of the unpack tensor geometry."""

import json
from pathlib import Path
from analysis_paths import DATA

MPK = DATA / "export-yolo11" / "yolov8_mpk.json"


def main() -> None:
    mpk = json.loads(MPK.read_text())
    for plugin in mpk["plugins"]:
        name = plugin.get("name", "")
        if not ("unpack" in name or "tuple_get_item" in name or name.startswith("dequantize")):
            continue
        params = (plugin.get("config_params") or {}).get("params") or {}
        print(name)
        for key in ("input_shapes", "output_shapes", "channel_params", "dims"):
            if key in params:
                value = params[key]
                if key == "channel_params":
                    value = [v[:2] if isinstance(v, list) else v for v in value]
                print("   ", key, "=", value)


if __name__ == "__main__":
    main()
