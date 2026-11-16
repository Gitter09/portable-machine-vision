"""Read the compiler's own numbers out of yolov8_mpk.json."""

import json
from pathlib import Path
from analysis_paths import DATA

MPK = DATA / "export-1280" / "yolov8_mpk.json"


def main() -> None:
    mpk = json.loads(MPK.read_text())
    print("name:", mpk.get("name"), "| sdk:", mpk.get("model_sdk_version"))
    print("input_nodes:", json.dumps(mpk.get("input_nodes"), indent=1))
    print()
    for plugin in mpk.get("plugins", []):
        params = (plugin.get("config_params") or {}).get("params") or {}
        print(f"{plugin.get('processor'):<5} {plugin.get('name'):<48} {plugin.get('kernel')}")
        for key in ("input_shapes", "output_shapes", "channel_params", "input_data_type", "output_data_type"):
            if key in params:
                value = params[key]
                if isinstance(value, list) and len(value) > 3:
                    value = [value[0], "...", value[-1]]
                print(f"      {key}: {value}")

    print()
    print("dequantize channel_params in full:")
    for plugin in mpk.get("plugins", []):
        if "dequant" not in str(plugin.get("name", "")).lower():
            continue
        params = (plugin.get("config_params") or {}).get("params") or {}
        shape = params.get("output_shapes")
        print(f"  {plugin.get('name')}: {params.get('channel_params')}  out={shape}")


if __name__ == "__main__":
    main()
