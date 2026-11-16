"""Generate the board-side configs for a compiled detector from its own MPK report.

Every quantisation scale, tensor size and slice geometry comes from
yolov8_mpk.json. Nothing is inherited from the previous model, because a stale
scale produces plausible-looking but wrong boxes rather than an error.

Usage:
  build_ic_terminal_configs.py --mpk yolov8_mpk.json --template-dir seed/.../board \
      --output-dir out --model-size 1280 --source 1280x720 --classes 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ITEM = re.compile(r"tuple_get_item_(\d+)_detessellation")

def compiler_numbers(mpk: dict, model_size: int) -> dict:
    """Pull input quantisation, MLA buffer size and per-tensor dequantisation."""
    input_scale = None
    for plugin in mpk["plugins"]:
        params = (plugin.get("config_params") or {}).get("params") or {}
        if plugin.get("name") == "quantize_0":
            input_scale = params["channel_params"][0]
    if input_scale is None:
        raise SystemExit("MPK has no quantize_0 plugin; cannot read the input scale")

    mla_size = None
    for plugin in mpk["plugins"]:
        params = (plugin.get("config_params") or {}).get("params") or {}
        if plugin.get("name") == "MLA_0_ofm_unpack_transform":
            mla_size = int(params["input_shapes"][0][-1])
    if mla_size is None:
        raise SystemExit("MPK has no MLA output unpack plugin")

    scales: dict[int, list[float]] = {}
    current = None
    for plugin in mpk["plugins"]:
        match = ITEM.search(plugin.get("name", ""))
        if match:
            current = int(match.group(1))
            continue
        if current is not None and str(plugin.get("name", "")).startswith("dequantize"):
            params = (plugin.get("config_params") or {}).get("params") or {}
            scales[current] = list(params["channel_params"][0])
    if sorted(scales) != list(range(6)):
        raise SystemExit(f"expected six dequantised tensors, found {sorted(scales)}")

    grid = model_size // 8
    derived = [grid, grid // 2, grid // 4, grid, grid // 2, grid // 4]
    # The compiler records the unpacked shape of each tensor. Trust it over the
    # derived guess, and fail loudly if the two disagree.
    shapes = {}
    for plugin in mpk["plugins"]:
        match = ITEM.search(plugin.get("name", ""))
        if not match:
            continue
        params = (plugin.get("config_params") or {}).get("params") or {}
        output = params.get("output_shapes") or []
        if output and len(output[0]) == 4:
            _, height, width, channels = output[0]
            shapes[int(match.group(1))] = (int(height), int(width), int(channels))
    if sorted(shapes) == list(range(6)):
        widths = [shapes[index][1] for index in range(6)]
        heights = [shapes[index][0] for index in range(6)]
        depths = [shapes[index][2] for index in range(6)]
        if widths != derived or heights != derived:
            raise SystemExit(
                f"compiler grids {widths} disagree with --model-size {model_size} "
                f"(expected {derived})")
    else:
        widths = heights = derived
        depths = None
    input_tiles = None
    output_tiles = {}
    for plugin in mpk["plugins"]:
        params = (plugin.get("config_params") or {}).get("params") or {}
        tile = params.get("slice_shape")
        if not isinstance(tile, list) or len(tile) != 3 or any(type(v) is not int or v <= 0 for v in tile):
            continue
        if plugin.get("name", "").startswith("tessellate_quantize_0_"):
            if params.get("align_c16") is not False or params.get("cblock") is not False:
                raise SystemExit("preprocessor supports only compiler input tiling without channel blocking")
            input_tiles = list(tile)
        match = ITEM.search(plugin.get("name", ""))
        if match:
            output_tiles[int(match.group(1))] = list(tile)
    return {
        "input_tiles": input_tiles,
        "output_tiles": [output_tiles[i] for i in range(6)] if sorted(output_tiles) == list(range(6)) else None,
        "input_scale": input_scale[0],
        "input_zero": input_scale[1],
        "mla_size": mla_size,
        "dq": [scales[index] for index in range(6)],
        "grids": widths,
        "heights": heights,
        "depths": depths,
    }


def assert_geometry_matches(mla: dict, box: dict) -> None:
    """Reject a config pair whose unpack and decode nodes describe different tensors.

    Logical tensor geometry and physical tile geometry are separate contracts.
    The original MLA node kept a 640-model logical geometry while the compiler
    emitted 64-channel heads on grid 160. A later defect equated tile height with
    logical height, despite the compiler's 2/1/1-row box tiles. Neither mistake
    necessarily prevents execution. The preserved original bad pair is
    ``tests/fixtures/unpack_640_bug/``.
    """
    for field in ("width", "height", "depth"):
        unpack = mla.get(f"input_{field}")
        decode = box.get(f"input_{field}")
        if unpack is None or decode is None:
            raise ValueError(f"{field} geometry is missing from the unpack or "
                             f"decode config")
        unpack_slice = mla.get(f"slice_{field}")
        decode_slice = box.get(f"slice_{field}")
        if unpack != decode:
            raise ValueError(f"{field} geometry disagrees: unpack {unpack} vs decoder {decode}")
        if unpack_slice != decode_slice or decode_slice is None or len(decode_slice) != len(decode):
            raise ValueError(f"{field} tile geometry disagrees: unpack {unpack_slice} vs decoder {decode_slice}")
        if any(type(tile) is not int or not 0 < tile <= full for tile, full in zip(decode_slice, decode)):
            raise ValueError(f"invalid {field} tile geometry {decode_slice} for tensor {decode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mpk", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", type=int, default=1280)
    parser.add_argument("--source", default="1280x720", help="camera frame WxH")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--elf-name", default="yolov8_stage1_mla.elf")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--decode-type", default=None, help="override decode_type")
    parser.add_argument("--box-channels", type=int, default=4, help="64 for a DFL head, 4 for decoded outputs")
    parser.add_argument("--sigmoid", choices=("true", "false"), default=None)
    arguments = parser.parse_args()

    numbers = compiler_numbers(json.loads(arguments.mpk.read_text()), arguments.model_size)
    names = [line.strip() for line in arguments.labels_file.read_text().splitlines() if line.strip()]
    if len(names) != arguments.classes:
        raise SystemExit(f"labels file has {len(names)} names, expected {arguments.classes}")
    source_width, source_height = (int(value) for value in arguments.source.lower().split("x"))
    grids = numbers["grids"]
    heights = numbers["heights"]
    declared = [arguments.box_channels] * 3 + [arguments.classes] * 3
    depth = numbers["depths"] or declared
    if numbers["depths"] and depth != declared:
        raise SystemExit(
            f"compiler tensor depths {depth} disagree with --box-channels "
            f"{arguments.box_channels} / --classes {arguments.classes}")
    if numbers["input_tiles"] is None or numbers["output_tiles"] is None:
        raise SystemExit("MPK lacks compiler input/output tessellation geometry; refusing to inherit template tiles")
    tiles = numbers["output_tiles"]
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    preproc = json.loads((arguments.template_dir / "0_preproc.json").read_text())
    preproc.update({
        "input_width": source_width, "input_height": source_height,
        "output_width": arguments.model_size, "output_height": arguments.model_size,
        "scaled_width": arguments.model_size, "scaled_height": arguments.model_size,
        "q_scale": numbers["input_scale"], "q_zp": numbers["input_zero"],
        "tile_height": numbers["input_tiles"][0], "tile_width": numbers["input_tiles"][1],
        "tile_channels": numbers["input_tiles"][2],
        "aspect_ratio": True, "padding_type": "CENTER", "scaling_type": "BILINEAR",
    })
    for pad in preproc["caps"]["sink_pads"]:
        for param in pad["params"]:
            if param.get("json_field") == "input_width":
                param["values"] = f"1 - 4096"
    (output / "0_preproc.json").write_text(json.dumps(preproc, indent=4) + "\n")

    mla = json.loads((arguments.template_dir / "0_process_mla.json").read_text())
    mla["simaai__params"]["outputs"][0]["size"] = numbers["mla_size"]
    mla["simaai__params"]["model_path"] = arguments.elf_name
    # The unpack node describes its own output geometry with input_* keys. Writing
    # them under output_* silently left the template's geometry in place, so the
    # node advertised a 640-sized grid to a 1280-sized tensor. input_* describes
    # logical frame geometry; slice_* MUST retain the compiler's physical tiles.
    geometry = {"width": list(grids), "height": list(heights), "depth": list(depth)}
    for suffix, value in geometry.items():
        mla[f"input_{suffix}"] = value
        mla[f"slice_{suffix}"] = [tile[{"height": 0, "width": 1, "depth": 2}[suffix]] for tile in tiles]
        mla.pop(f"output_{suffix}", None)
    (output / "0_process_mla.json").write_text(json.dumps(mla, indent=4) + "\n")

    box = json.loads((arguments.template_dir / "0_boxdecoder.json").read_text())
    box.update({
        "num_classes": arguments.classes,
        "labels": names,
        "detection_threshold": arguments.threshold,
        "nms_iou_threshold": arguments.nms_iou,
        "topk": arguments.topk,
        "original_width": source_width, "original_height": source_height,
        "model_width": arguments.model_size, "model_height": arguments.model_size,
        "input_width": list(grids), "input_height": list(heights),
        "input_depth": list(depth),
        "slice_width": [tile[1] for tile in tiles], "slice_height": [tile[0] for tile in tiles],
        "slice_depth": [tile[2] for tile in tiles],
        "dq_scale": [value[0] for value in numbers["dq"]],
        "dq_zp": [int(value[1]) for value in numbers["dq"]],
    })
    if arguments.decode_type:
        box["decode_type"] = arguments.decode_type
    if arguments.sigmoid is not None:
        box["sigmoid_on_probabilities"] = arguments.sigmoid == "true"
    box["buffers"] = {"input": [{"name": "simaai_process_mla", "size": numbers["mla_size"]}],
                      "output": box.get("buffers", {}).get("output", {"size": 580})}
    (output / "boxdecoder.json").write_text(json.dumps(box, indent=4) + "\n")

    try:
        assert_geometry_matches(mla, box)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps({
        "input_scale": numbers["input_scale"], "input_zero": numbers["input_zero"],
        "mla_buffer_bytes": numbers["mla_size"], "grids": grids,
        "input_tiles_hwc": numbers["input_tiles"], "output_tiles_hwc": tiles,
        "dq_scale": [round(value[0], 4) for value in numbers["dq"]],
        "dq_zp": [int(value[1]) for value in numbers["dq"]],
        "written": [str(output / name) for name in ("0_preproc.json", "0_process_mla.json", "boxdecoder.json")],
    }, indent=2))


if __name__ == "__main__":
    main()
