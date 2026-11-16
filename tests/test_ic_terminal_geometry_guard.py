"""The geometry guard must reject the config pair that shipped the defect."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_ic_terminal_configs import assert_geometry_matches

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "unpack_640_bug"


def buggy_pair() -> tuple[dict, dict]:
    mla = json.loads((FIXTURE / "0_process_mla.json").read_text())
    box = json.loads((FIXTURE / "boxdecoder.json").read_text())
    return mla, box


def declared_bytes(mla: dict) -> int:
    return sum(w * h * d for w, h, d in
               zip(mla["input_width"], mla["input_height"], mla["input_depth"]))


def test_the_preserved_bad_pair_is_rejected():
    mla, box = buggy_pair()
    with pytest.raises(ValueError) as error:
        assert_geometry_matches(mla, box)
    assert "geometry disagrees" in str(error.value)


def test_nothing_checked_the_sizes_that_were_different():
    """The two configs described different-sized tensor sets.

    The unpack node declared 80/40/20 grids at depth 4: 67,200 bytes of tensor
    geometry. The box decoder declared 160/80/40 at depths 64/64/64/4/4/4:
    2,284,800 bytes. Neither equals the 2,688,000-byte MLA buffer, which the
    compiler pads. Nothing compared any of these numbers, which is why the run
    started and decoded garbage.
    """
    mla, box = buggy_pair()
    assert declared_bytes(mla) == 67_200
    assert sum(w * h * d for w, h, d in
               zip(box["slice_width"], box["slice_height"],
                   box["slice_depth"])) == 2_284_800
    assert box["buffers"]["input"][0]["size"] == 2_688_000
    assert declared_bytes(mla) != 2_284_800


def test_names_a_field_that_disagrees():
    mla, box = buggy_pair()
    with pytest.raises(ValueError) as error:
        assert_geometry_matches(mla, box)
    message = str(error.value)
    assert any(field in message for field in ("width", "height", "depth"))


def test_a_depth_only_mismatch_is_caught():
    mla, box = buggy_pair()
    for field in ("width", "height"):
        mla[f"input_{field}"] = list(box[f"input_{field}"])
        box[f"slice_{field}"] = list(box[f"input_{field}"])
    with pytest.raises(ValueError) as error:
        assert_geometry_matches(mla, box)
    assert "depth" in str(error.value)


def test_matching_geometry_passes():
    mla, box = buggy_pair()
    for field in ("width", "height", "depth"):
        mla[f"input_{field}"] = list(box[f"input_{field}"])
        mla[f"slice_{field}"] = list(box[f"input_{field}"])
        box[f"slice_{field}"] = list(box[f"input_{field}"])
    assert_geometry_matches(mla, box)


def test_valid_physical_tiles_need_not_equal_logical_tensor_geometry():
    mla, box = buggy_pair()
    for field in ('width', 'height', 'depth'):
        values = [64, 64, 64, 4, 4, 4] if field == 'depth' else [160, 80, 40] * 2
        mla[f'input_{field}'] = box[f'input_{field}'] = values
        tile = [2, 1, 1, 160, 80, 40] if field == 'height' else values
        mla[f'slice_{field}'] = box[f'slice_{field}'] = tile
    assert_geometry_matches(mla, box)
    box['slice_height'] = box['input_height']
    with pytest.raises(ValueError, match='tile geometry'):
        assert_geometry_matches(mla, box)
