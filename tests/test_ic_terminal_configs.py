"""The config generator must read the compiler's numbers, not invent them."""

from __future__ import annotations

import json
import sys

import pytest

from tools.build_ic_terminal_configs import compiler_numbers, main


def sample_mpk(dq_scales: list[float] | None = None) -> dict:
    scales = dq_scales or [25.9, 19.0, 20.7, 10.7, 12.0, 33.6]
    plugins = [
        {"name": "quantize_0", "config_params": {"params": {"channel_params": [[255.5, -128]]}}},
        {"name": "tessellate_quantize_0_MLA_0/placeholder_0_0_tessellation_transform",
         "config_params": {"params": {}}},
        {"name": "MLA_0", "config_params": {"params": {}}},
        {"name": "MLA_0_ofm_unpack_transform",
         "config_params": {"params": {"input_shapes": [[1, 1075200]]}}},
    ]
    for index in range(6):
        depth = 64 if index < 3 else 4
        plugins.append({
            "name": f"detessellate_MLA_0/tuple_get_item_{index}_detessellation_transform",
            "config_params": {"params": {"input_shapes": [[1, 1]]}},
        })
        plugins.append({
            "name": f"dequantize_{index + 2}",
            "config_params": {"params": {"channel_params": [[scales[index], -100 - index]]}},
        })
    return {"plugins": plugins}


def shaped_mpk(depths: list[int], grids: list[int]) -> dict:
    """An MPK whose detessellation plugins publish the unpacked tensor shapes."""
    mpk = sample_mpk()
    for plugin in mpk["plugins"]:
        for index in range(6):
            if f"tuple_get_item_{index}_" in plugin["name"]:
                plugin["config_params"]["params"]["output_shapes"] = [
                    [1, grids[index], grids[index], depths[index]]]
    return mpk


def test_reads_input_scale_and_mla_buffer():
    numbers = compiler_numbers(sample_mpk(), 1280)
    assert numbers["input_scale"] == 255.5
    assert numbers["input_zero"] == -128
    assert numbers["mla_size"] == 1075200


def test_grids_follow_the_model_size():
    assert compiler_numbers(sample_mpk(), 1280)["grids"] == [160, 80, 40, 160, 80, 40]
    assert compiler_numbers(sample_mpk(), 640)["grids"] == [80, 40, 20, 80, 40, 20]


def test_dequantisation_is_kept_in_tensor_order():
    numbers = compiler_numbers(sample_mpk(), 1280)
    assert [round(value[0], 1) for value in numbers["dq"]] == [25.9, 19.0, 20.7, 10.7, 12.0, 33.6]
    assert [int(value[1]) for value in numbers["dq"]] == [-100, -101, -102, -103, -104, -105]


def test_missing_dequantisation_is_rejected():
    mpk = sample_mpk()
    mpk["plugins"] = [plugin for plugin in mpk["plugins"] if plugin["name"] != "dequantize_5"]
    try:
        compiler_numbers(mpk, 1280)
    except SystemExit as error:
        assert "six dequantised tensors" in str(error)
    else:
        raise AssertionError("expected a rejection when a tensor is missing")


def test_depths_come_from_the_compiler_not_the_flags():
    numbers = compiler_numbers(shaped_mpk([64, 64, 64, 4, 4, 4], [160, 80, 40] * 2), 1280)
    assert numbers["depths"] == [64, 64, 64, 4, 4, 4]
    assert numbers["grids"] == [160, 80, 40, 160, 80, 40]
    assert numbers["heights"] == [160, 80, 40, 160, 80, 40]


def test_compiler_grids_must_match_the_model_size():
    mpk = shaped_mpk([64, 64, 64, 4, 4, 4], [80, 40, 20] * 2)
    try:
        compiler_numbers(mpk, 1280)
    except SystemExit as error:
        assert "disagree with --model-size" in str(error)
    else:
        raise AssertionError("a 640 graph must not be configured as 1280")


def test_unshaped_mpk_falls_back_to_the_model_size():
    numbers = compiler_numbers(sample_mpk(), 1280)
    assert numbers["depths"] is None
    assert numbers["grids"] == [160, 80, 40, 160, 80, 40]


def tiled_mpk():
    mpk = shaped_mpk([64, 64, 64, 4, 4, 4], [160, 80, 40] * 2)
    mpk['plugins'][1]['config_params']['params'] = {
        'slice_shape': [1280, 64, 3], 'align_c16': False, 'cblock': False}
    for plugin in mpk['plugins']:
        if 'tuple_get_item_' in plugin['name']:
            index = int(plugin['name'].split('tuple_get_item_')[1].split('_')[0])
            grid = [160, 80, 40][index % 3]
            plugin['config_params']['params']['slice_shape'] = [
                [2, 1, 1][index] if index < 3 else grid, grid, 64 if index < 3 else 4]
    return mpk


def test_compiler_input_and_output_tiles_remain_distinct_from_frames():
    result = compiler_numbers(tiled_mpk(), 1280)
    assert result['input_tiles'] == [1280, 64, 3]
    assert result['output_tiles'] == [[2, 160, 64], [1, 80, 64], [1, 40, 64],
                                      [160, 160, 4], [80, 80, 4], [40, 40, 4]]
    assert result['heights'] == [160, 80, 40] * 2


def test_cli_writes_compiler_tiles_not_template_tiles(tmp_path, monkeypatch):
    mpk = tmp_path / 'mpk.json'
    mpk.write_text(json.dumps(tiled_mpk()))
    template = tmp_path / 'template'
    template.mkdir()
    (template / '0_preproc.json').write_text(json.dumps({
        'tile_width': 128, 'tile_height': 32, 'caps': {'sink_pads': []}}))
    (template / '0_process_mla.json').write_text(json.dumps({'simaai__params': {'outputs': [{'size': 0}]}}))
    (template / '0_boxdecoder.json').write_text(json.dumps({'buffers': {'output': {'size': 2404}}}))
    labels = tmp_path / 'labels'
    labels.write_text('IC_Fail\nTerminal_pass\nIC_Pass\nTerminal_Fail\n')
    output = tmp_path / 'generated'
    monkeypatch.setattr(sys, 'argv', ['configs', '--mpk', str(mpk), '--template-dir', str(template),
        '--output-dir', str(output), '--box-channels', '64', '--labels-file', str(labels)])
    main()
    pre = json.loads((output / '0_preproc.json').read_text())
    mla = json.loads((output / '0_process_mla.json').read_text())
    box = json.loads((output / 'boxdecoder.json').read_text())
    assert (pre['tile_height'], pre['tile_width'], pre['tile_channels']) == (1280, 64, 3)
    assert mla['input_height'] == box['input_height'] == [160, 80, 40] * 2
    assert mla['slice_height'] == box['slice_height'] == [2, 1, 1, 160, 80, 40]


def test_cli_rejects_missing_tiles_before_writing(tmp_path, monkeypatch):
    mpk = tmp_path / 'mpk.json'
    mpk.write_text(json.dumps(sample_mpk()))
    labels = tmp_path / 'labels'
    labels.write_text('IC_Fail\nTerminal_pass\nIC_Pass\nTerminal_Fail\n')
    output = tmp_path / 'generated'
    monkeypatch.setattr(sys, 'argv', ['configs', '--mpk', str(mpk), '--template-dir', str(tmp_path),
        '--output-dir', str(output), '--box-channels', '64', '--labels-file', str(labels)])
    with pytest.raises(SystemExit, match='tessellation geometry'):
        main()
    assert not output.exists()
