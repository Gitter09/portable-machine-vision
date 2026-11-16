"""Portable-only DFL diagnostic: never use the fixed station's 4-distance decode."""
import argparse
import json

import numpy as np
import pytest

from tools.analysis.portable_tensor_parity import LABELS, SoftwareTransforms, decode_heads, digest, prepare, transform_contract


def heads():
    return ([np.zeros((1, 64, grid, grid), dtype=np.float32) for grid in (4, 2, 1)]
            + [np.zeros((1, 4, grid, grid), dtype=np.float32) for grid in (4, 2, 1)])


def tiny_mpk():
    def plugin(name, kernel, params, source, outputs, processor='EV74'):
        return {'name': name, 'processor': processor,
                'config_params': {'kernel': kernel, 'params': params},
                'input_nodes': [{'name': source, 'size': 3072}], 'output_nodes': outputs}
    def output(name, size):
        return {'name': name, 'size': size}
    quant = {'num_bits': 8, 'rounding': 'TONEAREST', 'output_data_type': 'int8',
             'channel_params': [[2., -1]], 'input_shapes': [[1, 32, 32, 3]]}
    tess = {'slice_shape': [32, 8, 3], 'align_c16': False, 'cblock': False,
            'frame_type': 'int8', 'input_shapes': [[1, 32, 32, 3]]}
    sizes = [4*4*64, 2*2*64, 64, 4*4*16, 2*2*16, 16]
    plugins = [plugin('q', 'quantization_transform', quant, 'images', [output('q', 3072)]),
               plugin('t', 'tessellation_transform', tess, 'q', [output('t', 3072)]),
               plugin('mla', None, {}, 't', [output('mla', sum(sizes))], 'MLA'),
               plugin('unpack', 'unpack_transform', {}, 'mla', [output(f'packed_{i}', n) for i, n in enumerate(sizes)])]
    for i, grid in enumerate((4, 2, 1, 4, 2, 1)):
        channels = 64 if i < 3 else 4
        shape = [1, grid, grid, channels]
        detess = {'slice_shape': [1, grid, channels], 'frame_shape': shape, 'frame_type': 'int8',
                  'align_c16': True, 'cblock': True}
        dequant = {'channel_params': [[2., 0]], 'output_shapes': [shape]}
        name = f'bbox_{i}' if i < 3 else f'class_prob_{i-3}'
        plugins += [plugin(f'dt_{i}', 'detessellation_transform', detess, f'packed_{i}', [output(f'dt_{i}', np.prod(shape))]),
                    plugin(f'dq_{i}', 'dequantization_transform', dequant, f'dt_{i}', [output('dequant/'+name, 4*np.prod(shape))])]
    return {'plugins': plugins}


class FakeSoftwareSDK:
    """Test only dispatch/byte boundaries, not the proprietary SDK tile algorithm.

    Actual SDK tiling has its own on-board roundtrip preflight before inference.
    """
    @staticmethod
    def quantize(frame, channel_params, num_bits):
        assert num_bits == 8 and channel_params == [[2., -1]]
        return np.clip(np.rint(frame * 2 - 1), -128, 127).astype(np.int8)

    @staticmethod
    def tessellation(frame, slice_shape, align_c16, cblock):
        assert len(slice_shape) == 3 and (not cblock or align_c16)
        padding = (-frame.shape[-1]) % 16 if align_c16 else 0
        return np.pad(frame, ((0, 0), (0, 0), (0, 0), (0, padding))).reshape(1, -1)

    @staticmethod
    def detessellation(packed, frame_shape, frame_type, slice_shape, align_c16, cblock):
        channels = ((frame_shape[-1] + 15)//16)*16 if align_c16 else frame_shape[-1]
        return np.asarray(packed).view(frame_type).reshape(*frame_shape[:-1], channels)[..., :frame_shape[-1]]

    @staticmethod
    def dequantize(frame, channel_params):
        assert channel_params == [[2., 0]]
        return frame.astype(np.float32) / 2


def test_software_contract_uses_compiler_input_tile_not_native_parser_guess():
    contract = transform_contract(tiny_mpk())
    assert contract['tessellate']['slice_shape'] == [32, 8, 3]
    assert contract['input_bytes'] == 3072
    assert contract['heads'][3]['packed_bytes'] == 256  # logical class bytes are only 64


def test_software_input_roundtrip_and_output_unpack_padding():
    codec = SoftwareTransforms(tiny_mpk(), FakeSoftwareSDK)
    encoded = codec.encode(np.ones((32, 32, 3), dtype=np.float32))
    assert encoded.dtype == np.uint8 and encoded.size == 3072
    np.testing.assert_array_equal(encoded, 1)
    codec.check_output_layouts()
    raw = np.concatenate([np.full(h['packed_bytes'], i, np.int8) for i, h in enumerate(codec.contract['heads'])])
    outputs = codec.decode(raw)
    assert len(outputs) == 6
    for i, output in enumerate(outputs):
        np.testing.assert_array_equal(output, i / 2)
        assert list(output.shape) == codec.contract['heads'][i]['dequantize']['output_shapes'][0]


def test_software_codec_rejects_bad_frame_and_raw_sizes():
    codec = SoftwareTransforms(tiny_mpk(), FakeSoftwareSDK)
    with pytest.raises(ValueError, match='geometry'):
        codec.encode(np.zeros((16, 16, 3), np.float32))
    with pytest.raises(ValueError, match='byte count'):
        codec.decode(np.zeros(7, np.uint8))


@pytest.mark.parametrize('mutation', ['rounding', 'duplicate', 'head_order', 'output_size', 'multi_mla'])
def test_unsupported_compiler_contract_is_rejected(mutation):
    mpk = tiny_mpk()
    if mutation == 'rounding': mpk['plugins'][0]['config_params']['params']['rounding'] = 'GUESS'
    if mutation == 'duplicate': mpk['plugins'].append(mpk['plugins'][0])
    if mutation == 'head_order': mpk['plugins'][5]['output_nodes'][0]['name'] = 'dequant/class_prob_0'
    if mutation == 'output_size': mpk['plugins'][2]['output_nodes'][0]['size'] += 1
    if mutation == 'multi_mla':
        mpk['plugins'].append({'name': 'another_mla', 'processor': 'MLA'})
    with pytest.raises(ValueError):
        transform_contract(mpk)


def test_uniform_dfl_has_mean_distance_seven_and_half():
    result = decode_heads(heads(), side=32)
    assert result.shape == (1, 8, 21)
    np.testing.assert_allclose(result[0, :4, 0], [4, 4, 120, 120])
    np.testing.assert_allclose(result[0, :4, 16], [8, 8, 240, 240])
    np.testing.assert_allclose(result[0, :4, 20], [16, 16, 480, 480])
    np.testing.assert_allclose(result[:, 4:], .5)


def test_known_distances_grid_order_and_four_class_ids():
    values = heads()
    for side, bin_index in enumerate([1, 2, 3, 4]):
        values[0][:, side * 16:(side + 1) * 16] = -1000
        values[0][:, side * 16 + bin_index] = 1000
    values[3][:] = -1000
    values[3][:, 3] = 1000
    result = decode_heads(values, side=32)
    np.testing.assert_allclose(result[0, :4, 0], [12, 12, 32, 48])
    np.testing.assert_allclose(result[0, :2, 1], [20, 12])  # columns advance first
    assert np.argmax(result[0, 4:, 0]) == 3
    assert LABELS == ['IC_Fail', 'Terminal_pass', 'IC_Pass', 'Terminal_Fail']
    assert np.isfinite(result).all()


def test_nhwc_hardware_and_nchw_reference_are_identical():
    rng = np.random.default_rng(7)
    values = [rng.normal(size=h.shape).astype(np.float32) for h in heads()]
    a = decode_heads(values, 32)
    b = decode_heads([h.transpose(0, 2, 3, 1) for h in values], 32, 'NHWC')
    np.testing.assert_array_equal(a, b)
    assert (a[:, 2:4] >= 0).all()


@pytest.mark.parametrize('side,layout', [(0, 'NCHW'), (31, 'NCHW'), (32, 'guess'), (True, 'NCHW')])
def test_invalid_geometry_or_layout(side, layout):
    with pytest.raises(ValueError):
        decode_heads(heads(), side, layout)


def test_four_distance_fixed_station_heads_are_rejected():
    values = heads()
    values[0] = values[0][:, :4]
    with pytest.raises(ValueError, match='tensor 0'):
        decode_heads(values, 32)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_outputs_are_rejected(value):
    values = heads()
    values[5][0, 0, 0, 0] = value
    with pytest.raises(ValueError):
        decode_heads(values, 32)


def test_incomplete_or_integer_outputs_are_rejected():
    with pytest.raises(ValueError):
        decode_heads(heads()[:5], 32)
    values = heads()
    values[0] = values[0].astype(np.int8)
    with pytest.raises(ValueError):
        decode_heads(values, 32)


def test_preparation_rgb_normalisation_padding_and_fingerprint(tmp_path):
    import cv2
    image = np.full((8, 16, 3), [10, 20, 30], dtype=np.uint8)
    path = tmp_path / 'source.png'
    assert cv2.imwrite(str(path), image)
    output = tmp_path / 'prepared'
    args = argparse.Namespace(images=[path], output=output, side=32)
    prepare(args)
    with np.load(output / 'inputs.npz', allow_pickle=False) as values:
        tensor = values['input_0']
        assert tensor.shape == (32, 32, 3) and tensor.dtype == np.float32
        np.testing.assert_allclose(tensor[8, 0], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(tensor[0, 0], 114 / 255)
    manifest = json.loads((output / 'inputs.json').read_text())
    assert manifest['inputs_sha256'] == digest(output / 'inputs.npz')
    assert manifest['images'][0]['padding_left_top'] == [0, 8]
    with pytest.raises(FileExistsError):
        prepare(args)
