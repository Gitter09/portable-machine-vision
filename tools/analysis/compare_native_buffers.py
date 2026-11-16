"""Controlled board test: replay captured input bytes, then change only tiling.

No live config/model changes. Requires buffers from inspect_native_buffers.py.
Uses installed SDK software transforms and the same compiled ELF.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from portable_tensor_parity import SoftwareTransforms, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from mpk_parser import Parser
    from sima.operations import SimaModelRunner
    from ev_transforms import transforms as ops
    root = args.release.resolve()
    inventory = json.loads((root / 'release.json').read_text())
    for name, sha in inventory['assets'].items():
        if digest(root / 'assets' / name) != sha:
            raise ValueError('release asset mismatch: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    codec = SoftwareTransforms(json.loads((root / 'assets/provenance/mpk.json').read_text()), ops)
    pre = json.loads((root / 'assets/etc/0_preproc.json').read_text())
    tp, qp = codec.contract['tessellate'], codec.contract['quantize']
    shape = qp['input_shapes'][0]
    n = codec.contract['input_bytes']
    if np.prod(shape) != n:
        raise ValueError('test expects unpadded INT8 RGB input')
    native_tile = [pre['tile_height'], pre['tile_width'], pre['tile_channels']]
    records = json.loads((args.capture / 'buffers.json').read_text())['records']
    frames = sorted({r['frame_id'] for r in records if r['label'] == 'simaai_preprocess'})
    reports, arrays = [], {}
    runner = SimaModelRunner(Parser(str(root / 'assets/provenance/mpk.json')))
    runner.configure({'model_path': str(root / 'assets/share/processmla/yolov8_stage1_mla.elf')})
    for frame in frames:
        data = np.fromfile(args.capture / f'simaai_preprocess-{frame}.bin', np.uint8)
        if data.nbytes != 2 * n:
            raise ValueError('unexpected preprocessing buffer size')
        packed, plain = data[:n], data[n:].view(np.int8).reshape(shape)
        recovered_native = ops.detessellation(packed.view(np.int8), slice_shape=native_tile,
                                               frame_type='int8', frame_shape=shape, align_c16=False, cblock=False)
        recovered_compiler = ops.detessellation(packed.view(np.int8), frame_type='int8', frame_shape=shape,
                                                **codec.tile_kwargs(tp))
        match_native = float(np.mean(recovered_native == plain))
        match_compiler = float(np.mean(recovered_compiler == plain))
        if match_native != 1.0:
            raise ValueError('native layout hypothesis does not reproduce the companion RGB tensor')
        repacked = np.ascontiguousarray(ops.tessellation(plain, **codec.tile_kwargs(tp))).view(np.uint8).reshape(-1)
        if repacked.nbytes != n:
            raise ValueError('repacked input size differs')
        captured = np.fromfile(args.capture / f'simaai_process_mla-{frame}.bin', np.uint8)
        # The native runner may reuse its output allocation on the next call.
        original = np.ascontiguousarray(runner.run(packed)).view(np.uint8).reshape(-1).copy()
        corrected = np.ascontiguousarray(runner.run(repacked)).view(np.uint8).reshape(-1).copy()
        if captured.size != original.size:
            raise ValueError('captured/runtime raw output sizes differ')
        row = {'frame_id': frame, 'native_tile_hwc': native_tile, 'compiler_tile_hwc': tp['slice_shape'],
               'native_layout_matches_companion_fraction': match_native,
               'compiler_layout_matches_companion_fraction': match_compiler,
               'input_changed_byte_fraction': float(np.mean(packed != repacked)),
               'replayed_original_matches_captured_output_fraction': float(np.mean(original == captured)),
               'repacked_changes_only_layout': True}
        for name, raw in [('captured', captured), ('replayed', original), ('repacked', corrected)]:
            for i, head in enumerate(codec.decode(raw)):
                arrays[f'frame_{frame}_{name}_head_{i}'] = head
        arrays[f'frame_{frame}_prepared_input'] = ops.dequantize(plain, channel_params=qp['channel_params'])
        reports.append(row)
        print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / 'heads.npz', **arrays)
    (args.output / 'comparison.json').write_text(json.dumps({
        'scope': 'captured-frame controlled tiling comparison, no live configuration change',
        'release': str(root), 'elf_sha256': digest(root / 'assets/share/processmla/yolov8_stage1_mla.elf'),
        'frames': reports, 'heads_sha256': digest(args.output / 'heads.npz'),
    }, indent=2))


if __name__ == '__main__':
    main()
