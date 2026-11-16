"""Compare the portable ELF and FP32 exports on byte-identical prepared inputs.

Diagnostic only: no camera, GStreamer boxdecode, installed service, or fixed
station changes. Board mode needs installed SiMa Python APIs; compare mode needs
ONNX Runtime/OpenCV. All output directories must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LABELS = ['IC_Fail', 'Terminal_pass', 'IC_Pass', 'Terminal_Fail']
STRIDES = (8, 16, 32)
HEAD_NAMES = [f'bbox_{i}' for i in range(3)] + [f'class_prob_{i}' for i in range(3)]


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def decode_heads(heads, side=1280, layout='NCHW'):
    """YOLO11 DFL: 4 sides x 16 bins, followed by four class logits.

    Returns model-space cx,cy,w,h and class probabilities, shape [1,8,N].
    This is not the fixed station's four-distance decoder.
    """
    if type(side) is not int or side < 32 or side % 32 or layout not in ('NCHW', 'NHWC'):
        raise ValueError('invalid model side or layout')
    if len(heads) != 6:
        raise ValueError('expected six grouped box/class tensors')
    tensors = []
    for i, value in enumerate(heads):
        array = np.asarray(value)
        grid = side // STRIDES[i % 3]
        depth = 64 if i < 3 else len(LABELS)
        expected = (1, depth, grid, grid) if layout == 'NCHW' else (1, grid, grid, depth)
        if array.shape != expected or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f'invalid tensor {i}: {array.shape}, expected {expected}, finite floats')
        tensors.append(array if layout == 'NCHW' else array.transpose(0, 3, 1, 2))
    predictions = []
    for i, stride in enumerate(STRIDES):
        grid = side // stride
        logits = tensors[i].reshape(1, 4, 16, -1).astype(np.float32)
        probability = np.exp(logits - logits.max(axis=2, keepdims=True))
        probability /= probability.sum(axis=2, keepdims=True)
        distance = (probability * np.arange(16, dtype=np.float32)[None, None, :, None]).sum(axis=2)
        yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
        anchor = np.stack((xx + .5, yy + .5)).reshape(1, 2, -1).astype(np.float32)
        lower, upper = anchor - distance[:, :2], anchor + distance[:, 2:]
        boxes = np.concatenate(((lower + upper) / 2, upper - lower), axis=1) * stride
        scores = 1 / (1 + np.exp(-np.clip(tensors[i + 3].reshape(1, len(LABELS), -1), -80, 80)))
        predictions.append(np.concatenate((boxes, scores), axis=1))
    return np.concatenate(predictions, axis=2)


def detection_rows(prediction, threshold=.05, iou=.7):
    import cv2
    values = prediction[0].T
    ids = values[:, 4:].argmax(axis=1)
    scores = values[np.arange(len(values)), ids + 4]
    selected = scores >= threshold
    boxes = values[selected, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    scores, ids = scores[selected], ids[selected]
    keep = np.asarray(cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), ids.tolist(), threshold, iou)).reshape(-1)
    keep = sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:300]
    return [{'class_id': int(ids[i]), 'label': LABELS[int(ids[i])], 'score': float(scores[i]),
             'model_xywh': boxes[i].tolist()} for i in keep]


def prepare(args):
    import cv2
    args.output.mkdir(parents=True, exist_ok=False)
    arrays, rows = {}, []
    for i, path in enumerate(args.images):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('unreadable image: ' + str(path))
        height, width = image.shape[:2]
        scale = min(args.side / width, args.side / height)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        dx, dy = (args.side - resized.shape[1]) / 2, (args.side - resized.shape[0]) / 2
        left, right, top, bottom = round(dx - .1), round(dx + .1), round(dy - .1), round(dy + .1)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        arrays[f'input_{i}'] = np.ascontiguousarray(padded[:, :, ::-1], dtype=np.float32) / np.float32(255)
        rows.append({'key': f'input_{i}', 'image': str(path.resolve()), 'image_sha256': digest(path),
                     'original_size': [width, height], 'scale': scale, 'padding_left_top': [left, top]})
    np.savez_compressed(args.output / 'inputs.npz', **arrays)
    manifest = {'side': args.side, 'labels': LABELS, 'preprocessing': 'RGB float32 /255, centered padding 114',
                'inputs_sha256': digest(args.output / 'inputs.npz'), 'images': rows}
    (args.output / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def transform_contract(mpk):
    """Extract this single-MLA portable graph's transforms, never infer tiles.

    The SDK parser is still used for the model runner, but its CVU preparation
    config is not used to interpret compiler tiling or padded output buffers.
    """
    plugins = mpk['plugins']
    by_name = {p['name']: p for p in plugins}
    if len(by_name) != len(plugins):
        raise ValueError('duplicate MPK plugin names')

    def kernel(plugin):
        return plugin.get('config_params', {}).get('kernel')

    def parent(plugin):
        if len(plugin['input_nodes']) != 1:
            raise ValueError('expected one parent for ' + plugin['name'])
        return by_name[plugin['input_nodes'][0]['name']]

    mlas = [p for p in plugins if p['processor'] == 'MLA']
    if len(mlas) != 1:
        raise ValueError('diagnostic supports exactly one MLA segment')
    mla = mlas[0]
    tess = parent(mla)
    quant = parent(tess)
    if kernel(tess) != 'tessellation_transform' or kernel(quant) != 'quantization_transform':
        raise ValueError('unexpected input transform chain')
    qp, tp = quant['config_params']['params'], tess['config_params']['params']
    if qp['num_bits'] != 8 or qp['output_data_type'] != 'int8' or qp['rounding'] != 'TONEAREST' or tp['frame_type'] != 'int8':
        raise ValueError('unsupported input quantization contract')
    unpacks = [p for p in plugins if kernel(p) == 'unpack_transform'
               and len(p['input_nodes']) == 1 and p['input_nodes'][0]['name'] == mla['name']]
    if len(unpacks) != 1 or len(unpacks[0]['output_nodes']) != 6:
        raise ValueError('expected one six-tensor output unpack')
    unpack = unpacks[0]
    result = {'quantize': qp, 'tessellate': tp, 'input_bytes': mla['input_nodes'][0]['size'],
              'output_bytes': mla['output_nodes'][0]['size'], 'heads': []}
    for i, output in enumerate(unpack['output_nodes']):
        detess = [p for p in plugins if kernel(p) == 'detessellation_transform'
                  and len(p['input_nodes']) == 1 and p['input_nodes'][0]['name'] == output['name']]
        if len(detess) != 1:
            raise ValueError('ambiguous output detessellation')
        detess = detess[0]
        dequant = [p for p in plugins if kernel(p) == 'dequantization_transform'
                   and len(p['input_nodes']) == 1 and p['input_nodes'][0]['name'] == detess['output_nodes'][0]['name']]
        if len(dequant) != 1 or dequant[0]['output_nodes'][0]['name'].rsplit('/', 1)[-1] != HEAD_NAMES[i]:
            raise ValueError('unexpected output head order')
        result['heads'].append({'name': HEAD_NAMES[i], 'packed_bytes': output['size'],
                                'detessellate': detess['config_params']['params'],
                                'dequantize': dequant[0]['config_params']['params']})
    if sum(h['packed_bytes'] for h in result['heads']) != result['output_bytes']:
        raise ValueError('unpack sizes do not cover MLA output')
    return result


class SoftwareTransforms:
    """Call installed SDK software transforms with the original MPK parameters."""

    def __init__(self, mpk, operations):
        self.contract = transform_contract(mpk)
        self.ops = operations

    @staticmethod
    def tile_kwargs(params):
        return {key: params[key] for key in ('slice_shape', 'align_c16', 'cblock')}

    def encode(self, tensor):
        qp, tp = self.contract['quantize'], self.contract['tessellate']
        frame = np.ascontiguousarray(tensor[None])
        if list(frame.shape) != qp['input_shapes'][0]:
            raise ValueError('software input geometry differs from compiler')
        quantized = self.ops.quantize(frame, channel_params=qp['channel_params'], num_bits=qp['num_bits'])
        if quantized.dtype != np.int8:
            raise ValueError('unexpected software quantization dtype')
        packed = np.ascontiguousarray(self.ops.tessellation(quantized, **self.tile_kwargs(tp)))
        if packed.nbytes != self.contract['input_bytes']:
            raise ValueError('software input byte count differs from compiler')
        restored = self.ops.detessellation(packed, frame_shape=tp['input_shapes'][0],
                                           frame_type=tp['frame_type'], **self.tile_kwargs(tp))
        if not np.array_equal(restored, quantized):
            raise ValueError('software input tessellation roundtrip failed')
        return packed.view(np.uint8).reshape(-1)

    def decode(self, raw):
        packed = np.ascontiguousarray(raw).view(np.int8).reshape(-1)
        if packed.nbytes != self.contract['output_bytes']:
            raise ValueError('raw MLA output byte count differs from compiler')
        outputs, offset = [], 0
        for head in self.contract['heads']:
            dp, qp = head['detessellate'], head['dequantize']
            end = offset + head['packed_bytes']
            frame = self.ops.detessellation(packed[offset:end], frame_shape=dp['frame_shape'],
                                            frame_type=dp['frame_type'], **self.tile_kwargs(dp))
            output = self.ops.dequantize(frame, channel_params=qp['channel_params'])
            if list(output.shape) != qp['output_shapes'][0] or not np.isfinite(output).all():
                raise ValueError('invalid software-decoded output')
            outputs.append(output.astype(np.float32, copy=False))
            offset = end
        return outputs

    def check_output_layouts(self):
        """Validate SDK tile/padding inverses on synthetic integer heads first."""
        for head in self.contract['heads']:
            dp = head['detessellate']
            shape = tuple(dp['frame_shape'])
            frame = ((np.arange(np.prod(shape), dtype=np.int32) % 255) - 127).astype(np.int8).reshape(shape)
            packed = self.ops.tessellation(frame, **self.tile_kwargs(dp))
            if packed.nbytes != head['packed_bytes']:
                raise ValueError('output tile padding disagrees with compiler')
            restored = self.ops.detessellation(packed, frame_shape=dp['frame_shape'], frame_type=dp['frame_type'],
                                               **self.tile_kwargs(dp))
            if not np.array_equal(restored, frame):
                raise ValueError('software output tessellation roundtrip failed')


def board(args):
    from mpk_parser import Parser
    from sima.operations import SimaModelRunner
    root = args.release.resolve()
    inventory = json.loads((root / 'release.json').read_text())
    for name, sha in inventory['assets'].items():
        target = (root / 'assets' / name).resolve()
        if not target.is_relative_to(root / 'assets') or digest(target) != sha:
            raise ValueError('release asset mismatch: ' + name)
    manifest = json.loads(args.inputs.with_suffix('.json').read_text())
    if manifest['inputs_sha256'] != digest(args.inputs) or manifest['labels'] != LABELS:
        raise ValueError('input hash or class contract mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    print('STAGE parse_mpk', flush=True)
    parser = Parser(str(root / 'assets/provenance/mpk.json'))
    if args.transforms == 'software':
        from ev_transforms import transforms
        print('STAGE software_transform_preflight', flush=True)
        software = SoftwareTransforms(json.loads((root / 'assets/provenance/mpk.json').read_text()), transforms)
        (args.output / 'software-transform-contract.json').write_text(json.dumps(software.contract, indent=2))
        software.check_output_layouts()
        with np.load(args.inputs, allow_pickle=False) as samples:
            software.encode(samples[manifest['images'][0]['key']])
        encode, decode = software.encode, software.decode
    else:
        from sima.operations import SimaQuantTess, SimaDetessDequant
        print('STAGE configure_native_quantizer (known to abort in previous trial)', flush=True)
        quant = SimaQuantTess(parser)
        (args.output / 'quantizer-config.json').write_text(json.dumps(quant.configuration, indent=2))
        quant.configure({})
        dequant = SimaDetessDequant(parser)
        encode, decode = quant.run, dequant.run
    print('STAGE construct_model_runner', flush=True)
    runner = SimaModelRunner(parser)
    (args.output / 'runner-config.json').write_text(json.dumps(runner.configuration, indent=2))
    print('STAGE configure_model_runner', flush=True)
    runner.configure({'model_path': str(root / 'assets/share/processmla/yolov8_stage1_mla.elf')})
    outputs, rows = {}, []
    with np.load(args.inputs, allow_pickle=False) as samples:
        for item in manifest['images']:
            key = item['key']
            tensor = samples[key]
            if tensor.shape != (manifest['side'], manifest['side'], 3) or tensor.dtype != np.float32 or not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError('invalid prepared input ' + key)
            start = time.perf_counter()
            print('STAGE quantize ' + key, flush=True)
            quantized = encode(tensor)
            q = time.perf_counter()
            print('STAGE infer ' + key, flush=True)
            raw = runner.run(quantized)
            r = time.perf_counter()
            print('STAGE dequantize ' + key, flush=True)
            heads = decode(raw)
            end = time.perf_counter()
            decode_heads(heads, manifest['side'], 'NHWC')  # validate shapes/finite values
            for i, head in enumerate(heads):
                outputs[f'{key}_head_{i}'] = head
            rows.append({'key': key, 'head_shapes': [list(v.shape) for v in heads],
                         'timings_ms': {'quantize_tessellate': (q-start)*1000, 'mla': (r-q)*1000,
                                        'detessellate_dequantize': (end-r)*1000}})
            print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(args.output / 'outputs.npz', **outputs)
    report = {'release': str(root), 'release_inventory_sha256': digest(root / 'release.json'),
              'elf_sha256': digest(root / 'assets/share/processmla/yolov8_stage1_mla.elf'),
              'mpk_sha256': digest(root / 'assets/provenance/mpk.json'),
              'inputs_sha256': digest(args.inputs), 'outputs_sha256': digest(args.output / 'outputs.npz'),
              'side': manifest['side'], 'labels': LABELS, 'transforms': args.transforms, 'samples': rows}
    (args.output / 'board.json').write_text(json.dumps(report, indent=2))


def compare(args):
    import onnxruntime as ort
    manifest = json.loads(args.inputs.with_suffix('.json').read_text())
    device = json.loads((args.board / 'board.json').read_text())
    sha = digest(args.inputs)
    if sha != manifest['inputs_sha256'] or sha != device['inputs_sha256'] or digest(args.board / 'outputs.npz') != device['outputs_sha256']:
        raise ValueError('input/output fingerprint mismatch')
    if manifest['labels'] != LABELS or device['labels'] != LABELS or manifest['side'] != device['side']:
        raise ValueError('class/geometry mismatch')
    provenance = json.loads(args.compilation.read_text())
    if digest(args.raw) != provenance['source_onnx_sha256'] or device['elf_sha256'] != provenance['artifacts']['yolov8_stage1_mla.elf'] or device['mpk_sha256'] != provenance['artifacts']['yolov8_mpk.json']:
        raise ValueError('ELF/MPK/raw ONNX compilation lineage mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    raw = ort.InferenceSession(str(args.raw), sess_options=options, providers=['CPUExecutionProvider'])
    original = ort.InferenceSession(str(args.original), sess_options=options, providers=['CPUExecutionProvider'])
    if [v.name for v in raw.get_outputs()] != HEAD_NAMES:
        raise ValueError('unexpected raw ONNX output ordering')
    report = {'scope': 'identical_prepared_input_accelerator_comparison_not_accuracy', 'labels': LABELS,
              'inputs_sha256': sha, 'raw_onnx_sha256': digest(args.raw), 'original_onnx_sha256': digest(args.original),
              'board_report': device, 'confidence': .05, 'nms_iou': .7, 'samples': []}
    with np.load(args.inputs, allow_pickle=False) as samples, np.load(args.board / 'outputs.npz', allow_pickle=False) as hardware:
        for item in manifest['images']:
            key = item['key']
            tensor = np.ascontiguousarray(samples[key].transpose(2, 0, 1)[None])
            ref_heads = raw.run(None, {raw.get_inputs()[0].name: tensor})
            hw_heads = [hardware[f'{key}_head_{i}'] for i in range(6)]
            reference = decode_heads(ref_heads, manifest['side'])
            actual = decode_heads(hw_heads, manifest['side'], 'NHWC')
            uncut = original.run(None, {original.get_inputs()[0].name: tensor})[0]
            if reference.shape != uncut.shape:
                raise ValueError('original/raw decode geometry mismatch')
            head_metrics = []
            for ref, hw in zip(ref_heads, hw_heads):
                delta = hw.transpose(0, 3, 1, 2) - ref
                head_metrics.append({'max_abs_error': float(np.abs(delta).max()), 'mean_abs_error': float(np.abs(delta).mean()),
                                     'reference_range': [float(ref.min()), float(ref.max())],
                                     'hardware_range': [float(hw.min()), float(hw.max())]})
            row = {'key': key, 'input_image': item['image'], 'head_metrics': head_metrics,
                   'fp32_cut_parity': {'max_box_error_pixels': float(np.abs(reference[:, :4] - uncut[:, :4]).max()),
                                      'max_score_error': float(np.abs(reference[:, 4:] - uncut[:, 4:]).max())},
                   'original_detections': detection_rows(uncut), 'hardware_python_detections': detection_rows(actual),
                   'hardware_max_score': float(actual[:, 4:].max()),
                   'reference_max_score': float(uncut[:, 4:].max())}
            report['samples'].append(row)
            print(key, 'FP32:', len(row['original_detections']), 'hardware/Python:', len(row['hardware_python_detections']),
                  'max scores:', row['reference_max_score'], row['hardware_max_score'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    p = modes.add_parser('prepare')
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--side', type=int, default=1280, choices=[1280])
    p.add_argument('--output', type=Path, required=True)
    p = modes.add_parser('board')
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--transforms', choices=['software', 'native'], default='software',
                   help='native is for reproduction only: prior native quantizer setup aborted')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = modes.add_parser('compare')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--board', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--compilation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'board': board, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
    main()
