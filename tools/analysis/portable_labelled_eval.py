"""Held-out historical labels versus FP32 and compiled portable model outputs.

Not a live-camera acceptance test. No confidence tuning, retraining or release
changes. Test frames share source runs with training: not independent new parts.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from .portable_tensor_parity import LABELS, decode_heads, detection_rows, digest, prepare as prepare_inputs
else:
    from portable_tensor_parity import LABELS, decode_heads, detection_rows, digest, prepare as prepare_inputs


def load_labels(path, width, height):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError('invalid YOLO annotation: ' + str(path))
        cls = int(values[0])
        x, y, w, h = map(float, values[1:])
        if cls not in range(4) or not np.isfinite([x, y, w, h]).all() or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError('invalid class/normalized box: ' + str(path))
        rows.append({'class_id': cls, 'xywh': [(x-w/2)*width, (y-h/2)*height, w*width, h*height]})
    return rows


def box_iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x) * max(0, bottom-y)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection/union if union > 0 else 0.


def match(ground_truth, predictions, min_iou=.5, class_aware=True):
    """Confidence-ordered one-to-one matching; duplicate predictions remain FP."""
    unused = set(range(len(ground_truth)))
    pairs, false_positive = [], []
    for p in sorted(range(len(predictions)), key=lambda i: predictions[i]['score'], reverse=True):
        candidates = [(box_iou(predictions[p]['xywh'], ground_truth[g]['xywh']), g)
                      for g in unused if not class_aware or predictions[p]['class_id'] == ground_truth[g]['class_id']]
        overlap, g = max(candidates, default=(0., -1))
        if g >= 0 and overlap >= min_iou:
            unused.remove(g)
            pairs.append((g, p, overlap))
        else:
            false_positive.append(p)
    return pairs, false_positive, sorted(unused)


def metrics(rows, source):
    counts = [{'ground_truth': 0, 'tp': 0, 'fp': 0, 'fn': 0} for _ in LABELS]
    confusion = np.zeros((5, 5), dtype=int)  # GT rows, prediction columns; index 4 = background/missing
    for row in rows:
        gt, pred = row['ground_truth'], row[source]
        for g in gt:
            counts[g['class_id']]['ground_truth'] += 1
        pairs, fp, fn = match(gt, pred)
        for g, _, _ in pairs:
            counts[gt[g]['class_id']]['tp'] += 1
        for p in fp:
            counts[pred[p]['class_id']]['fp'] += 1
        for g in fn:
            counts[gt[g]['class_id']]['fn'] += 1
        pairs, fp, fn = match(gt, pred, class_aware=False)
        for g, p, _ in pairs:
            confusion[gt[g]['class_id'], pred[p]['class_id']] += 1
        for p in fp:
            confusion[4, pred[p]['class_id']] += 1
        for g in fn:
            confusion[gt[g]['class_id'], 4] += 1
    def rates(count):
        return dict(count, precision=count['tp']/(count['tp']+count['fp']) if count['tp']+count['fp'] else None,
                    recall=count['tp']/(count['tp']+count['fn']) if count['tp']+count['fn'] else None)
    total = {key: sum(c[key] for c in counts) for key in counts[0]}
    return {'overall': rates(total), 'classes': {label: rates(c) for label, c in zip(LABELS, counts)},
            'class_agnostic_confusion': {'labels': LABELS + ['background_or_missing'], 'gt_rows_prediction_columns': confusion.tolist()}}


def prepare(args):
    import cv2
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    report = json.loads((dataset / 'report.json').read_text())
    if report['classes'] != LABELS:
        raise ValueError('dataset class order differs')
    selected = [r for r in manifest if r['split'] == 'test']
    if not selected:
        raise ValueError('empty test split')
    other_hashes = {r['source_sha256'] for r in manifest if r['split'] != 'test'}
    if any(r['source_sha256'] in other_hashes for r in selected):
        raise ValueError('test source hash overlaps train/validation')
    calibration = json.loads(args.compilation.read_text())['calibration']
    images, records = [], []
    for row in selected:
        image = dataset / row['file']
        label = dataset / 'labels/test' / (image.stem + '.txt')
        if image.name in calibration or row['model_sha256'] in calibration.values():
            raise ValueError('test sample overlaps compiler calibration')
        if digest(image) != row['model_sha256']:
            raise ValueError('image differs from dataset manifest: ' + str(image))
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError('unreadable image: ' + str(image))
        height, width = pixels.shape[:2]
        records.append({'image': str(image), 'image_sha256': digest(image), 'label': str(label),
                        'label_sha256': digest(label), 'run': row['run'], 'frame': row['frame'],
                        'ground_truth': load_labels(label, width, height)})
        images.append(image)
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_inputs(SimpleNamespace(images=images, side=1280, output=args.output / 'inputs'))
    selection = {'scope': 'historical_test_split_not_live_acceptance', 'labels': LABELS,
                 'dataset': str(dataset), 'manifest_sha256': digest(dataset / 'manifest.json'),
                 'compilation_sha256': digest(args.compilation), 'samples': records,
                 'source_runs': dict(collections.Counter(r['run'] for r in selected)),
                 'test_train_val_source_hash_overlap': 0, 'test_calibration_name_or_model_hash_overlap': 0}
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print('Prepared', len(records), 'held-out historical test images.')


def source_predictions(rows, item):
    result = []
    scale = item['scale']
    left, top = item['padding_left_top']
    width, height = item['original_size']
    for row in rows:
        x, y, w, h = row['model_xywh']
        x1, y1 = max(0., (x-left)/scale), max(0., (y-top)/scale)
        x2, y2 = min(width, (x+w-left)/scale), min(height, (y+h-top)/scale)
        if x2 > x1 and y2 > y1:
            result.append({'class_id': row['class_id'], 'score': row['score'], 'xywh': [x1, y1, x2-x1, y2-y1]})
    return result


def evaluate(args):
    import cv2
    import onnxruntime as ort
    selection = json.loads((args.prepared / 'selection.json').read_text())
    inputs = args.prepared / 'inputs/inputs.npz'
    manifest = json.loads(inputs.with_suffix('.json').read_text())
    board = json.loads((args.board / 'board.json').read_text())
    compilation = json.loads(args.compilation.read_text())
    if digest(args.compilation) != selection['compilation_sha256']:
        raise ValueError('compilation record changed after preparation')
    if not (digest(inputs) == manifest['inputs_sha256'] == board['inputs_sha256']):
        raise ValueError('prepared/board input fingerprint mismatch')
    if digest(args.board / 'outputs.npz') != board['outputs_sha256']:
        raise ValueError('board output fingerprint mismatch')
    if board['elf_sha256'] != compilation['artifacts']['yolov8_stage1_mla.elf'] or board['mpk_sha256'] != compilation['artifacts']['yolov8_mpk.json']:
        raise ValueError('compiled artifact lineage differs')
    if board['labels'] != LABELS or selection['labels'] != LABELS or manifest['labels'] != LABELS:
        raise ValueError('class contract differs')
    if len(selection['samples']) != len(manifest['images']) or len(board['samples']) != len(selection['samples']):
        raise ValueError('sample counts differ')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.original), sess_options=options, providers=['CPUExecutionProvider'])
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
    if names != dict(enumerate(LABELS)):
        raise ValueError('FP32 model class metadata differs')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(inputs, allow_pickle=False) as samples, np.load(args.board / 'outputs.npz', allow_pickle=False) as hardware:
        for i, (record, item) in enumerate(zip(selection['samples'], manifest['images'])):
            if record['image_sha256'] != item['image_sha256'] or digest(record['label']) != record['label_sha256']:
                raise ValueError('image order or ground-truth fingerprint changed')
            key = item['key']
            tensor = np.ascontiguousarray(samples[key].transpose(2, 0, 1)[None])
            original = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            compiled = decode_heads([hardware[f'{key}_head_{j}'] for j in range(6)], layout='NHWC')
            # Lock settings before evaluating test data; match native r33 topk/NMS.
            row = dict(record, key=key)
            for name, prediction in [('fp32', original), ('compiled', compiled)]:
                row[name] = source_predictions(detection_rows(prediction, threshold=.05, iou=.45)[:100], item)
            rows.append(row)
            panels = []
            image = cv2.imread(record['image'])
            for name in ['ground_truth', 'fp32', 'compiled']:
                panel = image.copy()
                for det in row[name]:
                    x, y, w, h = map(round, det['xywh'])
                    color = (0, 200, 0) if name == 'ground_truth' else (0, 100, 255)
                    cv2.rectangle(panel, (x,y), (x+w,y+h), color, 2)
                    label = LABELS[det['class_id']] + (f" {det['score']:.2f}" if 'score' in det else '')
                    cv2.putText(panel, label, (max(0,x), max(20,y-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                cv2.putText(panel, name, (12,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                panels.append(cv2.resize(panel, (640,640)))
            cv2.imwrite(str(args.output / f'{i:02d}-{Path(record["image"]).stem}.jpg'), np.hstack(panels))
            print(key, 'GT', len(row['ground_truth']), 'FP32', len(row['fp32']), 'compiled', len(row['compiled']), flush=True)
    result = {'scope': selection['scope'], 'confidence': .05, 'nms_iou': .45, 'topk': 100,
              'matching_iou': .5, 'matching': 'confidence-ordered class-aware one-to-one; duplicates count as false positives',
              'limitations': ['Historical camera view; not a current live good/bad placement test.',
                              'Frames share three source runs with training; not 31 independent physical parts.',
                              'Compiled outputs use the verified software transform diagnostic, not native camera replay.',
                              'Fixed diagnostic settings, not a threshold recommendation or mAP estimate.'],
              'original_onnx_sha256': digest(args.original), 'selection': selection, 'board': board,
              'fp32': metrics(rows, 'fp32'), 'compiled': metrics(rows, 'compiled'), 'samples': rows}
    (args.output / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({name: result[name] for name in ['fp32','compiled']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--compilation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=prepare)
    p = sub.add_parser('evaluate')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--board', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--compilation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
