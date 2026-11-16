"""Board-only, bounded observation of an existing portable release's buffers.

Uses its unchanged diagnostic graph; no config/model edits, receiver, or health
qualification. Dumps a few buffers for layout analysis, not throughput testing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


STAGES = ('decoder', 'simaai_preprocess', 'simaai_process_mla', 'simaai_boxdecode')


def selected_frames(start, step, samples):
    if any(type(v) is not int for v in (start, step, samples)) or start < 0 or step < 1 or not 1 <= samples <= 64:
        raise ValueError('require nonnegative start, positive step, and 1..64 samples')
    return tuple(start + i*step for i in range(samples))


def missing_frames(records, targets):
    return {label: sorted(set(targets)-{r['frame_id'] for r in records if r['label'] == label})
            for label in STAGES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=20)
    p.add_argument('--frame-step', type=int, default=1)
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--seconds', type=float, default=8)
    args = p.parse_args()
    targets = selected_frames(args.start_frame, args.frame_step, args.samples)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
        p.error('seconds must be finite, positive and <=120')
    sys.path.insert(0, str(args.release.resolve() / 'tools'))
    import worker
    from support import check_release, load_deployment, prepare_runtime_configs
    Gst = worker.Gst
    Gst.init(None)
    root = args.release.resolve()
    check_release(root, worker.runtime_info(), worker.plugin_info())
    inventory_bytes = (root / 'release.json').read_bytes()
    inventory = json.loads(inventory_bytes)
    for name, expected in inventory['tool_hashes'].items():
        if hashlib.sha256((root / 'tools' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned launcher hash mismatch: ' + name)
    worker.validate_assets(root)
    args.output.mkdir(parents=True, exist_ok=False)
    configs = prepare_runtime_configs(root, args.output)
    pipeline = Gst.parse_launch(worker.graph(root, configs, True))
    config = load_deployment(root / 'assets/deployment.json')
    camera = pipeline.get_by_name('camera')
    camera.set_property('location', config['camera_uri'])
    camera.set_property('latency', config['rtsp_latency_ms'])
    Gst.util_set_object_arg(camera, 'protocols', config['rtsp_transport'])
    records, errors, seen = [], [], {name: set() for name in STAGES}
    duplicates = {name: 0 for name in STAGES}
    target_set = set(targets)

    def observe(pad, info, label):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            meta = buffer.get_custom_meta('GstSimaMeta')
            values = meta.get_structure() if meta is not None else None
            frame = values.get_value('frame-id') if values is not None and values.has_field('frame-id') else None
            if type(frame) is not int or frame not in target_set:
                return Gst.PadProbeReturn.OK
            stream = values.get_value('stream-id') if values.has_field('stream-id') else None
            if not isinstance(stream, str) or not stream:
                raise ValueError('selected frame has no stream identity')
            if frame in seen[label]:
                duplicates[label] += 1
                return Gst.PadProbeReturn.OK
            if records and stream != records[0]['stream_id']:
                raise ValueError('probe supports one source stream only')
            seen[label].add(frame)
            caps = pad.get_current_caps()
            row = {'label': label, 'frame_id': frame, 'stream_id': stream,
                   'caps': caps.to_string() if caps is not None else None,
                   'size': buffer.get_size(), 'memories': buffer.n_memory(),
                   'meta': values.to_string() if values is not None else None, 'memory_sizes': []}
            for i in range(buffer.n_memory()):
                row['memory_sizes'].append(list(buffer.peek_memory(i).get_sizes()))
            filename = f'{label}-{frame}.bin'
            data = buffer.extract_dup(0, buffer.get_size())
            (args.output / filename).write_bytes(data)
            row.update(file=filename, sha256=hashlib.sha256(data).hexdigest())
            records.append(row)
            print(json.dumps(row), flush=True)
        except Exception as error:
            errors.append(label + ':' + repr(error))
        return Gst.PadProbeReturn.OK

    for name in STAGES:
        pipeline.get_by_name(name).get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, observe, name)
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('pipeline refused PLAYING')
        deadline = time.monotonic() + args.seconds
        bus = pipeline.get_bus()
        while time.monotonic() < deadline:
            message = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR)
            if message is not None:
                errors.append(str(message.parse_error()))
                break
            if errors or all(len(seen[name]) == len(targets) for name in STAGES):
                break
    finally:
        pipeline.set_state(Gst.State.NULL)
        pipeline.run_dispose()
        missing = missing_frames(records, targets)
        (args.output / 'buffers.json').write_text(json.dumps({
            'records': records, 'errors': errors, 'requested_frame_ids': targets,
            'missing_by_stage': missing, 'duplicates_by_stage': duplicates,
            'selection': 'periodic frame IDs, no prediction/score selection',
            'release_inventory_sha256': hashlib.sha256(inventory_bytes).hexdigest(),
            'complete': not errors and not any(missing.values()) and not any(duplicates.values()),
        }, indent=2))
    return int(bool(errors) or any(missing.values()) or any(duplicates.values()))


if __name__ == '__main__':
    raise SystemExit(main())
