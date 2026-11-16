"""Board-only GStreamer worker. Launched in an isolated group by launch.py."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import threading
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from support import FrameTimestampCache, Health, SourceClock, atomic_json, check_release, file_hash, load_deployment, parse_native_boxes, prepare_runtime_configs

SYSTEM_PLUGINS = Path('/usr/lib/aarch64-linux-gnu/gstreamer-1.0').resolve()
FACTORIES = (
    'rtspsrc', 'rtph264depay', 'h264parse', 'simaaidecoder', 'tee', 'capsfilter',
    'simaaiprocesscvu', 'simaaiprocessmla', 'simaaiboxdecode', 'simaai-overlay2',
    'simaaiencoder', 'rtph264pay', 'udpsink', 'fakesink', 'rtpbin', 'rtpjitterbuffer',
)
ASSETS = (
    'etc/0_preproc.json', 'etc/0_process_mla.json', 'etc/boxdecoder.json',
    'share/overlay/labels', 'share/processmla/yolov8_stage1_mla.elf',
)
TOOLS = ('launch.py', 'worker.py', 'support.py', 'verify.py')


def runtime_info():
    text = subprocess.check_output([
        'dpkg-query', '-W', '-f=${Package}\t${Version}\n',
        'simaai-gst-plugins', 'simaai-mlart-modalix',
    ], text=True, timeout=10)
    return {'gstreamer': '.'.join(map(str, Gst.version()[:3])), 'packages': dict(line.split('\t') for line in text.splitlines())}


def plugin_info():
    result = {}
    for name in FACTORIES:
        factory = Gst.ElementFactory.find(name)
        if factory is None:
            raise ValueError('required system plugin is unavailable: ' + name)
        path = Path(factory.get_plugin().get_filename()).resolve()
        if not path.is_relative_to(SYSTEM_PLUGINS):
            raise ValueError('plugin was selected outside system directory: ' + name)
        result[name] = {'path': str(path), 'sha256': file_hash(path)}
    return result


def validate_assets(root):
    assets = root / 'assets'
    pre = json.loads((assets / 'etc/0_preproc.json').read_text())
    mla = json.loads((assets / 'etc/0_process_mla.json').read_text())
    box = json.loads((assets / 'etc/boxdecoder.json').read_text())
    if (pre['input_width'], pre['input_height']) != (box['original_width'], box['original_height']):
        raise ValueError('preprocessing input and box-decoder original geometry disagree')
    if (pre['output_width'], pre['output_height']) != (box['model_width'], box['model_height']):
        raise ValueError('preprocessing output and box-decoder model geometry disagree')
    if mla['simaai__params']['outputs'][0]['size'] != box['buffers']['input'][0]['size']:
        raise ValueError('MLA/box-decoder physical output sizes disagree')
    labels = (assets / 'share/overlay/labels').read_text().splitlines()
    if labels != box['labels'] or len(labels) != box['num_classes']:
        raise ValueError('label ordering/class count disagree')
    model = assets / 'share/processmla/yolov8_stage1_mla.elf'
    if Path(mla['simaai__params']['model_path']).resolve() != model.resolve():
        raise ValueError('MLA model path escapes the staged release')


def stage(source, destination, deployment):
    source, destination = source.resolve(), destination.resolve()
    config = load_deployment(deployment)
    if destination.is_relative_to(source):
        raise ValueError('staged release must be outside the original application')
    runtime, plugins = runtime_info(), plugin_info()
    if runtime != {'gstreamer': '1.22.0', 'packages': {'simaai-gst-plugins': '2.1.3', 'simaai-mlart-modalix': '2.1.3'}}:
        raise ValueError('this release targets the inspected 2.1.3 / GStreamer 1.22.0 runtime')
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        provenance = {}
        for name in ASSETS:
            original = source / name
            target = destination / 'assets' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            provenance[name] = file_hash(original)
        metadata = destination / 'assets/provenance/mpk.json'
        metadata.parent.mkdir()
        shutil.copyfile(source / 'etc/mpk.json', metadata)
        config_path = destination / 'assets/deployment.json'
        atomic_json(config_path, config)
        mla_path = destination / 'assets/etc/0_process_mla.json'
        mla = json.loads(mla_path.read_text())
        mla['simaai__params']['model_path'] = str(destination / 'assets/share/processmla/yolov8_stage1_mla.elf')
        atomic_json(mla_path, mla)
        tools = destination / 'tools'
        tools.mkdir()
        for name in TOOLS:
            shutil.copyfile(Path(__file__).with_name(name), tools / name)
        inventory = {
            'schema_version': 1,
            'runtime': runtime,
            'plugins': plugins,
            'assets': {str(path.relative_to(destination / 'assets')): file_hash(path) for path in (destination / 'assets').rglob('*') if path.is_file()},
            'tool_hashes': {name: file_hash(tools / name) for name in TOOLS},
            'original_asset_hashes': provenance,
            'model_reference_qualification': 'not_established',
        }
        validate_assets(destination)
        atomic_json(destination / 'release.json', inventory)
        check_release(destination, runtime, plugins)
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(json.dumps({'staged_release': str(destination), 'assets': len(inventory['assets']), 'plugins': len(plugins)}), flush=True)


def graph(root, configs, diagnostic):
    def config(name):
        return json.dumps(str(configs / name))
    sink = 'fakesink name=output sync=false' if diagnostic else 'udpsink name=output'
    return (
        'rtspsrc name=camera ! rtph264depay wait-for-keyframe=true ! h264parse ! '
        'video/x-h264,parsed=true,stream-format=byte-stream,alignment=au ! '
        'simaaidecoder name=decoder sima-allocator-type=2 next-element=CVU ! tee name=source ! '
        'video/x-raw ! simaaiprocesscvu name=simaai_preprocess num-buffers=5 config=' + config('0_preproc.json') + ' ! '
        'simaaiprocessmla name=simaai_process_mla multi-pipeline=true num-buffers=5 config=' + config('0_process_mla.json') + ' ! '
        'simaaiboxdecode name=simaai_boxdecode config=' + config('boxdecoder.json') + ' ! '
        'application/vnd.simaai.tensor ! overlay. source. ! video/x-raw ! '
        'simaai-overlay2 name=overlay render-info="input::decoder,bbox::simaai_boxdecode" labels-file=' + json.dumps(str(root / 'assets/share/overlay/labels')) + ' ! '
        'simaaiencoder name=encoder enc-bitrate=4000 ! h264parse ! rtph264pay name=payloader ! ' + sink
    )


def run(root, output, seconds, diagnostic):
    health = Health()
    pipeline = None
    interrupted = threading.Event()
    fatal = threading.Event()
    events = queue.Queue(maxsize=512)
    counts = {}
    frame_pts = FrameTimestampCache(capacity=512)
    source_clock = SourceClock()
    count_lock = threading.Lock()
    warnings = []
    last_marker_time = None
    marker_count = 0
    final = {'schema_version': 1, 'session': health.session, 'diagnostic': diagnostic, 'release_verified': False, 'shutdown_complete': False}
    detection_frames = 0
    evidence = ExitStack()
    evidence_ready = False

    def request_stop(_number, _frame):
        interrupted.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def fault(reason):
        with count_lock:
            health.fail(reason)
        fatal.set()

    def enqueue(event):
        try:
            events.put_nowait(event)
        except queue.Full:
            fault('evidence_queue_overflow')

    def observe(pad, info, label):
        nonlocal last_marker_time, marker_count
        if fatal.is_set():
            return Gst.PadProbeReturn.DROP
        buffers = info.get_buffer_list() if info.type & Gst.PadProbeType.BUFFER_LIST else None
        items = (buffers.get(i) for i in range(buffers.length())) if buffers is not None else (info.get_buffer(),)
        drop = False
        for buffer in items:
            if buffer is None:
                continue
            now = time.monotonic()
            pts = None if buffer.pts == Gst.CLOCK_TIME_NONE else buffer.pts
            pts_source = 'buffer' if pts is not None else 'none'
            frame_id = stream_id = stamp = None
            if label in ('decoder:src', 'simaai_boxdecode:src', 'encoder:src'):
                # The native SiMa elements drop GstBuffer.pts but preserve GstSimaMeta.
                # Only decoder output supplies time labels; key by stream AND frame.
                # Encoder output consumes them even when its own PTS is already set.
                meta = buffer.get_custom_meta('GstSimaMeta')
                values = meta.get_structure() if meta is not None else None
                frame_id = values.get_value('frame-id') if values is not None and values.has_field('frame-id') else None
                stream_id = values.get_value('stream-id') if values is not None and values.has_field('stream-id') else None
                stamp = values.get_value('timestamp') if values is not None and values.has_field('timestamp') else None
                if label == 'decoder:src':
                    try:
                        clock = pipeline.get_clock()
                        if clock is None:
                            raise ValueError('pipeline clock unavailable')
                        running_time = clock.get_time() - pipeline.get_base_time()
                        # GstSimaMeta time is board uptime, not stream-relative PTS.
                        # Anchor once here; never re-time individual encoded frames.
                        media_time = source_clock.convert(stamp, running_time)
                        frame_pts.remember(stream_id, frame_id, media_time)
                    except ValueError as error:
                        fault('source_timestamp_error:' + str(error))
                        return Gst.PadProbeReturn.DROP
                if label == 'encoder:src':
                    carried = frame_pts.take(stream_id, frame_id)
                    if pts is None and carried is not None:
                        try:
                            buffer.pts = carried
                        except Exception:
                            fault('timestamp_restore_failed')
                            return Gst.PadProbeReturn.DROP
                        pts = carried
                        pts_source = 'meta_running_time'
            with count_lock:
                row = counts.setdefault(label, {'buffers': 0, 'missing_pts': 0, 'max_delivery_gap_seconds': 0, 'pts_source': {}})
                if 'last_delivery' in row:
                    row['max_delivery_gap_seconds'] = max(row['max_delivery_gap_seconds'], now-row['last_delivery'])
                row['pts_source'][pts_source] = row['pts_source'].get(pts_source, 0) + 1
                row.update(buffers=row['buffers'] + 1, missing_pts=row['missing_pts'] + int(pts is None), last_delivery=now)
            if label == 'decoder:src':
                caps = pad.get_current_caps()
                shape = caps.get_structure(0) if caps is not None else None
                if shape is None or (shape.get_value('width'), shape.get_value('height')) != (1280, 720):
                    fault('unsupported_camera_geometry')
                    return Gst.PadProbeReturn.DROP
            if label == 'simaai_boxdecode:src':
                if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0 or not isinstance(stream_id, str) or not stream_id:
                    fault('invalid_detection_identity')
                    return Gst.PadProbeReturn.DROP
                size = buffer.get_size()
                if not 4 <= size <= 4 + 24 * box_contract['topk']:
                    fault('invalid_native_detection_size')
                    return Gst.PadProbeReturn.DROP
                try:
                    boxes = parse_native_boxes(buffer.extract_dup(0, size), box_contract['labels'], box_contract['topk'])
                except ValueError:
                    fault('invalid_native_detections')
                    return Gst.PadProbeReturn.DROP
                enqueue(('detections', frame_id, stream_id, pts, now, boxes))
            if label == 'encoder:src':
                with count_lock:
                    if not health.running:
                        return Gst.PadProbeReturn.DROP
                    admitted = health.observe(frame_id, stream_id, pts, now)
                enqueue(('frame', frame_id, stream_id, pts, now))
                if not diagnostic and not admitted:
                    fatal.set()
                    drop = True
            if label == 'output:sink':
                if buffer.get_size() < 12:
                    fault('invalid_output_rtp')
                    drop = True
                    continue
                _flags, flags, _sequence, timestamp, _ssrc = struct.unpack('!BBHII', buffer.extract_dup(0, 12))
                if flags & 128:
                    marker_count += 1
                    if last_marker_time is not None:
                        delta = (timestamp-last_marker_time) & 0xffffffff
                        if delta == 0 or delta >= 0x80000000:
                            with count_lock:
                                health.fail('nonadvancing_rtp_time')
                            if not diagnostic:
                                fatal.set()
                                drop = True
                    last_marker_time = timestamp
                if diagnostic:
                    enqueue(('packet', time.time_ns(), buffer.extract_dup(0, buffer.get_size())))
        return Gst.PadProbeReturn.DROP if drop else Gst.PadProbeReturn.OK

    def status():
        with count_lock:
            value = health.snapshot(time.monotonic())
            value['boundaries'] = {name: dict(row) for name, row in counts.items()}
        value.update(diagnostic=diagnostic, release_verified=final['release_verified'], rtp_frame_markers=marker_count,
                     timestamp_cache=frame_pts.snapshot(), source_clock=source_clock.snapshot())
        if marker_count < 2 and value['state'] == 'live':
            value.update(state='starting', media_ready=False)
        value['application_qualification'] = 'not_established'
        return value

    def drain_events():
        nonlocal detection_frames
        for _ in range(512):
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'frame':
                _, frame_id, stream_id, pts, now = event
                frames.write(json.dumps({'session': health.session, 'frame_id': frame_id, 'stream_id': stream_id, 'pts': pts, 'monotonic_seconds': now}) + '\n')
            elif event[0] == 'detections':
                _, frame_id, stream_id, pts, now, boxes = event
                detections.write(json.dumps({
                    'session': health.session, 'frame_id': frame_id, 'stream_id': stream_id,
                    'pts': pts, 'monotonic_seconds': now, 'detections': boxes,
                    'coordinate_contract': 'vendor_native_xywh_unqualified',
                    'qualification': 'not_established',
                }, allow_nan=False) + '\n')
                detection_frames += 1
            else:
                _, wall_ns, data = event
                packets.write(struct.pack('!QI', wall_ns, len(data)) + data)

    try:
        config = load_deployment(root / 'assets/deployment.json')
        inventory = json.loads((root / 'release.json').read_text())
        check_release(root, runtime_info(), plugin_info())
        if set(inventory['assets']) != set(ASSETS) | {'deployment.json', 'provenance/mpk.json'}:
            raise ValueError('release does not contain the required asset set')
        if inventory.get('tool_hashes') != {name: file_hash(root / 'tools' / name) for name in TOOLS}:
            raise ValueError('launcher code differs from pinned release')
        validate_assets(root)
        box_contract = json.loads((root / 'assets/etc/boxdecoder.json').read_text())
        final['release_verified'] = True
        configs = prepare_runtime_configs(root, output)
        pipeline = Gst.parse_launch(graph(root, configs, diagnostic))
        camera = pipeline.get_by_name('camera')
        camera.set_property('location', config['camera_uri'])
        camera.set_property('latency', config['rtsp_latency_ms'])
        Gst.util_set_object_arg(camera, 'protocols', config['rtsp_transport'])
        if not diagnostic:
            sink = pipeline.get_by_name('output')
            sink.set_property('host', config['receiver_host'])
            sink.set_property('port', config['receiver_port'])
        names = ('decoder', 'simaai_preprocess', 'simaai_process_mla', 'simaai_boxdecode', 'overlay', 'encoder', 'output') if diagnostic else ('decoder', 'simaai_boxdecode', 'encoder', 'output')
        for name in names:
            element = pipeline.get_by_name(name)
            for pad in element.pads:
                if diagnostic or pad.get_direction() == Gst.PadDirection.SRC or name == 'output':
                    pad.add_probe(Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST, observe, name + ':' + pad.name)
        frames = evidence.enter_context((output / 'frames.jsonl').open('x'))
        packets = evidence.enter_context((output / 'packets.bin').open('xb'))
        detections = evidence.enter_context((output / 'detections.jsonl').open('x'))
        evidence_ready = True
        atomic_json(output / 'health.json', status())
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise ValueError('pipeline refused PLAYING state')
        bus = pipeline.get_bus()
        deadline = time.monotonic() + seconds
        next_status = 0
        while True:
            drain_events()
            current = status()
            if time.monotonic() >= next_status:
                atomic_json(output / 'health.json', current)
                frames.flush()
                packets.flush()
                detections.flush()
                next_status = time.monotonic() + .25
            if 'startup_timeout' in current['errors']:
                fault('startup_timeout')
            if current['last_output_age_seconds'] is not None and current['last_output_age_seconds'] >= 10:
                fault('output_timeout')
            if fatal.is_set() or interrupted.is_set() or time.monotonic() >= deadline:
                break
            message = bus.timed_pop_filtered(50 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    error, _debug = message.parse_error()
                    fault('gstreamer_error:' + message.src.name + ':' + str(error))
                elif message.type == Gst.MessageType.WARNING:
                    warning, _debug = message.parse_warning()
                    if len(warnings) < 20:
                        warnings.append(str(warning))
                else:
                    fault('unexpected_end_of_stream')
    except Exception as error:
        health.fail(type(error).__name__ + ':' + str(error))
    finally:
        fatal.set()
        with count_lock:
            health.stop()
        try:
            atomic_json(output / 'health.json', status())
        finally:
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
                    pipeline.run_dispose()
            finally:
                try:
                    if evidence_ready:
                        drain_events()
                finally:
                    evidence.close()
        final.update(status(), shutdown_complete=True, warnings=warnings, detection_frames=detection_frames)
        atomic_json(output / 'health.json', final)
        atomic_json(output / 'worker.json', final)
    return 1 if final['errors'] or not final['frames'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('stage')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('release', type=Path)
    prepare.add_argument('--config', type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('release', type=Path)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--seconds', type=float, required=True)
    execute.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    Gst.init(None)
    if args.command == 'stage':
        stage(args.source, args.release, args.config)
        return 0
    return run(args.release.resolve(), args.output, args.seconds, args.diagnostic)


if __name__ == '__main__':
    raise SystemExit(main())
