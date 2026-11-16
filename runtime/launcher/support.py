"""Standard-library contracts shared by the offline Portable launcher and checks."""
from collections import OrderedDict
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from urllib.parse import unquote, urlsplit


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def parse_native_boxes(data, labels, max_detections=100):
    """Decode native box records without asserting their coordinate semantics."""
    if not 4 <= len(data) <= 4 + 24 * max_detections:
        raise ValueError('native detection buffer is outside its bounded capacity')
    count = struct.unpack_from('<I', data)[0]
    if count > max_detections or len(data) < 4 + count * 24:
        raise ValueError('native detection count exceeds capacity or available records')
    result = []
    for index in range(count):
        x, y, width, height, score, class_id = struct.unpack_from('<4IfI', data, 4 + index * 24)
        if not 0 <= score <= 1 or class_id >= len(labels):
            raise ValueError('native detection has an invalid score or class')
        result.append({
            'class_id': class_id, 'label': labels[class_id], 'score': score,
            'native_box': {'x': x, 'y': y, 'width': width, 'height': height},
        })
    return result


def load_deployment(path):
    config = json.loads(Path(path).read_text())
    keys = {'camera_uri', 'receiver_host', 'receiver_port', 'rtsp_transport', 'rtsp_latency_ms'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('deployment configuration has missing or unsupported keys')
    uri = config['camera_uri']
    try:
        if not isinstance(uri, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in unquote(uri)):
            raise ValueError()
        parsed = urlsplit(uri)
        if parsed.scheme not in ('rtsp', 'rtsps') or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise ValueError()
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError()
    except ValueError:
        raise ValueError('camera URI must be RTSP(S), without credentials, fragments or whitespace') from None
    try:
        if not isinstance(config['receiver_host'], str):
            raise ValueError()
        ipaddress.ip_address(config['receiver_host'])
    except ValueError:
        raise ValueError('receiver_host must be a numeric IPv4 or IPv6 address') from None
    for key, lower, upper in (('receiver_port', 1, 65535), ('rtsp_latency_ms', 0, 10000)):
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(key + ' is outside its supported integer range')
    if config['rtsp_transport'] not in ('tcp', 'udp'):
        raise ValueError('rtsp_transport must be tcp or udp')
    return config


def check_release(root, runtime, plugins):
    root = Path(root)
    inventory = json.loads((root / 'release.json').read_text())
    if not isinstance(inventory, dict) or type(inventory.get('schema_version')) is not int or inventory['schema_version'] != 1:
        raise ValueError('unsupported release schema')
    if inventory.get('runtime') != runtime:
        raise ValueError('runtime/package versions differ from pinned release')
    if not plugins or inventory.get('plugins') != plugins:
        raise ValueError('selected plugin files differ from pinned release')
    assets = inventory.get('assets')
    if not isinstance(assets, dict) or not assets:
        raise ValueError('release asset inventory is empty or invalid')
    base = (root / 'assets').resolve()
    for relative, digest in assets.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('asset path escapes release')
        target = (base / path).resolve()
        if not target.is_relative_to(base):
            raise ValueError('asset symlink escapes release')
        try:
            actual = file_hash(target)
        except OSError:
            raise ValueError('release asset is missing: ' + relative) from None
        if actual != digest:
            raise ValueError('release asset hash differs: ' + relative)
    return assets


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name, delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_runtime_configs(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('run output must be outside the immutable release')
    configs = output / 'configs'
    configs.mkdir(parents=True, exist_ok=False)
    for name in ('0_preproc.json', '0_process_mla.json', 'boxdecoder.json'):
        shutil.copyfile(root / 'assets/etc' / name, configs / name)
    return configs


def stop_process(process, grace=(8.0, 3.0, 2.0)):
    """Stop only a child created with start_new_session=True, within its budget."""
    try:
        if os.getpgid(process.pid) != process.pid:
            raise ValueError('child must own an isolated process group')
    except ProcessLookupError:
        pass
    sent = []
    for number, budget in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), grace):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            break
        sent.append(number.name)
        deadline = time.monotonic() + budget
        while True:
            process.poll()  # Reap the leader even when descendants keep its group alive.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(.01, remaining))
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
    return {
        'exit_code': process.poll(),
        'forced': any(name != 'SIGINT' for name in sent),
        'reaped': process.returncode is not None,
        'signals': sent,
    }


class SourceClock:
    """Translate source time to pipeline running time using one fixed offset.

    Anchor once at the first decoded frame, never at encoder arrival. All
    subsequent source intervals are preserved exactly; clock resets are not
    hidden by re-anchoring. This does not measure capture/optical latency.
    """

    def __init__(self):
        self._source_origin = self._running_origin = None
        self._lock = threading.Lock()

    def convert(self, source_time, running_time):
        for name, value in (('source', source_time), ('pipeline', running_time)):
            if type(value) is not int or not 0 <= value < (1 << 64) - 1:
                raise ValueError('invalid ' + name + ' clock time')
        with self._lock:
            if self._source_origin is None:
                self._source_origin, self._running_origin = source_time, running_time
            translated = self._running_origin + source_time - self._source_origin
            if source_time < self._source_origin or not 0 <= translated < (1 << 64) - 1:
                raise ValueError('source clock reset or translated time outside range')
            return translated

    def snapshot(self):
        with self._lock:
            return {
                'source_origin_ns': self._source_origin,
                'running_origin_ns': self._running_origin,
                'offset_ns': None if self._source_origin is None else self._running_origin - self._source_origin,
                'method': 'fixed_offset_at_first_decoder_frame',
            }


class FrameTimestampCache:
    """Bounded source timestamps, consumed once by (stream id, frame id).

    Only decoder output supplies entries. Internal locking covers probes on
    different streaming threads as well as health snapshots. Evicted or missing
    entries return None; callers must never invent a replacement timestamp.
    """

    def __init__(self, capacity=512):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('timestamp capacity must be a positive integer')
        self.capacity = capacity
        self._pending = OrderedDict()
        self._lock = threading.Lock()
        self._peak = self._stored = self._consumed = self._misses = self._evicted = 0

    @staticmethod
    def _valid_identity(stream_id, frame_id):
        return isinstance(stream_id, str) and bool(stream_id) and type(frame_id) is int and frame_id >= 0

    def remember(self, stream_id, frame_id, timestamp):
        if not self._valid_identity(stream_id, frame_id):
            raise ValueError('invalid source frame identity')
        # UINT64_MAX is Gst.CLOCK_TIME_NONE, not a usable time label.
        if type(timestamp) is not int or not 0 <= timestamp < (1 << 64) - 1:
            raise ValueError('invalid source media timestamp')
        key = (stream_id, frame_id)
        with self._lock:
            if key in self._pending:
                if self._pending[key] != timestamp:
                    raise ValueError('conflicting source media timestamp')
                return
            if len(self._pending) == self.capacity:
                self._pending.popitem(last=False)
                self._evicted += 1
            self._pending[key] = timestamp
            self._stored += 1
            self._peak = max(self._peak, len(self._pending))

    def take(self, stream_id, frame_id):
        with self._lock:
            stamp = self._pending.pop((stream_id, frame_id), None) if self._valid_identity(stream_id, frame_id) else None
            if stamp is None:
                self._misses += 1
            else:
                self._consumed += 1
            return stamp

    def snapshot(self):
        with self._lock:
            return {
                'capacity': self.capacity, 'pending': len(self._pending),
                'peak_pending': self._peak, 'stored': self._stored,
                'consumed': self._consumed, 'misses': self._misses,
                'evicted': self._evicted,
            }


class Health:
    """Frame-flow health, not a model-accuracy or image/result qualification."""

    def __init__(self, started=None, session=None, startup_timeout=10.0, stale_after=1.0):
        self.started = time.monotonic() if started is None else started
        self.session = str(uuid.uuid4()) if session is None else session
        self.startup_timeout = startup_timeout
        self.stale_after = stale_after
        self.last_output = None
        self.last_pts = None
        self.last_frame_id = None
        self.stream_id = None
        self.frames = 0
        self.valid_frames = 0
        self.errors = set()
        self.running = True

    def fail(self, reason):
        self.errors.add(reason)

    def observe(self, frame_id, stream_id, pts, now):
        if not self.running:
            return False
        self.frames += 1
        self.last_output = now
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0 or not stream_id:
            self.fail('missing_frame_identity')
        else:
            if self.stream_id is not None and stream_id != self.stream_id:
                self.fail('stream_changed_without_restart')
            if self.last_frame_id is not None and frame_id <= self.last_frame_id:
                self.fail('nonadvancing_frame_identity')
            self.stream_id = stream_id
            self.last_frame_id = frame_id
        if not isinstance(pts, int) or isinstance(pts, bool) or pts < 0:
            self.fail('missing_media_time')
        else:
            if self.last_pts is not None and pts <= self.last_pts:
                self.fail('nonadvancing_media_time')
            else:
                self.valid_frames += 1
            self.last_pts = pts
        return not self.errors

    def stop(self):
        self.running = False

    def snapshot(self, now):
        age = None if self.last_output is None else max(0, now - self.last_output)
        errors = sorted(self.errors)
        if self.last_output is None and now - self.started >= self.startup_timeout:
            errors = sorted(set(errors) | {'startup_timeout'})
        if not self.running:
            state = 'error' if errors else 'stopped'
        elif age is not None and age >= self.stale_after:
            state = 'stale'
        elif errors:
            state = 'error'
        elif self.valid_frames >= 2:
            state = 'live'
        else:
            state = 'starting'
        return {
            'schema_version': 1,
            'session': self.session,
            'state': state,
            'running': self.running,
            'media_ready': state == 'live',
            'last_output_age_seconds': age,
            'frames': self.frames,
            'valid_timed_frames': self.valid_frames,
            'last_frame_id': self.last_frame_id,
            'stream_id': self.stream_id,
            'errors': errors,
        }
