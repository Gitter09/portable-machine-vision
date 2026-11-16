"""Timestamp storage contracts and the real worker probe, without board plugins."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from runtime.launcher.support import FrameTimestampCache, Health, SourceClock, parse_native_boxes


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        FrameTimestampCache(capacity)


def test_completed_frames_do_not_accumulate():
    cache = FrameTimestampCache()
    for frame in range(10_000):
        cache.remember('camera', frame, frame * 33_333_333)
        assert cache.take('camera', frame) == frame * 33_333_333
    assert cache.snapshot() == {
        'capacity': 512, 'pending': 0, 'peak_pending': 1,
        'stored': 10_000, 'consumed': 10_000, 'misses': 0, 'evicted': 0,
    }


def test_streams_with_same_frame_number_do_not_share_timestamps():
    cache = FrameTimestampCache()
    cache.remember('A', 1, 100)
    cache.remember('B', 1, 200)
    assert cache.take('A', 1) == 100
    assert cache.take('A', 1) is None
    assert cache.take('B', 1) == 200


def test_unfinished_frames_are_bounded_and_oldest_is_evicted():
    cache = FrameTimestampCache(3)
    for frame in range(10_000):
        cache.remember('camera', frame, frame)
        assert cache.snapshot()['pending'] <= 3
    assert cache.snapshot()['evicted'] == 9997
    assert cache.snapshot()['peak_pending'] == 3
    assert cache.take('camera', 0) is None
    assert cache.take('camera', 9997) == 9997


def test_duplicates_do_not_overwrite_or_refresh_oldest_entry():
    cache = FrameTimestampCache(2)
    cache.remember('A', 1, 100)
    cache.remember('A', 2, 200)
    cache.remember('A', 1, 100)
    with pytest.raises(ValueError, match='conflicting'):
        cache.remember('A', 1, 101)
    assert cache.snapshot()['stored'] == 2
    cache.remember('A', 3, 300)
    assert cache.take('A', 1) is None
    assert cache.take('A', 2) == 200


@pytest.mark.parametrize('stream,frame', [('', 1), (None, 1), ([], 1), ('A', -1), ('A', True), ('A', 1.0), ('A', [])])
def test_invalid_identity_is_not_stored_or_matched(stream, frame):
    cache = FrameTimestampCache()
    with pytest.raises(ValueError, match='identity'):
        cache.remember(stream, frame, 100)
    assert cache.take(stream, frame) is None
    assert cache.snapshot()['pending'] == 0


@pytest.mark.parametrize('stamp', [None, True, -1, 1.5, 2**64 - 1, 2**64])
def test_invalid_time_is_not_stored(stamp):
    cache = FrameTimestampCache()
    with pytest.raises(ValueError, match='timestamp'):
        cache.remember('A', 1, stamp)
    assert cache.take('A', 1) is None


def test_zero_time_and_out_of_order_completion():
    cache = FrameTimestampCache()
    cache.remember('A', 0, 0)
    cache.remember('A', 1, 100)
    assert cache.take('A', 1) == 100
    assert cache.take('A', 0) == 0


def test_concurrent_streaming_threads():
    cache = FrameTimestampCache(8)

    def process(stream):
        for frame in range(1000):
            cache.remember(stream, frame, frame)
            assert cache.take(stream, frame) == frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(process, ['A', 'B', 'C', 'D']))
    assert cache.snapshot()['pending'] == 0
    assert cache.snapshot()['consumed'] == 4000
    assert cache.snapshot()['evicted'] == 0


NONE = 2**64 - 1


class Buffer:
    def __init__(self, frame=1, stream='A', stamp=100, pts=NONE, meta=True):
        self.pts = pts
        self.metadata = {'frame-id': frame, 'stream-id': stream, 'timestamp': stamp} if meta else None

    def get_custom_meta(self, _name):
        if self.metadata is None:
            return None
        values = SimpleNamespace(has_field=lambda k: k in self.metadata, get_value=self.metadata.get)
        return SimpleNamespace(get_structure=lambda: values)

    def get_size(self):
        return 4

    def extract_dup(self, _offset, _size):
        return struct.pack('<I', 0)


@pytest.fixture
def probe():
    """Compile the actual nested callback; only GStreamer buffer/pad APIs are fake.

    This exercises worker wiring, not just the cache, without importing `gi`.
    Native plugin behaviour still requires board tests.
    """
    path = Path(__file__).resolve().parents[1] / 'runtime/launcher/worker.py'
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    observe = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'observe')
    observe.body = [n for n in observe.body if not isinstance(n, ast.Nonlocal)]
    cache, health, fatal, events = FrameTimestampCache(3), Health(), threading.Event(), []

    def fault(reason):
        health.fail(reason)
        fatal.set()

    namespace = {
        'Gst': SimpleNamespace(CLOCK_TIME_NONE=NONE, PadProbeType=SimpleNamespace(BUFFER_LIST=1),
                               PadProbeReturn=SimpleNamespace(OK=0, DROP=1)),
        'fatal': fatal, 'time': time, 'frame_pts': cache, 'count_lock': threading.Lock(),
        'counts': {}, 'health': health, 'enqueue': events.append, 'diagnostic': True,
        'fault': fault, 'box_contract': {'topk': 100, 'labels': ['IC']},
        'parse_native_boxes': parse_native_boxes, 'source_clock': SourceClock(),
        'pipeline': SimpleNamespace(get_clock=lambda: SimpleNamespace(get_time=lambda: 100),
                                    get_base_time=lambda: 100),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[observe], type_ignores=[])), str(path), 'exec'), namespace)
    shape = SimpleNamespace(get_value=lambda k: {'width': 1280, 'height': 720}[k])
    pad = SimpleNamespace(get_current_caps=lambda: SimpleNamespace(get_structure=lambda _: shape))

    def call(label, buffer):
        info = SimpleNamespace(type=0, get_buffer=lambda: buffer)
        return namespace['observe'](pad, info, label)

    return SimpleNamespace(call=call, cache=cache, health=health, fatal=fatal, events=events, namespace=namespace)


def test_worker_consumes_all_completed_frame_labels(probe):
    for frame in range(10_000):
        probe.call('decoder:src', Buffer(frame=frame, stamp=frame))
        encoded = Buffer(frame=frame)
        probe.call('encoder:src', encoded)
        assert encoded.pts == frame
    assert probe.cache.snapshot()['pending'] == 0
    assert probe.health.valid_frames == 10_000
    assert not probe.health.errors


def test_worker_uses_stream_and_frame_for_restore(probe):
    probe.call('decoder:src', Buffer(stream='A', stamp=100))
    probe.call('decoder:src', Buffer(stream='B', stamp=200))
    encoded = Buffer(stream='A')
    probe.call('encoder:src', encoded)
    assert encoded.pts == 0
    assert probe.cache.take('B', 1) == 100


def test_worker_consumes_cache_even_with_native_pts(probe):
    probe.call('decoder:src', Buffer())
    encoded = Buffer(pts=150)
    probe.call('encoder:src', encoded)
    assert encoded.pts == 150
    assert probe.cache.snapshot()['pending'] == 0


def test_late_boxdecode_does_not_recreate_consumed_entry(probe):
    probe.call('decoder:src', Buffer())
    probe.call('encoder:src', Buffer())
    probe.call('simaai_boxdecode:src', Buffer(stamp=999))
    assert probe.cache.snapshot()['pending'] == 0
    assert probe.events[-1][0] == 'detections'


def test_cache_miss_does_not_fabricate_time(probe):
    encoded = Buffer(stream='unknown')
    probe.call('encoder:src', encoded)
    assert encoded.pts == NONE
    assert 'missing_media_time' in probe.health.errors


def test_evicted_frame_does_not_borrow_another_time(probe):
    for frame in range(4):
        probe.call('decoder:src', Buffer(frame=frame, stamp=frame))
    encoded = Buffer(frame=0)
    probe.call('encoder:src', encoded)
    assert encoded.pts == NONE
    assert 'missing_media_time' in probe.health.errors


@pytest.mark.parametrize('buffer', [Buffer(meta=False), Buffer(stamp=NONE), Buffer(frame=True)])
def test_invalid_source_metadata_stops_probe(probe, buffer):
    assert probe.call('decoder:src', buffer) == 1
    assert probe.fatal.is_set()
    assert any(e.startswith('source_timestamp_error:') for e in probe.health.errors)
    assert probe.cache.snapshot()['pending'] == 0


def test_conflicting_source_time_stops_probe(probe):
    probe.call('decoder:src', Buffer(stamp=100))
    assert probe.call('decoder:src', Buffer(stamp=200)) == 1
    assert probe.fatal.is_set()
    assert probe.cache.take('A', 1) == 0


def test_clock_conversion_removes_uptime_and_preserves_intervals():
    clock = SourceClock()
    uptime = 179_755_000_000_000
    assert clock.convert(uptime, 2_000_000_000) == 2_000_000_000
    # Delivery jitter must not change spacing or re-anchor the clock.
    assert clock.convert(uptime + 33_333_333, 2_500_000_000) == 2_033_333_333
    assert clock.convert(uptime + 66_666_666, 2_510_000_000) == 2_066_666_666
    assert clock.snapshot()['offset_ns'] == 2_000_000_000 - uptime


@pytest.mark.parametrize('bad', [None, True, -1, 1.5, NONE, NONE + 1])
@pytest.mark.parametrize('field', ['source', 'pipeline'])
def test_invalid_clock_input_does_not_establish_anchor(bad, field):
    clock = SourceClock()
    with pytest.raises(ValueError, match='invalid'):
        clock.convert(bad if field == 'source' else 100, bad if field == 'pipeline' else 0)
    assert clock.snapshot()['offset_ns'] is None


def test_clock_reset_is_not_hidden_by_reanchoring():
    clock = SourceClock()
    clock.convert(100, 50)
    with pytest.raises(ValueError, match='reset'):
        clock.convert(99, 60)
    assert clock.snapshot()['source_origin_ns'] == 100
    assert clock.convert(110, 70) == 60


def test_translated_clock_cannot_become_clock_time_none():
    clock = SourceClock()
    clock.convert(0, 100)
    with pytest.raises(ValueError, match='range'):
        clock.convert(NONE - 1, 200)


def test_worker_restores_pipeline_time_not_board_uptime(probe):
    probe.namespace['pipeline'] = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(get_time=lambda: 9_000_000_000),
        get_base_time=lambda: 7_000_000_000,
    )
    uptime = 179_755_000_000_000
    probe.call('decoder:src', Buffer(frame=1, stamp=uptime))
    encoded = Buffer(frame=1)
    probe.call('encoder:src', encoded)
    assert encoded.pts == 2_000_000_000
    probe.call('decoder:src', Buffer(frame=2, stamp=uptime + 33_333_333))
    encoded = Buffer(frame=2)
    probe.call('encoder:src', encoded)
    assert encoded.pts == 2_033_333_333
    assert not probe.health.errors


def test_unavailable_pipeline_clock_stops_instead_of_inventing_time(probe):
    probe.namespace['pipeline'] = SimpleNamespace(get_clock=lambda: None)
    assert probe.call('decoder:src', Buffer()) == 1
    assert probe.fatal.is_set()
    assert probe.cache.snapshot()['stored'] == 0


def test_non_diagnostic_missing_time_drops_output(probe):
    probe.namespace['diagnostic'] = False
    assert probe.call('encoder:src', Buffer()) == 1
    assert probe.fatal.is_set()
