"""Read-only steady-state benchmark of an existing portable live session.

Never starts/stops a graph or receiver. Class presence is not accuracy; delivery
intervals are not optical latency. Keep the owner-confirmed specimen unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import struct
import time

LABELS = ['IC_Fail', 'Terminal_pass', 'IC_Pass', 'Terminal_Fail']
ROOT = Path(__file__).resolve().parents[2]


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*percent/100
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def interval_stats(values):
    return {'count': len(values), 'p50_ms': percentile(values, 50),
            'p95_ms': percentile(values, 95), 'p99_ms': percentile(values, 99),
            'max_ms': max(values, default=None)}


def process_stat(text):
    fields = text[text.rindex(')')+2:].split()  # field 3 onward; comm may have spaces
    return {'cpu_ticks': int(fields[11])+int(fields[12]), 'start_ticks': int(fields[19])}


def clock_ticks(auxv):
    if len(auxv) % 16:
        raise ValueError('expected the portable board ELF64 little-endian auxv')
    for tag, value in struct.iter_unpack('<QQ', auxv):
        if tag == 17 and value > 0:  # Linux AT_CLKTCK; do not guess 100
            return value
    raise ValueError('AT_CLKTCK unavailable')


def total_cpu(text):
    lines = text.splitlines()
    values = list(map(int, lines[0].split()[1:9]))  # guest is already included in user/nice
    return {'total': sum(values), 'idle': values[3]+values[4],
            'logical_cpus': sum(bool(re.match(r'^cpu\d+\s', line)) for line in lines)}


def viewer_progress(path):
    if not path.exists():
        return None
    with path.open('rb') as source:
        source.seek(max(0, path.stat().st_size-16384))
        text = source.read(16384).decode('utf-8', errors='replace')
    hits = re.findall(r'([\d.]+)\s+M-V:\s*([-+\d.]+)\s+fd=\s*(\d+)', text)
    if not hits:
        return None
    clock, difference, dropped = hits[-1]
    return {'clock_seconds': float(clock), 'master_minus_video_seconds': float(difference),
            'reported_drops': int(dropped)}


def complete_rows(data, starts_midfile=False):
    if starts_midfile:
        data = data.partition(b'\n')[2]
    if data and not data.endswith(b'\n'):
        data = data.rpartition(b'\n')[0]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def cohort(frames, detections, start, end):
    session, stream = start['session'], start['stream_id']
    def selected(row):
        return (row['session'] == session and row['stream_id'] == stream
                and start['last_frame_id'] < row['frame_id'] <= end['last_frame_id'])
    frames = [r for r in frames if selected(r)]
    lookup = {}
    for row in detections:
        if selected(row):
            key = row['frame_id']
            if key in lookup:
                raise ValueError('duplicate detection identity')
            lookup[key] = row
    ids = [r['frame_id'] for r in frames]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate encoder identity')
    expected = end['frames']-start['frames']
    if len(frames) != expected or any(i not in lookup for i in ids):
        raise ValueError('window logs incomplete relative to health counters')
    return frames, [lookup[i] for i in ids]


def prediction_summary(rows, expected):
    result = {'scope': 'native class-presence only, not precision/recall or part-level verdicts',
              'records': len(rows), 'expected_labels': expected, 'both_or_all_expected': 0,
              'any_unexpected': 0, 'empty': 0, 'classes': {label: {'frames_present': 0,
              'boxes': 0, 'frames_with_multiple_candidates': 0} for label in LABELS}}
    scores = {label: [] for label in LABELS}
    for row in rows:
        labels = [d['label'] for d in row['detections']]
        present = set(labels)
        result['both_or_all_expected'] += set(expected).issubset(present)
        result['any_unexpected'] += bool(present-set(expected))
        result['empty'] += not labels
        for label in LABELS:
            n = labels.count(label)
            c = result['classes'][label]
            c['frames_present'] += n > 0
            c['boxes'] += n
            c['frames_with_multiple_candidates'] += n > 1
        for det in row['detections']:
            scores[det['label']].append(det['score'])
    for label, values in scores.items():
        result['classes'][label].update(score_min=min(values, default=None), score_median=statistics.median(values) if values else None,
                                        score_max=max(values, default=None))
    return result


def summarize(samples, frames, detections, expected, ticks):
    first, last = samples[0], samples[-1]
    elapsed = last['board_uptime']-first['board_uptime']
    count = last['health']['frames']-first['health']['frames']
    stamps = [r['monotonic_seconds'] for r in frames]
    arrival = [(b-a)*1000 for a,b in zip(stamps, stamps[1:])]
    pts = [r['pts'] for r in frames]
    media = [(b-a)/1e6 for a,b in zip(pts, pts[1:])]
    worker_cpu, board_cpu = [], []
    for a,b in zip(samples, samples[1:]):
        dt = b['board_uptime']-a['board_uptime']
        if dt <= 0 or a['process']['start_ticks'] != b['process']['start_ticks']:
            raise ValueError('process epoch or sampling time changed')
        worker_cpu.append((b['process']['cpu_ticks']-a['process']['cpu_ticks'])/ticks/dt*100)
        total = b['cpu']['total']-a['cpu']['total']
        idle = b['cpu']['idle']-a['cpu']['idle']
        if total > 0:
            board_cpu.append((total-idle)/total*100)
    rss = [s['rss_kib'] for s in samples]
    viewer = None
    if first['viewer'] and last['viewer']:
        drops = last['viewer']['reported_drops']-first['viewer']['reported_drops']
        clock = last['viewer']['clock_seconds']-first['viewer']['clock_seconds']
        if drops < 0 or clock < 0:
            raise ValueError('viewer restarted during benchmark')
        viewer = {'reported_drop_delta': drops, 'video_clock_advance_seconds': clock,
                  'scope': 'FFplay counter, not RTP packet loss, rendered FPS or optical latency'}
    return {'window_seconds': elapsed, 'encoder_frames': count,
            'encoder_fps_from_health_window': count/elapsed,
            'encoder_fps_from_delivery_span': (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else None,
            'encoder_delivery_intervals': interval_stats(arrival), 'media_pts_intervals': interval_stats(media),
            'encoder_frame_id_gaps': (max(0,frames[0]['frame_id']-first['health']['last_frame_id']-1) if frames else 0)
                + sum(max(0,b['frame_id']-a['frame_id']-1) for a,b in zip(frames,frames[1:])),
            'valid_timed_frame_delta': last['health']['valid_timed_frames']-first['health']['valid_timed_frames'],
            'rtp_marker_delta': last['health']['rtp_frame_markers']-first['health']['rtp_frame_markers'],
            'board_errors': sorted({e for s in samples for e in s['health']['errors']}),
            'not_media_ready_samples': sum(not s['health']['media_ready'] for s in samples),
            'worker_cpu_percent_one_core': {'mean': statistics.mean(worker_cpu), 'max': max(worker_cpu)},
            'board_cpu_percent_all_cores': {'mean': statistics.mean(board_cpu), 'max': max(board_cpu),
                                          'logical_cpus': first['cpu']['logical_cpus']},
            'worker_rss_kib': {'min': min(rss), 'max': max(rss), 'end_minus_start': rss[-1]-rss[0]},
            'board_mem_available_kib': {'min': min(s['mem_available_kib'] for s in samples),
                                      'max': max(s['mem_available_kib'] for s in samples)},
            'timestamp_cache': {'max_sampled_pending': max(s['health']['timestamp_cache']['pending'] for s in samples),
                                **{key+'_delta': last['health']['timestamp_cache'][key]-first['health']['timestamp_cache'][key]
                                   for key in ('misses','evicted')}},
            'viewer': viewer, 'predictions': prediction_summary(detections, expected)}


def main():
    import paramiko
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True, help='existing host .session.json')
    parser.add_argument('--case', required=True, help='operator-confirmed specimen state')
    parser.add_argument('--expected', nargs='+', choices=LABELS, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--credentials', type=Path, default=ROOT/'config/credentials.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not math.isfinite(args.interval) or not .25 <= args.interval <= args.seconds:
        parser.error('require finite seconds >= interval >= 0.25')
    args.output.mkdir(parents=True, exist_ok=False)
    session = json.loads(args.session.read_text(encoding='utf-8'))
    root = session['board_output']
    pid = int(session['remote_worker_pid'])
    local = Path(session['output'])
    credentials = json.loads(args.credentials.read_text(encoding='utf-8'))['board']
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    samples = []
    try:
        client.connect(credentials['host'], username=credentials['user'], password=credentials['password'], timeout=15)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(15)
        def read(path):
            with sftp.open(path, 'rb') as f:
                return f.read()
        argv = read(f'/proc/{pid}/cmdline').decode().split('\0')
        release = argv[argv.index('run')+1]
        if root not in argv or str(PurePosixPath(release)/'tools/worker.py') not in argv:
            raise ValueError('worker identity differs from the requested session')
        inv = read(release+'/release.json')
        (args.output/'release.json').write_bytes(inv)
        ticks = clock_ticks(read(f'/proc/{pid}/auxv'))
        boot_id = read('/proc/sys/kernel/random/boot_id').decode().strip()
        def sample():
            h = json.loads(read(root+'/health.json'))
            if not h['running'] or h['diagnostic'] or not h['release_verified'] or h['errors']:
                raise ValueError('existing pipeline not healthy verified live UDP: '+str(h['errors']))
            if not samples and not h['media_ready']:
                raise ValueError('pipeline must already be media-ready for a steady-state benchmark')
            if samples and h['session'] != samples[0]['health']['session']:
                raise ValueError('pipeline session changed')
            status = read(f'/proc/{pid}/status').decode()
            mem = read('/proc/meminfo').decode()
            return {'host_wall_ns': time.time_ns(), 'host_monotonic': time.monotonic(),
                    'board_uptime': float(read('/proc/uptime').split()[0]),
                    'health': h, 'process': process_stat(read(f'/proc/{pid}/stat').decode()),
                    'cpu': total_cpu(read('/proc/stat').decode()),
                    'rss_kib': int(re.search(r'^VmRSS:\s+(\d+)',status,re.M)[1]),
                    'mem_available_kib': int(re.search(r'^MemAvailable:\s+(\d+)',mem,re.M)[1]),
                    'viewer': viewer_progress(local/'receiver.log')}
        # Include overlap for buffered logs; cohort is strictly gated by encoder IDs.
        offsets = {name: max(0,sftp.stat(root+'/'+name).st_size-131072) for name in ('frames.jsonl','detections.jsonl')}
        samples.append(sample())
        began = time.monotonic()
        next_poll = began+args.interval
        print('Measuring existing live stream; keep specimen unchanged:',args.case,flush=True)
        with (args.output/'samples.jsonl').open('x',encoding='utf-8') as log:
            log.write(json.dumps(samples[0])+'\n');log.flush()
            while time.monotonic()-began < args.seconds:
                time.sleep(max(0,min(next_poll,began+args.seconds)-time.monotonic()))
                samples.append(sample())
                log.write(json.dumps(samples[-1])+'\n');log.flush()
                next_poll += args.interval
        time.sleep(.6)  # let the worker flush the terminal encoder cohort to disk
        data = {}
        for name,offset in offsets.items():
            length = sftp.stat(root+'/'+name).st_size-offset
            with sftp.open(root+'/'+name,'rb') as source:
                source.seek(offset);chunk=source.read(length)
            (args.output/('raw-window-'+name)).write_bytes(chunk)
            data[name] = complete_rows(chunk, offset>0)
        first, last = samples[0]['health'], samples[-1]['health']
        if samples[-1]['process']['start_ticks'] != samples[0]['process']['start_ticks']:
            raise ValueError('worker process restarted')
        frames, detections = cohort(data['frames.jsonl'],data['detections.jsonl'],first,last)
        report = {'case':args.case,'scope':'steady-state existing native pipeline; owner-confirmed labels, not box accuracy',
                  'source_session':str(args.session),'release':release,'release_inventory_sha256':hashlib.sha256(inv).hexdigest(),
                  'board_output':root,'worker_pid':pid,'boot_id':boot_id,'clock_ticks_per_second':ticks,
                  'log_offsets':offsets,'first_health':first,'last_health':last,
                  'limitations':['No stream restart, no cold-start measurement.',
                                 'Worker CPU/RSS excludes native service memory; total CPU includes all board work and sampler SSH.',
                                 'No MLA utilization, power, optical/network latency or independent source/box labels measured.',
                                 'No rendered/client decoded FPS or RTP packet-loss measurement from this observer.',
                                 'Native raw scores, including near-threshold values, are not post-filtered.',
                                 'Framing accepted by owner; no new pixel-clearance measurement in this read-only run.'],
                  'metrics':summarize(samples,frames,detections,args.expected,ticks)}
        (args.output/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(report['metrics'],indent=2),flush=True)
    except Exception as error:
        (args.output/'benchmark-error.json').write_text(json.dumps({'error':type(error).__name__+': '+str(error),
                                                                  'samples':len(samples),'stream_not_stopped_by_observer':True},indent=2))
        raise
    finally:
        client.close()  # only this observer's connection; never the owning live wrapper


if __name__ == '__main__':
    main()
