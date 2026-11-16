"""Separate raw-RTP timing checks from archival viewing-video consistency."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
from support import Health


def verify_media(path):
    issues = set()
    packets = frames = markers = 0
    last_sequence = last_frame_time = ssrc = current_time = None
    fragment = False
    frame_valid = True
    with Path(path).open('rb') as capture:
        while True:
            prefix = capture.read(12)
            if not prefix:
                break
            if len(prefix) != 12:
                issues.add('truncated_record_header')
                break
            _wall_ns, length = struct.unpack('!QI', prefix)
            if not 12 <= length <= 65535:
                issues.add('invalid_record_length')
                break
            packet = capture.read(length)
            if len(packet) != length:
                issues.add('truncated_packet')
                break
            packets += 1
            flags, payload_flags, sequence, timestamp, source = struct.unpack_from('!BBHII', packet)
            offset = 12 + (flags & 15) * 4
            end = len(packet)
            if flags >> 6 != 2 or payload_flags & 127 != 96 or offset > end:
                issues.add('invalid_rtp_header')
                frame_valid = False
                continue
            if flags & 16:
                if offset + 4 > end:
                    issues.add('truncated_rtp_extension')
                    frame_valid = False
                    continue
                offset += 4 + struct.unpack_from('!H', packet, offset + 2)[0] * 4
            if flags & 32:
                padding = packet[-1]
                if padding == 0 or padding > end - offset:
                    issues.add('invalid_rtp_padding')
                    frame_valid = False
                    continue
                end -= padding
            if offset >= end:
                issues.add('empty_or_truncated_rtp_payload')
                frame_valid = False
                continue
            if ssrc is not None and source != ssrc:
                issues.add('stream_changed_without_new_capture')
                fragment = False
            ssrc = source
            if last_sequence is not None and (sequence - last_sequence) & 65535 != 1:
                issues.add('sequence_discontinuity')
                fragment = False
                frame_valid = False
            last_sequence = sequence
            if current_time is not None and timestamp != current_time:
                issues.add('timestamp_changed_before_frame_end')
                fragment = False
                frame_valid = False
            current_time = timestamp
            payload = packet[offset:end]
            nal_type = payload[0] & 31
            valid = True
            if 1 <= nal_type <= 23:
                if fragment:
                    valid = False
                fragment = False
            elif nal_type == 24:
                index = 1
                if fragment:
                    valid = False
                fragment = False
                while index + 2 <= len(payload):
                    size = struct.unpack_from('!H', payload, index)[0]
                    index += 2
                    if not size or index + size > len(payload) or payload[index] & 128 or not 1 <= payload[index] & 31 <= 23:
                        valid = False
                        break
                    index += size
                valid = valid and index == len(payload) and index > 1
            elif nal_type == 28:
                if len(payload) < 2 or not 1 <= payload[1] & 31 <= 23:
                    valid = False
                else:
                    start, finish = bool(payload[1] & 128), bool(payload[1] & 64)
                    identity = (payload[0] & 96, payload[1] & 31)
                    valid = not (start and finish) and (not fragment if start else fragment == identity)
                    fragment = identity if valid and not finish else False
            else:
                valid = False
            valid = valid and not (payload[0] & 128)
            if not valid:
                issues.add('malformed_or_unsupported_h264_payload')
                frame_valid = False
                fragment = False
            if payload_flags & 128:
                markers += 1
                if fragment:
                    issues.add('unfinished_h264_fragment')
                    frame_valid = False
                if last_frame_time is not None:
                    delta = (timestamp - last_frame_time) & 0xffffffff
                    if delta == 0 or delta >= 0x80000000:
                        issues.add('nonadvancing_frame_time')
                last_frame_time = timestamp
                frames += int(frame_valid)
                current_time = None
                fragment = False
                frame_valid = True
    if current_time is not None or fragment:
        issues.add('incomplete_final_frame')
    if frames < 2:
        issues.add('insufficient_complete_frames')
    return {
        'scope': 'rtp_media_timing_only',
        'passed': not issues,
        'packets': packets,
        'frames': frames,
        'raw_frame_markers': markers,
        'issues': sorted(issues),
        'not_checked': ['decoded_pixels', 'pairing', 'numerical_reference', 'recovery', 'optical_latency', 'soak'],
    }


def verify_archive(root):
    """Retain the historical experiment's narrow, explicitly non-readiness check."""
    root = Path(root)
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        raise ValueError('ffprobe is required for archival verification')
    decoded = {}
    for receiver, video in (
        ('native-receiver', 'annotated-baseline-30fps.mp4'),
        ('return-receiver', 'annotated-return-30fps.mp4'),
    ):
        summary = json.loads((root / receiver / 'summary.json').read_text())
        if not isinstance(summary, dict) or not isinstance(summary.get('counters'), dict):
            raise ValueError('archive receiver metadata must contain a counters object')
        counters = summary['counters']
        output = subprocess.run([
            ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=codec_name,width,height,nb_read_frames', '-of', 'json', str(root / video),
        ], capture_output=True, text=True, check=True)
        streams = json.loads(output.stdout)['streams']
        if len(streams) != 1:
            raise ValueError('archive must contain one selected video stream')
        stream = streams[0]
        frames = int(stream['nb_read_frames'])
        if stream['codec_name'] != 'h264' or (stream['width'], stream['height']) != (1280, 720):
            raise ValueError('unexpected archived video format')
        if frames != counters['frame_markers']:
            raise ValueError('archived decoded/marker count mismatch')
        if any(value for key, value in counters.items() if key.startswith(('rejected_', 'invalid_', 'unsupported_', 'orphan_')) or key in ('missing_sequence_numbers', 'duplicates', 'out_of_order')):
            raise ValueError('archive contains rejected, missing or reordered packets')
        decoded[video] = frames
    for name in ('native', 'restart', 'baseline'):
        run = json.loads((root / name / 'run.json').read_text())
        if run['exit_code'] != 0 or run['early_exit'] or run['measurement_elapsed_seconds'] < run['warmup_seconds'] + run['steady_seconds']:
            raise ValueError('incomplete or failed archived run: ' + name)
    return {'scope': 'archival_consistency_only', 'passed': True, 'decoded_frames': decoded, 'live_readiness': 'not_verified'}


def verify_run(root):
    """Validate raw run evidence; absent hardware qualification remains unverified."""
    root = Path(root)
    issues = set()
    frames = 0
    try:
        supervisor = json.loads((root / 'supervisor.json').read_text())
        worker = json.loads((root / 'worker.json').read_text())
        if supervisor['exit_code'] != 0 or supervisor['forced'] or not supervisor['reaped'] or supervisor['stop_reason']:
            issues.add('failed_or_forced_process_outcome')
        if not worker['shutdown_complete'] or not worker['release_verified'] or worker['errors']:
            issues.add('worker_failed_or_release_unverified')
        health = Health(started=0, session=worker['session'])
        last_time = None
        with (root / 'frames.jsonl').open() as source:
            for line in source:
                event = json.loads(line)
                now = event['monotonic_seconds']
                if event['session'] != health.session:
                    issues.add('session_mismatch')
                if last_time is not None and now <= last_time:
                    issues.add('nonadvancing_observation_clock')
                if last_time is not None and now-last_time >= 1:
                    issues.add('stale_frame_gap')
                health.observe(event['frame_id'], event['stream_id'], event['pts'], now)
                last_time = now
                frames += 1
        issues.update(health.errors)
        if frames < 2 or frames != worker['frames']:
            issues.add('insufficient_or_incomplete_frame_evidence')
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.add('missing_or_invalid_run_evidence:' + str(error))
    return {
        'scope': 'live_readiness',
        'passed': False,
        'run_integrity_passed': not issues,
        'frames': frames,
        'issues': sorted(issues),
        'not_verified': ['pairing', 'numerical_reference', 'recovery_trials', 'normal_playback', 'optical_latency', 'four_hour_soak'],
        'qualification': 'not_established',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('media', 'archive', 'run'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    try:
        result = {'media': verify_media, 'archive': verify_archive, 'run': verify_run}[args.mode](args.path)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        scope = {'media': 'rtp_media_timing_only', 'archive': 'archival_consistency_only', 'run': 'live_readiness'}[args.mode]
        result = {'scope': scope, 'passed': False, 'issues': [str(error)]}
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
