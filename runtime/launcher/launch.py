"""Stage or run the pinned Portable pipeline without installing system services."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from support import atomic_json, stop_process


def clean_environment(registry):
    env = os.environ.copy()
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'GST_TRACERS', 'GST_DEBUG_FILE'):
        env.pop(name, None)
    env.update({
        'GST_PLUGIN_PATH': '',
        'GST_PLUGIN_PATH_1_0': '',
        'GST_PLUGIN_SYSTEM_PATH': '/usr/lib/aarch64-linux-gnu/gstreamer-1.0',
        'GST_PLUGIN_SYSTEM_PATH_1_0': '/usr/lib/aarch64-linux-gnu/gstreamer-1.0',
        'GST_REGISTRY': str(registry),
        'GST_DEBUG': '1',
    })
    return env


def persist_outcome(output, metadata, failed):
    """Attempt process outcome and health invalidation independently."""
    errors = []
    try:
        atomic_json(output / 'supervisor.json', metadata)
    except OSError as error:
        errors.append('supervisor.json: ' + str(error))
    if failed:
        try:
            health: dict[str, Any] = json.loads((output / 'health.json').read_text())
            if not isinstance(health, dict) or not isinstance(health.get('errors', []), list):
                raise ValueError('invalid worker health')
        except (OSError, ValueError):
            health = {'errors': ['worker_health_unavailable']}
        health.update(schema_version=1, state='error', running=False, media_ready=False, supervisor=metadata)
        health['errors'] = sorted(set(health.get('errors', [])) | {'worker_failed_or_forced_shutdown'})
        try:
            atomic_json(output / 'health.json', health)
        except OSError as error:
            errors.append('health.json: ' + str(error))
    return errors


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('stage')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('release', type=Path)
    prepare.add_argument('--config', type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('release', type=Path)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--seconds', type=float, default=60)
    execute.add_argument('--diagnostic', action='store_true', help='local fakesink and raw capture; never sends broken RTP')
    args = parser.parse_args()
    worker = Path(__file__).with_name('worker.py')
    with tempfile.TemporaryDirectory(prefix='portable-gst-') as temporary:
        env = clean_environment(Path(temporary) / 'registry.bin')
        if args.command == 'stage':
            command = [sys.executable, str(worker), 'stage', str(args.source), str(args.release), '--config', str(args.config)]
            try:
                return subprocess.run(command, env=env, timeout=30, check=False).returncode
            except subprocess.TimeoutExpired:
                print('release staging exceeded 30 seconds', file=sys.stderr)
                return 1
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            parser.error('seconds must be a finite positive number')
        release = args.release.resolve()
        if worker.resolve() != (release / 'tools/worker.py').resolve():
            parser.error('run the staged release/tools/launch.py, not the staging copy')
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        output = args.output.resolve()
        metadata = {'schema_version': 1, 'release': str(release), 'diagnostic': args.diagnostic, 'wall_started_ns': time.time_ns()}
        atomic_json(output / 'supervisor.json', metadata)
        process = None
        interrupted = False
        stop_reason = None
        cleanup = {'exit_code': None, 'forced': False, 'reaped': False, 'signals': []}
        started = time.monotonic()
        previous = {}
        def request_stop(_number, _frame):
            nonlocal interrupted
            interrupted = True
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, request_stop)
        try:
            command = [sys.executable, str(worker), 'run', str(release), '--output', str(output), '--seconds', str(args.seconds)]
            if args.diagnostic:
                command.append('--diagnostic')
            with (output / 'worker.log').open('x') as log:
                process = subprocess.Popen(command, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.seconds + 15
                while process.poll() is None and not interrupted:
                    if time.monotonic() >= deadline:
                        stop_reason = 'watchdog_timeout'
                        break
                    try:
                        process.wait(timeout=.1)
                    except subprocess.TimeoutExpired:
                        pass
        except Exception as error:
            stop_reason = type(error).__name__ + ':' + str(error)
        finally:
            try:
                if process is not None:
                    cleanup = stop_process(process)
            except Exception as error:
                stop_reason = 'cleanup_failed:' + str(error)
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)
                metadata.update(cleanup, stop_reason=stop_reason, interrupted=interrupted, elapsed_seconds=time.monotonic()-started)
                failed = cleanup['forced'] or not cleanup['reaped'] or cleanup['exit_code'] != 0 or bool(stop_reason)
                write_errors = persist_outcome(output, metadata, failed)
                for error in write_errors:
                    print(error, file=sys.stderr)
        print(json.dumps(metadata, indent=2))
        return 0 if not failed and not write_errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
