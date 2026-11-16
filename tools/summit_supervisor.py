"""Sequential renewal of verified bounded portable sessions; never steals a decoder."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import tempfile


@contextmanager
def private_profile(settings):
    """A prompt-once secret is available to children only through a private temp file."""
    with tempfile.TemporaryDirectory(prefix='portable-vision-auth-') as directory:
        path = Path(directory)/'link.json'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as file:
            json.dump(settings, file)
        yield path


def read_report(path):
    try:
        if path.stat().st_size > 1024 * 1024:
            return {}
        report = json.loads(path.read_text())
        return report if isinstance(report, dict) else {}
    except (OSError, ValueError):
        return {}


def renewal_decision(code, output):
    """A completed host process alone is not proof that its board worker stopped."""
    session = read_report(output/'session.json')
    worker = read_report(output/'worker.json')
    if session.get('state') in ('stopped', 'stop-requested') or worker.get('stop_requested'):
        return 'stop'
    if session.get('stop_error') or type(session.get('remote_exit')) is not int:
        return 'unsafe'
    if worker.get('state') not in ('completed', 'stopped', 'error'):
        return 'unsafe'
    if code == 0 and session.get('state') == 'completed' and session['remote_exit'] == 0 and worker.get('state') == 'completed':
        return 'renew'
    return 'retry'


class SingleOwner:
    """OS advisory lock is released on host crash; no stale PID-file lockout."""
    def __init__(self, path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):
                    self.file.write(b'0'); self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise RuntimeError('Another summit supervisor owns this session directory') from error
        return self

    def __exit__(self, *_):
        self.file.close()


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        if os.name == 'nt':
            child.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=25)
    except subprocess.TimeoutExpired:
        # Only our process group, never a broad board or host process-name kill.
        if os.name == 'nt':
            child.terminate()
        else:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=5)


def run_child(command, seconds, lock_fd=None):
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    if os.name != 'nt' and lock_fd is not None:
        # Keep ownership if the supervisor dies while its launcher is still running.
        options['pass_fds'] = (lock_fd,)
    child = subprocess.Popen(command, **options)
    try:
        return child.wait(timeout=seconds + 180)
    except subprocess.TimeoutExpired:
        stop_child(child)
        return 124
    except KeyboardInterrupt:
        stop_child(child)
        raise


class SessionSupervisor:
    def __init__(self, command, root, seconds=300, max_sessions=0, max_failures=3, min_free_bytes=2 * 1024**3):
        self.command = command
        self.root = Path(root)
        self.seconds = seconds
        self.max_sessions = max_sessions
        self.max_failures = max_failures
        self.min_free_bytes = min_free_bytes

    def status(self, state, output=None):
        record = {'state': state, 'updated_unix': time.time(), 'session': output.name if output else None}
        temporary = self.root/'summit-status.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(self.root/'summit-status.json')
        print('Summit supervisor:', state, record['session'] or '', flush=True)

    def run(self, runner=run_child, sleep=time.sleep):
        self.root.mkdir(parents=True, exist_ok=True)
        failures = completed = 0
        try:
            while not self.max_sessions or completed < self.max_sessions:
                if shutil.disk_usage(self.root).free < self.min_free_bytes:
                    self.status('low-disk-space-manual-check-required')
                    return 1
                output = self.root/f'summit-{time.time_ns()}'
                self.status('starting', output)
                code = runner(self.command + ['--output', str(output)], self.seconds)
                decision = renewal_decision(code, output)
                if decision == 'stop':
                    self.status('stopped', output)
                    return 0
                if code == 124 or decision == 'unsafe':
                    self.status('cleanup-unconfirmed-manual-check-required', output)
                    return 1
                if decision == 'renew':
                    completed += 1
                    failures = 0
                    self.status('session-completed', output)
                else:
                    failures += 1
                    if failures >= self.max_failures:
                        self.status('retry-limit-reached', output)
                        return 1
                    self.status('retrying-confirmed-ended-worker', output)
                if not self.max_sessions or completed < self.max_sessions:
                    sleep(min(30, 2 ** (failures + 1)))
            self.status('completed')
            return 0
        except KeyboardInterrupt:
            self.status('stopped-by-presenter')
            return 130
