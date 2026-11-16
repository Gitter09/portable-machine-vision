"""Start pinned portable video plus a local viewer; Q or Ctrl+C stops both.

No services, release edits, feed killing or automatic restarts. Run from the
workstation named by the pinned deployment. A finite duration bounds orphan risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = '/media/nvme/portable-release-20261114-r33'


def remote_command(release, output, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    release, output = PurePosixPath(release), PurePosixPath(output)
    if not release.is_absolute() or not output.is_absolute() or '..' in release.parts + output.parts:
        raise ValueError('absolute normalized board paths required')
    if output == release or release in output.parents:
        raise ValueError('output must be outside the pinned release')
    return "echo PORTABLE_PID=$$; exec " + shlex.join([
        'python3', '-u', str(release / 'tools/launch.py'), 'run', str(release),
        '--output', str(output), '--seconds', str(seconds)])


def sdp(host, port):
    socket.inet_pton(socket.AF_INET, host)
    if type(port) is not int or not 1024 <= port <= 65534:
        raise ValueError('invalid receiver port')
    return (f'v=0\no=- 0 0 IN IP4 127.0.0.1\ns=Portable annotated debug video\n'
            f'c=IN IP4 {host}\nt=0 0\nm=video {port} RTP/AVP 96\n'
            'a=rtpmap:96 H264/90000\na=fmtp:96 packetization-mode=1\n')


def receiver_command(binary, source, headless, frames, progress):
    common = ['-protocol_whitelist', 'file,udp,rtp', '-i', str(source)]
    if headless:
        if type(frames) is not int or frames <= 0:
            raise ValueError('positive frame count required')
        return [binary, '-hide_banner', '-loglevel', 'warning', *common, '-an',
                '-frames:v', str(frames), '-progress', str(progress), '-f', 'null', '-']
    return [binary, '-hide_banner', '-loglevel', 'warning', '-stats', '-window_title',
            'Portable annotated DEBUG - Q to stop', *common]


def remote_python(client, code):
    _, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(code), timeout=15)
    text, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error.strip() or text.strip())
    return text


def stop_launcher(client, pid, release, output):
    # Check both exact argv tokens before signalling; never kill by process name.
    code = ("import os,signal,pathlib\n"
            f"p=pathlib.Path('/proc/{int(pid)}/cmdline')\n"
            "if p.exists():\n"
            " a=p.read_bytes().decode().split('\\0')\n"
            f" if {str(PurePosixPath(release) / 'tools/launch.py')!r} not in a or {output!r} not in a: raise SystemExit('PID identity changed; refusing signal')\n"
            f" os.kill({int(pid)},signal.SIGTERM)\n")
    remote_python(client, code)


def busy_processes(client):
    return json.loads(remote_python(client, """import pathlib,json,os
busy=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().decode().split('\\0')
 except (OSError,UnicodeError):
  continue
 names=[pathlib.Path(x).name for x in a]
 if any(x in names for x in ['gst-launch-1.0','feed_supervisor.sh','worker.py','launch.py','portable_tensor_parity.py','inspect_native_buffers.py','compare_native_buffers.py']):
  busy.append({'pid':int(p.parent.name),'programs':[x for x in names if x.endswith('.py') or x=='gst-launch-1.0']})
print(json.dumps(busy))
"""))


def run(args):
    import paramiko
    binary = shutil.which('ffmpeg' if args.headless else 'ffplay')
    if not binary:
        raise RuntimeError('FFmpeg/FFplay is not on PATH')
    # Validate before opening sockets or starting either process.
    receiver_command(binary, 'receiver.sdp', args.headless, args.frames, 'progress.txt')
    board_output = '/media/nvme/ic-terminal-live-' + args.output.name
    command = remote_command(args.release, board_output, args.seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    credentials = json.loads(args.credentials.read_text())['board']
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    receiver = channel = sftp = None
    pid = None
    result = {'release': args.release, 'board_output': board_output, 'headless': args.headless,
              'requested_seconds': args.seconds, 'errors': []}
    try:
        client.connect(credentials['host'], username=credentials['user'], password=credentials['password'], timeout=15)
        sftp = client.open_sftp()
        busy = busy_processes(client)
        if busy:
            raise RuntimeError('Board occupied; stop its current job explicitly: ' + json.dumps(busy))
        try:
            sftp.stat(board_output)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError('Board output already exists; choose a new output basename: ' + board_output)
        with sftp.open(args.release + '/release.json') as source:
            manifest = json.load(source)
        with sftp.open(args.release + '/assets/deployment.json') as source:
            deployment_bytes = source.read()
        if hashlib.sha256(deployment_bytes).hexdigest() != manifest['assets']['deployment.json']:
            raise RuntimeError('Pinned deployment hash differs')
        config = json.loads(deployment_bytes)
        host, port = config['receiver_host'], config['receiver_port']
        source = args.output / 'receiver.sdp'
        source.write_text(sdp(host, port))
        # Refuse a moved workstation or occupied RTP/RTCP port, rather than send blindly.
        for p in (port, port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.bind((host, p))
        (args.output / 'release.json').write_text(json.dumps(manifest, indent=2))
        with (args.output / 'receiver.log').open('x') as log, (args.output / 'launcher.log').open('xb') as remote_log:
            receiver = subprocess.Popen(receiver_command(binary, source, args.headless, args.frames, args.output / 'progress.txt'),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(.5)
            if receiver.poll() is not None:
                raise RuntimeError('Receiver exited before board startup; inspect receiver.log')
            channel = client.get_transport().open_session(timeout=15)
            channel.set_combine_stderr(True)
            channel.exec_command(command)
            channel.settimeout(15)
            prefix = b''
            while b'\n' not in prefix:
                chunk = channel.recv(1024)
                if not chunk:
                    raise RuntimeError('Launcher closed before reporting PID')
                prefix += chunk
            line, remainder = prefix.split(b'\n', 1)
            if not line.startswith(b'PORTABLE_PID='):
                raise RuntimeError('Unexpected launcher response: ' + line.decode(errors='replace'))
            pid = int(line.split(b'=')[1])
            result['launcher_pid'] = pid
            remote_log.write(remainder)
            print(f'Started {args.release}; receiver {host}:{port}. Q in viewer or Ctrl+C here stops both.', flush=True)
            end = time.monotonic() + args.seconds + 30
            while not channel.exit_status_ready() and receiver.poll() is None:
                if time.monotonic() > end:
                    raise RuntimeError('Host watchdog expired')
                while channel.recv_ready():
                    remote_log.write(channel.recv(65536))
                time.sleep(.1)
            if channel.exit_status_ready():
                result['launcher_exit'] = channel.recv_exit_status()
    except KeyboardInterrupt:
        result['user_stop'] = True
    except Exception as error:
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if pid is not None:
            try:
                if channel is not None and not channel.exit_status_ready():
                    stop_launcher(client, pid, args.release, board_output)
                    deadline = time.monotonic() + 20
                    while not channel.exit_status_ready() and time.monotonic() < deadline:
                        time.sleep(.1)
                    if not channel.exit_status_ready():
                        raise RuntimeError('Launcher did not stop; check board before another run')
                result['launcher_exit'] = channel.recv_exit_status()
            except Exception as error:
                result['errors'].append('board_stop: ' + str(error))
        if receiver is not None:
            natural_exit = receiver.poll()
            result['receiver_natural_exit'] = natural_exit
            if natural_exit is None:
                receiver.terminate()
                try:
                    receiver.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    receiver.kill()
                    receiver.wait(timeout=5)
            result['receiver_exit'] = receiver.returncode
        if sftp is not None and pid is not None:
            for name in ('worker.json', 'health.json', 'supervisor.json'):
                try:
                    sftp.get(board_output + '/' + name, str(args.output / name))
                except OSError as error:
                    result['errors'].append('fetch ' + name + ': ' + str(error))
        client.close()
        worker_path = args.output / 'worker.json'
        if worker_path.exists():
            worker = json.loads(worker_path.read_text())
            result['board_frames'] = worker.get('frames')
            if worker.get('errors') or not worker.get('frames'):
                result['errors'].append('Board run unsuccessful: ' + str(worker.get('errors')))
        if result.get('launcher_exit', 1) != 0:
            result['errors'].append('Launcher did not exit cleanly')
        if not args.headless and result.get('receiver_natural_exit') not in (None, 0):
            result['errors'].append('Viewer exited unexpectedly; inspect receiver.log')
        if args.headless:
            progress = args.output / 'progress.txt'
            counts = [int(x.split('=', 1)[1]) for x in progress.read_text().splitlines() if x.startswith('frame=')] if progress.exists() else []
            result['decoded_frames'] = max(counts, default=0)
            if result.get('receiver_natural_exit') != 0 or result['decoded_frames'] < args.frames:
                result['errors'].append('Bounded receiver decode did not complete')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    return 1 if result['errors'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, default=ROOT / 'config/credentials.json')
    parser.add_argument('--release', default=DEFAULT_RELEASE)
    parser.add_argument('--seconds', type=float, default=600, help='finite maximum; default ten minutes')
    parser.add_argument('--output', type=Path, required=True, help='new local evidence directory')
    parser.add_argument('--headless', action='store_true', help='bounded FFmpeg decode instead of a viewer')
    parser.add_argument('--frames', type=int, default=900, help='headless decoded frame target')
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
