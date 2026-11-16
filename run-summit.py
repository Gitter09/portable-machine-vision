"""Opt-in sequential summit sessions, without modifying the frozen board runtime.

Start the dashboard separately: python3 frontend/serve.py (auto-discovers sessions).
Each worker retains its 20..300s bound. Ctrl+C stops renewal and requests cleanup.
"""
import argparse
from contextlib import ExitStack
from functools import partial
import getpass
import json
from pathlib import Path
import sys
from tools.summit_supervisor import SessionSupervisor, SingleOwner, private_profile, run_child

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='existing private link profile')
    parser.add_argument('--ask-password', action='store_true', help='prompt once; reuse via a permission-restricted temporary profile, deleted on normal exit')
    parser.add_argument('--ssh-key', type=Path)
    parser.add_argument('--board-host')
    parser.add_argument('--receiver-host')
    parser.add_argument('--ssh-user')
    parser.add_argument('--package', type=Path, default=ROOT/'deployment/portable-v1')
    parser.add_argument('--session-root', type=Path, default=ROOT/'local/runs')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--max-sessions', type=int, default=0, help='0 renews clean sessions until Ctrl+C')
    parser.add_argument('--max-failures', type=int, default=3)
    args = parser.parse_args()
    if not 20 <= args.seconds <= 300 or args.max_sessions < 0 or not 1 <= args.max_failures <= 10:
        parser.error('seconds must be 20..300; max-sessions >=0; max-failures 1..10')
    return parser, args


def load_profile(parser, args):
    path = args.config.expanduser().resolve()
    try:
        settings = json.loads(path.read_text())
        board = settings.get('board', {})
        if not isinstance(board, dict):
            raise ValueError('board must be an object')
        key = args.ssh_key or board.get('key_filename')
        if not args.ask_password and not key and not board.get('password'):
            parser.error('Interactive authentication requires --ask-password (once), or use an existing SSH key/private password profile.')
        if key and not Path(key).expanduser().is_file():
            parser.error('Configured SSH key file does not exist')
        return path, settings
    except (OSError, ValueError, AttributeError, TypeError):
        parser.error('Cannot read private link profile')


def child_command(args, profile):
    command = [sys.executable, '-B', str(ROOT/'run-portable.py'), '--config', str(profile),
               '--package', str(args.package.expanduser().resolve()), '--seconds', str(args.seconds), '--headless']
    for key in ('ssh_key', 'board_host', 'receiver_host', 'ssh_user'):
        value = getattr(args, key)
        if value is not None:
            command += ['--'+key.replace('_', '-'), str(value)]
    return command


def main():
    parser, args = arguments()
    profile, settings = load_profile(parser, args)
    root = args.session_root.expanduser().resolve()
    try:
        with ExitStack() as stack:
            # This lock is independent of --session-root, so two roots cannot bypass it.
            owner = stack.enter_context(SingleOwner(ROOT/'local/summit-supervisor.lock'))
            if args.ask_password:
                password = getpass.getpass('Board SSH password (this supervised run only): ')
                if not password:
                    parser.error('Empty password; refusing to start')
                settings.setdefault('board', {})['password'] = password
                profile = stack.enter_context(private_profile(settings))
            supervisor = SessionSupervisor(child_command(args, profile), root, args.seconds, args.max_sessions, args.max_failures)
            raise SystemExit(supervisor.run(runner=partial(run_child, lock_fd=owner.file.fileno())))
    except RuntimeError as error:
        parser.exit(1, str(error)+'\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Stopped before session launch.\n')


if __name__ == '__main__':
    main()
