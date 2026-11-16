"""Host-only supervision policy: never run a camera or SSH during these tests."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tools.summit_supervisor import SessionSupervisor, SingleOwner, renewal_decision, private_profile


def reports(output, state='completed', worker='completed', remote_exit=0, **extra):
    output.mkdir(exist_ok=True)
    (output/'session.json').write_text(json.dumps({'state':state,'remote_exit':remote_exit,**extra}))
    (output/'worker.json').write_text(json.dumps({'state':worker,'stop_requested':worker=='stopped'}))


@pytest.mark.parametrize('state,worker,remote_exit,expected',[
    ('completed','completed',0,'renew'),
    ('stopped','stopped',0,'stop'),
    ('failed','error',1,'retry'),
    ('running','running',None,'unsafe'),
    ('failed','completed',None,'unsafe'),
])
def test_renewal_policy(tmp_path,state,worker,remote_exit,expected):
    reports(tmp_path,state,worker,remote_exit)
    assert renewal_decision(0 if state=='completed' else 1,tmp_path) == expected


def test_cleanup_errors_and_missing_evidence_fail_closed(tmp_path):
    assert renewal_decision(1,tmp_path) == 'unsafe'
    reports(tmp_path,stop_error='SSH unavailable')
    assert renewal_decision(1,tmp_path) == 'unsafe'


def test_successive_sessions_are_sequential_and_have_unique_output(tmp_path):
    calls=[]
    def runner(command,seconds):
        output=Path(command[-1]); calls.append(output)
        assert seconds==20
        reports(output)
        return 0
    supervisor=SessionSupervisor(['fake-launcher'],tmp_path,seconds=20,max_sessions=3)
    assert supervisor.run(runner=runner,sleep=lambda _:None)==0
    assert len(set(calls))==3
    assert json.loads((tmp_path/'summit-status.json').read_text())['state']=='completed'


def test_failures_are_bounded_and_uncertain_cleanup_is_not_restarted(tmp_path):
    calls=[]
    def runner(command,_seconds):
        output=Path(command[-1]);calls.append(output)
        reports(output,state='failed',worker='error',remote_exit=1)
        return 1
    supervisor=SessionSupervisor(['fake-launcher'],tmp_path,max_failures=3)
    assert supervisor.run(runner=runner,sleep=lambda _:None)==1
    assert len(calls)==3
    calls.clear()
    def uncertain(command,_seconds):
        calls.append(command)
        return 1
    assert supervisor.run(runner=uncertain,sleep=lambda _:None)==1
    assert len(calls)==1


def test_presenter_stop_and_host_timeout_do_not_renew(tmp_path):
    def stopped(command,_seconds):
        reports(Path(command[-1]),state='stopped',worker='stopped')
        return 0
    supervisor=SessionSupervisor(['fake-launcher'],tmp_path)
    assert supervisor.run(runner=stopped,sleep=lambda _:pytest.fail('must not retry'))==0
    def timeout(command,_seconds):
        reports(Path(command[-1]))
        return 124
    assert supervisor.run(runner=timeout,sleep=lambda _:pytest.fail('must not retry'))==1


def test_low_disk_space_prevents_launch_without_deleting_evidence(tmp_path):
    supervisor=SessionSupervisor(['fake-launcher'],tmp_path,min_free_bytes=10**30)
    assert supervisor.run(runner=lambda *_:pytest.fail('must not launch'))==1
    assert json.loads((tmp_path/'summit-status.json').read_text())['state']=='low-disk-space-manual-check-required'


def test_single_owner_lock_is_released_after_close(tmp_path):
    lock=tmp_path/'summit.lock'
    with SingleOwner(lock):
        with pytest.raises(RuntimeError):
            with SingleOwner(lock):
                pass
    with SingleOwner(lock):
        pass


@pytest.mark.skipif(sys.platform=='win32',reason='POSIX descriptor inheritance')
def test_launcher_can_retain_lock_after_parent_releases_it(tmp_path):
    lock=tmp_path/'summit.lock'
    with SingleOwner(lock) as owner:
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.3)'],pass_fds=(owner.file.fileno(),))
    try:
        with pytest.raises(RuntimeError):
            with SingleOwner(lock):
                pass
    finally:
        child.wait(timeout=3)
    with SingleOwner(lock):
        pass


def test_prompt_profile_is_private_and_removed_on_exit():
    import os
    with private_profile({'board':{'password':'test-only-placeholder'}}) as path:
        assert json.loads(path.read_text())['board']['password']=='test-only-placeholder'
        if os.name != 'nt':
            assert path.stat().st_mode & 0o777 == 0o600
    assert not path.exists()


def test_cli_requires_noninteractive_auth_before_starting_any_child(tmp_path):
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'board':{}}))
    result=subprocess.run([sys.executable,'run-summit.py','--config',str(profile)],capture_output=True,text=True)
    assert result.returncode==2
    assert 'Interactive authentication' in result.stderr
