"""Frozen local fixture checks happen before any board connection."""
import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/analysis'))
from run_latest_frame_view import fixture_files,publish_latest
import run_latest_frame_view as viewer


def fixture(root):
    assets={}
    for name in ('provenance/mpk.json','share/processmla/yolov8_stage1_mla.elf'):
        p=root/'assets'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'frozen');assets[name]=hashlib.sha256(b'frozen').hexdigest()
    (root/'release.json').write_text(json.dumps({'assets':assets}))


def test_publish_retries_windows_reader_lock(tmp_path,monkeypatch):
    path=tmp_path/'latest.json';path.write_text('{"old":true}')
    real=viewer.os.replace;calls=[]
    def transient(a,b):
        calls.append(1)
        if len(calls)<3:raise PermissionError('reader holds file')
        return real(a,b)
    monkeypatch.setattr(viewer.os,'replace',transient)
    assert publish_latest(path,{'new':True},retries=3)
    assert json.loads(path.read_text())=={'new':True} and len(calls)==3


def test_publish_preserves_old_snapshot_when_locked(tmp_path,monkeypatch):
    path=tmp_path/'latest.json';path.write_text('{"old":true}')
    def locked(*args):raise PermissionError('locked')
    monkeypatch.setattr(viewer.os,'replace',locked)
    assert not publish_latest(path,{'new':True},retries=1)
    assert json.loads(path.read_text())=={'old':True}


def test_frozen_fixture_complete(tmp_path):
    fixture(tmp_path)
    assert len(fixture_files(tmp_path))==3


def test_frozen_fixture_tamper_refused(tmp_path):
    fixture(tmp_path);(tmp_path/'assets/provenance/mpk.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):fixture_files(tmp_path)


def test_frozen_fixture_missing_model_refused(tmp_path):
    fixture(tmp_path);(tmp_path/'release.json').write_text('{"assets":{}}')
    with pytest.raises(ValueError,match='missing'):fixture_files(tmp_path)


def test_frozen_fixture_path_escape_refused(tmp_path):
    fixture(tmp_path);data=json.loads((tmp_path/'release.json').read_text());data['assets']['../../outside']='x';(tmp_path/'release.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='path'):fixture_files(tmp_path)
