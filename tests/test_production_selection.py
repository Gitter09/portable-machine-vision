import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import promote_production_pipeline as production
from package_coordinate_fallback import launchers


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(production,'REPO',tmp_path)
    package=tmp_path/'release';package.mkdir();session=tmp_path/'session';session.mkdir()
    modes=dict(feature_roi=True,coordinates_only=True,track_pose=True,optimized=True,fast_transforms=True)
    manifest=dict(package_id='r1',release_channel='production',cpp_input_kernel='parallel8',cpp_library_sha256='native',board_model_sha256='elf',mpk_sha256='mpk',files={},**modes)
    worker=dict(state='completed',errors=[],cpp_input_kernel='parallel8',cpp_library_sha256='native',elf_sha256='elf',mpk_sha256='mpk',located=1,published_results=1,inference_frames=1,**modes)
    (package/'MANIFEST.json').write_text(json.dumps(manifest));package.with_suffix('.zip').write_bytes(b'archive')
    (session/'worker.json').write_text(json.dumps(worker));(session/'receiver.json').write_text(json.dumps(dict(host_errors=[],matched_at_arrival=1,metadata_results=1)))
    (session/'session.json').write_text(json.dumps(dict(state='completed',remote_exit=0)))
    config=dict(schema='portable-production-selection/v1',state='active',package='release',release_id='r1',manifest_sha256=production.digest(package/'MANIFEST.json'),archive_sha256=production.digest(package.with_suffix('.zip')),cpp_library_sha256='native')
    return config,session


def test_production_smoke_accepted(tmp_path,monkeypatch):
    c,s=fixture(tmp_path,monkeypatch);assert production.validate(c,s)[1]['state']=='completed'


@pytest.mark.parametrize('field,value',[('feature_roi',False),('cpp_library_sha256','wrong'),('state','error'),('elf_sha256','wrong'),('published_results',0)])
def test_production_wrong_runtime_refused(tmp_path,monkeypatch,field,value):
    c,s=fixture(tmp_path,monkeypatch);p=s/'worker.json';w=json.loads(p.read_text());w[field]=value;p.write_text(json.dumps(w))
    with pytest.raises(ValueError):production.validate(c,s)


def test_package_tamper_refused(tmp_path,monkeypatch):
    c,s=fixture(tmp_path,monkeypatch);(tmp_path/'release/MANIFEST.json').write_text('{}')
    with pytest.raises(ValueError,match='fingerprint'):production.validate(c,s)


def test_unmatched_smoke_refused(tmp_path,monkeypatch):
    c,s=fixture(tmp_path,monkeypatch);p=s/'receiver.json';r=json.loads(p.read_text());r['matched_at_arrival']=0;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='matched'):production.validate(c,s)


def test_production_launch_flags_and_isolated_environment():
    run,setup=launchers('release-r1',True,'parallel8',True)
    for flag in ('--track-pose','--feature-roi','--cpp-library','parallel8'):assert flag in run
    assert 'production-envs' in run and 'production-envs' in setup and 'production-runs' in run
    assert '[int]$Seconds = 300' in run and ' -B ' in run
    assert '--no-index' in setup


def test_legacy_fallback_defaults_unchanged():
    run,setup=launchers('fallback-r1')
    assert '--feature-roi' not in run and '--cpp-library' not in run
    assert 'fallback-envs' in setup and '[int]$Seconds = 180' in run


def test_unsafe_release_id_refused():
    with pytest.raises(ValueError):launchers('release; bad')
