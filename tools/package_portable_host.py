"""Freeze network-portable candidate beside accepted Windows production release."""
import argparse,json,shutil,subprocess,sys,zipfile
from pathlib import Path
from package_latest_frame_fallback import VERIFY,digest
REPO=Path(__file__).resolve().parents[1]


def build(a):
    out=a.output.resolve();archive=out.with_suffix('.zip')
    if out.exists() or archive.exists():raise FileExistsError('new immutable output required')
    base=a.base.resolve();manifest=json.loads((base/'MANIFEST.json').read_text())
    subprocess.run([sys.executable,'-B',str(base/'verify.py')],check=True)
    out.mkdir(parents=True)
    def copy(source,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for folder in ('release','native','reference','provenance'):
        shutil.copytree(base/folder,out/folder)
    runtime=('latest_frame_stream.py','latest_frame_graph.py','crop_geometry.py','fast_crop_inference.py','pcb_pose_tracker.py','coordinate_publisher.py','fused_input.py')
    for name in runtime:copy(REPO/'runtime/launcher'/name,'runtime/launcher/'+name)
    copy(REPO/'runtime/launcher/native/fused_input.cpp','runtime/launcher/native/fused_input.cpp')
    analysis=('run_coordinate_browser.py','browser_coordinate_view.py','coordinate_browser.html','portable_network.py','portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py')
    for name in analysis:copy(REPO/'tools/analysis'/name,'tools/analysis/'+name)
    copy(REPO/'run-portable.py','run-portable.py');copy(REPO/'tools/setup_portable_host.py','setup-host.py')
    copy(REPO/'config/portable-link.example.json','config/portable-link.example.json');copy(base/'requirements.txt','requirements.txt')
    shutil.copytree(base/'wheels',out/'host-wheels/windows-amd64');shutil.copytree(a.mac_wheels,out/'host-wheels/macos-arm64')
    for name in ('portable-network-stop-ready-20261116','portable-network-restart-20261116'):
        for f in ('worker.json','receiver.json','session.json','stop-validation.json'):
            source=REPO.parent/'IC_TERMINAL_TRAIN'/name/f
            if source.exists():copy(source,'evidence/'+name+'/'+f)
    copy(REPO/'openwiki/operations/mac-private-link-20261116.md','README.md')
    (out/'verify.py').write_text(VERIFY,encoding='utf-8')
    files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    for key in ('proof_path','owner_path','owner_sources_match_proof'):manifest.pop(key,None)
    manifest.update(package_id=out.name,release_channel='portable-candidate',network_portable=True,mac_execution_verified=False,
        source_production_id=json.loads((base/'MANIFEST.json').read_text())['package_id'],credentials_included=False,files=files)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run([sys.executable,'-B',str(out/'verify.py')],check=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('ZIP CRC failed')
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'files':len(files)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','mac-wheels','output'):p.add_argument('--'+n,type=Path,required=True)
    build(p.parse_args())
