"""Freeze coordinate-only browser + pose reuse, including offline host wheels."""
from __future__ import annotations
import argparse,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
from package_latest_frame_fallback import VERIFY,digest
REPO=Path(__file__).resolve().parents[1]

RUN=r'''param(
 [string]$Python = "$PSScriptRoot\..\..\fallback-envs\coordinate-browser-pose-20261116-v1\Scripts\python.exe",
 [string]$Credentials = "$PSScriptRoot\..\..\..\portable-vision\config\credentials.json",
 [ValidateRange(20,300)][int]$Seconds = 180,
 [switch]$Headless
)
$ErrorActionPreference = "Stop"
if (!(Test-Path $Python)) { throw "Run setup.ps1 first, or supply -Python for your browser-view environment" }
& $Python -B "$PSScriptRoot\verify.py"
if ($LASTEXITCODE -ne 0) { throw "Package integrity failed" }
if (!(Test-Path $Credentials)) { throw "External credentials required; never included in package" }
$output=[IO.Path]::GetFullPath("$PSScriptRoot\..\..\fallback-runs\coordinate-pose-"+[DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss-fff"))
$runArgs=@("$PSScriptRoot\tools\analysis\run_coordinate_browser.py","--fixture","$PSScriptRoot\release","--credentials",$Credentials,"--output",$output,"--seconds","$Seconds","--track-pose")
if ($Headless) { $runArgs += "--headless" }
& $Python -B @runArgs
if ($LASTEXITCODE -ne 0) { throw "Trial failed; inspect $output" }
'''
SETUP=r'''param(
 [string]$Python="python",
 [string]$Environment="$PSScriptRoot\..\..\fallback-envs\coordinate-browser-pose-20261116-v1"
)
$ErrorActionPreference="Stop"
& $Python -B "$PSScriptRoot\verify.py"
if ($LASTEXITCODE -ne 0) { throw "Package integrity failed" }
& $Python -c "import sys,struct; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Requires CPython3.11 x64 on Windows'"
if ($LASTEXITCODE -ne 0) { throw "Wrong Python version/architecture" }
if (Test-Path $Environment) { throw "Environment already exists; use it via run.ps1 or choose a NEW -Environment" }
& $Python -m venv $Environment
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
& "$Environment\Scripts\python.exe" -m pip install --no-index --find-links "$PSScriptRoot\wheels" -r "$PSScriptRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "Offline dependency install failed" }
Write-Output "Ready: $Environment"
'''


def launchers(package_id,feature_roi=False,cpp_kernel=None,production=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',package_id):raise ValueError('unsafe package id')
    if cpp_kernel not in (None,'parallel8'):raise ValueError('unsupported release kernel')
    run=RUN.replace('coordinate-browser-pose-20261116-v1',package_id);setup=SETUP.replace('coordinate-browser-pose-20261116-v1',package_id)
    extra=[]
    if feature_roi:extra.append('$runArgs += "--feature-roi"')
    if cpp_kernel:extra.append('$runArgs += @("--cpp-library","$PSScriptRoot\\native\\libportable_input.so","--cpp-kernel","'+cpp_kernel+'")')
    if extra:run=run.replace('if ($Headless)','\n'.join(extra)+'\nif ($Headless)')
    if production:
        run=run.replace('fallback-envs','production-envs').replace('fallback-runs','production-runs').replace('[int]$Seconds = 180','[int]$Seconds = 300')
        setup=setup.replace('fallback-envs','production-envs')
    return run,setup


def build(args):
    out=args.output.resolve();archive=out.with_suffix('.zip')
    if out.exists() or archive.exists() or archive.with_suffix('.zip.sha256').exists():raise FileExistsError('immutable destination exists')
    proof=json.loads((args.proof/'worker.json').read_text());receiver=json.loads((args.proof/'receiver.json').read_text())
    feature_roi=bool(proof.get('feature_roi'));cpp_kernel=proof.get('cpp_input_kernel')
    if cpp_kernel:
        if cpp_kernel!='parallel8':raise ValueError('only verified parallel8 native release supported')
        if digest(args.owner/'source-snapshot/libportable_input.so')!=proof['cpp_library_sha256']:raise ValueError('native library differs from proof')
        preflight=json.loads((args.proof/'transform-preflight.json').read_text())['cpp_input']
        if not preflight['packed_bytes_exact'] or preflight['library_sha256']!=proof['cpp_library_sha256'] or preflight['kernel']!=cpp_kernel:raise ValueError('native preflight differs')
    if proof['state']!='completed' or proof['errors'] or not proof['coordinates_only'] or not proof['track_pose'] or receiver['host_errors']:
        raise ValueError('successful coordinate+pose proof required')
    if receiver['matched_at_arrival']!=receiver['metadata_results'] or receiver['metadata_results']!=proof['inference_frames']:raise ValueError('all-frame-matched proof required')
    for p in (args.proof/'source-snapshot').iterdir():
        if p.name=='run_coordinate_browser.py':continue  # Host launch-only argument validation/packaging adaptation.
        if p.is_file() and digest(p)!=digest(args.owner/'source-snapshot'/p.name):raise ValueError('owner source differs: '+p.name)
    inv=json.loads((args.fixture/'release.json').read_text());assets=(args.fixture/'assets').resolve()
    for name,sha in inv['assets'].items():
        p=(assets/name).resolve()
        if not p.is_relative_to(assets) or digest(p)!=sha:raise ValueError('fixture differs')
    if digest(assets/'share/processmla/yolov8_stage1_mla.elf')!=proof['elf_sha256'] or digest(assets/'provenance/mpk.json')!=proof['mpk_sha256']:raise ValueError('wrong model')
    out.mkdir(parents=True)
    def copy(src,name):
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    runtime=('latest_frame_stream.py','latest_frame_graph.py','crop_geometry.py','fast_crop_inference.py','pcb_pose_tracker.py','coordinate_publisher.py')
    analysis=('portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py','browser_coordinate_view.py','coordinate_browser.html')
    for name in runtime:copy(args.owner/'source-snapshot'/name,'runtime/launcher/'+name)
    for name in analysis:copy(args.owner/'source-snapshot'/name,'tools/analysis/'+name)
    if cpp_kernel:
        copy(args.owner/'source-snapshot/fused_input.py','runtime/launcher/fused_input.py')
        copy(args.owner/'source-snapshot/fused_input.cpp','runtime/launcher/native/fused_input.cpp')
        copy(args.owner/'source-snapshot/libportable_input.so','native/libportable_input.so')
    # Only entrypoint adaptation: explicit local fixture/credentials instead of old workspace paths.
    copy(REPO/'tools/analysis/run_coordinate_browser.py','tools/analysis/run_coordinate_browser.py')
    shutil.copytree(args.fixture,out/'release');shutil.copytree(args.wheels,out/'wheels');copy(args.requirements,'requirements.txt')
    for name in ('worker.json','receiver.json','transform-preflight.json'):copy(args.proof/name,'evidence/'+name)
    for name in ('geometry-audit-final.json','feature-geometry-audit.json','stage-summary.json','browser.png'):
        if (args.proof/name).is_file():copy(args.proof/name,'evidence/'+name)
    copy(args.owner/'session.json','evidence/owner-session-at-packaging.json')
    for name in ('compilation.json','export-lineage.json'):copy(args.export/name,'provenance/'+name)
    for name in ('yolo11n_fused.onnx','yolo11n_six.onnx'):copy(args.export/name,'reference/'+name)
    behaviour='tracker-feature-roi-20261116.md' if feature_roi else 'coordinate-browser-pose-20261116.md'
    copy(REPO/'openwiki/operations'/behaviour,'provenance/validated-behaviour.md')
    run,setup=launchers(out.name,feature_roi,cpp_kernel,args.production)
    (out/'run.ps1').write_text(run,encoding='utf-8');(out/'setup.ps1').write_text(setup,encoding='utf-8');(out/'verify.py').write_text(VERIFY,encoding='utf-8')
    (out/'README.md').write_text('''# Latest fallback: browser coordinates + PCB pose tracking

Owner-approved diagnostic: smooth raw camera video, coordinate-only board results,
browser canvas on EXACT decoded-NV12-hash-matched inspection frames. Separate live
pane; no stale live overlays. ~12.6 inspections/s, ~30 decoded video frames/s in
completed60s proof. Geometry tracking refreshes fully every0.5s or on confidence
failure; UNKNOWN/stale/unmatched clears overlays. No region gate or C++ fusion.

## Restore / run (PowerShell)

Extract to a NEW folder. Requires Windows x64 CPython3.11 and existing board SDK:

```powershell
.\\setup.ps1             # Creates NEW env OUTSIDE package, installs bundled wheels offline
.\\run.ps1               # Opens browser;180s bounded default
.\\run.ps1 -Seconds 300
# Existing compatible environment:
.\\run.ps1 -Python C:\\path\\to\\browser-view-env\\Scripts\\python.exe
```

Credentials are deliberately EXTERNAL. Default workspace path is
portable-vision/config/credentials.json; override -Credentials after moving.
Receiver172.21.30.230 must be local, UDP5010(video)/5012(metadata) free. Browser
HTTP is127.0.0.1 only. Requires no external FFmpeg executable: PyAV wheels include
host codec libraries. Installed SiMa SDK/GStreamer and Python itself are NOT bundled.

All runtime/model/helper/UI sources and dependencies are frozen here. Wheels pin
9 actual host dependency versions (Windows/CPython3.11). No dependency on old replay
board folders, mutable project sources or MODEL environment. run.ps1 verifies files,
uploads frozen assets/scripts to a NEW job and refuses occupied board hardware.
New outputs go OUTSIDE package under fallback-runs. Keep this package immutable.

Both flags (coordinate-only/optimized transforms and pose reuse) are explicitly
selected by packaged controller/launcher. No silent switch back to the slow path.
Model, calibration and thresholds unchanged. No confidence/region tuning.

Closing browser does not immediately stop board:20..300s self-bound applies.
No r33/service/vendor/fixed-station modifications. This is not production accuracy,
optical latency, tracker stress or soak acceptance. Read provenance for caveats.

Board/runtime and host rendering sources match the accepted owner-view snapshot.
Packaging adapts only host entrypoint fixture/credentials to explicit package paths.
Completed60s proof is separate from the live owner-session snapshot, which may
still say running at packaging. No new camera trial is required to verify archive.
SHA manifest and ZIP SHA are integrity checks, not a publisher signature.
''',encoding='utf-8')
    readme=out/'README.md';text=readme.read_text(encoding='utf-8')
    if feature_roi or cpp_kernel:
        text=text.replace('~12.6 inspections/s','~16 inspections/s').replace('No region gate or C++ fusion.','No inspection-region gate. Frozen feature-ROI flag and C++8 input library are enabled.').replace('No dependency on old replay','No dependency on old replay')
        text+='\nThis version bundles the SHA-verified AArch64 native library and source. Board OS libgomp.so.1 is required. Feature ROI only reduces feature-detection work; inspection remains eligible at all plate positions.\n'
    if args.production:
        text=text.replace('Owner-approved diagnostic:','Owner-approved operational baseline:').replace('180s bounded default','300s bounded default')
        text='# Owner-selected operational production pipeline\n\n'+text+'\nSelected for operations by owner approval, not a claim of independently measured production accuracy/optical latency. Default launcher remains bounded; service/autostart and frontend supervision are separate work.\n'
    readme.write_text(text,encoding='utf-8')
    files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest={'package_id':out.name,'latest_fallback':True,'coordinates_only':True,'track_pose':True,'optimized':True,'fast_transforms':True,
        'release_channel':'production' if args.production else 'fallback','feature_roi':feature_roi,'cpp_input_kernel':cpp_kernel,'cpp_library_sha256':proof.get('cpp_library_sha256'),
        'credentials_included':False,'board_model_sha256':proof['elf_sha256'],'mpk_sha256':proof['mpk_sha256'],
        'proof_path':str(args.proof.resolve()),'owner_path':str(args.owner.resolve()),'owner_sources_match_proof':True,
        'proof_inference_fps':proof['inference_fps'],'proof_decoded_arrival_fps':receiver['decoded_arrival_fps'],'files':files}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(out/'verify.py')],check=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('archive CRC failed')
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps({'package':str(out),'zip':str(archive),'zip_sha256':digest(archive),'bytes':archive.stat().st_size,'files':len(files)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('proof','owner','fixture','export','wheels','requirements','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--production',action='store_true');build(p.parse_args())
