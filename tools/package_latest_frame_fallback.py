"""Freeze the proven latest-frame pipeline without credentials or board changes."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


VERIFY='''"""Offline integrity verification. Does not connect to the board."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
if actual != set(m['files'])|{'MANIFEST.json'}:raise SystemExit('FAIL: missing/extra package files')
for name,sha in m['files'].items():
 p=root/name
 if p.is_symlink() or not p.resolve().is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise SystemExit('FAIL: '+name)
print('PASS: '+str(len(m['files']))+' files verified; package '+m['package_id'])
'''

RUN=r'''param(
 [string]$Python = "$PSScriptRoot\..\..\..\MODEL\.venv\Scripts\python.exe",
 [string]$Credentials = "$PSScriptRoot\..\..\..\portable-vision\config\credentials.json",
 [string]$ReceiverHost = "172.21.30.230",
 [ValidateRange(20,300)][int]$Seconds = 180,
 [switch]$Headless
)
$ErrorActionPreference = "Stop"
& $Python -B "$PSScriptRoot\verify.py"
if ($LASTEXITCODE -ne 0) { throw "Package integrity failed" }
if (!(Test-Path $Credentials)) { throw "Provide external -Credentials JSON; credentials are intentionally not packaged" }
$output = [IO.Path]::GetFullPath("$PSScriptRoot\..\..\fallback-runs\latest-frame-" + [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss-fff"))
$runArgs = @("$PSScriptRoot\tools\analysis\run_latest_frame_view.py", "--fixture", "$PSScriptRoot\release", "--credentials", $Credentials, "--output", $output, "--receiver-host", $ReceiverHost, "--seconds", "$Seconds")
if ($Headless) { $runArgs += "--headless" }
& $Python -B @runArgs
if ($LASTEXITCODE -ne 0) { throw "Trial failed; inspect $output" }
'''


def build(args):
    out=args.output.resolve();archive=out.with_suffix('.zip')
    if out.exists() or archive.exists() or archive.with_suffix('.zip.sha256').exists():raise FileExistsError('immutable output/archive already exists')
    proof=json.loads((args.verified_view/'worker.json').read_text())
    if proof['state']!='completed' or proof['errors'] or proof['video_fps']<25:raise ValueError('source view not a successful smooth-video run')
    fixture=args.fixture.resolve();inv=json.loads((fixture/'release.json').read_text())
    for name,sha in inv['assets'].items():
        p=(fixture/'assets'/name).resolve()
        if not p.is_relative_to(fixture/'assets') or digest(p)!=sha:raise ValueError('fixture hash/path differs')
    if digest(fixture/'assets/share/processmla/yolov8_stage1_mla.elf')!=proof['elf_sha256'] or digest(fixture/'assets/provenance/mpk.json')!=proof['mpk_sha256']:raise ValueError('model differs from verified run')
    optimized=bool(proof.get('optimized',False));fast=bool(proof.get('fast_transforms',False))
    out.mkdir(parents=True)
    def copy(src,dest):
        dst=out/dest;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for name in ('latest_frame_stream.py','latest_frame_graph.py','crop_geometry.py'):copy(args.verified_view/'source-snapshot'/name,'runtime/launcher/'+name)
    if optimized or fast:copy(args.verified_view/'source-snapshot/fast_crop_inference.py','runtime/launcher/fast_crop_inference.py')
    for helper in ('pcb_pose_tracker.py','coordinate_publisher.py'):
        source=args.verified_view/'source-snapshot'/helper
        if source.is_file():copy(source,'runtime/launcher/'+helper)
    for name in ('portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py'):copy(args.verified_view/'source-snapshot'/name,'tools/analysis/'+name)
    copy(REPO/'tools/analysis/run_latest_frame_view.py','tools/analysis/run_latest_frame_view.py')
    copy(REPO/'tools/portable_live_view.py','tools/portable_live_view.py')
    copy(args.verified_view/'index.html','tools/analysis/latest_frame_review.html')
    shutil.copytree(fixture,out/'release')
    for name in ('worker.json','session.json'):copy(args.verified_view/name,'evidence/owner-view/'+name)
    host_log=args.verified_view.parent/(args.verified_view.name+'.log')
    if host_log.is_file():copy(host_log,'evidence/owner-view/host-controller.log')
    for name in ('compilation.json','export-lineage.json'):copy(args.export/name,'provenance/'+name)
    for name in ('yolo11n_fused.onnx','yolo11n_six.onnx'):copy(args.export/name,'reference/'+name)
    behaviour='latest-frame-performance-20261116.md' if optimized else 'latest-frame-video-20261116.md'
    copy(REPO/'openwiki/operations'/behaviour,'provenance/validated-behaviour.md')
    flags=(' --optimized' if optimized else '')+(' --fast-transforms' if fast else '')
    launcher=RUN.replace('if ($Headless)',('$runArgs += @('+','.join('"'+f+'"' for f in flags.split())+')\nif ($Headless)') if flags else 'if ($Headless)')
    (out/'verify.py').write_text(VERIFY,encoding='utf-8');(out/'run.ps1').write_text(launcher,encoding='utf-8')
    (out/'README.md').write_text('''# Frozen latest-frame fallback

Smooth unannotated camera video (~30FPS) plus independent latest-frame compiled
inspection (see MANIFEST.json for measured rate/mode), shown only on its own timestamped browser snapshot.
No stale boxes on live video; UNKNOWN/STALE/disconnected states remain explicit.
Host launcher retries transient Windows browser-read file locks; repeated locks
keep the old snapshot visible/stale instead of terminating the viewer. This is a
host reliability fix; frozen board-side inference behaviour is unchanged.
No favourable-region gate, tracking or threshold tuning. The launcher explicitly restores
the optimized/fast-transform flags recorded in MANIFEST.json; no silent slow-mode fallback.

## Verify / run (PowerShell)

From this extracted package:

```powershell
& ..\\..\\..\\MODEL\\.venv\\Scripts\\python.exe .\\verify.py
.\\run.ps1
# Optional bounded headless capture:
.\\run.ps1 -Headless -Seconds 45
```

The defaults locate this workstation's existing MODEL environment and EXTERNAL
portable-vision/config/credentials.json. Override -Python and -Credentials after
moving the package. Credentials/passwords are intentionally NOT included.
Requires Windows Python3.11 with NumPy/OpenCV/Paramiko, FFmpeg/FFplay on PATH,
and the existing portable Modalix SDK/GStreamer environment (not bundled).
Approved receiver172.21.30.230:5010 must be locally assigned. Maximum300s per run.

Every run verifies package hashes, uploads these frozen assets/scripts into a NEW
board directory, refuses an occupied board, and creates output OUTSIDE this
package under IC_TERMINAL_TRAIN/fallback-runs/. No dependency on the old replay
fixture on the board, mutable model/export folders or the experiment source tree.
No edits/restarts of r33, services, vendor SDK or fixed-station board.

Closing FFplay alone does not immediately end inference; the time bound applies.
The browser serves only loopback and closes with the controller. Model outcomes
remain diagnostic, NOT independent production accuracy or full latency acceptance.

Restore by extracting the ZIP to a fresh folder, verifying it, then running the
launcher. Do not edit this package for experiments; copy it to a new candidate.
MANIFEST.json fingerprints every packaged file. ZIP .sha256 fingerprints the
archive; neither is a cryptographic publisher signature. Python dependencies and
installed board plugins are environmental prerequisites, not frozen binaries here.
''',encoding='utf-8')
    files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    if any('credentials' in name.lower() for name in files):raise ValueError('credential file unexpectedly included')
    manifest={'package_id':out.name,'scope':'frozen fallback, not production-qualified release',
              'verified_view':str(args.verified_view.resolve()),'elf_sha256':proof['elf_sha256'],'mpk_sha256':proof['mpk_sha256'],
              'verified_video_fps':proof['video_fps'],'verified_inference_fps':proof['inference_fps'],
              'optimized':optimized,'fast_transforms':fast,'snapshot_publish_retry':True,'credentials_included':False,'files':files}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(out/'verify.py')],check=True)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps({'package':str(out),'zip':str(archive),'zip_sha256':digest(archive),'files':len(files),'bytes':archive.stat().st_size},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('verified-view','fixture','export','output'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())
