"""Cross-platform launcher for a NETWORK-PORTABLE frozen package (not Windows r2)."""
import argparse,datetime,json,platform,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--config',type=Path,required=True,help='local-only board/camera profile')
    p.add_argument('--board-host');p.add_argument('--receiver-host');p.add_argument('--ssh-user');p.add_argument('--ssh-key',type=Path);p.add_argument('--ask-password',action='store_true')
    p.add_argument('--seconds',type=int,default=300);p.add_argument('--headless',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not 20<=a.seconds<=300:p.error('bounded20..300s until continuous supervision is qualified')
    if platform.system()=='Darwin' and platform.machine().lower()!='arm64':p.error('Use native arm64 Python on the M3 Mac')
    package=a.package.expanduser().resolve();profile=a.config.expanduser().resolve()
    subprocess.run([sys.executable,'-B',str(package/'verify.py')],check=True)
    manifest=json.loads((package/'MANIFEST.json').read_text())
    if not manifest.get('network_portable'):p.error('This package has Windows-only/fixed-network runtime; use the portable candidate')
    if not all(manifest.get(k) for k in ('coordinates_only','track_pose','feature_roi')) or manifest.get('cpp_input_kernel')!='parallel8':p.error('package does not declare the approved runtime profile')
    if not profile.exists():p.error('Create a local link profile from config/portable-link.example.json; do not put passwords in Git')
    root=Path.home()/'Library/Application Support/PortableVision/runs' if platform.system()=='Darwin' else Path.home()/'.local/share/portable-vision/runs'
    output=a.output or root/(manifest['package_id']+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    command=[sys.executable,'-B',str(package/'tools/analysis/run_coordinate_browser.py'),'--credentials',str(profile),'--fixture',str(package/'release'),'--cpp-library',str(package/'native/libportable_input.so'),'--cpp-kernel','parallel8','--track-pose','--feature-roi','--output',str(output),'--seconds',str(a.seconds)]
    for key in ('board_host','receiver_host','ssh_user','ssh_key'):
        value=getattr(a,key)
        if value is not None:command+=['--'+key.replace('_','-'),str(value)]
    if a.ask_password:command+=['--ask-password']
    if a.headless:command+=['--headless']
    print('Running release:',manifest['package_id'],'on',platform.system(),platform.machine(),flush=True)
    print('Evidence:',output,flush=True)
    child=subprocess.Popen(command)
    try:code=child.wait()
    except KeyboardInterrupt:
        print('Waiting for host and owned board-worker cleanup...',flush=True)
        try:code=child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            print('Host cleanup did not finish; board self-time-bound remains. Check recorded session before restarting.',file=sys.stderr)
            raise SystemExit(130)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
