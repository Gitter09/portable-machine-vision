"""Bounded receiver-first smooth video + separate latest-result browser panel."""
from __future__ import annotations
import argparse,functools,hashlib,http.server,json,os,shlex,shutil,socket,subprocess,sys,threading,time,webbrowser
from pathlib import Path
import paramiko
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'tools'))
from portable_live_view import sdp


def publish_latest(path,result,retries=100):
    """Windows HTTP readers may briefly deny replacing the displayed JSON."""
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2))
    for attempt in range(retries):
        try:os.replace(tmp,path);return True
        except PermissionError:
            if attempt+1<retries:time.sleep(.01)
    return False  # Keep the previous snapshot; retry on next poll, not crash video.


def fixture_files(root):
    root=root.resolve();assets=root/'assets'
    inventory=json.loads((root/'release.json').read_text())
    result=[root/'release.json']
    for name,sha in inventory['assets'].items():
        path=(assets/name).resolve()
        if not path.is_relative_to(assets) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('fixture asset path/hash differs')
        result.append(path)
    required={'provenance/mpk.json','share/processmla/yolov8_stage1_mla.elf'}
    if not required.issubset(inventory['assets']):raise ValueError('fixture missing model assets')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--receiver-host',default='172.21.30.230');p.add_argument('--headless',action='store_true');p.add_argument('--optimized',action='store_true');p.add_argument('--fast-transforms',action='store_true')
    p.add_argument('--credentials',type=Path,default=REPO/'config/credentials.json')
    p.add_argument('--fixture',type=Path,help='local frozen replay fixture; no dependency on old board copies')
    args=p.parse_args()
    local_assets=fixture_files(args.fixture) if args.fixture else None
    if not 20<=args.seconds<=300:raise ValueError('bound20..300 seconds')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:probe.bind((args.receiver_host,0))
    args.output.mkdir(parents=True,exist_ok=False)
    b=json.loads(args.credentials.read_text())['board'];client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(b['host'],username=b['user'],password=b['password'],timeout=15)
    def execute(command):
        _,o,e=client.exec_command(command,timeout=30);out=o.read().decode();err=e.read().decode()
        if o.channel.recv_exit_status():raise RuntimeError(out+err)
        return out
    active=execute("ps -eo pid,args | grep -E 'launch.py|worker.py|gst-launch|feed_supervisor|crop_stream.py|latest_frame_stream.py|portable_tensor_parity.py board' | grep -v grep || true")
    if active.strip():raise RuntimeError('board occupied; no processes displaced: '+active)
    root='/media/nvme/'+args.output.name
    execute(f'test ! -e {root} && mkdir {root}')
    sftp=client.open_sftp()
    if local_assets is None:
        execute(f'cp -a /media/nvme/compiled-crop-replay-20261116/replay-fixture {root}/release')
    else:
        for file in local_assets:
            destination=root+'/release/'+file.relative_to(args.fixture.resolve()).as_posix()
            execute('mkdir -p '+shlex.quote(destination.rsplit('/',1)[0]))
            sftp.put(str(file),destination)
    (args.output/'source-snapshot').mkdir()
    sources=[REPO/'runtime/launcher'/n for n in ('latest_frame_stream.py','latest_frame_graph.py','crop_geometry.py')]+[REPO/'tools/analysis'/n for n in ('portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py')]
    fast_helper=REPO/'runtime/launcher/fast_crop_inference.py'
    if fast_helper.exists():sources.append(fast_helper)
    for helper in ('pcb_pose_tracker.py','coordinate_publisher.py'):
        path=REPO/'runtime/launcher'/helper
        if path.exists():sources.append(path)
    for f in sources:sftp.put(str(f),root+'/'+f.name);shutil.copyfile(f,args.output/'source-snapshot'/f.name)
    shutil.copyfile(Path(__file__).with_name('latest_frame_review.html'),args.output/'index.html')
    (args.output/'receiver.sdp').write_text(sdp(args.receiver_host,5010))
    common=['-hide_banner','-loglevel','warning','-protocol_whitelist','file,udp,rtp','-analyzeduration','1000000','-probesize','100000','-i',str((args.output/'receiver.sdp').resolve())]
    if args.headless:
        command=[shutil.which('ffmpeg'),*common,'-an','-frames:v','600','-c:v','copy','-progress',str((args.output/'progress.txt').resolve()),'-n',str((args.output/'received.mkv').resolve())]
    else:command=[shutil.which('ffplay'),'-window_title','LIVE CAMERA — asynchronous inspection is in browser',*common]
    server=None;receiver=None
    try:
        if not args.headless:
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self,*args):pass
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(args.output.resolve())))
            threading.Thread(target=server.serve_forever,daemon=True).start()
        with (args.output/'receiver.log').open('w') as log:
            receiver=subprocess.Popen(command,stdout=log,stderr=log)
            cmd=shlex.join(['timeout','-k','5s',str(args.seconds+30)+'s','python3','-u',root+'/latest_frame_stream.py','--release',root+'/release','--output',root+'/run','--rtsp','rtsp://192.168.1.10:5005/routecam','--seconds',str(args.seconds),'--host',args.receiver_host]+(['--optimized'] if args.optimized else [])+(['--fast-transforms'] if args.fast_transforms else []))
            _,stdout,stderr=client.exec_command(cmd+' > '+root+'/worker.log 2>&1',timeout=args.seconds+45)
            session={'host_pid':os.getpid(),'receiver_pid':receiver.pid,'remote_directory':root,'remote_command':cmd,'bound_seconds':args.seconds,
                     'started_unix':time.time(),'panel_url':f'http://127.0.0.1:{server.server_port}/' if server else None,'state':'running'}
            (args.output/'session.json').write_text(json.dumps(session,indent=2))
            if server:webbrowser.open(session['panel_url'])
            last=-1;deadline=time.monotonic()+args.seconds+35
            while not stdout.channel.exit_status_ready():
                if time.monotonic()>deadline:raise TimeoutError('remote exceeded bounded controller wait')
                try:
                    with sftp.open(root+'/run/latest.json') as f:result=json.loads(f.read())
                    if result['sequence']!=last:
                        name=result['image']
                        if Path(name).name!=name or not name.startswith('inspection-'):raise ValueError('unsafe result image path')
                        if not (args.output/name).exists():sftp.get(root+'/run/'+name,str(args.output/name))
                        result['received_unix']=time.time()
                        if publish_latest(args.output/'latest.json',result):last=result['sequence']
                except FileNotFoundError:pass
                time.sleep(.25)
            code=stdout.channel.recv_exit_status()
            for name in ('worker.log','run/worker.json','run/frames.jsonl','run/transform-preflight.json'):
                try:sftp.get(root+'/'+name,str(args.output/Path(name).name))
                except FileNotFoundError:pass
            if args.headless:
                try:receiver.wait(timeout=5)
                except subprocess.TimeoutExpired:receiver.terminate();receiver.wait(timeout=5)
            elif receiver.poll() is None:receiver.terminate();receiver.wait(timeout=5)
            session.update(state='completed' if code==0 else 'failed',remote_exit=code,receiver_exit=receiver.returncode,finished_unix=time.time())
            (args.output/'session.json').write_text(json.dumps(session,indent=2))
            if code:raise RuntimeError('board failed; see worker.log')
            if args.headless and receiver.returncode:raise RuntimeError('headless receiver failed; see receiver.log')
            print((args.output/'worker.json').read_text())
    finally:
        if receiver is not None and receiver.poll() is None:receiver.terminate();receiver.wait(timeout=5)
        if server:server.shutdown();server.server_close()
        client.close()


if __name__=='__main__':main()
