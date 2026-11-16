"""Bounded coordinate-only inspection browser: RTP5010 + JSON5012, loopback UI.
No board JPEG/overlay. Exact decoded-NV12 hashes pair inspected frames with boxes.
"""
from __future__ import annotations
import argparse,getpass,http.server,json,re,shlex,shutil,threading,time,webbrowser
from pathlib import Path
import paramiko
from browser_coordinate_view import BrowserReceiver
from portable_network import resolve_link,configure_host_keys,connect_ssh,stop_owned_worker
REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--credentials',type=Path,default=REPO/'config/credentials.json');p.add_argument('--fixture',type=Path,default=REPO.parent/'IC_TERMINAL_TRAIN/fallbacks/latest-frame-optimized-20261116-v2/release')
    p.add_argument('--cpp-library',type=Path);p.add_argument('--cpp-kernel',choices=('scalar','neon','parallel2','parallel4','parallel8','parallel16'),default='parallel8')
    p.add_argument('--board-host');p.add_argument('--ssh-user');p.add_argument('--ssh-key',type=Path);p.add_argument('--ask-password',action='store_true');p.add_argument('--rtsp')
    p.add_argument('--feature-roi',action='store_true');p.add_argument('--audit-inputs',action='store_true');p.add_argument('--track-pose',action='store_true');p.add_argument('--headless',action='store_true');p.add_argument('--receiver-host',help='optional local IPv4 override; default auto-selects route to board');args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.output.name):p.error('output directory name must use letters/digits/dot/underscore/hyphen')
    if args.feature_roi and not args.track_pose:p.error('--feature-roi requires --track-pose')
    if not 20<=args.seconds<=300:raise ValueError('bounded20..300s')
    settings=json.loads(args.credentials.read_text()) if args.credentials.exists() else {}
    b=settings.get('board',{});rtsp=args.rtsp or settings.get('camera',{}).get('rtsp')
    if not rtsp or not rtsp.startswith('rtsp://'):p.error('configure camera.rtsp or supply --rtsp')
    link=resolve_link(args.board_host or b.get('host'),args.receiver_host);args.receiver_host=link['receiver_ip']
    print('Link:',json.dumps(link),flush=True)
    password=getpass.getpass('Board SSH password: ') if args.ask_password else b.get('password')
    key=args.ssh_key or b.get('key_filename')
    client=paramiko.SSHClient();configure_host_keys(client,b.get('host_key_sha256'))
    connect_ssh(client,link,args.ssh_user or b.get('user','sima'),password,str(Path(key).expanduser()) if key else None)
    args.output.mkdir(parents=True,exist_ok=False);hub=BrowserReceiver(args.receiver_host,audit_dir=args.output/'audit' if args.audit_inputs else None)
    def execute(command):
        _,o,e=client.exec_command(command,timeout=30);out=o.read().decode();err=e.read().decode()
        if o.channel.recv_exit_status():raise RuntimeError(out+err)
        return out
    busy=execute("ps -eo pid,args | grep -E 'latest_frame_stream.py|crop_stream.py|launch.py|worker.py|gst-launch|feed_supervisor' | grep -v grep || true")
    if busy.strip():raise RuntimeError('board occupied; refusing to displace: '+busy)
    root='/media/nvme/'+args.output.name;execute(f'test ! -e {root} && mkdir {root}')
    sf=client.open_sftp();snapshot=args.output/'source-snapshot';snapshot.mkdir()
    fixture=args.fixture.resolve()
    inv=json.loads((fixture/'release.json').read_text())
    import hashlib
    for name,sha in inv['assets'].items():
        f=(fixture/'assets'/name).resolve()
        if not f.is_relative_to((fixture/'assets').resolve()) or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise ValueError('fixture asset changed')
    for f in fixture.rglob('*'):
        if f.is_file():
            dst=root+'/release/'+f.relative_to(fixture).as_posix();execute('mkdir -p '+shlex.quote(dst.rsplit('/',1)[0]));sf.put(str(f),dst)
    sources=[REPO/'runtime/launcher'/n for n in ('latest_frame_stream.py','latest_frame_graph.py','crop_geometry.py','fast_crop_inference.py','pcb_pose_tracker.py','coordinate_publisher.py')]+[REPO/'tools/analysis'/n for n in ('portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py')]
    for f in sources:sf.put(str(f),root+'/'+f.name);shutil.copyfile(f,snapshot/f.name)
    for name in ('run_coordinate_browser.py','browser_coordinate_view.py','coordinate_browser.html','portable_network.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    cpp_args=[]
    if args.cpp_library:
        library=args.cpp_library.resolve();library_sha=hashlib.sha256(library.read_bytes()).hexdigest()
        sf.put(str(library),root+'/libportable_input.so');shutil.copyfile(library,snapshot/'libportable_input.so')
        sf.put(str(REPO/'runtime/launcher/fused_input.py'),root+'/fused_input.py');shutil.copyfile(REPO/'runtime/launcher/fused_input.py',snapshot/'fused_input.py')
        shutil.copyfile(REPO/'runtime/launcher/native/fused_input.cpp',snapshot/'fused_input.cpp')
        cpp_args=['--cpp-library',root+'/libportable_input.so','--cpp-sha256',library_sha,'--cpp-kernel',args.cpp_kernel]
    page=Path(__file__).with_name('coordinate_browser.html').read_bytes()
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            path=self.path.split('?')[0]
            try:
                if path=='/live.mjpg':
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.end_headers();last=None
                    while not hub.stop.is_set():
                        with hub.lock:key=hub.latest
                        if key and key!=last:
                            image=hub.image(key)
                            if image:self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n');last=key
                        time.sleep(.01)
                    return
                if path=='/':data=page;typ='text/html; charset=utf-8'
                elif path=='/inspection.json':data=json.dumps(hub.current()).encode();typ='application/json'
                elif path=='/stats.json':data=json.dumps(hub.summary()).encode();typ='application/json'
                elif path.startswith('/frames/') and path.endswith('.jpg'):
                    data=hub.image(path[len('/frames/'):-4]);typ='image/jpeg'
                    if data is None:self.send_error(404);return
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();hub.start()
    cmd=shlex.join(['timeout','-k','5s',str(args.seconds+30)+'s','python3','-u',root+'/latest_frame_stream.py','--release',root+'/release','--output',root+'/run','--rtsp',rtsp,'--seconds',str(args.seconds),'--host',args.receiver_host,'--optimized','--fast-transforms','--coordinates-only']+(['--track-pose'] if args.track_pose else [])+(['--feature-roi'] if args.feature_roi else [])+cpp_args)
    if cpp_args:cmd='env OMP_WAIT_POLICY=PASSIVE OMP_DYNAMIC=FALSE '+cmd
    remote_channel=None;remote_started=False;session={'remote_directory':root,'state':'starting'}
    try:
        remote_started=True
        _,o,e=client.exec_command(cmd+' > '+root+'/worker.log 2>&1',timeout=args.seconds+45)
        remote_channel=o.channel
        session={'link':link,'host_platform':__import__('platform').platform(),'host_python':__import__('sys').executable,'remote_directory':root,'command':cmd.replace(shlex.quote(rtsp),'<configured-camera>'),'panel_url':f'http://127.0.0.1:{server.server_port}/','state':'running','started_unix':time.time()}
        (args.output/'session.json').write_text(json.dumps(session,indent=2));print('Browser:',session['panel_url'],flush=True)
        if not args.headless:webbrowser.open(session['panel_url'])
        end=time.monotonic()+args.seconds+35
        while not o.channel.exit_status_ready():
            if hub.errors:raise RuntimeError('host receiver failed; stopping owned board worker')
            if time.monotonic()>end:raise TimeoutError('remote bound exceeded')
            time.sleep(.2)
        code=o.channel.recv_exit_status()
        for n in ('worker.log','run/worker.json','run/frames.jsonl','run/published.jsonl','run/transform-preflight.json'):
            destination=args.output/Path(n).name
            try:
                sf.stat(root+'/'+n);sf.get(root+'/'+n,str(destination))
            except FileNotFoundError:
                if destination.exists() and destination.stat().st_size==0:destination.unlink()
        stats=hub.summary();(args.output/'receiver.json').write_text(json.dumps(stats,indent=2));(args.output/'last-result.json').write_text(json.dumps(hub.current(),indent=2))
        worker_state=json.loads((args.output/'worker.json').read_text()).get('state') if (args.output/'worker.json').exists() else None
        session.update(state=('stopped' if worker_state=='stopped' else 'completed') if code==0 else 'failed',remote_exit=code);(args.output/'session.json').write_text(json.dumps(session,indent=2))
        if code or stats['host_errors']:raise RuntimeError('trial failed or stopped during startup; inspect worker/receiver logs')
        if session['state']!='stopped' and not stats['matched_at_arrival']:raise RuntimeError('no exact frame matches; do not overlay unmatched coordinates')
        print((args.output/'worker.json').read_text());print(json.dumps(stats,indent=2))
    except KeyboardInterrupt:
        session['state']='stop-requested';print('Stopping this owned board job...',flush=True)
    except Exception as error:
        session.update(state='failed',error=str(error));raise
    finally:
        if remote_started and (remote_channel is None or not remote_channel.exit_status_ready()):
            try:
                session['stop_result']=stop_owned_worker(client,root+'/latest_frame_stream.py',root+'/run')
                deadline=time.monotonic()+5
                while remote_channel is not None and not remote_channel.exit_status_ready() and time.monotonic()<deadline:time.sleep(.1)
                if remote_channel is not None and remote_channel.exit_status_ready():
                    session['remote_exit']=remote_channel.recv_exit_status();session['state']='failed' if session.get('error') else 'stopped'
                    for name in ('worker.json','frames.jsonl'):
                        try:sf.get(root+'/run/'+name,str(args.output/name))
                        except FileNotFoundError:pass
            except Exception as error:session['stop_error']=str(error)
        if session.get('state')=='stop-requested' and remote_channel is not None and remote_channel.exit_status_ready():session['state']='stopped'
        (args.output/'session.json').write_text(json.dumps(session,indent=2))
        hub.close();server.shutdown();server.server_close();client.close()


if __name__=='__main__':main()
