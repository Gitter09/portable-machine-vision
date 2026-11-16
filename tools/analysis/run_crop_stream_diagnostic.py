"""Receiver-first bounded crop streaming trial; owns only its own child processes."""
from __future__ import annotations
import argparse,json,shlex,shutil,subprocess,time,socket
from pathlib import Path
import paramiko
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'tools'))
from portable_live_view import sdp


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=45);p.add_argument('--receiver-host',default='172.21.30.13');args=p.parse_args()
    if not 15<=args.seconds<=300:raise ValueError('bounded15..300 seconds')
    # Refuse a nonlocal receiver address BEFORE any board work.
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:probe.bind((args.receiver_host,0))
    args.output.mkdir(parents=True,exist_ok=False)
    credentials=json.loads((REPO/'config/credentials.json').read_text());b=credentials['board']
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect(b['host'],username=b['user'],password=b['password'],timeout=15)
    def execute(command,timeout=30):
        _,o,e=client.exec_command(command,timeout=timeout);out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
        if code:raise RuntimeError(f'exit{code}: {out}\n{err}')
        return out
    active=execute("ps -eo pid,args | grep -E 'launch.py|worker.py|gst-launch|feed_supervisor|crop_stream.py|portable_tensor_parity.py board' | grep -v grep || true")
    if active.strip():raise RuntimeError('occupied board; not stopping others: '+active)
    root='/media/nvme/'+args.output.name
    execute(f'test ! -e {root} && mkdir {root} && cp -a /media/nvme/compiled-crop-replay-20261116/replay-fixture {root}/release')
    sftp=client.open_sftp()
    for local in [REPO/'runtime/launcher/crop_stream.py',REPO/'runtime/launcher/crop_geometry.py']+[REPO/'tools/analysis'/n for n in ('portable_hybrid_crop.py','portable_canny_crop.py','portable_tensor_parity.py')]:sftp.put(str(local),root+'/'+local.name)
    (args.output/'receiver.sdp').write_text(sdp(args.receiver_host,5010))
    command=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','warning','-protocol_whitelist','file,udp,rtp','-analyzeduration','1000000','-probesize','100000','-i',str((args.output/'receiver.sdp').resolve()),'-an','-frames:v','30','-c:v','copy','-progress',str((args.output/'progress.txt').resolve()),'-n',str((args.output/'received.mkv').resolve())]
    with (args.output/'receiver.log').open('w') as log:
        receiver=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            time.sleep(1)
            if receiver.poll() is not None:raise RuntimeError('receiver exited before camera start')
            cmd=shlex.join(['timeout','-k','5s',str(args.seconds+30)+'s','python3','-u',root+'/crop_stream.py','--release',root+'/release','--output',root+'/run','--rtsp','rtsp://192.168.1.10:5005/routecam','--seconds',str(args.seconds),'--host',args.receiver_host])
            _,o,e=client.exec_command(cmd+' > '+root+'/worker.log 2>&1',timeout=args.seconds+45)
            code=o.channel.recv_exit_status()
            sftp.get(root+'/worker.log',str(args.output/'worker.log'))
            try:snapshots=tuple(a.filename for a in sftp.listdir_attr(root+'/run') if a.filename.startswith('sample-'))
            except FileNotFoundError:snapshots=()
            for name in ('worker.json','frames.jsonl','encoded.h264')+snapshots:
                try:sftp.get(root+'/run/'+name,str(args.output/name))
                except FileNotFoundError:pass
            try:receiver.wait(timeout=5)
            except subprocess.TimeoutExpired:receiver.terminate();receiver.wait(timeout=5)
            (args.output/'session.json').write_text(json.dumps({'remote_directory':root,'remote_command':cmd,'remote_exit':code,'receiver_exit':receiver.returncode,'receiver_pid':receiver.pid,'bound_seconds':args.seconds,'live_release_modified':False},indent=2))
            if code:raise RuntimeError('stream worker failed; inspect worker.log')
            if receiver.returncode:raise RuntimeError('receiver failed/no30frames; inspect receiver.log/progress')
        finally:
            if receiver.poll() is None:receiver.terminate();receiver.wait(timeout=5)
            client.close()
    print((args.output/'worker.json').read_text())


if __name__=='__main__':main()
