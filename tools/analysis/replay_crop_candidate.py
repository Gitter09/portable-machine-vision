"""Isolated compiled crop-model replay on portable board, no camera/runtime changes."""
from __future__ import annotations
import argparse
import hashlib
import json
import shlex
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import paramiko
import portable_tensor_parity as parity

REPO=Path(__file__).resolve().parents[2]


def run(args):
    if args.output.exists():raise FileExistsError('replay output exists; never overwrite')
    reviewed=json.loads((args.evaluation/'reviewed-labels.snapshot.json').read_text(encoding='utf-8'))
    evaluation=json.loads((args.evaluation/'evaluation.json').read_text(encoding='utf-8'))
    compilation=json.loads((args.export/'compilation.json').read_text(encoding='utf-8'))
    if parity.digest(args.export/'yolo11n_fused.onnx')!=evaluation['models']['new_crop_candidate']['sha256']:raise ValueError('FP32 candidate differs')
    if parity.digest(args.export/compilation['source_onnx'])!=compilation['source_onnx_sha256']:raise ValueError('compiled source lineage differs')
    for name,sha in compilation['artifacts'].items():
        if parity.digest(args.export/name)!=sha:raise ValueError('compiled asset differs')
    args.output.mkdir(parents=True)
    images=[args.pack/'model-inputs'/f"{r['id']}.png" for r in reviewed['annotations']]
    selection=json.loads((args.pack/'selection.json').read_text(encoding='utf-8'))
    expected={r['id']:r for r in selection['sources']}
    for r,image in zip(reviewed['annotations'],images):
        if parity.digest(image)!=expected[r['id']]['model_image_sha256']:raise ValueError('validation model input changed')
    parity.prepare(SimpleNamespace(images=images,side=1280,output=args.output/'inputs'))
    fixture=args.output/'replay-fixture';assets=fixture/'assets';(assets/'share/processmla').mkdir(parents=True);(assets/'provenance').mkdir()
    shutil.copyfile(args.export/'yolov8_stage1_mla.elf',assets/'share/processmla/yolov8_stage1_mla.elf')
    shutil.copyfile(args.export/'yolov8_mpk.json',assets/'provenance/mpk.json')
    inventory={'kind':'isolated replay fixture; NOT a staged live release','assets':{str(p.relative_to(assets)).replace('\\','/'):parity.digest(p) for p in assets.rglob('*') if p.is_file()}}
    (fixture/'release.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    c=json.loads((REPO/'config/credentials.json').read_text(encoding='utf-8'))['board']
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(c['host'],username=c['user'],password=c['password'],timeout=15)
    remote='/media/nvme/'+args.output.name
    def execute(command,timeout=30):
        _,stdout,stderr=client.exec_command(command,timeout=timeout)
        out=stdout.read().decode('utf-8','replace');err=stderr.read().decode('utf-8','replace');code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError(f'remote exit {code}: {out}\n{err}')
        return out
    preflight=execute("date -u; ps -eo pid,args | grep -E 'launch.py|worker.py|gst-launch|feed_supervisor|portable_tensor_parity' | grep -v grep || true")
    # Only date is expected. Do not stop or displace someone else's job.
    if len(preflight.strip().splitlines())!=1:raise RuntimeError('board process preflight not empty: '+preflight)
    execute('test ! -e '+shlex.quote(remote)+' && mkdir '+shlex.quote(remote))
    files=list(fixture.rglob('*'))+list((args.output/'inputs').rglob('*'))
    sftp=client.open_sftp()
    for p in files:
        if not p.is_file():continue
        destination=remote+'/'+str(p.relative_to(args.output)).replace('\\','/')
        execute('mkdir -p '+shlex.quote(destination.rsplit('/',1)[0]))
        sftp.put(str(p),destination)
    script=Path(parity.__file__);sftp.put(str(script),remote+'/portable_tensor_parity.py')
    command=f'timeout -k 5s 300s python3 -u {remote}/portable_tensor_parity.py board --transforms software --release {remote}/replay-fixture --inputs {remote}/inputs/inputs.npz --output {remote}/board'
    started=time.time()
    _,stdout,stderr=client.exec_command(command+' > '+remote+'/board.log 2>&1',timeout=330)
    code=stdout.channel.recv_exit_status();duration=time.time()-started
    sftp.get(remote+'/board.log',str(args.output/'board.log'))
    (args.output/'execution.json').write_text(json.dumps({'host':c['host'],'host_key_sha256_hex':hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest(),
        'remote_directory':remote,'preflight':preflight,'command':command,'exit_code':code,'elapsed_seconds':duration,
        'camera_decoder_used':False,'live_release_modified':False,'script_sha256':parity.digest(script)},indent=2),encoding='utf-8')
    if code:raise RuntimeError('board replay failed; see board.log')
    (args.output/'board').mkdir()
    for name in ('board.json','outputs.npz','software-transform-contract.json','runner-config.json'):sftp.get(remote+'/board/'+name,str(args.output/'board'/name))
    board=json.loads((args.output/'board/board.json').read_text())
    if parity.digest(args.output/'board/outputs.npz')!=board['outputs_sha256']:raise ValueError('downloaded board output hash mismatch')
    if board['elf_sha256']!=compilation['artifacts']['yolov8_stage1_mla.elf'] or board['mpk_sha256']!=compilation['artifacts']['yolov8_mpk.json']:raise ValueError('board model lineage differs')
    sftp.close();client.close()
    print('Verified compiled board replay:',len(board['samples']),'images;',round(duration,2),'seconds; no camera or live release changes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pack','evaluation','export','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
