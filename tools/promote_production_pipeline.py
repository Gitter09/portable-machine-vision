"""Activate an immutable production package ONLY after its packaged smoke test."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config,session):
    if config.get('schema')!='portable-production-selection/v1' or config.get('state')!='active':raise ValueError('invalid production selection')
    package=(REPO/config['package']).resolve();manifest=package/'MANIFEST.json'
    if digest(manifest)!=config['manifest_sha256'] or digest(package.with_suffix('.zip'))!=config['archive_sha256']:raise ValueError('release fingerprint differs')
    m=json.loads(manifest.read_text());w=json.loads((session/'worker.json').read_text());r=json.loads((session/'receiver.json').read_text());s=json.loads((session/'session.json').read_text())
    if m['release_channel']!='production' or m['package_id']!=config['release_id']:raise ValueError('wrong production release')
    if s['state']!='completed' or s['remote_exit'] or w['state']!='completed' or w['errors'] or r['host_errors']:raise ValueError('packaged smoke failed')
    if not all(w.get(k) and m.get(k) for k in ('feature_roi','coordinates_only','track_pose','optimized','fast_transforms')):raise ValueError('production features not enabled')
    if w['cpp_input_kernel']!='parallel8' or m['cpp_input_kernel']!='parallel8' or w['cpp_library_sha256']!=m['cpp_library_sha256'] or m['cpp_library_sha256']!=config['cpp_library_sha256']:raise ValueError('native artifact/kernel differs')
    if w['elf_sha256']!=m['board_model_sha256'] or w['mpk_sha256']!=m['mpk_sha256']:raise ValueError('model differs')
    if not w['located'] or w['published_results']!=w['inference_frames'] or r['matched_at_arrival']!=r['metadata_results'] or r['metadata_results']!=w['published_results']:raise ValueError('no complete matched smoke')
    for name,sha in m['files'].items():
        p=(package/name).resolve()
        if not p.is_relative_to(package) or digest(p)!=sha:raise ValueError('package changed: '+name)
        if (name.startswith('runtime/launcher/') or name.startswith('tools/analysis/') or name=='native/libportable_input.so') and p.suffix in ('.py','.cpp','.html','.so'):
            if digest(session/'source-snapshot'/p.name)!=sha:raise ValueError('deployed smoke source differs: '+name)
    return package,w,r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--verified-session',type=Path,required=True)
    args=p.parse_args();config=json.loads(args.selection.read_text());package,w,r=validate(config,args.verified_session)
    subprocess.run([str((REPO/config['python']).resolve()),'-B',str(package/'verify.py')],check=True)
    config.update(promoted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_session=str(args.verified_session.resolve()),
                  smoke_inference_fps=w['inference_fps'],smoke_decoded_fps=r['decoded_arrival_fps'])
    target=REPO/'config/production-pipeline.json';history=args.selection.parent/'history';history.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    if target.exists():shutil.copyfile(target,history/(stamp+'-previous.json'))
    text=json.dumps(config,indent=2);tmp=target.with_suffix('.tmp.json');tmp.write_text(text,encoding='utf-8');os.replace(tmp,target)
    (history/(stamp+'-promoted.json')).write_text(text,encoding='utf-8')
    print('PRODUCTION SELECTED:',config['release_id']);print('Verified smoke:',args.verified_session)


if __name__=='__main__':main()
