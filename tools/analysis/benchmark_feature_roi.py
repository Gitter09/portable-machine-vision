"""Camera-free GFTT ROI parity and timing on exact audited camera frames."""
import argparse,hashlib,json,statistics,time
from pathlib import Path
import cv2
import numpy as np
from pcb_pose_tracker import seed_features


def run(args):
    if args.output.exists():raise FileExistsError('new output required')
    cases=[];parity=[]
    for path in sorted(args.inputs.glob('*.json')):
        record=json.loads(path.read_text());quad=record.get('pose_seed_quad')
        if quad is None:continue
        raw=path.with_suffix('.nv12').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['frame_sha256']:raise ValueError('audited frame hash differs')
        image=cv2.cvtColor(np.frombuffer(raw,np.uint8).reshape(1080,1280),cv2.COLOR_YUV2BGR_NV12);grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        a,full=seed_features(grey,quad,False);b,roi=seed_features(grey,quad,True)
        exact=(a is None and b is None) or (a is not None and b is not None and np.array_equal(a,b))
        if not exact:raise ValueError('feature points/order differ: '+path.stem)
        parity.append({'frame':path.stem,'ordered_points_exact':True,'feature_count':full['feature_count'],
            'roi_area_pixels':roi['feature_area_pixels'],'full_area_pixels':grey.size})
        if len(cases)<3:cases.append((grey,quad))
    if not cases:raise ValueError('no usable audited frames')
    samples={'full':[],'roi':[]}
    for i in range(args.repeats+3):
        grey,quad=cases[i%len(cases)]
        for local in ((False,True) if i%2 else (True,False)):
            _,stats=seed_features(grey,quad,local)
            if i>=3:samples['roi' if local else 'full'].append(stats['feature_seed_ms'])
    result={'scope':'same exact camera frames, unchanged detector mask/parameters; no camera/MLA used','opencv':cv2.__version__,
      'parity':parity,'repeats':args.repeats,'mean_ms':{k:statistics.mean(v) for k,v in samples.items()},
      'median_ms':{k:statistics.median(v) for k,v in samples.items()}}
    args.output.mkdir();(args.output/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps({'frames':len(parity),'mean_ms':result['mean_ms'],'median_ms':result['median_ms']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=40)
    run(p.parse_args())
