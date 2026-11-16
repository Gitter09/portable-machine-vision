"""Prepare a reviewed, TRAIN-ONLY crop pool and reproducible baked augmentation.

Never invents validation/test samples or starts training. All original parents
are already development data. Validation must come from separate reserved data.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

s=importlib.util.spec_from_file_location('box_helpers',Path(__file__).with_name('prepare_cropped_training_review.py'))
boxes=importlib.util.module_from_spec(s);s.loader.exec_module(boxes);base=boxes.base
SIDE=1280
VARIANTS=('base','rotate180','exposure','geometry','mild_blur')
POLICY={'seed':20261115,'side':SIDE,'padding':114,'variants':list(VARIANTS),
        'exposure_alpha':[.90,1.10],'exposure_beta':[-10,10],
        'rotation_degrees':[-8,8],'geometry_occupancy':[.85,.98],
        'blur_kernel':3,'blur_sigma':[.6,1.0],
        'scope':'baked TRAIN ONLY; five images per parent do not create independent captures',
        'no_clahe_sharpening_hue_flip_mosaic':True}


def validate_review(draft,review):
    for key in ('schema','classes','source_review_sha256','hybrid_results_sha256'):
        if draft[key]!=review[key]:raise ValueError('review pack contract differs')
    if len(draft['annotations'])!=len(review['annotations']):raise ValueError('review count differs')
    old={r['id']:r for r in draft['annotations']};seen=set();changed=0
    for r in review['annotations']:
        if r['id'] not in old or r['id'] in seen:raise ValueError('review ID differs/duplicates')
        seen.add(r['id']);o=old[r['id']]
        for key in ('image','image_sha256','width','height','state','capture_group','split'):
            if r[key]!=o[key]:raise ValueError('review source/state differs')
        if r['review_status']!='reviewed' or not r.get('reviewer'):raise ValueError('unreviewed image')
        if len(r['boxes'])!=len(o['boxes']):raise ValueError('target count differs')
        for b,ob in zip(r['boxes'],o['boxes']):
            if any(b[k]!=ob[k] for k in ('target','class_id','label')):raise ValueError('review target/class differs')
            boxes.yolo_line(b,r['width'],r['height'])
        changed+=r['boxes']!=o['boxes']
    return changed


def check_verbal_approval(draft_path,approval):
    if approval['draft_sha256']!=base.digest(draft_path) or approval['images']!=42 or approval['boxes']!=84:
        raise ValueError('conversation approval is not bound to this exact first batch')
    if 'owner verbal approval' not in approval['approval_type']:raise ValueError('unsupported first-batch review provenance')


def seed_for(parent,variant):
    return int.from_bytes(hashlib.sha256(f"{POLICY['seed']}|{parent}|{variant}".encode()).digest()[:8],'little')


def letterbox(image):
    h,w=image.shape[:2];scale=min(SIDE/w,SIDE/h);nw,nh=round(w*scale),round(h*scale)
    left=round((SIDE-nw)/2-.1);top=round((SIDE-nh)/2-.1)
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    out=cv2.copyMakeBorder(resized,top,SIDE-nh-top,left,SIDE-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    return out,np.float64([[nw/w,0,left],[0,nh/h,top],[0,0,1]])


def transform_boxes(source,matrix):
    result=[]
    for b in source:
        x0,y0,x1,y1=b['xyxy'];q=np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]])
        p=cv2.perspectiveTransform(q,np.float64(matrix))[0];lo=p.min(axis=0);hi=p.max(axis=0)
        if np.any(lo<-.001) or np.any(hi>SIDE+.001):raise ValueError('augmentation clips a target; refusing silent label clipping/drop')
        r={**b,'xyxy':[float(max(0,lo[0])),float(max(0,lo[1])),float(min(SIDE,hi[0])),float(min(SIDE,hi[1]))]}
        boxes.yolo_line(r,SIDE,SIDE);result.append(r)
    return result


def augment(image,source,parent,variant):
    if variant not in VARIANTS:raise ValueError('unknown augmentation')
    rng=np.random.default_rng(seed_for(parent,variant));params={};h,w=image.shape[:2]
    if variant=='rotate180':
        out,matrix=letterbox(cv2.rotate(image,cv2.ROTATE_180))
        matrix=matrix@np.float64([[-1,0,w],[0,-1,h],[0,0,1]])
        params={'angle':180,'box_coordinates':'continuous image-edge convention'}
    elif variant=='exposure':
        alpha=float(rng.uniform(.9,1.1));beta=float(rng.uniform(-10,10))
        pixels=np.clip(image.astype(np.float32)*alpha+beta,0,255).astype(np.uint8)
        out,matrix=letterbox(pixels);params={'alpha':alpha,'beta':beta,'padding_unchanged':114}
    elif variant=='geometry':
        angle=float(rng.uniform(-8,8));occupancy=float(rng.uniform(.85,.98));radians=np.deg2rad(angle)
        extent_w=abs(w*np.cos(radians))+abs(h*np.sin(radians));extent_h=abs(w*np.sin(radians))+abs(h*np.cos(radians))
        scale=min(SIDE/extent_w,SIDE/extent_h)*occupancy
        affine=cv2.getRotationMatrix2D((w/2,h/2),angle,scale);affine[:,2]+=np.array([SIDE/2-w/2,SIDE/2-h/2])
        out=cv2.warpAffine(image,affine,(SIDE,SIDE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))
        matrix=np.vstack([affine,[0,0,1]]);params={'angle':angle,'occupancy':occupancy,'scale':float(scale),'full_image_contained':True}
    else:
        out,matrix=letterbox(image)
        if variant=='mild_blur':
            sigma=float(rng.uniform(.6,1.0));out=cv2.GaussianBlur(out,(3,3),sigma);params={'sigma':sigma,'kernel':3,'domain':'after model resize; mild, not native-small-feature blur'}
    return out,transform_boxes(source,matrix),matrix,params


def render(image,labels,caption):
    view=cv2.resize(image,(320,320))
    for b in labels:
        x0,y0,x1,y1=np.int32(np.array(b['xyxy'])/4);colour=(0,210,255) if b['target']=='IC' else (210,20,210)
        cv2.rectangle(view,(x0,y0),(x1,y1),colour,1);cv2.putText(view,b['label'],(max(0,x0),max(12,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,colour,1)
    view=cv2.copyMakeBorder(view,26,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240));cv2.putText(view,caption,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
    return view


def run(args):
    first=json.loads((args.first/'annotations.draft.json').read_text(encoding='utf-8'))
    approval=json.loads((args.second/'first-batch-owner-confirmation.json').read_text(encoding='utf-8'))
    check_verbal_approval(args.first/'annotations.draft.json',approval)
    draft=json.loads((args.second/'annotations.draft.json').read_text(encoding='utf-8'));review=json.loads(args.review.read_text(encoding='utf-8'))
    changed=validate_review(draft,review)
    parents=[];hashes=set();ids=set()
    for pack,rows,basis in [(args.first,first['annotations'],'owner conversation approval of unchanged draft'),(args.second,review['annotations'],'named batch02 JSON review')]:
        for row in rows:
            path=pack/row['image']
            if not path.resolve().is_relative_to(pack.resolve()):raise ValueError('unsafe source path')
            if base.digest(path)!=row['image_sha256'] or row['id'] in ids or row['image_sha256'] in hashes:raise ValueError('source changed or duplicated')
            ids.add(row['id']);hashes.add(row['image_sha256'])
            if pack==args.second and row.get('original_split')!='train':raise ValueError('historical holdout cannot enter training')
            for b in row['boxes']:boxes.yolo_line(b,row['width'],row['height'])
            parents.append({'record':row,'path':path,'approval_basis':basis})
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('images/train','labels/train','native-parents','previews','provenance'):(args.output/sub).mkdir(parents=True)
    records=[];lineage=[];gallery=[];base_counts={n:0 for n in boxes.CLASSES}
    for parent in parents:
        row=parent['record'];image=cv2.imread(str(parent['path']));panels=[]
        if image.shape[:2]!=(row['height'],row['width']):raise ValueError('native dimensions differ')
        shutil.copyfile(parent['path'],args.output/'native-parents'/f"{row['id']}.png")
        for b in row['boxes']:base_counts[boxes.CLASSES[b['class_id']]]+=1
        lineage.append({'id':row['id'],'source_crop_sha256':row['image_sha256'],'state':row['state'],
                        'capture_group':row['capture_group'],'split':'train','approval_basis':parent['approval_basis'],
                        'source_ancestor_sha256':row.get('source_ancestor_sha256'),'boxes':row['boxes']})
        for variant in VARIANTS:
            out,bs,matrix,params=augment(image,row['boxes'],row['id'],variant);name=row['id']+'__'+variant
            image_path=args.output/'images/train'/f'{name}.png';label_path=args.output/'labels/train'/f'{name}.txt'
            cv2.imwrite(str(image_path),out,[cv2.IMWRITE_PNG_COMPRESSION,3]);label_path.write_text('\n'.join(boxes.yolo_line(b,SIDE,SIDE) for b in bs)+'\n',encoding='utf-8')
            records.append({'id':name,'parent_id':row['id'],'split':'train','capture_group':row['capture_group'],
                            'variant':variant,'rng_seed':seed_for(row['id'],variant),'params':params,'native_to_model':matrix.tolist(),
                            'image':f'images/train/{name}.png','label':f'labels/train/{name}.txt',
                            'image_sha256':base.digest(image_path),'label_sha256':base.digest(label_path),'boxes':bs})
            panels.append(render(out,bs,variant))
        view=np.hstack(panels);cv2.imwrite(str(args.output/'previews'/f"{row['id']}.jpg"),view)
        gallery.append(f'<section><h2>{html.escape(row["id"])} — {html.escape(row["state"])}</h2><a href="previews/{row["id"]}.jpg"><img loading="lazy" src="previews/{row["id"]}.jpg"></a></section>')
        print(row['id'],'five train-only views',flush=True)
    summary={'original_parents':len(parents),'training_images':len(records),'boxes':sum(len(r['boxes']) for r in records),
             'original_class_counts':base_counts,'augmented_class_counts':{n:sum(b['class_id']==i for r in records for b in r['boxes']) for i,n in enumerate(boxes.CLASSES)},
             'batch02_box_records_changed':changed,'validation_images':0,'test_images':0,
             'independence':'90 originals are correlated; 450 views are NOT 450 independent examples',
             'training_ready':False,'blockers':['Separate validation dataset/explicit evaluation plan not supplied; do not split augmented siblings or use train as validation.']}
    manifest={'classes':boxes.CLASSES,'policy':POLICY,'summary':summary,'parents':lineage,'images':records,
              'first_draft_sha256':base.digest(args.first/'annotations.draft.json'),'first_approval_sha256':base.digest(args.second/'first-batch-owner-confirmation.json'),
              'second_review_sha256':base.digest(args.review),'builder_sha256':base.digest(__file__),
              'label_helper_sha256':base.digest(Path(boxes.__file__)),'opencv_version':cv2.__version__,
              'normalization':'Stored PNGs are UINT8 BGR/OpenCV encoding. Training loader must convert RGB and divide by 255 once; do not save normalized values to PNG.',
              'split_policy':'All current parents are irrevocably TRAIN for this candidate. Reserve external captures/heldouts; no val/test links or training YAML invented.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output/'training-readiness.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    overrides={'imgsz':1280,'rect':False,'mosaic':0.,'mixup':0.,'copy_paste':0.,'degrees':0.,'translate':0.,'scale':0.,'shear':0.,'perspective':0.,'fliplr':0.,'flipud':0.,'hsv_h':0.,'hsv_s':0.,'hsv_v':0.}
    (args.output/'recommended-loader-overrides.json').write_text(json.dumps({'reason':'Augmentation baked already; prevent double online augmentation. NO model/epoch/device/run selected; NO training invocation.','options':overrides},indent=2),encoding='utf-8')
    for name,path in [('first-draft.json',args.first/'annotations.draft.json'),('first-approval.json',args.second/'first-batch-owner-confirmation.json'),('batch02-reviewed.json',args.review),('batch02-draft.json',args.second/'annotations.draft.json')]:shutil.copyfile(path,args.output/'provenance'/name)
    intro='<!doctype html><meta charset="utf-8"><title>Training augmentation review</title><style>body{font:16px system-ui;margin:22px;background:#222;color:#eee}a{color:#9df}img{width:100%;height:auto}section{border-top:1px solid #777;margin-top:25px}</style><h1>Training-only preprocessing and augmentation</h1><p>90 original parents, five views each: base / 180° / exposure / mild geometry-scale / mild blur. All labels transformed with pixels; no target clipping/drop. Not 450 independent examples. Validation/test are NOT augmented or invented. Training has NOT started.</p><p>PNG inputs are 1280-square, padding 114; loader RGB/255 normalization still required. No CLAHE/sharpening/hue changes/mosaic. All 90 preview strips shown below.</p>'
    (args.output/'index.html').write_text(intro+'\n'.join(gallery),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('first','second','review','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
