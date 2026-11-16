"""Create UNREVIEWED four-class box proposals on native hybrid crops.

Transfers human-reviewed site geometry, not prior model predictions. A projected
footprint is only a seed for the NEW training-box review. No training dataset YAML,
train/test split, model invocation or board access is produced.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('crop_review_helpers',Path(__file__).with_name('portable_canny_crop.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CLASSES=['IC_Fail','Terminal_pass','IC_Pass','Terminal_Fail']


def projected_box(polygon,matrix,width,height):
    points=cv2.perspectiveTransform(np.float32([polygon]),np.float64(matrix))[0]
    if not np.isfinite(points).all():raise ValueError('invalid projected points')
    low=points.min(axis=0);high=points.max(axis=0)
    if np.any(low < -1e-3) or high[0]>width+1e-3 or high[1]>height+1e-3:raise ValueError('target outside crop; do not silently truncate training label')
    low=np.maximum(low,0);high=np.minimum(high,[width,height])
    if np.any(high-low<1):raise ValueError('degenerate draft box')
    return [float(low[0]),float(low[1]),float(high[0]),float(high[1])]


def yolo_line(box,width,height):
    x0,y0,x1,y1=box['xyxy'];cid=box['class_id']
    if cid not in range(4) or not np.isfinite([x0,y0,x1,y1]).all() or not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('invalid training box')
    return f'{cid} {(x0+x1)/2/width:.9f} {(y0+y1)/2/height:.9f} {(x1-x0)/width:.9f} {(y1-y0)/height:.9f}'


def make_record(source,crop,path,matrix,source_sha):
    height,width=crop.shape[:2];boxes=[]
    for target in ('IC','terminal'):
        s=source['sites'][target]
        expected=(0 if s['presence']=='missing' else 2) if target=='IC' else (3 if s['presence']=='missing' else 1)
        if s['presence'] not in ('missing','present') or s['legacy_class_id']!=expected:raise ValueError('presence/class mismatch')
        boxes.append({'target':target,'class_id':expected,'label':CLASSES[expected],
                      'xyxy':projected_box(s['polygon'],matrix,width,height),
                      'proposal_source':'AABB of projected owner-reviewed SITE polygon; not tight body ground truth'})
    return {'id':source['id'],'image':f"images/{source['id']}.png",'image_sha256':base.digest(path),
            'width':width,'height':height,'state':source['state'],'capture_group':source['capture_group'],
            'source_image_sha256':source_sha,'source_to_crop':np.asarray(matrix).tolist(),
            'split':'development_pool_unsplit','boxes':boxes,'review_status':'pending','reviewer':None,
            'review_notes':'','training_eligible':False,'review_priority':'high' if source['state']=='both_present' else 'normal'}


def render(crop,row):
    h,w=crop.shape[:2];scale=min(560/w,330/h);resized=cv2.resize(crop,(round(w*scale),round(h*scale)))
    marked=resized.copy()
    for b in row['boxes']:
        x0,y0,x1,y1=np.int32(np.array(b['xyxy'])*scale);colour=(0,190,255) if b['target']=='IC' else (210,30,210)
        cv2.rectangle(marked,(x0,y0),(x1,y1),colour,2)
        cv2.putText(marked,b['label'],(max(0,x0),max(16,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1)
    canvas=np.full((390,1120,3),240,np.uint8);canvas[45:45+resized.shape[0],:resized.shape[1]]=resized
    canvas[45:45+marked.shape[0],560:560+marked.shape[1]]=marked
    cv2.putText(canvas,row['id']+' | '+row['state']+' | UNREVIEWED training-box proposals',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    return canvas


def run(args):
    review=json.loads(args.review.read_text(encoding='utf-8'));original=json.loads(args.original.read_text(encoding='utf-8'))
    base.validate_review(review,original,args.images)
    hybrid=json.loads((args.hybrid/'results.json').read_text(encoding='utf-8'))
    if hybrid['review_sha256']!=base.digest(args.review):raise ValueError('review/crop version mismatch')
    crops={r['id']:r for r in hybrid['results']['colour_seed_canny']['frames']};prepared=[]
    for source in review['annotations']:
        c=crops[source['id']]
        if c['status']!='located':raise ValueError('missing crop; no selective omission')
        image=cv2.imread(str(args.images/source['image']));regenerated,_,matrix=base.crop_pixels(image,c['crop_quad'])
        path=args.hybrid/'colour_seed_canny'/f"crop-{source['id']}.png";crop=cv2.imread(str(path))
        if crop is None or not np.array_equal(crop,regenerated) or not np.allclose(matrix,c['source_to_crop'],atol=1e-8,rtol=0):raise ValueError('crop pixel/transform mismatch')
        row=make_record(source,crop,path,matrix,source['source_image_sha256'])
        for b in row['boxes']:yolo_line(b,row['width'],row['height'])
        prepared.append((row,crop,path))
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('images','draft-yolo-DO-NOT-TRAIN','review-images','review-sheets'):(args.output/sub).mkdir()
    rows=[];tiles=[]
    for row,crop,path in prepared:
        shutil.copyfile(path,args.output/row['image'])
        lines=[yolo_line(b,row['width'],row['height']) for b in row['boxes']]
        (args.output/'draft-yolo-DO-NOT-TRAIN'/f"{row['id']}.txt").write_text('\n'.join(lines)+'\n',encoding='utf-8')
        tile=render(crop,row);cv2.imwrite(str(args.output/'review-images'/f"{row['id']}.jpg"),tile);rows.append(row);tiles.append(tile)
    for start in range(0,len(tiles),6):cv2.imwrite(str(args.output/'review-sheets'/f'page-{start//6+1:02}.jpg'),np.vstack(tiles[start:start+6]))
    counts={name:sum(b['class_id']==i for r in rows for b in r['boxes']) for i,name in enumerate(CLASSES)}
    result={'schema':'portable-cropped-training-boxes/v1','status':'UNREVIEWED proposals; training blocked',
            'classes':CLASSES,'coordinate_contract':'xyxy original NATIVE crop pixels; axis-aligned boxes',
            'annotation_contract':{'IC_Pass':'visible IC body and its own pins, excluding neighbours','Terminal_pass':'visible terminal housing, excluding wires and unrelated parts',
                                   'IC_Fail':'empty intended IC footprint including its pads','Terminal_Fail':'empty intended terminal footprint including holes/pads'},
            'preprocessing':'frozen hybrid colour-seed/local-Canny native crop; no CLAHE/sharpening/illumination enhancement; later model letterbox must match serving',
            'source_review_sha256':base.digest(args.review),'hybrid_results_sha256':base.digest(args.hybrid/'results.json'),
            'builder_sha256':base.digest(__file__),'helper_sha256':base.digest(Path(base.__file__)),
            'split_policy':'unsplit development pool; independent capture/specimen validation required; never randomly split adjacent frames',
            'class_counts':counts,'approved_images':0,'training_eligible':False,'annotations':rows}
    (args.output/'annotations.draft.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'classes.txt').write_text('\n'.join(CLASSES)+'\n',encoding='utf-8')
    (args.output/'DO-NOT-TRAIN.txt').write_text('All proposals require new training-box review. No data.yaml or train/val split. Review before finalization; current coverage insufficient for independent four-class training evaluation.\n',encoding='utf-8')
    template=Path(__file__).with_name('cropped_training_review.html').read_text(encoding='utf-8')
    (args.output/'review.html').write_text(template.replace('/*ANNOTATIONS*/null',json.dumps(result).replace('<','\\u003c')),encoding='utf-8')
    shutil.copyfile(args.review,args.output/'source-site-review.snapshot.json');shutil.copyfile(args.hybrid/'results.json',args.output/'hybrid-results.snapshot.json')
    print(json.dumps({'images':len(rows),'boxes':sum(counts.values()),'class_counts':counts,'approved':0,'output':str(args.output)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','original','images','hybrid','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
