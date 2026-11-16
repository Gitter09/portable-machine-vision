"""Build a portable, explicitly UNREVIEWED PCB pose/site annotation pack.

Offline only. Existing experiment geometry is a proposal, not ground truth.
Manual overrides are separately recorded; source images are copied byte-for-byte.
No training export or model/runtime changes are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

LABELS = ['IC_Fail', 'Terminal_pass', 'IC_Pass', 'Terminal_Fail']
ORDER = ['header_top', 'terminal_top', 'terminal_bottom', 'header_bottom']
SITE_BOUNDS = {'IC': [.46,.26,.67,.74], 'terminal': [.74,.08,1.08,.92]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_quad(points, width, height):
    a=np.asarray(points,dtype=np.float32)
    if a.shape!=(4,2) or not np.isfinite(a).all(): raise ValueError('quad must be finite 4x2')
    if np.any(a<0) or np.any(a[:,0]>=width) or np.any(a[:,1]>=height): raise ValueError('quad outside image')
    if not cv2.isContourConvex(a) or cv2.contourArea(a,oriented=True)<=1: raise ValueError('quad must be convex, positive image-coordinate winding')
    return a


def projected_sites(quad):
    matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),np.float32(quad))
    result={}
    for name,(x0,y0,x1,y1) in SITE_BOUNDS.items():
        p=np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]])
        result[name]=cv2.perspectiveTransform(p,matrix)[0].astype(float).tolist()
    return result


def direction(quad):
    q=np.asarray(quad,dtype=float);header=(q[0]+q[3])/2;terminal=(q[1]+q[2])/2
    return {'header_center':header.tolist(),'terminal_end_center':terminal.tolist(),
            'header_to_terminal_degrees_clockwise_from_image_x':float(np.degrees(np.arctan2(*(terminal-header)[::-1]))%360)}


def make_annotation(item, pose, override, shape, image_sha):
    height,width=shape[:2]
    if override:
        quad=override['board_quad'];method='assistant manual proposal from source/grid'
        notes=override['notes'];priority='high'
    elif pose.get('status')=='located':
        quad=pose['board_quad'];method='unreliable colour-experiment proposal'
        notes='Machine-seeded; not independently verified. Review substrate edges, orientation and both sites.';priority='normal'
    else:
        raise ValueError(f"missing manual proposal for {item['id']}")
    validate_quad(quad,width,height)
    sites=projected_sites(quad)
    for p in sites.values():validate_quad(p,width,height)
    state=item['state']
    if state not in ('both_present','both_missing'): raise ValueError('state requires explicit presence mapping')
    present=state=='both_present'
    return {'id':item['id'],'image':f"images/{item['id']}.png",'source_image_sha256':image_sha,
            'source_path':item['path'],'width':width,'height':height,'frame_id':item.get('frame_id'),
            'capture_group':'both-present-probe-20261115' if present else 'both-missing-periodic-probe-20261115',
            'split':'development_only_not_held_out','state':state,
            'state_provenance':'owner-confirmed specimen state inherited from source manifest; not individual site geometry approval',
            'board_quad':np.asarray(quad,dtype=float).tolist(),'orientation':direction(quad),
            'sites':{name:{'polygon':poly,'presence':'present' if present else 'missing',
                           'legacy_class_id':(2 if present else 0) if name=='IC' else (1 if present else 3),
                           'geometry_source':'provisional template projection; footprint, not tight component-body box'} for name,poly in sites.items()},
            'review_status':'pending','reviewer':None,'review_priority':priority,'proposal_method':method,
            'notes':notes,'training_eligible':False,
            'geometry_independently_reviewed':False}


def labelme_record(row):
    shapes=[{'label':'PCB','points':row['board_quad'],'group_id':None,'shape_type':'polygon','flags':{}}]
    shapes.extend({'label':name+'_site','points':site['polygon'],'group_id':None,'shape_type':'polygon','flags':{}} for name,site in row['sites'].items())
    shapes.append({'label':'header_to_terminal','points':[row['orientation']['header_center'],row['orientation']['terminal_end_center']],
                   'group_id':None,'shape_type':'line','flags':{}})
    return {'version':'5.5.0','flags':{'unreviewed':True},'shapes':shapes,
            'imagePath':'../'+row['image'],'imageData':None,'imageHeight':row['height'],'imageWidth':row['width'],
            'portable_metadata':{'id':row['id'],'state':row['state'],'sha256':row['source_image_sha256'],
                                 'corner_order':ORDER,'status':'pending; NOT training ground truth'}}


def crop_bounds(row):
    points=np.array(row['board_quad']+[p for s in row['sites'].values() for p in s['polygon']])
    low=np.floor(points.min(axis=0)-35).astype(int);high=np.ceil(points.max(axis=0)+35).astype(int)
    return [max(0,int(low[0])),max(0,int(low[1])),min(row['width'],int(high[0])),min(row['height'],int(high[1]))]


def render_row(image,row):
    left,top,right,bottom=crop_bounds(row);crop=image[top:bottom,left:right]
    scale=min(480/(right-left),320/(bottom-top));size=(round(crop.shape[1]*scale),round(crop.shape[0]*scale))
    out=np.full((390,960,3),245,np.uint8)
    raw=cv2.resize(crop,size);out[55:55+size[1],:size[0]]=raw
    marked=raw.copy()
    for name,points,color in [('PCB',row['board_quad'],(255,180,0))]+[(name,s['polygon'],(0,210,255) if name=='IC' else (220,0,220)) for name,s in row['sites'].items()]:
        p=np.int32((np.array(points)-[left,top])*scale)
        cv2.polylines(marked,[p],True,color,2)
        cv2.putText(marked,name,tuple(p[0]),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
        if name=='PCB':
            for k,v in enumerate(p):cv2.circle(marked,tuple(v),3,color,-1);cv2.putText(marked,str(k),tuple(v+[3,-3]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
    d=row['orientation'];a=tuple(np.int32((np.array(d['header_center'])-[left,top])*scale));b=tuple(np.int32((np.array(d['terminal_end_center'])-[left,top])*scale))
    cv2.arrowedLine(marked,a,b,(40,180,40),1,tipLength=.12)
    out[55:55+size[1],480:480+size[0]]=marked
    cv2.putText(out,f"{row['id']} | {row['state']} | UNREVIEWED | {row['review_priority']} priority",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
    cv2.putText(out,'Original pixels                         Proposed PCB / IC / terminal sites',(10,48),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
    cv2.putText(out,'Arrow: header -> terminal. Corners 0,3 at header; 1,2 at terminal. Sites may be inferred.',(10,382),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1)
    return out


def build(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    experiment=json.loads(args.proposals.read_text(encoding='utf-8'))
    proposals={r['id']:r['pose'] for r in experiment['frames']}
    overrides=json.loads(args.overrides.read_text(encoding='utf-8'))
    items=manifest['images'];ids=[i['id'] for i in items]
    if len(ids)!=len(set(ids)) or any(not i or Path(i).name!=i or not all(c.isalnum() or c in '-_' for c in i) for i in ids): raise ValueError('unsafe/duplicate image ID')
    if set(overrides)-set(ids):raise ValueError('unknown override ID')
    # Validate all input evidence and geometry before creating output.
    prepared=[]
    for item in items:
        path=Path(item['path'])
        if digest(path)!=item['sha256']: raise ValueError('source hash differs')
        image=cv2.imread(str(path))
        if image is None:raise ValueError('unreadable source')
        row=make_annotation(item,proposals[item['id']],overrides.get(item['id']),image.shape,digest(path))
        prepared.append((path,image,row))
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('images','labelme','review-images','review-sheets'):(args.output/sub).mkdir()
    records=[];tiles=[]
    for path,image,row in prepared:
        shutil.copyfile(path,args.output/row['image'])
        assert digest(args.output/row['image'])==row['source_image_sha256']
        (args.output/'labelme'/f"{row['id']}.json").write_text(json.dumps(labelme_record(row),indent=2),encoding='utf-8')
        tile=render_row(image,row);cv2.imwrite(str(args.output/'review-images'/f"{row['id']}.jpg"),tile)
        records.append(row);tiles.append(tile)
    for start in range(0,len(tiles),6):
        cv2.imwrite(str(args.output/'review-sheets'/f'page-{start//6+1:02}.jpg'),np.vstack(tiles[start:start+6]))
    data={'schema':'portable-pose-sites-draft/v1','status':'UNREVIEWED; NOT independent ground truth or a training export',
          'class_contract':LABELS,'board_corner_order':ORDER,'coordinate_system':'original image pixel x,y; clockwise image-coordinate winding',
          'site_contract':'expected mounting footprint for each target, also when component is absent; NOT tight visible-body boxes',
          'source_manifest_sha256':digest(args.manifest),'proposal_results_sha256':digest(args.proposals),
          'manual_overrides_sha256':digest(args.overrides),'builder_sha256':digest(__file__),
          'split_policy':'All existing images are development-only. Do not split adjacent images/rotations into train and test. New independent captures required for validation.',
          'coverage':{**{state:sum(r['state']==state for r in records) for state in ('both_missing','both_present','ic_missing_terminal_present')},'approved_geometry':0},
          'annotations':records}
    (args.output/'annotations.draft.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    for name,path in [('input-manifest.json',args.manifest),('proposal-results.json',args.proposals),('manual-overrides.json',args.overrides)]:shutil.copyfile(path,args.output/name)
    template=Path(__file__).with_name('portable_site_review.html').read_text(encoding='utf-8')
    (args.output/'review.html').write_text(template.replace('/*ANNOTATION_DATA*/null',json.dumps(data).replace('<','\\u003c')),encoding='utf-8')
    print(json.dumps({'images':len(records),'high_priority':sum(r['review_priority']=='high' for r in records),
                      'approved':0,'output':str(args.output)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','proposals','overrides','output'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())
