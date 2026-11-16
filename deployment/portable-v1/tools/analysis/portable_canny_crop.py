"""Offline Canny-based full-frame PCB crop experiment against reviewed geometry.

No labels enter localization; review polygons are read only after selection for
coverage evaluation. No camera, model inference, training or deployment changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

PRESETS={'low':(30,90),'medium':(50,150),'high':(80,200)}
POLICY={'blur_kernel':5,'blur_sigma':1.2,'close_kernel':3,'close_iterations':2,
        'long_side_range':[100,300],'short_side_range':[25,120],
        'area_range':[2000,25000],'aspect_range':[1.5,5.0],
        'min_rectangularity':.65,'padding_source_pixels':16,
        'ambiguous_score_ratio':.90,'coverage_diagnostic_cutoff':.98}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rect_quad(rect,padding=0):
    center=np.float32(rect[0]);w,h=rect[1];corners=cv2.boxPoints(rect)
    if w<h: # use a long edge to define horizontal crop; NOT semantic orientation
        u=corners[1]-corners[0]
    else:u=corners[2]-corners[1]
    u=u/np.linalg.norm(u);v=np.float32([-u[1],u[0]])
    long=max(w,h)+2*padding;short=min(w,h)+2*padding
    quad=np.float32([center-u*long/2-v*short/2,center+u*long/2-v*short/2,
                     center+u*long/2+v*short/2,center-u*long/2+v*short/2])
    return quad


def polygon_coverage(inner,outer):
    a=np.float32(inner);b=np.float32(outer)
    area=abs(cv2.contourArea(a))
    if area<=0:raise ValueError('zero polygon area')
    intersection,_=cv2.intersectConvexConvex(a,b)
    return float(np.clip(intersection/area,0,1))


def find_crop(image,thresholds,close_iterations=2):
    """Canny + closed contours + geometry/texture ranking; NO review geometry."""
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(5,5),1.2)
    edges=cv2.Canny(blur,*thresholds,L2gradient=True)
    if close_iterations not in (0,1,2):raise ValueError('unsupported closing ablation')
    connected=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=close_iterations) if close_iterations else edges
    contours,_=cv2.findContours(connected,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=float(cv2.contourArea(contour))
        if area<900:continue
        rect=cv2.minAreaRect(contour);a,b=rect[1];long=max(a,b);short=min(a,b)
        if not 100<=long<=300 or not 25<=short<=120:continue
        if not 2000<=long*short<=25000 or not 1.5<=long/short<=5:continue
        rectangularity=area/(long*short)
        if rectangularity<.65:continue
        quad=rect_quad(rect)
        mask=np.zeros(image.shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.int32(quad),1)
        density=float(np.count_nonzero(edges[mask>0])/max(1,np.count_nonzero(mask)))
        score=float(rectangularity*min(density/.12,1)*min(long*short/5000,1))
        candidates.append({'quad':quad.tolist(),'padded_quad':rect_quad(rect,16).tolist(),
                           'score':score,'rectangularity':rectangularity,'edge_density':density,
                           'long_side':float(long),'short_side':float(short)})
    candidates.sort(key=lambda c:c['score'],reverse=True)
    # Collapse overlapping outer/inner contours from the SAME object. This does
    # not suppress spatially distinct contenders to force a single PCB.
    distinct=[]
    for c in candidates:
        if any(polygon_coverage(c['quad'],d['quad'])>.65 and polygon_coverage(d['quad'],c['quad'])>.65 for d in distinct):continue
        distinct.append(c)
    result={'candidate_count':len(distinct),'candidates':distinct[:5]}
    if not distinct:return {**result,'status':'unknown','reason':'no plausible edge rectangle'},edges
    chosen=distinct[0]
    if len(distinct)>1 and distinct[1]['score']>=.9*chosen['score']:
        return {**result,'status':'unknown','reason':'spatially distinct candidate ambiguity'},edges
    q=np.array(chosen['padded_quad'])
    if np.any(q[:,0]<0) or np.any(q[:,0]>=image.shape[1]) or np.any(q[:,1]<0) or np.any(q[:,1]>=image.shape[0]):
        return {**result,'status':'unknown','reason':'padded crop extends beyond source'},edges
    return {**result,'status':'located','crop_quad':chosen['padded_quad'],
            'semantic_orientation':'unresolved; Canny rectangle has 180-degree ambiguity'},edges


def crop_pixels(image,quad):
    q=np.float32(quad);width=max(2,round(float(np.linalg.norm(q[1]-q[0]))));height=max(2,round(float(np.linalg.norm(q[3]-q[0]))))
    dst=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    matrix=cv2.getPerspectiveTransform(q,dst)
    crop=cv2.warpPerspective(image,matrix,(width,height))
    scale=min(512/width,256/height);resized=cv2.resize(crop,(max(1,round(width*scale)),max(1,round(height*scale))))
    display=np.full((256,512,3),114,np.uint8);y=(256-resized.shape[0])//2;x=(512-resized.shape[1])//2
    display[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return crop,display,matrix


def validate_review(review,original,image_root):
    if review['schema']!=original['schema'] or review['class_contract']!=original['class_contract'] or review['board_corner_order']!=original['board_corner_order']:
        raise ValueError('annotation contract changed')
    old={r['id']:r for r in original['annotations']}
    if len(review['annotations'])!=len(old) or len({r['id'] for r in review['annotations']})!=len(old):raise ValueError('annotation count/duplicates differ')
    for r in review['annotations']:
        o=old[r['id']]
        for key in ('image','source_image_sha256','width','height','capture_group','state','split'):
            if r[key]!=o[key]:raise ValueError('source identity/state/split differs')
        if r['review_status']!='reviewed' or not r.get('reviewer') or not r.get('geometry_independently_reviewed'):
            raise ValueError('geometry not human reviewed')
        path=image_root/r['image']
        if not path.resolve().is_relative_to(image_root.resolve()):raise ValueError('unsafe image path')
        if digest(path)!=r['source_image_sha256']:raise ValueError('source image hash differs')
        for name,s in r['sites'].items():
            if s['presence']!=o['sites'][name]['presence'] or s['legacy_class_id']!=o['sites'][name]['legacy_class_id']:raise ValueError('site class/state differs')
        if set(r['sites'])!={'IC','terminal'}:raise ValueError('site names differ')
        for quad in [r['board_quad']]+[s['polygon'] for s in r['sites'].values()]:
            q=np.float32(quad)
            if q.shape!=(4,2) or not np.isfinite(q).all() or not cv2.isContourConvex(q) or cv2.contourArea(q,oriented=True)<=1:raise ValueError('invalid reviewed quad')
            if np.any(q<0) or np.any(q[:,0]>=r['width']) or np.any(q[:,1]>=r['height']):raise ValueError('reviewed quad outside source')
    return {'images':len(old),'reviewed':len(old),'board_quads_changed':sum(r['board_quad']!=old[r['id']]['board_quad'] for r in review['annotations']),
            'site_records_changed':sum(r['sites']!=old[r['id']]['sites'] for r in review['annotations'])}


def evaluate(quad,row):
    parts={'PCB':row['board_quad'],**{name:s['polygon'] for name,s in row['sites'].items()}}
    coverage={name:polygon_coverage(p,quad) for name,p in parts.items()}
    exact={name:all(cv2.pointPolygonTest(np.float32(quad),tuple(map(float,v)),False)>=0 for v in p) for name,p in parts.items()}
    return {'coverage':coverage,'all_parts_at_least_98_percent':all(v>=.98 for v in coverage.values()),
            'all_vertices_inside':all(exact.values()),'vertices_inside_by_part':exact,
            'crop_area_fraction':float(cv2.contourArea(np.float32(quad))/(row['width']*row['height']))}


def run(args):
    review=json.loads(args.review.read_text(encoding='utf-8'));original=json.loads(args.original.read_text(encoding='utf-8'))
    validation=validate_review(review,original,args.images)
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.review,args.output/'annotations.review-export.snapshot.json')
    report={'scope':'Exploratory reviewed DEVELOPMENT images, not held-out detector precision/recall. No inference or deployment.',
            'review_sha256':digest(args.review),'original_sha256':digest(args.original),'script_sha256':digest(__file__),
            'review_validation':validation,'policy':{**POLICY,'close_iterations':args.close_iterations},'presets':PRESETS,'results':{}}
    for name,thresholds in PRESETS.items():
        folder=args.output/name;folder.mkdir();rows=[];panels=[]
        for r in review['annotations']:
            image=cv2.imread(str(args.images/r['image']))
            found,edges=find_crop(image,thresholds,args.close_iterations)
            cv2.imwrite(str(folder/f"edges-{r['id']}.png"),edges)
            canvas=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
            for c in found['candidates']:cv2.polylines(canvas,[np.int32(c['quad'])],True,(200,100,0),1)
            display=np.full((256,512,3),80,np.uint8)
            row={'id':r['id'],'state':r['state'],**found}
            if found['status']=='located':
                row['evaluation']=evaluate(found['crop_quad'],r)
                crop,display,matrix=crop_pixels(image,found['crop_quad'])
                row['source_to_crop']=matrix.tolist();row['native_crop_shape']=list(crop.shape)
                cv2.imwrite(str(folder/f"crop-{r['id']}.png"),crop)
                cv2.imwrite(str(folder/f"rescaled-{r['id']}.png"),display)
                cv2.polylines(canvas,[np.int32(found['crop_quad'])],True,(0,255,255),2)
            for p in [r['board_quad']]+[s['polygon'] for s in r['sites'].values()]:cv2.polylines(canvas,[np.int32(p)],True,(0,220,0),1)
            cv2.imwrite(str(folder/f"overlay-{r['id']}.jpg"),canvas)
            title=r['id']+' '+found['status']
            if 'evaluation' in row:title+=' min coverage '+str(round(100*min(row['evaluation']['coverage'].values()),1))+'%'
            tile=cv2.copyMakeBorder(display,30,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240));cv2.putText(tile,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1);panels.append(tile);rows.append(row)
        for start in range(0,len(panels),8):
            chunk=panels[start:start+8]
            if len(chunk)%2:chunk.append(np.full_like(chunk[0],240))
            cv2.imwrite(str(folder/f'review-page-{start//8+1:02}.jpg'),np.vstack([np.hstack(chunk[i:i+2]) for i in range(0,len(chunk),2)]))
        located=[r for r in rows if r['status']=='located']
        summary={'images':len(rows),'located':len(located),'unknown':len(rows)-len(located),
                 'all_parts_98_percent':sum(r['evaluation']['all_parts_at_least_98_percent'] for r in located),
                 'all_vertices_inside':sum(r['evaluation']['all_vertices_inside'] for r in located),
                 'unsafe_located':sum(not r['evaluation']['all_parts_at_least_98_percent'] for r in located),
                 'median_crop_area_fraction':float(np.median([r['evaluation']['crop_area_fraction'] for r in located])) if located else None,
                 'by_state':{state:{'images':sum(r['state']==state for r in rows),'located':sum(r['state']==state for r in located),
                                   'all_parts_98_percent':sum(r['state']==state and r['evaluation']['all_parts_at_least_98_percent'] for r in located)} for state in sorted({r['state'] for r in rows})}}
        report['results'][name]={'summary':summary,'frames':rows};print(name,json.dumps(summary),flush=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Validated owner review:',json.dumps(validation))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','original','images','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--close-iterations',type=int,choices=(0,1,2),default=2)
    run(p.parse_args())
