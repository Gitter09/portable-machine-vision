"""Offline Canny + blue-substrate crop ablation. No live/model changes.

Selectors receive pixels only. Human geometry enters evaluation or explicitly
synthetic board-removed controls, never positive crop selection.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

_spec=importlib.util.spec_from_file_location('canny_helpers',Path(__file__).with_name('portable_canny_crop.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
POLICY={'hsv_low':[90,65,30],'hsv_high':[140,255,255], 'canny':[30,90],
        'edge_close_iterations':1,'group_close_kernel':15,'min_blue_pixels':500,
        'min_blue_fraction':.15,'long_range':[100,300],'short_range':[25,120],
        'aspect_range':[1.5,5.0],'area_range':[2000,25000],
        'seed_ambiguity_ratio':.75,'canny_ambiguity_ratio':.9,
        'edge_search_margin_pixels':32,'output_padding_pixels':16,
        'minimum_local_edge_pixels':30,'preservation_cutoff':.98}
METHODS=('canny_colour_gate','colour_group_only','colour_seed_canny')


def blue_mask(image):
    return cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),tuple(POLICY['hsv_low']),tuple(POLICY['hsv_high']))


def quad_mask(shape,quad):
    mask=np.zeros(shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.int32(quad),1)
    return mask


def appearance(mask,quad):
    region=quad_mask(mask.shape,quad)>0
    count=int(np.count_nonzero(mask[region]))
    return {'blue_pixels':count,'blue_fraction':float(count/max(1,np.count_nonzero(region)))}


def valid_source_quad(quad,shape):
    q=np.asarray(quad)
    return bool(np.all(q[:,0]>=0) and np.all(q[:,0]<shape[1]) and np.all(q[:,1]>=0) and np.all(q[:,1]<shape[0]))


def choose_canny(candidates,mask,shape,total=None):
    # Existing helper stores up to five candidates. Refuse truncated lists rather
    # than silently ignoring a lower-ranked spatial contender.
    if total is not None and total>len(candidates):return {'status':'unknown','reason':'candidate list truncated'}
    accepted=[];rejected=[]
    for candidate in candidates:
        c={**candidate,**appearance(mask,candidate['quad'])}
        if c['blue_pixels']>=500 and c['blue_fraction']>=.15:accepted.append(c)
        else:rejected.append(c)
    info={'appearance_candidates':accepted,'appearance_rejected':rejected}
    if not accepted:return {**info,'status':'unknown','reason':'no Canny rectangle with PCB colour evidence'}
    # Keep pre-existing geometry score; colour is a gate, not a label-based rank.
    accepted.sort(key=lambda c:c['score'],reverse=True)
    if len(accepted)>1 and accepted[1]['score']>=.9*accepted[0]['score']:
        return {**info,'status':'unknown','reason':'multiple coloured Canny candidates'}
    q=accepted[0]['padded_quad']
    if not valid_source_quad(q,shape):return {**info,'status':'unknown','reason':'crop beyond source'}
    return {**info,'status':'located','crop_quad':q,'seed_quad':accepted[0]['quad']}


def colour_seeds(mask):
    connected=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    contours,_=cv2.findContours(connected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        if cv2.contourArea(contour)<500:continue
        rect=cv2.minAreaRect(cv2.convexHull(contour));a,b=rect[1];long=max(a,b);short=min(a,b)
        if not 100<=long<=300 or not 25<=short<=120:continue
        if not 1.5<=long/short<=5 or not 2000<=long*short<=25000:continue
        quad=base.rect_quad(rect);evidence=appearance(mask,quad)
        if evidence['blue_pixels']<500 or evidence['blue_fraction']<.15:continue
        candidates.append({'quad':quad.tolist(),'rect':[[float(v) for v in rect[0]],[float(v) for v in rect[1]],float(rect[2])],**evidence})
    return sorted(candidates,key=lambda c:c['blue_pixels'],reverse=True)


def from_seed(candidates,edges,shape,refine):
    info={'seed_candidates':candidates}
    if not candidates:return {**info,'status':'unknown','reason':'no plausible grouped substrate'}
    if len(candidates)>1 and candidates[1]['blue_pixels']>=.75*candidates[0]['blue_pixels']:
        return {**info,'status':'unknown','reason':'multiple plausible blue substrates'}
    seed=candidates[0];rect=seed['rect'];q=np.float32(seed['quad']);rect_points=q
    local_count=0
    if refine:
        # Rectangular LOCAL search around the colour seed, not a hand-chosen
        # camera ROI. Grey plate/bracket edges outside it cannot move the crop.
        center=q.mean(axis=0);u=q[1]-q[0];long=float(np.linalg.norm(u));u/=long
        v=q[3]-q[0];short=float(np.linalg.norm(v));v/=short
        yy,xx=np.nonzero(edges);points=np.column_stack((xx,yy)).astype(np.float32)
        delta=points-center
        inside=(np.abs(delta@u)<=long/2+32)&(np.abs(delta@v)<=short/2+32)
        local=points[inside];local_count=len(local)
        if local_count<30:return {**info,'status':'unknown','reason':'insufficient local Canny support','local_edge_pixels':local_count}
        # Include substrate corners so refinement never shrinks away the seed.
        rect_points=np.vstack([q,local])
    fitted=cv2.minAreaRect(np.float32(rect_points))
    crop=base.rect_quad(fitted,16)
    if not valid_source_quad(crop,shape):return {**info,'status':'unknown','reason':'crop beyond source'}
    return {**info,'status':'located','crop_quad':crop.tolist(),'seed_quad':seed['quad'],
            'local_edge_pixels':local_count,'refinement_used':refine,
            'semantic_orientation':'unresolved 180-degree ambiguity'}


def select_all(image):
    mask=blue_mask(image)
    canny,edges=base.find_crop(image,(30,90),1)
    candidates=colour_seeds(mask)
    return {'canny_colour_gate':choose_canny(canny['candidates'],mask,image.shape,canny['candidate_count']),
            'colour_group_only':from_seed(candidates,edges,image.shape,False),
            'colour_seed_canny':from_seed(candidates,edges,image.shape,True)},mask,edges


def synthetic_without_board(image,row):
    # NEGATIVE EVALUATION CONTROL ONLY. This uses annotations to remove targets;
    # its result is never supplied to positive localization or used for tuning.
    out=image.copy()
    points=np.vstack([np.float32(row['board_quad'])]+[np.float32(s['polygon']) for s in row['sites'].values()])
    x,y,w,h=cv2.boundingRect(points);x0=max(0,x-35);y0=max(0,y-35);x1=min(image.shape[1],x+w+35);y1=min(image.shape[0],y+h+35)
    out[y0:y1,x0:x1]=[190,190,190]
    return out


def summary(rows):
    located=[r for r in rows if r['status']=='located'];good=[r for r in located if r['evaluation']['all_parts_at_least_98_percent']]
    return {'images':len(rows),'located':len(located),'preserved_98_percent':len(good),
            'all_vertices_inside':sum(r['evaluation']['all_vertices_inside'] for r in located),
            'unsafe_selected':len(located)-len(good),'unknown':len(rows)-len(located),
            'safe_median_crop_area_fraction':float(np.median([r['evaluation']['crop_area_fraction'] for r in good])) if good else None,
            'by_state':{state:{'images':sum(r['state']==state for r in rows),'preserved':sum(r['state']==state for r in good),'unknown':sum(r['state']==state and r['status']!='located' for r in rows)} for state in sorted({r['state'] for r in rows})}}


def run(args):
    review=json.loads(args.review.read_text(encoding='utf-8'));original=json.loads(args.original.read_text(encoding='utf-8'))
    validation=base.validate_review(review,original,args.images)
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.review,args.output/'annotations.review-export.snapshot.json')
    for method in METHODS:(args.output/method).mkdir()
    frames={method:[] for method in METHODS};panels={method:[] for method in METHODS};negatives=[]
    for row in review['annotations']:
        image=cv2.imread(str(args.images/row['image']));results,mask,edges=select_all(image)
        cv2.imwrite(str(args.output/f"mask-{row['id']}.png"),mask);cv2.imwrite(str(args.output/f"edges-{row['id']}.png"),edges)
        for method,result in results.items():
            record={'id':row['id'],'state':row['state'],**result};folder=args.output/method
            canvas=image.copy();display=np.full((256,512,3),80,np.uint8)
            if result['status']=='located':
                record['evaluation']=base.evaluate(result['crop_quad'],row)
                crop,display,matrix=base.crop_pixels(image,result['crop_quad']);record['source_to_crop']=matrix.tolist()
                cv2.imwrite(str(folder/f"crop-{row['id']}.png"),crop);cv2.imwrite(str(folder/f"rescaled-{row['id']}.png"),display)
                cv2.polylines(canvas,[np.int32(result['crop_quad'])],True,(0,220,255),2)
            if 'seed_quad' in result:cv2.polylines(canvas,[np.int32(result['seed_quad'])],True,(255,160,0),1)
            for p in [row['board_quad']]+[s['polygon'] for s in row['sites'].values()]:cv2.polylines(canvas,[np.int32(p)],True,(0,220,0),1)
            cv2.imwrite(str(folder/f"overlay-{row['id']}.jpg"),canvas)
            caption=row['id']+' '+result['status']
            if 'evaluation' in record:caption+=' min coverage '+str(round(100*min(record['evaluation']['coverage'].values()),1))+'%'
            tile=cv2.copyMakeBorder(display,30,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240));cv2.putText(tile,caption,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),1)
            frames[method].append(record);panels[method].append(tile)
        negative=synthetic_without_board(image,row);nresults,_,_=select_all(negative)
        negatives.append({'id':row['id'],'control':'synthetic neutral rectangular removal using review geometry; not real empty-plate data',
                          'results':{method:{'status':r['status'],'reason':r.get('reason'),'crop_quad':r.get('crop_quad')} for method,r in nresults.items()}})
    report={'scope':'Exploratory DEVELOPMENT crop containment, NOT held-out detection accuracy or runtime qualification',
            'policy':POLICY,'review_validation':validation,'review_sha256':base.digest(args.review),
            'script_sha256':base.digest(__file__),'canny_helper_sha256':base.digest(Path(base.__file__)),
            'results':{},'synthetic_negative_controls':negatives}
    for method in METHODS:
        tiles=panels[method]
        for start in range(0,len(tiles),8):
            chunk=tiles[start:start+8]
            if len(chunk)%2:chunk.append(np.full_like(chunk[0],240))
            cv2.imwrite(str(args.output/method/f'review-page-{start//8+1:02}.jpg'),np.vstack([np.hstack(chunk[i:i+2]) for i in range(0,len(chunk),2)]))
        stats=summary(frames[method]);stats['synthetic_negative_selected']=sum(n['results'][method]['status']=='located' for n in negatives)
        report['results'][method]={'summary':stats,'frames':frames[method]};print(method,json.dumps(stats),flush=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','original','images','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
