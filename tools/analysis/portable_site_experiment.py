"""Offline, detector-independent blue-PCB pose/site feasibility experiment.

Not a production preprocessor or accuracy evaluation. No model or board access.
Geometry uses blue substrate and the neutral-dark header at one end, not the IC
or terminal's presence. Site polygons are provisional reference annotations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np

SIZE = (320, 144)
# Normalized substrate coordinates, header on LEFT. Provisional, review required.
SITES = {'IC': [0.46, 0.26, 0.67, 0.74], 'terminal': [0.74, 0.08, 1.08, 0.92]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def site_quad(bounds):
    x0,y0,x1,y1 = bounds
    return np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]) * np.float32(SIZE)


def locate(image):
    """Return UNKNOWN on clipping, ambiguity or insufficient header asymmetry."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90,65,30), (140,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not 500 <= area <= 25000: continue
        rect=cv2.minAreaRect(contour); w,h=rect[1]
        if min(w,h)<=0 or not 1.7<=max(w,h)/min(w,h)<=4.8: continue
        if area/(w*h)<0.55: continue
        candidates.append(rect)
    if len(candidates)!=1:
        return {'status':'unknown','reason':f'{len(candidates)} plausible blue boards'}
    corners=cv2.boxPoints(candidates[0]).astype(np.float32)
    if np.any(corners[:,0]<5) or np.any(corners[:,0]>=image.shape[1]-5) or np.any(corners[:,1]<5) or np.any(corners[:,1]>=image.shape[0]-5):
        return {'status':'unknown','reason':'board touches image boundary'}
    # Opposite short-edge midpoints are the longitudinal ends. Initial direction
    # arbitrary; header score below resolves 180-degree ambiguity.
    edges=np.roll(corners,-1,axis=0)-corners
    k=int(np.argmin(np.linalg.norm(edges,axis=1)))
    left=(corners[k]+corners[(k+1)%4])/2
    right=(corners[(k+2)%4]+corners[(k+3)%4])/2
    u=right-left; length=float(np.linalg.norm(u)); u/=length
    v=np.float32([-u[1],u[0]])
    width=float(min(candidates[0][1]))
    src=np.float32([left-v*width/2,right-v*width/2,right+v*width/2,left+v*width/2])
    dst=np.float32([[0,0],[SIZE[0],0],[SIZE[0],SIZE[1]],[0,SIZE[1]]])
    matrix=cv2.getPerspectiveTransform(src,dst)
    # Inspect slightly beyond both substrate ends; never use target-site colour
    # as a direction cue. Header is neutral-dark, blue substrate isn't neutral.
    shift=np.float64([[1,0,32],[0,1,0],[0,0,1]])
    ext=cv2.warpPerspective(image,shift@matrix,(SIZE[0]+64,SIZE[1]),borderValue=(255,255,255))
    eh=cv2.cvtColor(ext,cv2.COLOR_BGR2HSV)
    dark=(eh[:,:,1]<100)&(eh[:,:,2]<105)
    a=float(dark[20:124,0:76].mean()); b=float(dark[20:124,-76:].mean())
    if max(a,b)<0.035 or abs(a-b)<0.025:
        return {'status':'unknown','reason':'header direction ambiguous','header_scores':[a,b]}
    if b>a:
        flip=np.float64([[-1,0,SIZE[0]],[0,-1,SIZE[1]],[0,0,1]])
        matrix=flip@matrix
        src=src[[2,3,0,1]]
    inv=np.linalg.inv(matrix)
    polygons={name:cv2.perspectiveTransform(site_quad(bounds)[None],inv)[0].tolist() for name,bounds in SITES.items()}
    return {'status':'located','board_quad':src.tolist(),'source_to_canonical':matrix.tolist(),
            'sites':polygons,'header_scores':[a,b], 'long_side_pixels':length, 'short_side_pixels':width}


class ReferenceLocator:
    """ORB/RANSAC using stable reference features outside both target sites.

    Reference pose is provisional. Never update it from detections or test frames.
    This is a second exploratory method, not a validated alignment gate.
    """
    def __init__(self, image):
        self.reference_pose=locate(image)
        if self.reference_pose['status']!='located': raise ValueError('reference pose unknown')
        self.orb=cv2.ORB_create(nfeatures=1800,edgeThreshold=10,patchSize=21,fastThreshold=7)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        mask=self.mask(image)
        region=np.zeros(image.shape[:2],np.uint8)
        quad=np.float32(self.reference_pose['board_quad']); center=quad.mean(axis=0)
        cv2.fillConvexPoly(region,np.int32(center+1.2*(quad-center)),255)
        mask=cv2.bitwise_and(mask,region)
        for poly in self.reference_pose['sites'].values():
            cv2.fillConvexPoly(mask,np.int32(poly),0)
        self.kp,self.des=self.features(image,mask)
        self.reference_keypoints=len(self.kp)

    @staticmethod
    def mask(image):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        blue=cv2.inRange(hsv,(90,65,30),(140,255,255))
        return cv2.dilate(blue,np.ones((19,19),np.uint8))

    def features(self,image,mask):
        return self.orb.detectAndCompute(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None,fx=2,fy=2),cv2.resize(mask,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))

    def __call__(self,image):
        kp,des=self.features(image,self.mask(image))
        if des is None or self.des is None:return {'status':'unknown','reason':'no descriptors'}
        pairs=self.matcher.knnMatch(self.des,des,k=2)
        good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
        info={'matches':len(good),'reference_keypoints':self.reference_keypoints}
        if len(good)<8:return {'status':'unknown','reason':'insufficient ORB matches',**info}
        a=np.float32([self.kp[m.queryIdx].pt for m in good])/2
        b=np.float32([kp[m.trainIdx].pt for m in good])/2
        cv2.setRNGSeed(42)
        affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=3000,confidence=.995)
        n=int(inliers.sum()) if inliers is not None else 0;info['inliers']=n
        if affine is None or n<8 or n/len(good)<.5:return {'status':'unknown','reason':'RANSAC support insufficient',**info}
        use=inliers.ravel().astype(bool)
        if np.linalg.norm(np.ptp(a[use],axis=0))<35:return {'status':'unknown','reason':'inliers too localized',**info}
        scale=float(np.hypot(affine[0,0],affine[0,1]));info['scale']=scale
        if not .75<scale<1.3:return {'status':'unknown','reason':'implausible scale',**info}
        transform=np.vstack([affine,[0,0,1]])
        canonical=np.float64(self.reference_pose['source_to_canonical'])@np.linalg.inv(transform)
        inv=np.linalg.inv(canonical)
        quad=cv2.perspectiveTransform(np.float32([[[0,0],[320,0],[320,144],[0,144]]]),inv)[0]
        if np.any(quad[:,0]<5) or np.any(quad[:,0]>=image.shape[1]-5) or np.any(quad[:,1]<5) or np.any(quad[:,1]>=image.shape[0]-5):
            return {'status':'unknown','reason':'mapped board clipped',**info}
        return {'status':'located','board_quad':quad.tolist(),'source_to_canonical':canonical.tolist(),
                'sites':{name:cv2.perspectiveTransform(site_quad(bounds)[None],inv)[0].tolist() for name,bounds in SITES.items()},**info}


def point_in_site(point, polygon):
    return cv2.pointPolygonTest(np.float32(polygon),tuple(map(float,point)),False)>=0


def gate(detections, pose):
    """Counterfactual center gate ONLY; never creates pass/fail verdicts."""
    kept=[]; rejected=[]
    for det in detections:
        x,y,w,h=det['xywh']; name='IC' if det['class_id'] in (0,2) else 'terminal'
        (kept if pose['status']=='located' and point_in_site((x+w/2,y+h/2),pose['sites'][name]) else rejected).append(det)
    return kept,rejected


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    comparisons={r['frame_id']:r for r in json.loads(args.comparison.read_text(encoding='utf-8'))['frames']} if args.comparison else {}
    locator=ReferenceLocator(cv2.imread(str(args.reference))) if args.reference else locate
    rows=[]; panels=[]
    for item in manifest['images']:
        path=Path(item['path'])
        if digest(path)!=item['sha256']: raise ValueError('source hash differs')
        image=cv2.imread(str(path))
        if image is None: raise ValueError(f'cannot read {path}')
        start=time.perf_counter(); pose=locator(image); ms=(time.perf_counter()-start)*1000
        row={**item,'pose':pose,'localization_ms':ms}
        canvas=image.copy()
        if pose['status']=='located':
            cv2.polylines(canvas,[np.int32(pose['board_quad'])],True,(255,255,0),2)
            for name,poly in pose['sites'].items():
                cv2.polylines(canvas,[np.int32(poly)],True,(0,255,255),2)
                cv2.putText(canvas,name,tuple(np.int32(poly[0])),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
            normalized=cv2.warpPerspective(image,np.float64(pose['source_to_canonical']),SIZE)
            for name,bounds in SITES.items():
                cv2.polylines(normalized,[np.int32(site_quad(bounds))],True,(0,255,255),1)
            cv2.imwrite(str(args.output/f"canonical-{item['id']}.png"),normalized)
        if item.get('frame_id') in comparisons and item['state']=='both_missing':
            detections=comparisons[item['frame_id']]['native']; kept,rejected=gate(detections,pose)
            row['counterfactual_gate']={'kept':kept,'rejected':rejected}
            for d in detections:
                x,y,w,h=map(round,d['xywh']); color=(0,200,0) if d in kept else (0,0,255)
                cv2.rectangle(canvas,(x,y),(x+w,y+h),color,1)
        title=f"{item['id']} {item['state']} {pose['status']}"
        cv2.putText(canvas,title,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3)
        cv2.putText(canvas,title,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        cv2.imwrite(str(args.output/f"overlay-{item['id']}.jpg"),canvas)
        if pose['status']=='located':
            tile=cv2.copyMakeBorder(normalized,28,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240))
            cv2.putText(tile,item['id'],(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);panels.append(tile)
        rows.append(row)
    summary={'images':len(rows),'located':sum(r['pose']['status']=='located' for r in rows),
             'unknown':sum(r['pose']['status']!='located' for r in rows),
             'scope':'Exploratory same-capture geometry, provisional site labels, NOT labelled precision/recall or production performance',
             'manifest_sha256':digest(args.manifest),'script_sha256':digest(__file__), 'site_bounds':SITES,
             'method':'ORB/RANSAC stable-reference' if args.reference else 'blue silhouette + dark header',
             'reference_sha256':digest(args.reference) if args.reference else None,
             'state_counts':{state:sum(r['state']==state for r in rows) for state in sorted({r['state'] for r in rows})},
             'reference_is_in_sample':bool(args.reference and any(r['sha256']==digest(args.reference) for r in rows)),
             'timing_scope':'single host invocation, includes cold first frame; NOT board FPS or latency'}
    for cid in range(4):
        before=[d for r in rows for d in r.get('counterfactual_gate',{}).get('kept',[])+r.get('counterfactual_gate',{}).get('rejected',[]) if d['class_id']==cid]
        after=[d for r in rows for d in r.get('counterfactual_gate',{}).get('kept',[]) if d['class_id']==cid]
        summary[f'class_{cid}_before_after']=[len(before),len(after)]
        summary[f'class_{cid}_rejected_unknown']=sum(d['class_id']==cid for r in rows if r['pose']['status']!='located' for d in r.get('counterfactual_gate',{}).get('rejected',[]))
    if panels:
        while len(panels)%4:panels.append(np.full_like(panels[0],240))
        cv2.imwrite(str(args.output/'canonical-contact-sheet.jpg'),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)]))
    (args.output/'results.json').write_text(json.dumps({'summary':summary,'frames':rows},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--comparison',type=Path);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
