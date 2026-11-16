"""Bounded optical-flow pose reuse; confidence failure clears state/relocalizes.
Tracking is geometry only, never evidence of component presence.
"""
import time
import cv2
import numpy as np
from fast_crop_inference import select_crop
import portable_hybrid_crop as hybrid


def seed_features(grey,quad,local_roi=False):
    """Same mask/detector parameters; optional ROI includes an8px filter halo."""
    start=time.perf_counter();mask=np.zeros(grey.shape,np.uint8)
    polygon=np.int32(quad);cv2.fillConvexPoly(mask,polygon,255)
    x0=y0=0;y1,x1=grey.shape
    if local_roi:
        x,y,w,h=cv2.boundingRect(polygon)
        x0=max(0,x-8);y0=max(0,y-8);x1=min(x1,x+w+8);y1=min(y1,y+h+8)
    points=None
    if x1>x0 and y1>y0:
        points=cv2.goodFeaturesToTrack(grey[y0:y1,x0:x1],maxCorners=80,qualityLevel=.01,minDistance=3,mask=mask[y0:y1,x0:x1],blockSize=3)
        if points is not None and local_roi:points+=np.float32([x0,y0])
    return points,{'feature_seed_ms':1000*(time.perf_counter()-start),'feature_roi':local_roi,
        'feature_area_pixels':max(0,x1-x0)*max(0,y1-y0),'feature_full_pixels':grey.size,
        'feature_count':0 if points is None else len(points),'feature_bounds_xyxy':[x0,y0,x1,y1]}


class PoseTracker:
    def __init__(self,refresh_seconds=.5,feature_roi=False):
        self.refresh_seconds=refresh_seconds;self.feature_roi=feature_roi;self.clear();self.full=0;self.tracked=0;self.failures=0

    def clear(self):
        self.grey=None;self.points=None;self.found=None;self.refreshed=None;self.previous_time=None;self.seed_stats={}

    def seed(self,image,grey,found,now):
        points,stats=seed_features(grey,found['seed_quad'],self.feature_roi)
        if points is not None and len(points)>=8:
            self.grey=grey;self.points=points;self.found=found;self.refreshed=now;self.previous_time=now
        else:self.clear()
        self.seed_stats=stats

    def update(self,image,now):
        grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);reason='initial_or_periodic'
        if self.found is not None and 0<now-self.previous_time<=.25 and now-self.refreshed<self.refresh_seconds:
            proposed,forward,_=cv2.calcOpticalFlowPyrLK(self.grey,grey,self.points,None,winSize=(21,21),maxLevel=3)
            if proposed is not None:
                back,backward,_=cv2.calcOpticalFlowPyrLK(grey,self.grey,proposed,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    good=(forward.ravel()>0)&(backward.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<=1.)
                    old=self.points[good].reshape(-1,2);new=proposed[good].reshape(-1,2)
                    if len(old)>=8 and np.mean(good)>=.6:
                        matrix,inliers=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=300,confidence=.99,refineIters=10)
                        if matrix is not None and inliers is not None:
                            ratio=float(np.mean(inliers));scale=float(np.hypot(matrix[0,0],matrix[1,0]));angle=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
                            fitted=cv2.transform(old[None],matrix)[0];error=float(np.median(np.linalg.norm(fitted-new,axis=1)))
                            found=dict(self.found)
                            for key in ('crop_quad','seed_quad'):found[key]=cv2.transform(np.float32([found[key]]),matrix)[0].tolist()
                            shift=float(np.linalg.norm(np.mean(found['seed_quad'],axis=0)-np.mean(self.found['seed_quad'],axis=0)))
                            blue=hybrid.appearance(hybrid.blue_mask(image),found['seed_quad'])
                            if (ratio>=.75 and error<=1.5 and .95<=scale<=1.05 and abs(angle)<=25 and shift<=80 and
                                hybrid.valid_source_quad(found['crop_quad'],image.shape) and blue['blue_pixels']>=500 and blue['blue_fraction']>=.15):
                                self.grey=grey;self.points=new[inliers.ravel()>0].reshape(-1,1,2);self.found=found;self.previous_time=now;self.tracked+=1
                                return found,{'source':'tracked','inlier_ratio':ratio,'median_error_px':error,'age_since_full_seconds':now-self.refreshed}
            self.failures+=1;reason='tracking_unreliable'
        self.clear();start=time.perf_counter();found=select_crop(image);localize_ms=1000*(time.perf_counter()-start);self.full+=1
        if found['status']=='located':self.seed(image,grey,found,now)
        return found,{'source':'full' if found['status']=='located' else 'unknown','reason':reason,'age_since_full_seconds':0.,
            'full_localize_ms':localize_ms,**self.seed_stats}
