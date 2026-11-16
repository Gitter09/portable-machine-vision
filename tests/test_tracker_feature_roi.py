import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/launcher'));sys.path.insert(0,str(ROOT/'tools/analysis'))
from pcb_pose_tracker import seed_features,PoseTracker


@pytest.mark.parametrize('angle',[0,17,88])
def test_ordered_features_equal(angle):
    grey=np.random.default_rng(3).integers(0,256,(720,1280),dtype=np.uint8)
    quad=cv2.boxPoints(((610,340),(210,65),angle))
    old,_=seed_features(grey,quad);new,stats=seed_features(grey,quad,True)
    assert np.array_equal(old,new)
    assert stats['feature_area_pixels']<grey.size/10


def test_halo_at_camera_edge():
    grey=np.random.default_rng(7).integers(0,256,(180,260),dtype=np.uint8)
    quad=np.float32([[1,1],[100,1],[100,50],[1,50]])
    a,_=seed_features(grey,quad);b,stats=seed_features(grey,quad,True)
    assert np.array_equal(a,b) and stats['feature_bounds_xyxy'][:2]==[0,0]


def test_featureless_seed_remains_none():
    grey=np.full((720,1280),180,np.uint8);quad=np.float32([[50,50],[250,50],[250,120],[50,120]])
    a,_=seed_features(grey,quad);b,stats=seed_features(grey,quad,True)
    assert a is None and b is None and stats['feature_count']==0


def test_roi_flag_requires_tracking_before_board_work(tmp_path):
    import subprocess
    output=tmp_path/'must-not-create'
    r=subprocess.run([sys.executable,str(ROOT/'tools/analysis/run_coordinate_browser.py'),'--output',str(output),'--feature-roi','--headless'],capture_output=True,text=True)
    assert r.returncode==2 and '--feature-roi requires --track-pose' in r.stderr
    assert not output.exists()


def test_roi_tracker_removal_clears_cached_pose():
    image=np.full((720,1280,3),190,np.uint8);cv2.rectangle(image,(480,300),(700,370),(180,50,20),-1)
    for x in range(500,680,15):cv2.rectangle(image,(x,320),(x+5,345),(220,220,220),-1)
    tracker=PoseTracker(feature_roi=True);found,_=tracker.update(image,1.)
    assert found['status']=='located' and tracker.seed_stats['feature_roi']
    found,meta=tracker.update(np.full_like(image,190),1.05)
    assert found['status']=='unknown' and tracker.found is None
