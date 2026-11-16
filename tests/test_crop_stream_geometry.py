import importlib.util
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/analysis'))
from prepare_augmented_training_pool import letterbox as training_letterbox
from portable_tensor_parity import detection_rows as diagnostic_rows
spec=importlib.util.spec_from_file_location('crop_geometry',ROOT/'runtime/launcher/crop_geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


@pytest.mark.parametrize('shape',[(93,201),(111,309),(220,80)])
def test_training_serving_letterbox_agree(shape):
    image=np.random.default_rng(3).integers(0,256,(*shape,3),dtype=np.uint8)
    a,m=g.letterbox(image);b,n=training_letterbox(image)
    assert np.array_equal(a,b) and np.array_equal(m,n)


def test_inverse_source_mapping():
    m=np.float64([[2,0,20],[0,3,100],[0,0,1]])
    q=g.source_quad([40,130,20,60],np.linalg.inv(m))
    assert np.allclose(q,[[10,10],[20,10],[20,30],[10,30]])


def test_class_aware_opencv46_nms_matches_reference():
    a=np.zeros((1,8,4),np.float32)
    a[0,:4,:]=np.array([[100,100,100,400],[100,100,100,400],[30,30,30,30],[30,30,30,30]])
    a[0,4,:]=[.8,.7,0,0];a[0,5,2]=.9;a[0,7,3]=.4
    ours=g.detection_rows(a);reference=diagnostic_rows(a,threshold=.05,iou=.45)
    assert [(r['class_id'],r['score'],r['model_xywh']) for r in ours]==[(r['class_id'],r['score'],r['model_xywh']) for r in reference]


def test_explicit_private_receiver_endpoints():
    assert g.approved_receiver('172.21.30.13',5010)
    assert g.approved_receiver('172.21.30.230',5010)
    assert not g.approved_receiver('172.21.30.230',5011)
    assert g.approved_receiver('192.168.1.30',5010)
    assert g.approved_receiver('169.254.1.2',5010)
    assert not g.approved_receiver('8.8.8.8',5010)
    assert not g.approved_receiver('127.0.0.1',5010)


def test_empty_nms():
    assert g.detection_rows(np.zeros((1,8,2),np.float32))==[]
