"""Provisional crop label geometry/contract. Does not approve training labels."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('training_review',Path(__file__).parents[1]/'tools/analysis/prepare_cropped_training_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_projection_is_four_corner_axis_aligned_envelope():
    matrix=np.float64([[0,-1,100],[1,0,5],[0,0,1]])
    assert m.projected_box([[10,20],[40,20],[40,60],[10,60]],matrix,200,200)==[40,15,80,45]


def test_outside_crop_not_silently_truncated():
    with pytest.raises(ValueError,match='outside'):m.projected_box([[-5,5],[10,5],[10,20],[-5,20]],np.eye(3),100,100)


def test_degenerate_box_refused():
    with pytest.raises(ValueError,match='degenerate'):m.projected_box([[1,1],[1.1,1],[1.1,1.1],[1,1.1]],np.eye(3),100,100)


def test_normalized_yolo_mapping():
    line=m.yolo_line({'class_id':3,'xyxy':[10,20,30,60]},100,100)
    cls,x,y,w,h=map(float,line.split())
    np.testing.assert_allclose([cls,x,y,w,h],[3,.2,.4,.2,.4])


@pytest.mark.parametrize('box',[{'class_id':4,'xyxy':[1,1,5,5]},{'class_id':0,'xyxy':[-1,1,5,5]},{'class_id':0,'xyxy':[5,1,1,5]},{'class_id':0,'xyxy':[1,1,101,5]}])
def test_invalid_yolo_box_refused(box):
    with pytest.raises(ValueError):m.yolo_line(box,100,100)


@pytest.mark.parametrize('present,expected',[(False,[0,3]),(True,[2,1])])
def test_class_contract_and_review_reset(tmp_path,present,expected):
    path=tmp_path/'crop.png';image=np.full((100,200,3),100,np.uint8);cv2.imwrite(str(path),image)
    source={'id':'sample','state':'both_present' if present else 'both_missing','capture_group':'capture',
            'review_status':'reviewed','reviewer':'owner','sites':{}}
    for target,cid in zip(['IC','terminal'],expected):
        source['sites'][target]={'presence':'present' if present else 'missing','legacy_class_id':cid,
                                'polygon':[[20,20],[40,20],[40,40],[20,40]]}
    r=m.make_record(source,image,path,np.eye(3),'sha')
    assert [b['class_id'] for b in r['boxes']]==expected
    assert r['review_status']=='pending' and r['reviewer'] is None and not r['training_eligible']
    assert r['split']=='development_pool_unsplit'
    np.testing.assert_array_equal(image,np.full((100,200,3),100,np.uint8))


def test_render_keeps_original_pixels_unchanged():
    image=np.full((100,200,3),100,np.uint8);before=image.copy()
    row={'id':'sample','state':'both_missing','boxes':[{'target':'IC','label':'IC_Fail','xyxy':[10,10,30,30]}]}
    view=m.render(image,row)
    assert view.shape==(390,1120,3)
    np.testing.assert_array_equal(image,before)
