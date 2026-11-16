"""Offline crop/annotation checks, not hardware/detector qualification."""
import copy
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('canny_crop',Path(__file__).parents[1]/'tools/analysis/portable_canny_crop.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def board():
    im=np.full((500,700,3),210,np.uint8)
    cv2.rectangle(im,(240,200),(420,260),(100,35,15),-1)
    for x in range(260,400,20):cv2.rectangle(im,(x,215),(x+9,240),(180,180,180),-1)
    return im


def test_blank_unknown():
    r,e=m.find_crop(np.full((500,700,3),210,np.uint8),(50,150))
    assert r['status']=='unknown' and not np.count_nonzero(e)


@pytest.mark.parametrize('angle',[0,25,80,-35])
def test_rotated_rectangle_is_found_without_labels(angle):
    im=board();aff=cv2.getRotationMatrix2D((330,230),angle,1)
    image=cv2.warpAffine(im,aff,(700,500),borderValue=(210,210,210))
    r,e=m.find_crop(image,(50,150),1)
    assert r['status']=='located'
    assert 'unresolved' in r['semantic_orientation']
    original=np.float32([[[240,200],[420,200],[420,260],[240,260]]])
    expected=cv2.transform(original,aff)[0]
    assert m.polygon_coverage(expected,r['crop_quad'])>.98
    crop,display,matrix=m.crop_pixels(image,r['crop_quad'])
    assert crop.shape[1]>crop.shape[0] and display.shape==(256,512,3)
    mapped=cv2.perspectiveTransform(np.float32([r['crop_quad']]),matrix)[0]
    np.testing.assert_allclose(mapped[0],[0,0],atol=1e-3)


def test_duplicate_edge_contours_not_distinct_ambiguity():
    r,_=m.find_crop(board(),(30,90))
    assert r['status']=='located' and r['candidate_count']==1


def test_two_rectangles_are_unknown_not_forced_choice():
    image=np.full((700,1000,3),210,np.uint8)
    image[:500,:700]=board();image[190:690,300:1000]=np.minimum(image[190:690,300:1000],board())
    r,_=m.find_crop(image,(50,150),1)
    assert r['status']=='unknown' and r['candidate_count']>=2


def test_coverage_not_intersection_over_crop():
    target=[[0,0],[10,0],[10,10],[0,10]]
    assert m.polygon_coverage(target,[[0,0],[5,0],[5,10],[0,10]])==pytest.approx(.5)
    assert m.polygon_coverage(target,[[20,0],[30,0],[30,10],[20,10]])==0


def test_invalid_close_iterations_refused():
    with pytest.raises(ValueError):m.find_crop(board(),(50,150),4)


def evidence(tmp_path):
    path=tmp_path/'images/example.png';path.parent.mkdir();cv2.imwrite(str(path),board())
    r={'id':'example','image':'images/example.png','source_image_sha256':m.digest(path),'width':700,'height':500,
       'capture_group':'one','state':'both_missing','split':'development_only_not_held_out',
       'board_quad':[[240,200],[420,200],[420,260],[240,260]],
       'sites':{'IC':{'polygon':[[300,210],[320,210],[320,240],[300,240]],'presence':'missing','legacy_class_id':0},
                'terminal':{'polygon':[[370,210],[400,210],[400,250],[370,250]],'presence':'missing','legacy_class_id':3}},
       'review_status':'pending','reviewer':None,'geometry_independently_reviewed':False}
    original={'schema':'test','class_contract':['0','1','2','3'],'board_corner_order':['a','b','c','d'],'annotations':[r]}
    review=copy.deepcopy(original);rr=review['annotations'][0];rr.update(review_status='reviewed',reviewer='human',geometry_independently_reviewed=True)
    return original,review


def test_review_validated_and_crop_evaluated_afterwards(tmp_path):
    original,review=evidence(tmp_path)
    result=m.validate_review(review,original,tmp_path)
    assert result['reviewed']==1
    r=review['annotations'][0];eval=m.evaluate([[220,180],[440,180],[440,280],[220,280]],r)
    assert eval['all_vertices_inside'] and eval['all_parts_at_least_98_percent']


@pytest.mark.parametrize('change',['pending','hash','class','split','duplicate','contract','geometry'])
def test_bad_review_refused(tmp_path,change):
    original,review=evidence(tmp_path);r=review['annotations'][0]
    if change=='pending':r['review_status']='pending'
    if change=='hash':(tmp_path/r['image']).write_bytes(b'changed')
    if change=='class':r['sites']['IC']['legacy_class_id']=2
    if change=='split':r['split']='test'
    if change=='duplicate':review['annotations']*=2
    if change=='contract':review['board_corner_order'].reverse()
    if change=='geometry':r['board_quad'][0]=[-1,0]
    with pytest.raises(ValueError):m.validate_review(review,original,tmp_path)
