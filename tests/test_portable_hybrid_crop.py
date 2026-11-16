"""Hybrid crop helpers and controls; no real-world accuracy qualification."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('hybrid_crop',Path(__file__).parents[1]/'tools/analysis/portable_hybrid_crop.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def scene(green=False):
    image=np.full((500,700,3),210,np.uint8)
    cv2.rectangle(image,(240,200),(420,260),(110,35,15),-1)
    for x in range(270,400,20):cv2.rectangle(image,(x,215),(x+7,240),(170,170,170),-1)
    if green:cv2.rectangle(image,(210,180),(255,275),(45,120,40),-1)
    return image


def test_blank_and_neutral_bracket_refused():
    image=np.full((500,700,3),210,np.uint8);cv2.rectangle(image,(240,200),(420,260),(70,70,70),-1)
    results,mask,edges=m.select_all(image)
    assert not mask.any() and edges.any()
    assert all(r['status']=='unknown' for r in results.values())


@pytest.mark.parametrize('angle',[0,25,80,-35])
def test_hybrid_rotated_substrate_preserved(angle):
    image=scene();aff=cv2.getRotationMatrix2D((330,230),angle,1)
    rotated=cv2.warpAffine(image,aff,(700,500),borderValue=(210,210,210));results,_,_=m.select_all(rotated)
    result=results['colour_seed_canny'];assert result['status']=='located'
    expected=cv2.transform(np.float32([[[240,200],[420,200],[420,260],[240,260]]]),aff)[0]
    assert m.base.polygon_coverage(expected,result['crop_quad'])>.98
    assert m.base.polygon_coverage(result['seed_quad'],result['crop_quad'])>.999


def test_local_edges_recover_green_component_overhang():
    results,_,_=m.select_all(scene(green=True));target=[[210,180],[255,180],[255,275],[210,275]]
    colour=results['colour_group_only'];hybrid=results['colour_seed_canny']
    assert colour['status']==hybrid['status']=='located'
    assert m.base.polygon_coverage(target,colour['crop_quad'])<.98
    assert m.base.polygon_coverage(target,hybrid['crop_quad'])>=.98


def test_close_groups_small_gap_without_label_geometry():
    image=scene();image[195:266,325:333]=210
    seeds=m.colour_seeds(m.blue_mask(image))
    assert len(seeds)==1 and seeds[0]['blue_pixels']>=500


def test_multiple_substrates_are_unknown():
    image=np.full((700,1000,3),210,np.uint8)
    cv2.rectangle(image,(120,140),(300,200),(110,35,15),-1)
    cv2.rectangle(image,(570,420),(750,480),(110,35,15),-1)
    results,_,_=m.select_all(image)
    assert all(r['status']=='unknown' for r in results.values())


def test_insufficient_local_edge_support_unknown():
    mask=m.blue_mask(scene());seeds=m.colour_seeds(mask)
    result=m.from_seed(seeds,np.zeros_like(mask),mask.shape,True)
    assert result['status']=='unknown' and result['local_edge_pixels']==0


def test_truncated_canny_candidates_not_silently_accepted():
    result=m.choose_canny([],m.blue_mask(scene()),scene().shape,total=6)
    assert result['status']=='unknown' and 'truncated' in result['reason']


def test_selection_does_not_mutate_source():
    image=scene();before=image.copy();m.select_all(image)
    np.testing.assert_array_equal(image,before)


def test_synthetic_removal_is_separate_and_preserves_original():
    image=scene();before=image.copy()
    row={'board_quad':[[240,200],[420,200],[420,260],[240,260]],'sites':{}}
    negative=m.synthetic_without_board(image,row)
    np.testing.assert_array_equal(image,before)
    assert not m.blue_mask(negative).any()
    assert all(r['status']=='unknown' for r in m.select_all(negative)[0].values())


def test_crop_bounds_guard():
    assert m.valid_source_quad([[1,1],[10,1],[10,10],[1,10]],(20,20))
    assert not m.valid_source_quad([[-1,1],[10,1],[10,10],[1,10]],(20,20))


def test_coverage_summary_counts_unknown_separately():
    stats=m.summary([{'state':'missing','status':'unknown'},
                     {'state':'missing','status':'located','evaluation':{'all_parts_at_least_98_percent':False,'all_vertices_inside':False,'crop_area_fraction':.04}},
                     {'state':'missing','status':'located','evaluation':{'all_parts_at_least_98_percent':True,'all_vertices_inside':True,'crop_area_fraction':.03}}])
    assert stats['preserved_98_percent']==stats['unsafe_selected']==stats['unknown']==1
    assert stats['safe_median_crop_area_fraction']==.03
