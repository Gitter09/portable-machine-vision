"""Training-only augmentation label integrity and review gates."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('aug_pool',Path(__file__).parents[1]/'tools/analysis/prepare_augmented_training_pool.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def example():
    image=np.full((100,200,3),[90,70,40],np.uint8)
    labels=[{'class_id':0,'target':'IC','label':'IC_Fail','xyxy':[70,30,90,60]},
            {'class_id':1,'target':'terminal','label':'Terminal_pass','xyxy':[130,10,190,90]}]
    return image,labels


@pytest.mark.parametrize('variant',m.VARIANTS)
def test_variants_are_deterministic_in_bounds_and_preserve_classes(variant):
    image,labels=example();before=image.copy();out,bs,mat,params=m.augment(image,labels,'parent',variant)
    repeat,rbs,rmat,rparams=m.augment(image,labels,'parent',variant)
    assert out.shape==(1280,1280,3) and out.dtype==np.uint8
    np.testing.assert_array_equal(out,repeat);np.testing.assert_array_equal(mat,rmat);assert bs==rbs and params==rparams
    assert [b['class_id'] for b in bs]==[0,1]
    for b in bs:m.boxes.yolo_line(b,1280,1280)
    np.testing.assert_array_equal(image,before);assert labels==example()[1]


def test_base_exact_letterbox_and_rotate180_mapping():
    image,labels=example();base,bb,_,_=m.augment(image,labels,'x','base');rot,rb,_,_=m.augment(image,labels,'x','rotate180')
    assert np.all(base[:320]==114) and np.all(base[960:]==114)
    for a,b in zip(bb,rb):
        x0,y0,x1,y1=a['xyxy'];np.testing.assert_allclose(b['xyxy'],[1280-x1,1280-y1,1280-x0,1280-y0],atol=1e-4)


def test_cropping_target_refused_instead_of_silent_drop():
    _,labels=example()
    with pytest.raises(ValueError,match='clips'):m.transform_boxes(labels,np.float64([[1,0,-100],[0,1,0],[0,0,1]]))


def test_seed_is_parent_variant_specific():
    assert m.seed_for('a','geometry')==m.seed_for('a','geometry')
    assert len({m.seed_for(p,v) for p in ['a','b'] for v in m.VARIANTS})==10


def review_records():
    _,bs=example();r={'id':'sample','image':'images/sample.png','image_sha256':'hash','width':200,'height':100,
                     'state':'mixed','capture_group':'one','split':'train','review_status':'pending','boxes':bs}
    draft={'schema':'v1','classes':m.boxes.CLASSES,'source_review_sha256':None,'hybrid_results_sha256':'hash','annotations':[r]}
    review=copy.deepcopy(draft);review['annotations'][0].update(review_status='reviewed',reviewer='human')
    return draft,review


def test_reviewed_changes_used_instead_of_stale_draft_boxes():
    draft,review=review_records();review['annotations'][0]['boxes'][0]['xyxy'][0]+=2
    assert m.validate_review(draft,review)==1


@pytest.mark.parametrize('change',['unreviewed','source','class','duplicate'])
def test_bad_review_refused(change):
    draft,review=review_records();r=review['annotations'][0]
    if change=='unreviewed':r['review_status']='pending'
    if change=='source':r['image_sha256']='changed'
    if change=='class':r['boxes'][0]['class_id']=2
    if change=='duplicate':review['annotations']*=2
    with pytest.raises(ValueError):m.validate_review(draft,review)


def test_verbal_approval_bound_to_exact_draft(tmp_path):
    path=tmp_path/'first.json';path.write_text('original')
    approval={'draft_sha256':m.base.digest(path),'images':42,'boxes':84,'approval_type':'owner verbal approval of unchanged first batch'}
    m.check_verbal_approval(path,approval);path.write_text('changed')
    with pytest.raises(ValueError):m.check_verbal_approval(path,approval)
