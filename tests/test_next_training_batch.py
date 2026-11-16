"""Historical batch selection/leakage guards, not model accuracy."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('next_batch',Path(__file__).parents[1]/'tools/analysis/prepare_next_training_batch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_known_letterbox_unpads_pixels_without_resize():
    image=np.full((1280,1280,3),128,np.uint8);image[280:1000]=42
    report={'transform':{'mode':'letterbox','scale':1,'crop_size':1280,'pad_value':128}}
    native=m.unpad(image,report)
    assert native.shape==(720,1280,3) and np.all(native==42)
    native[0]=0;assert np.all(image[280]==42)


def test_bad_padding_refused():
    image=np.full((1280,1280,3),128,np.uint8);image[0,0]=0
    with pytest.raises(ValueError):m.unpad(image,{'transform':{'mode':'letterbox','scale':1,'crop_size':1280,'pad_value':128}})


def test_source_box_unpadding(tmp_path):
    p=tmp_path/'label.txt';p.write_text('2 0.5 0.5 0.1 0.1\n')
    box=m.source_boxes(p)[0];assert box['class_id']==2
    np.testing.assert_allclose(box['polygon'],[[576,296],[704,296],[704,424],[576,424]])


def test_label_in_padding_refused(tmp_path):
    p=tmp_path/'label.txt';p.write_text('2 0.5 0.1 0.1 0.1\n')
    with pytest.raises(ValueError):m.source_boxes(p)


def test_angle_distance_wraps():
    assert m.angle_distance(359,1)==2 and m.angle_distance(10,190)==180


def test_farthest_angles_are_deterministic_unique_and_spread():
    rows=[{'id':str(i),'angle':i} for i in range(0,360,10)]
    result=m.diverse_select(rows,4)
    assert [r['angle'] for r in result]==[0,90,180,270]
    assert result==m.diverse_select(list(reversed(rows)),4)


def test_insufficient_sources_not_duplicated():
    with pytest.raises(ValueError):m.diverse_select([{'id':'one','angle':0}],2)


def records():
    return [{'split':'train','run':'present','source_sha256':'a','model_sha256':'A'},
            {'split':'test','run':'present','source_sha256':'b','model_sha256':'B'},
            {'split':'val','run':'absent','source_sha256':'c','model_sha256':'C'},
            {'split':'train','run':'terminal_absent','source_sha256':'d','model_sha256':'D'}]


def test_only_target_training_runs_selected():
    rows=records();assert m.guard_manifest(rows)==rows[:1]


@pytest.mark.parametrize('field',['source_sha256','model_sha256'])
def test_holdout_hash_overlap_refused(field):
    rows=records();rows[0][field]=rows[1][field]
    with pytest.raises(ValueError,match='holdout'):m.guard_manifest(rows)
