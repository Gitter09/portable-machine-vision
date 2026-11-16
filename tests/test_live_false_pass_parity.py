"""Matched-input diagnostic helpers; no native SDK, camera or model execution."""
import numpy as np
import pytest
from tools.analysis.live_false_pass_parity import anchor_index, best_match, iou, save_or_verify_arrays, source_xywh


def test_source_coordinates_remove_letterbox_and_clip():
    assert source_xywh([10,290,20,30])==[10.,10.,20.,30.]
    assert source_xywh([-5,270,20,30])==[0.,0.,15.,20.]
    assert source_xywh([1270,990,30,30])==[1270.,710.,10.,10.]


def test_iou_handles_empty_and_disjoint_boxes():
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert iou([0,0,0,10],[0,0,0,10])==0
    assert iou([0,0,10,10],[20,20,10,10])==0


def prediction():
    p=np.zeros((1,8,3),np.float32)
    p[0,:4]=np.array([[10,20,30],[300,300,300],[4,4,4],[8,8,8]])
    p[0,6]=.4
    return p


def test_anchor_is_recovered_by_geometry_when_scores_repeat():
    assert anchor_index(prediction(),{'class_id':2,'score':.4,'model_xywh':[18,296,4,8]})==1


def test_unknown_anchor_refused():
    with pytest.raises(ValueError):
        anchor_index(prediction(),{'class_id':2,'score':.4,'model_xywh':[100,100,4,8]})


def test_resume_verifies_exact_arrays_without_rewriting(tmp_path):
    p=tmp_path/'frame.npz';a={'input':np.arange(4,dtype=np.float32)}
    assert save_or_verify_arrays(p,a)=='written'
    before=p.read_bytes()
    assert save_or_verify_arrays(p,a,True)=='verified_existing'
    assert p.read_bytes()==before


@pytest.mark.parametrize('arrays',[{'other':np.arange(4)}, {'input':np.arange(5)}])
def test_resume_rejects_valid_but_different_data(tmp_path,arrays):
    p=tmp_path/'frame.npz';save_or_verify_arrays(p,{'input':np.arange(4)})
    with pytest.raises(ValueError):save_or_verify_arrays(p,arrays,True)


def test_resume_rebuilds_incomplete_archive(tmp_path):
    p=tmp_path/'frame.npz';p.write_bytes(b'PK\\x03')
    assert save_or_verify_arrays(p,{'input':np.arange(4)},True)=='written'
    with np.load(p,allow_pickle=False) as z:assert np.array_equal(z['input'],np.arange(4))


def test_no_implicit_resume(tmp_path):
    p=tmp_path/'frame.npz';save_or_verify_arrays(p,{'input':np.arange(4)})
    with pytest.raises(FileExistsError):save_or_verify_arrays(p,{'input':np.arange(4)})


def test_same_class_match_does_not_borrow_other_class():
    rows=[{'class_id':0,'xywh':[0,0,10,10]}, {'class_id':2,'xywh':[20,20,10,10]}]
    assert best_match([0,0,10,10],rows)['detection']['class_id']==0
    assert best_match([0,0,10,10],rows,2)['iou']==0
    assert best_match([0,0,10,10],[],2)['detection'] is None
