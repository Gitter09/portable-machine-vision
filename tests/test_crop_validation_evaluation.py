import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/analysis'))
from evaluate_cropped_validation import ground_truth,failure_details
from portable_labelled_eval import metrics


def box(cid=0,score=None):
    r={'class_id':cid,'xywh':[10.,20.,30.,40.]}
    if score is not None:r['score']=score
    return r


def test_ground_truth_conversion():
    assert ground_truth([{'class_id':0,'xyxy':[10,20,40,60]}])==[box()]


def test_duplicate_is_false_positive():
    gt=[box()];pred=[box(score=.9),box(score=.8)];e=failure_details(gt,pred)
    assert e['false_positive']==[1] and e['false_negative']==[]
    assert e['unmatched_prediction_overlaps'][0]['best_gt_iou']==1.
    assert metrics([{'ground_truth':gt,'p':pred}],'p')['overall']['fp']==1


def test_wrong_presence_is_not_true_positive():
    e=failure_details([box(0)],[box(2,.9)])
    assert e['false_positive']==[0] and e['false_negative']==[0]
    assert e['unmatched_prediction_overlaps'][0]['best_gt_class']==0
