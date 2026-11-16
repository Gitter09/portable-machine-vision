"""FP32 review preparation/render helpers, not inference accuracy tests."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('crop_detection_review',Path(__file__).parents[1]/'tools/analysis/portable_crop_detection_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_preparation_is_rgb_float32_and_centered():
    image=np.full((3,5,3),[10,20,30],np.uint8);before=image.copy()
    tensor,meta=m.prepare(image,32)
    assert tensor.shape==(1,3,32,32) and tensor.dtype==np.float32 and tensor.flags.c_contiguous
    assert meta['resized_size']==[32,19] and meta['padding_left_top']==[0,6]
    np.testing.assert_allclose(tensor[0,:,6,0],np.array([30,20,10])/255,atol=1e-7)
    np.testing.assert_allclose(tensor[0,:,0,0],114/255,atol=1e-7)
    assert hashlib.sha256(tensor.tobytes()).hexdigest()==meta['tensor_sha256']
    np.testing.assert_array_equal(image,before)


def test_rounded_xy_scale_mapping_roundtrip():
    _,meta=m.prepare(np.zeros((3,5,3),np.uint8),32);sx,sy=meta['scale_xy'];left,top=meta['padding_left_top']
    pred=np.zeros((1,8,1),np.float32);pred[0,:4,0]=[2*sx+left,sy+top,2*sx,sy];pred[0,4,0]=.9
    result=m.image_boxes(pred,meta)
    assert len(result)==1 and result[0]['class_id']==0
    np.testing.assert_allclose(result[0]['xywh'],[1,.5,2,1],atol=1e-6)


def test_class_aware_nms_and_fixed_diagnostic_threshold():
    meta={'scale_xy':[1,1],'padding_left_top':[0,0],'original_size':[100,100]}
    pred=np.zeros((1,8,3),np.float32);pred[0,:4,:]=np.array([[20,20,40],[20,20,40],[8,8,8],[8,8,8]])
    pred[0,4,0]=.9;pred[0,5,1]=.8;pred[0,6,2]=.049
    result=m.image_boxes(pred,meta)
    assert [r['class_id'] for r in result]==[0,1]
    assert m.CONFIDENCE==.05 and m.NMS_IOU==.45 and m.TOPK==100


def test_model_padding_only_detection_not_exported():
    meta={'scale_xy':[1,1],'padding_left_top':[0,50],'original_size':[100,20]}
    pred=np.zeros((1,8,1),np.float32);pred[0,:4,0]=[20,10,8,8];pred[0,4,0]=.9
    assert m.image_boxes(pred,meta)==[]


def test_empty_predictions_remain_empty():
    meta={'scale_xy':[1,1],'padding_left_top':[0,0],'original_size':[100,100]}
    assert m.image_boxes(np.zeros((1,8,2),np.float32),meta)==[]


def test_crop_projection_keeps_all_four_corners():
    boxes=[{'class_id':0,'label':'IC_Fail','score':.7,'xywh':[10,20,30,40]}]
    transform=np.float64([[0,-1,100],[1,0,5],[0,0,1]])
    result=m.transform_polygons(boxes,transform)
    np.testing.assert_allclose(result[0]['polygon'],[[80,15],[80,45],[40,45],[40,15]])
    reverse=np.linalg.inv(transform)
    import cv2
    recovered=cv2.perspectiveTransform(np.float32([result[0]['polygon']]),reverse)[0]
    np.testing.assert_allclose(recovered,m.box_polygon(boxes[0]),atol=1e-6)


@pytest.mark.parametrize('size',[None,(1280,640)])
def test_render_does_not_mutate_pixels_or_invent_boxes(size):
    image=np.full((120,300,3),80,np.uint8);before=image.copy()
    canvas=m.draw(image,[],'No detections',size=size)
    assert canvas.shape[:2]==((162,300) if size is None else (682,1280))
    np.testing.assert_array_equal(image,before)


def test_display_can_handle_outside_crop_boxes():
    image=np.full((120,300,3),80,np.uint8)
    boxes=[{'class_id':3,'label':'Terminal_Fail','score':.1,'xywh':[1000,1000,20,20]}]
    np.testing.assert_array_equal(m.draw(image,boxes,'same'),m.draw(image,[],'same'))
