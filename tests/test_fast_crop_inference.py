import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/launcher'));sys.path.insert(0,str(ROOT/'tools/analysis'))
from fast_crop_inference import rgb_tensor,decode_candidates,FastTransforms,select_crop
from portable_tensor_parity import decode_heads
from crop_geometry import detection_rows


def test_rgb_is_exact():
    image=np.random.default_rng(1).integers(0,256,(83,121,3),dtype=np.uint8)
    expected=np.ascontiguousarray(image[:,:,::-1],dtype=np.float32)/np.float32(255)
    assert np.array_equal(rgb_tensor(image),expected)


def tensors():
    rng=np.random.default_rng(4)
    return [rng.normal(0,1,(1,s,s,64)).astype(np.float32) for s in (8,4,2)]+[rng.normal(-6,2,(1,s,s,4)).astype(np.float32) for s in (8,4,2)]


@pytest.mark.parametrize('threshold',[.05,.25,.5])
def test_gate_retains_same_boxes_and_scores(threshold):
    heads=tensors();a=detection_rows(decode_heads(heads,64,'NHWC'),threshold=threshold);b=detection_rows(decode_candidates(heads,64,threshold),threshold=threshold)
    assert [r['class_id'] for r in a]==[r['class_id'] for r in b]
    assert np.allclose([r['score'] for r in a],[r['score'] for r in b],atol=0,rtol=0)
    assert np.allclose([r['model_xywh'] for r in a],[r['model_xywh'] for r in b],atol=1e-4,rtol=0)


def contract():
    heads=[]
    for i in range(6):
        g=(160,80,40)[i%3];depth=64 if i<3 else 4;t=(2,1,1)[i] if i<3 else g
        heads.append({'packed_bytes':g*g*(64 if i<3 else 16),'detessellate':{'slice_shape':[t,g,depth],
          'align_c16':True,'cblock':True,'frame_type':'int8','frame_shape':[1,g,g,depth]},'dequantize':{'channel_params':[[10.,-1]]}})
    return {'quantize':{'num_bits':8,'rounding':'TONEAREST','channel_params':[[255.12385843826655,-128]]},
      'tessellate':{'slice_shape':[1280,64,3],'align_c16':False,'cblock':False,'input_shapes':[[1,1280,1280,3]]},
      'input_bytes':4915200,'output_bytes':2688000,'heads':heads}


def test_fast_quantization_all_uint8_values():
    f=FastTransforms(contract());im=np.arange(1280*1280*3,dtype=np.uint32).astype(np.uint8).reshape(1280,1280,3)
    normalized=rgb_tensor(im);q=np.clip(np.rint(normalized.astype(np.float64)*255.12385843826655-128),-128,127).astype(np.int8)
    packed=np.ascontiguousarray(q.reshape(1280,20,64,3).transpose(1,0,2,3)).view(np.uint8).reshape(-1)
    assert np.array_equal(f.encode_bgr(im),packed)


def test_sparse_packed_predictions_equal_full_decode():
    f=FastTransforms(contract());raw=np.random.default_rng(7).integers(0,256,2688000,dtype=np.uint8)
    full=decode_heads(f.decode(raw),1280,'NHWC');mask=full[0,4:].max(axis=0)>=.05
    sparse=f.predictions(raw)
    assert np.array_equal(full[:,4:,mask],sparse[:,4:])
    assert np.allclose(full[:,:4,mask],sparse[:,:4],rtol=0,atol=.001)


def test_fast_layout_refused():
    c=contract();c['tessellate']['slice_shape']=[32,128,3]
    with pytest.raises(ValueError):FastTransforms(c)


def test_selector_unknown_preserved():
    import portable_hybrid_crop as h
    im=np.full((720,1280,3),190,np.uint8)
    assert select_crop(im)==h.select_all(im)[0]['colour_seed_canny']


def test_empty_gate():
    heads=tensors()
    for h in heads[3:]:h.fill(-80)
    assert decode_candidates(heads,64).shape==(1,8,0)
