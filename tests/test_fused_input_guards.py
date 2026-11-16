"""Host guards; actual native bytes/ABI/buffer guards are exercised on board."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/launcher'))
from fused_input import FusedInputPacker
import fused_input


def test_unknown_kernel_refused_before_loading():
    with pytest.raises(ValueError,match='kernel'):FusedInputPacker('unused.so',None,'unknown')


def test_wrong_layout_refused_before_loading():
    ref=SimpleNamespace(contract={'input_bytes':0,'tessellate':{'slice_shape':[32,128,3]}})
    with pytest.raises(ValueError,match='layout'):FusedInputPacker('unused.so',ref,'scalar')


@pytest.mark.parametrize('image',[np.zeros((10,10,3),np.uint8),np.zeros((1280,1280,3),np.float32),np.zeros((1280,1280,3),np.uint8)[:,::-1]])
def test_bad_image_refused_before_native_call(image):
    obj=object.__new__(FusedInputPacker)
    with pytest.raises(ValueError,match='contiguous'):obj.encode_bgr(image)


def test_wrong_abi_refused(monkeypatch):
    class Function:
        def __call__(self):return 99
    lib=SimpleNamespace(portable_pack_abi=Function(),portable_pack_has_neon=Function())
    monkeypatch.setattr(fused_input.ctypes,'CDLL',lambda path:lib)
    ref=SimpleNamespace(contract={'input_bytes':4915200,'tessellate':{'slice_shape':[1280,64,3]}})
    with pytest.raises(ValueError,match='ABI'):FusedInputPacker('unused.so',ref,'scalar')
