"""Checked ctypes wrapper for portable-only C++ input fusion. CDLL releases GIL."""
import ctypes
from pathlib import Path
import numpy as np


class FusedInputPacker:
    def __init__(self,library,reference,kernel='neon'):
        if kernel not in ('scalar','neon','parallel2','parallel4','parallel8','parallel16'):raise ValueError('invalid C++ kernel')
        # reference is already guarded against unsupported MPK geometry/quantization.
        c=reference.contract
        if c['input_bytes']!=4915200 or c['tessellate']['slice_shape']!=[1280,64,3]:raise ValueError('unsupported input layout')
        self.library=Path(library).resolve();self.lib=ctypes.CDLL(str(self.library));self.kernel=kernel
        self.lib.portable_pack_abi.restype=ctypes.c_uint
        self.lib.portable_pack_has_neon.restype=ctypes.c_uint
        if self.lib.portable_pack_abi()!=1:raise ValueError('C++ packer ABI differs')
        if kernel=='neon' and not self.lib.portable_pack_has_neon():raise ValueError('AArch64 NEON unavailable')
        self.threads=int(kernel.removeprefix('parallel')) if kernel.startswith('parallel') else None
        if self.threads:
            self.lib.portable_pack_has_openmp.restype=ctypes.c_uint
            if not self.lib.portable_pack_has_openmp():raise ValueError('OpenMP unavailable')
        self.function=getattr(self.lib,'portable_pack_parallel' if self.threads else 'portable_pack_'+kernel)
        pointer=np.ctypeslib.ndpointer(dtype=np.uint8,flags=('C_CONTIGUOUS','ALIGNED'))
        self.function.argtypes=[pointer,ctypes.c_size_t,pointer,ctypes.c_size_t,pointer,ctypes.c_size_t];self.function.restype=ctypes.c_int
        if self.threads:self.function.argtypes=self.function.argtypes+[ctypes.c_uint]
        self.lut=np.ascontiguousarray(reference.quant.view(np.uint8));self.bytes=c['input_bytes']

    def encode_bgr(self,image):
        if image.dtype!=np.uint8 or image.shape!=(1280,1280,3) or not image.flags.c_contiguous:raise ValueError('C++ input requires contiguous1280 BGR uint8')
        output=np.empty(self.bytes,np.uint8)
        code=self.call(image,output)
        if code:raise RuntimeError('C++ input failure '+str(code))
        return output

    def call(self,image,output,input_bytes=None):
        args=[image,image.nbytes if input_bytes is None else input_bytes,self.lut,self.lut.nbytes,output,output.nbytes]
        if self.threads:args.append(self.threads)
        return self.function(*args)

    def preflight(self,reference):
        cases=[np.arange(self.bytes,dtype=np.uint32).astype(np.uint8).reshape(1280,1280,3),
               np.random.default_rng(51).integers(0,256,(1280,1280,3),dtype=np.uint8)]
        for image in cases:
            if not np.array_equal(self.encode_bgr(image),reference.encode_bgr(image)):raise ValueError('C++ packed input byte mismatch')
        return {'kernel':self.kernel,'packed_bytes_exact':True,'synthetic_cases':len(cases),'all_256_levels':True}
