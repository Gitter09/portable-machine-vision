"""Lossless candidate optimizations; frozen thresholds/model/crop unchanged.
Gate DFL work by the SAME class probabilities used by downstream thresholding.
No candidate that could pass that threshold is removed.
"""
import cv2
import numpy as np


class FastTransforms:
    """Exact uint8 LUT quantization/dequantization and known MPK tile inverses.

    Refuses other layouts. Installed SDK remains unchanged; preflight compares
    complete packed bytes and all dequantized tensor elements with that SDK.
    """
    def __init__(self,contract):
        self.contract=contract;q=contract['quantize'];t=contract['tessellate']
        if (q['num_bits']!=8 or q['rounding']!='TONEAREST' or len(q['channel_params'])!=1 or
            t['slice_shape']!=[1280,64,3] or t['align_c16'] or t['cblock'] or
            t['input_shapes']!=[[1,1280,1280,3]] or contract['input_bytes']!=4915200):raise ValueError('unsupported fast input contract')
        scale,zp=q['channel_params'][0]
        values=(np.arange(256,dtype=np.float32)/np.float32(255)).astype(np.float64)
        self.quant=np.clip(np.rint(values*scale+zp),-128,127).astype(np.int8)
        self.tables=[]
        for i,h in enumerate(contract['heads']):
            d=h['detessellate'];grid=(160,80,40)[i%3];depth=64 if i<3 else 4;tile=(2,1,1)[i] if i<3 else grid
            if (d['frame_type']!='int8' or not d['align_c16'] or not d['cblock'] or d['frame_shape']!=[1,grid,grid,depth] or
                d['slice_shape']!=[tile,grid,depth] or len(h['dequantize']['channel_params'])!=1 or
                h['packed_bytes']!=grid*grid*(64 if i<3 else 16)):raise ValueError('unsupported fast output contract')
            scale,zp=h['dequantize']['channel_params'][0]
            self.tables.append(((np.arange(256,dtype=np.uint8).view(np.int8).astype(np.float64)-zp)/scale).astype(np.float32))
        if len(self.tables)!=6 or contract['output_bytes']!=2688000:raise ValueError('unsupported fast head count/size')

    def encode_bgr(self,image):
        if image.shape!=(1280,1280,3) or image.dtype!=np.uint8:raise ValueError('fast input requires prepared uint8 BGR')
        quantized=cv2.LUT(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),self.quant)
        return np.ascontiguousarray(quantized.reshape(1280,20,64,3).transpose(1,0,2,3)).view(np.uint8).reshape(-1)

    def decode(self,raw):
        data=np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
        if data.nbytes!=self.contract['output_bytes']:raise ValueError('raw MLA bytes differ')
        result=[];offset=0
        for i,h in enumerate(self.contract['heads']):
            grid=(160,80,40)[i%3];block=data[offset:offset+h['packed_bytes']];offset+=h['packed_bytes']
            if i<3:
                tile=(2,1,1)[i];frame=block.reshape(grid//tile,4,tile,grid,16).transpose(0,2,3,1,4).reshape(1,grid,grid,64)
            else:frame=block.reshape(1,grid,grid,16)[:,:,:,:4]
            depth=64 if i<3 else 4
            result.append(cv2.LUT(frame.reshape(grid,grid,depth),self.tables[i]).reshape(1,grid,grid,depth))
        return result

    def predictions(self,raw,threshold=.05):
        """Score LUT first, then unpack/dequantize only retained box anchors."""
        if not 0<threshold<1:raise ValueError('invalid threshold')
        data=np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
        if data.nbytes!=self.contract['output_bytes']:raise ValueError('raw MLA bytes differ')
        offsets=np.cumsum([0]+[h['packed_bytes'] for h in self.contract['heads']]);predictions=[]
        for i,stride in enumerate((8,16,32)):
            grid=(160,80,40)[i];tile=(2,1,1)[i]
            classes=data[offsets[i+3]:offsets[i+4]].reshape(grid,grid,16)[:,:,:4]
            lut=1/(1+np.exp(-np.clip(self.tables[i+3],-80,80)))
            scores=cv2.LUT(classes,lut).reshape(-1,4)
            indexes=np.flatnonzero(scores.max(axis=1)>=threshold)
            if not len(indexes):continue
            blocks=data[offsets[i]:offsets[i+1]].reshape(grid//tile,4,tile,grid,16)
            selected=blocks[indexes//(grid*tile),:,(indexes//grid)%tile,indexes%grid,:]
            logits=cv2.LUT(selected,self.tables[i]).transpose(1,2,0)[None]
            probability=np.exp(logits-logits.max(axis=2,keepdims=True));probability/=probability.sum(axis=2,keepdims=True)
            distance=(probability*np.arange(16,dtype=np.float32)[None,None,:,None]).sum(axis=2)
            anchor=np.stack((indexes%grid+.5,indexes//grid+.5)).reshape(1,2,-1).astype(np.float32)
            lower,upper=anchor-distance[:,:2],anchor+distance[:,2:]
            boxes=np.concatenate(((lower+upper)/2,upper-lower),axis=1)*stride
            predictions.append(np.concatenate((boxes,scores[indexes].T[None]),axis=1))
        return np.concatenate(predictions,axis=2) if predictions else np.empty((1,8,0),np.float32)

    def preflight(self,reference):
        image=np.arange(1280*1280*3,dtype=np.uint32).astype(np.uint8).reshape(1280,1280,3)
        if not np.array_equal(self.encode_bgr(image),reference.encode(rgb_tensor(image))):raise ValueError('fast packed input parity failed')
        raw=np.random.default_rng(42).integers(0,256,self.contract['output_bytes'],dtype=np.uint8)
        reference_heads=reference.decode(raw)
        if not all(np.array_equal(a,b) for a,b in zip(self.decode(raw),reference_heads)):raise ValueError('fast output parity failed')
        from portable_tensor_parity import decode_heads
        full=decode_heads(reference_heads,1280,'NHWC');expected=full[:,:,full[0,4:].max(axis=0)>=.05];actual=self.predictions(raw,.05)
        if actual.shape!=expected.shape or not np.array_equal(actual[:,4:],expected[:,4:]) or not np.allclose(actual[:,:4],expected[:,:4],atol=.001,rtol=0):raise ValueError('fast sparse decoder parity failed')
        return {'packed_bytes_exact':True,'all_256_input_levels':True,'six_dequantized_heads_exact':True,
                'sparse_candidate_scores_exact':True,'max_box_error_px':float(np.max(np.abs(actual[:,:4]-expected[:,:4]))),
                'synthetic_candidate_count':actual.shape[2]}
        from portable_tensor_parity import decode_heads
        expected=decode_heads(reference.decode(raw),1280,'NHWC');mask=expected[0,4:].max(axis=0)>=.05;expected=expected[:,:,mask]
        actual=self.predictions(raw)
        if actual.shape!=expected.shape or not np.array_equal(actual[:,4:],expected[:,4:]) or not np.allclose(actual[:,:4],expected[:,:4],rtol=0,atol=.001):raise ValueError('sparse packed DFL parity failed')


def select_crop(image):
    """Same frozen colour-seed/Canny result, omitting unused ablation branches."""
    import portable_hybrid_crop as hybrid
    mask=hybrid.blue_mask(image)
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(5,5),1.2)
    edges=cv2.Canny(blur,30,90,L2gradient=True)
    return hybrid.from_seed(hybrid.colour_seeds(mask),edges,image.shape,True)


def rgb_tensor(image):
    result=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
    np.divide(result,np.float32(255),out=result)
    return result


def decode_candidates(heads,side=1280,threshold=.05):
    if len(heads)!=6 or side<32 or side%32 or not 0<threshold<1:raise ValueError('invalid head/side/threshold contract')
    for i,a in enumerate(heads):
        g=side//(8,16,32)[i%3];depth=64 if i<3 else 4
        if a.shape!=(1,g,g,depth) or not np.issubdtype(a.dtype,np.floating) or not np.isfinite(a).all():raise ValueError('invalid finite NHWC head')
    predictions=[]
    for i,stride in enumerate((8,16,32)):
        grid=side//stride
        scores=1/(1+np.exp(-np.clip(heads[i+3].reshape(-1,4),-80,80)))
        indexes=np.flatnonzero(scores.max(axis=1)>=threshold)
        if not len(indexes):continue
        logits=heads[i].reshape(-1,4,16)[indexes].transpose(1,2,0)[None].astype(np.float32)
        probability=np.exp(logits-logits.max(axis=2,keepdims=True));probability/=probability.sum(axis=2,keepdims=True)
        distance=(probability*np.arange(16,dtype=np.float32)[None,None,:,None]).sum(axis=2)
        anchor=np.stack((indexes%grid+.5,indexes//grid+.5)).reshape(1,2,-1).astype(np.float32)
        lower,upper=anchor-distance[:,:2],anchor+distance[:,2:]
        boxes=np.concatenate(((lower+upper)/2,upper-lower),axis=1)*stride
        predictions.append(np.concatenate((boxes,scores[indexes].T[None]),axis=1))
    return np.concatenate(predictions,axis=2) if predictions else np.empty((1,8,0),np.float32)
