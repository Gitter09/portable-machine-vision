"""Camera-free on-board C++ byte parity and interleaved timing benchmark."""
import argparse,json,platform,statistics,time
from pathlib import Path
import numpy as np
from portable_tensor_parity import transform_contract,digest
from fast_crop_inference import FastTransforms,rgb_tensor
from fused_input import FusedInputPacker


def run(args):
    if args.output.exists():raise FileExistsError('new benchmark directory required')
    reference=FastTransforms(transform_contract(json.loads(args.mpk.read_text())))
    kernels={k:FusedInputPacker(args.library,reference,k) for k in ('scalar','neon','parallel2','parallel4','parallel8','parallel16')}
    preflight={k:v.preflight(reference) for k,v in kernels.items()}
    manifest=json.loads(args.inputs.with_suffix('.json').read_text())
    if digest(args.inputs)!=manifest['inputs_sha256']:raise ValueError('prepared input hash differs')
    cases=[];parity=[]
    with np.load(args.inputs,allow_pickle=False) as arrays:
        for row in manifest['images']:
            rgb=arrays[row['key']];image=np.ascontiguousarray(np.rint(rgb*255).astype(np.uint8)[:,:,::-1])
            if not np.array_equal(rgb_tensor(image),rgb):raise ValueError('uint8 source reconstruction not exact')
            expected=reference.encode_bgr(image)
            for name,kernel in kernels.items():
                actual=kernel.encode_bgr(image)
                if not np.array_equal(actual,expected):raise ValueError('packed bytes differ '+row['key']+' '+name)
                guarded=np.full(kernel.bytes+128,165,np.uint8);out=guarded[64:-64]
                if kernel.call(image,out)!=0 or not np.array_equal(out,expected):raise ValueError('guarded result mismatch')
                if np.any(guarded[:64]!=165) or np.any(guarded[-64:]!=165):raise ValueError('C++ buffer overrun')
                if kernel.call(image,out,input_bytes=0)!=-1:raise ValueError('C++ invalid-length guard missing')
            parity.append({'key':row['key'],'packed_sha256':__import__('hashlib').sha256(expected.tobytes()).hexdigest(),'both_kernels_exact':True})
            if len(cases)<3:cases.append(image)
    methods={'opencv_numpy':reference.encode_bgr,**{k:v.encode_bgr for k,v in kernels.items()}};samples={k:[] for k in methods};names=list(methods)
    for i in range(args.repeats+5):
        image=cases[i%len(cases)]
        for name in names[i%len(names):]+names[:i%len(names)]:
            start=time.perf_counter();methods[name](image);elapsed=1000*(time.perf_counter()-start)
            if i>=5:samples[name].append(elapsed)
    result={'scope':'camera-free same prepared frames; byte parity then interleaved microbenchmark, not live FPS',
        'library_sha256':digest(args.library),'mpk_sha256':digest(args.mpk),'inputs_sha256':digest(args.inputs),'machine':platform.machine(),
        'preflight':preflight,'parity':parity,'iterations':args.repeats,
        'timings_ms':{k:{'mean':statistics.mean(v),'median':statistics.median(v),'p95':float(np.percentile(v,95))} for k,v in samples.items()}}
    args.output.mkdir();(args.output/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['timings_ms'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mpk','library','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=60);run(p.parse_args())
