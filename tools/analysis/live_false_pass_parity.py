"""Matched live pixels/native boxes/compiled tensors versus FP32; no tuning.

Decode mode only uses SDK software transforms on captured buffers, not inference.
Compare mode examines a missing-IC specimen. It is not labelled accuracy testing.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
if __package__:
    from .portable_tensor_parity import LABELS, SoftwareTransforms, decode_heads, detection_rows, digest
else:
    from portable_tensor_parity import LABELS, SoftwareTransforms, decode_heads, detection_rows, digest


def source_xywh(model_box):
    x,y,w,h=map(float,model_box)
    left,top=max(0.,x),max(0.,y-280)
    right,bottom=min(1280.,x+w),min(720.,y+h-280)
    return [left,top,max(0.,right-left),max(0.,bottom-top)]


def iou(a,b):
    overlap=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    union=a[2]*a[3]+b[2]*b[3]-overlap
    return overlap/union if union>0 else 0.


def anchor_index(prediction,row):
    values=prediction[0].T
    boxes=values[:,:4].copy();boxes[:,:2]-=boxes[:,2:]/2
    ids=values[:,4:].argmax(axis=1)
    valid=(ids==row['class_id']) & (np.abs(values[:,4+row['class_id']]-row['score'])<1e-6)
    distance=np.max(np.abs(boxes-np.asarray(row['model_xywh'],np.float32)),axis=1)
    distance[~valid]=np.inf
    index=int(distance.argmin())
    if distance[index]>1e-3:raise ValueError('cannot recover exact selected hardware anchor')
    return index


def best_match(box,rows,cls=None):
    candidates=[(iou(box,r['xywh']),i) for i,r in enumerate(rows) if cls is None or r['class_id']==cls]
    score,index=max(candidates,default=(0.,-1))
    return {'iou':score,'index':index,'detection':rows[index] if index>=0 else None}


def save_or_verify_arrays(path, arrays, resume=False):
    """Resume only if all existing arrays exactly reproduce current raw captures."""
    if path.exists():
        if not resume:raise FileExistsError(path)
        try:
            with np.load(path,allow_pickle=False) as saved:
                same=set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items())
        except (OSError,EOFError,ValueError,zipfile.BadZipFile):
            same=None  # incomplete archive from an interrupted write
        if same is False:raise ValueError('existing decoded arrays differ from capture: '+str(path))
        if same is True:return 'verified_existing'
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.partial',delete=False) as out:
        temporary=Path(out.name)
        np.savez_compressed(out,**arrays)
    temporary.replace(path)
    return 'written'


def decode(args):
    from ev_transforms import transforms
    sys.path.insert(0,str(args.release/'tools'))
    from support import parse_native_boxes
    inventory=json.loads((args.release/'release.json').read_text())
    for name,sha in inventory['assets'].items():
        if digest(args.release/'assets'/name)!=sha:raise ValueError('release asset differs: '+name)
    for name,sha in inventory['tool_hashes'].items():
        if digest(args.release/'tools'/name)!=sha:raise ValueError('release tool differs: '+name)
    capture=json.loads((args.capture/'buffers.json').read_text())
    if not capture.get('complete') or capture['release_inventory_sha256']!=digest(args.release/'release.json'):
        raise ValueError('capture incomplete or release differs')
    records={(r['label'],r['frame_id']):r for r in capture['records']}
    def raw(label,frame):
        r=records[label,frame];path=(args.capture/r['file']).resolve()
        if not path.is_relative_to(args.capture.resolve()) or digest(path)!=r['sha256']:
            raise ValueError('capture buffer fingerprint differs')
        return path.read_bytes()
    codec=SoftwareTransforms(json.loads((args.release/'assets/provenance/mpk.json').read_text()),transforms)
    pre=json.loads((args.release/'assets/etc/0_preproc.json').read_text())
    if [pre[k] for k in ('input_width','input_height','output_width','output_height')]!=[1280,720,1280,1280]:
        raise ValueError('this comparison requires pinned portable 1280x720 to 1280-square geometry')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    codec.check_output_layouts()
    rows=[];n=codec.contract['input_bytes'];qp=codec.contract['quantize'];tp=codec.contract['tessellate']
    for f in capture['requested_frame_ids']:
        arrays={}
        for i,head in enumerate(codec.decode(np.frombuffer(raw('simaai_process_mla',f),np.uint8).copy())):arrays[f'head_{i}']=head
        prebuffer=np.frombuffer(raw('simaai_preprocess',f),np.int8).copy()
        if prebuffer.size!=2*n:raise ValueError('preprocessor packed/companion size differs')
        plain=prebuffer[n:].reshape(qp['input_shapes'][0])
        inverse=transforms.detessellation(prebuffer[:n],frame_shape=tp['input_shapes'][0],frame_type=tp['frame_type'],**codec.tile_kwargs(tp))
        if not np.array_equal(inverse,plain):raise ValueError('compiler input layout does not match companion RGB')
        arrays['input']=transforms.dequantize(plain,channel_params=qp['channel_params'])
        name=f'frame-{f}.npz';action=save_or_verify_arrays(args.output/name,arrays,args.resume)
        source=raw('decoder',f);source_name=f'source-{f}.nv12'
        caps=records['decoder',f]['caps']
        if 'NV12' not in caps or len(source)!=1280*720*3//2:raise ValueError('unexpected source format/stride')
        source_path=args.output/source_name
        if source_path.exists() and source_path.read_bytes()!=source:raise ValueError('existing source pixels differ')
        if not source_path.exists():source_path.write_bytes(source)
        rows.append({'frame_id':f,'stream_id':records['decoder',f]['stream_id'],'data':name,'data_sha256':digest(args.output/name),
                     'source':source_name,'source_sha256':digest(args.output/source_name),'source_caps':caps,
                     'native':parse_native_boxes(raw('simaai_boxdecode',f),LABELS)})
        print('Decoded frame',f,action,flush=True)
    (args.output/'decoded.json').write_text(json.dumps({'release':str(args.release),'release_inventory_sha256':digest(args.release/'release.json'),
        'elf_sha256':inventory['assets']['share/processmla/yolov8_stage1_mla.elf'],
        'mpk_sha256':inventory['assets']['provenance/mpk.json'],'labels':LABELS,'frames':rows,
        'input_scope':'FP32 receives captured quantized RGB dequantized to float; not the pre-quantization float image',
        'selection':capture['selection'],'compiler_input_inverse_matches_companion':True},indent=2))
    shutil.copyfile(args.capture/'buffers.json',args.output/'buffers.json')


def compare(args):
    import cv2
    import onnxruntime as ort
    index=json.loads((args.decoded/'decoded.json').read_text())
    provenance=json.loads(args.compilation.read_text())
    if index['labels']!=LABELS or digest(args.raw)!=provenance['source_onnx_sha256'] or index['elf_sha256']!=provenance['artifacts']['yolov8_stage1_mla.elf'] or index['mpk_sha256']!=provenance['artifacts']['yolov8_mpk.json']:
        raise ValueError('class/model compilation lineage differs')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    original=ort.InferenceSession(str(args.original),sess_options=options,providers=['CPUExecutionProvider'])
    raw=ort.InferenceSession(str(args.raw),sess_options=options,providers=['CPUExecutionProvider'])
    if ast.literal_eval(original.get_modelmeta().custom_metadata_map['names'])!=dict(enumerate(LABELS)):
        raise ValueError('reference class metadata differs')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    def detections(pred):
        result=detection_rows(pred,.05,.45)[:100]
        for r in result:r['xywh']=source_xywh(r['model_xywh'])
        return result
    def draw(image,boxes,title):
        result=image.copy()
        for r in boxes:
            x,y,w,h=map(round,r['xywh']);color=(0,100,255) if r['class_id']==2 else (0,170,0)
            cv2.rectangle(result,(x,y),(x+w,y+h),color,1)
            cv2.putText(result,f"{LABELS[r['class_id']]} {r['score']:.3f}",(max(0,x),max(14,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.putText(result,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        return result
    for item in index['frames']:
        path=args.decoded/item['data'];source_path=args.decoded/item['source']
        if digest(path)!=item['data_sha256'] or digest(source_path)!=item['source_sha256']:raise ValueError('downloaded data fingerprint differs')
        with np.load(path,allow_pickle=False) as z:
            tensor=np.ascontiguousarray(z['input'].transpose(0,3,1,2))
            hw=decode_heads([z[f'head_{i}'] for i in range(6)],layout='NHWC')
            fp=original.run(None,{original.get_inputs()[0].name:tensor})[0]
            cut=decode_heads(raw.run(None,{raw.get_inputs()[0].name:tensor}))
        box_error=float(np.max(np.abs(fp[:,:4]-cut[:,:4])));score_error=float(np.max(np.abs(fp[:,4:]-cut[:,4:])))
        if box_error>1e-3 or score_error>1e-5:raise ValueError('FP32 raw/fused decode parity failed')
        fpd,hwd=detections(fp),detections(hw)
        native=[{'class_id':r['class_id'],'score':r['score'],'xywh':[r['native_box'][k] for k in ('x','y','width','height')]} for r in item['native']]
        row={'frame_id':item['frame_id'],'stream_id':item['stream_id'],'native':native,'fp32':fpd,'compiled_python':hwd,
             'fp32_raw_fused_max_box_error':box_error,'fp32_raw_fused_max_score_error':score_error,'native_false_pass_candidates':[]}
        parity=[]
        for n in native:
            if n['score']>=.05:
                m=best_match(n['xywh'],hwd,n['class_id'])
                parity.append({'native':n,'matched_iou':m['iou'],
                    'max_xywh_error':max(abs(a-b) for a,b in zip(n['xywh'],m['detection']['xywh'])) if m['iou']>=.8 else None,
                    'score_error':abs(n['score']-m['detection']['score']) if m['iou']>=.8 else None})
            if n['class_id']!=2:continue
            hm=best_match(n['xywh'],hwd,2);fm=best_match(n['xywh'],fpd,2)
            candidate={'native':n,'compiled_python_match':hm,'fp32_same_class_match':fm,
                       'fp32_best_any_class':best_match(n['xywh'],fpd)}
            if hm['iou']>=.8:
                a=anchor_index(hw,hm['detection']);candidate.update(anchor_index=a,
                  fp32_class_scores_same_anchor=fp[0,4:,a].tolist(),compiled_class_scores_same_anchor=hw[0,4:,a].tolist(),
                  fp32_best_class_same_anchor=int(fp[0,4:,a].argmax()),
                  fp32_same_anchor_xywh=source_xywh([float(fp[0,0,a]-fp[0,2,a]/2),float(fp[0,1,a]-fp[0,3,a]/2),float(fp[0,2,a]),float(fp[0,3,a])]))
            row['native_false_pass_candidates'].append(candidate)
        row['native_python_parity_strict_score_005']=parity
        image=cv2.cvtColor(np.fromfile(source_path,np.uint8).reshape(1080,1280),cv2.COLOR_YUV2BGR_NV12)
        f=item['frame_id'];cv2.imwrite(str(args.output/f'source-{f}.png'),image)
        panels=[draw(image,[],f'{f} source: BOTH MISSING'),draw(image,fpd,'FP32 same quantized input'),draw(image,native,'Native compiled')]
        cv2.imwrite(str(args.output/f'comparison-{f}.jpg'),np.hstack([cv2.resize(p,(640,360)) for p in panels]))
        for j,c in enumerate(row['native_false_pass_candidates']):
            x,y,w,h=c['native']['xywh'];left=max(0,min(1120,round(x+w/2-80)));top=max(0,min(560,round(y+h/2-80)))
            cv2.imwrite(str(args.output/f'false-pass-{f}-{j}.jpg'),np.hstack([cv2.resize(p[top:top+160,left:left+160],(480,480)) for p in panels]))
        rows.append(row)
        print(f,'native IC_Pass',len(row['native_false_pass_candidates']),'FP32 IC_Pass',sum(r['class_id']==2 for r in fpd),flush=True)
    summary={'sampled_frames':len(rows),'native_ic_pass_frames':sum(bool(r['native_false_pass_candidates']) for r in rows),
             'fp32_ic_pass_frames':sum(any(d['class_id']==2 for d in r['fp32']) for r in rows),
             'native_ic_pass_boxes':sum(len(r['native_false_pass_candidates']) for r in rows),
             'native_pass_boxes_matching_fp32_iou_half':sum(c['fp32_same_class_match']['iou']>=.5 for r in rows for c in r['native_false_pass_candidates']),
             'native_pass_boxes_matching_compiled_python_iou_08':sum(c['compiled_python_match']['iou']>=.8 for r in rows for c in r['native_false_pass_candidates'])}
    result={'scope':'matched sampled missing-IC frames, not labelled accuracy; no tuning','confidence':.05,'nms_iou':.45,
            'original_sha256':digest(args.original),'raw_sha256':digest(args.raw),'decoded_manifest_sha256':digest(args.decoded/'decoded.json'),
            'capture':index,'summary':summary,'frames':rows,
            'limitations':['Periodic sample is not a baseline recall estimate.','Native values below strict Python .05 may have no counterpart.',
                           'FP32 uses dequantized captured input, not pre-quantization floats.','No independently annotated inspection-site boxes.']}
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('decode');d.add_argument('--release',type=Path,required=True);d.add_argument('--capture',type=Path,required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--resume',action='store_true',help='verify and reuse exact existing arrays; rebuild incomplete archives');d.set_defaults(function=decode)
    c=sub.add_parser('compare');c.add_argument('--decoded',type=Path,required=True);c.add_argument('--original',type=Path,required=True);c.add_argument('--raw',type=Path,required=True);c.add_argument('--compilation',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.set_defaults(function=compare)
    args=p.parse_args();args.function(args)


if __name__=='__main__':main()
