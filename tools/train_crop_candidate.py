"""Owner-authorized train-now -> export -> compile candidate, no live deployment.

Training is fixed-length without held-out validation; no train-as-val substitution
or validation-selected checkpoint. Uses final last.pt. Validation can be prepared
independently while GPU training runs. Installed Ultralytics/SDK are not edited.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools/analysis'))
from portable_tensor_parity import LABELS, digest, decode_heads


def status(root,stage,**extra):
    path=root/'status.json';tmp=root/'status.tmp.json'
    tmp.write_text(json.dumps({'stage':stage,'pid':os.getpid(),'updated_unix':time.time(),**extra},indent=2),encoding='utf-8');os.replace(tmp,path)


class EmptyValidationDataset:
    labels=[]
    def __len__(self):return 0


class EmptyValidationLoader:
    dataset=EmptyValidationDataset()
    num_workers=0
    def __len__(self):return 0
    def __iter__(self):return iter(())
    def reset(self):pass
    def close(self):pass


def make_trainer():
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    class TrainWithoutValidation(DetectionTrainer):
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode=='val':
                if dataset_path is not None:raise ValueError('this run must not consume validation or train-as-val')
                return EmptyValidationLoader()
            return super().get_dataloader(dataset_path,batch_size,rank,mode)
        def get_validator(self):
            return SimpleNamespace(metrics=SimpleNamespace(keys=[]))
        def validate(self):
            print('Held-out validation deferred by owner; no metrics/fitness fabricated.',flush=True)
            return {},None
        def final_eval(self):
            print('Final held-out evaluation deferred; export FINAL last.pt, not validation-selected best.pt.',flush=True)
    return TrainWithoutValidation


def train(args):
    import torch
    import ultralytics
    if ultralytics.__version__!='8.4.144':raise ValueError('trainer API version differs from inspected version')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; do not silently train on CPU')
    manifest=json.loads((args.pool/'manifest.json').read_text(encoding='utf-8'))
    if manifest['classes']!=LABELS or manifest['summary']['training_images']!=450:raise ValueError('dataset contract differs')
    for r in manifest['images']:
        if r['split']!='train' or digest(args.pool/r['image'])!=r['image_sha256'] or digest(args.pool/r['label'])!=r['label_sha256']:
            raise ValueError('training input hash/split differs')
    if args.output.exists():raise FileExistsError('candidate directory exists; no overwrite')
    args.output.mkdir(parents=True)
    data={'path':str(args.pool.resolve()).replace('\\','/'),'train':'images/train','val':None,'names':dict(enumerate(LABELS)),'nc':4}
    import yaml
    (args.output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    config={'model':str(args.init.resolve()),'data':str((args.output/'data.yaml').resolve()),'epochs':args.epochs,'batch':8,'imgsz':1280,
            'device':0,'workers':0,'seed':42,'deterministic':True,'project':str(args.output.resolve()),'name':'train','exist_ok':False,
            'val':False,'patience':0,'save':True,'save_period':10,'plots':False,'amp':False,'cache':False,
            'optimizer':'AdamW','lr0':.001,'lrf':.01,'cos_lr':True,'warmup_epochs':3.,'warmup_bias_lr':0.,'weight_decay':.0005,
            'mosaic':0.,'mixup':0.,'copy_paste':0.,'degrees':0.,'translate':0.,'scale':0.,'shear':0.,'perspective':0.,
            'fliplr':0.,'flipud':0.,'hsv_h':0.,'hsv_s':0.,'hsv_v':0.,'close_mosaic':0,'rect':False,'multi_scale':0.,'verbose':True}
    record={'owner_authorization':'Train now, compile after training; validation preparation in parallel, not a prerequisite.',
            'init_sha256':digest(args.init),'dataset_manifest_sha256':digest(args.pool/'manifest.json'),'config':config,
            'checkpoint_selection':'fixed final epoch last.pt; NOT validation-selected','ultralytics':ultralytics.__version__,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'script_sha256':digest(Path(__file__)),
            'compile_queued':not args.no_compile,'model_contract':LABELS}
    (args.output/'job.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    shutil.copyfile(__file__,args.output/'runner-source.py')
    status(args.output,'training',epochs=args.epochs,validation='deferred; no val dataset or fabricated scores')
    trainer=make_trainer()(overrides=config)
    trainer.train()
    checkpoint=Path(trainer.last)
    if not checkpoint.is_file():raise RuntimeError('training finished without final checkpoint')
    status(args.output,'training_complete',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
    return checkpoint


def export_and_compile(args,checkpoint):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO
    out=args.output/'export';out.mkdir()
    status(args.output,'exporting',checkpoint=str(checkpoint))
    model=YOLO(str(checkpoint))
    if model.names!=dict(enumerate(LABELS)):raise ValueError('trained class names differ')
    model.to('cpu')
    exported=Path(model.export(format='onnx',imgsz=1280,dynamic=False,simplify=False,opset=17,half=False,device='cpu',nms=False))
    fused=out/'yolo11n_fused.onnx';shutil.copyfile(exported,fused)
    spec=importlib.util.spec_from_file_location('sdk_model_export',REPO/'vision_sdk/model.py');sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
    raw=sdk.export_six_heads(fused,out/'yolo11n_six.onnx',4,64)
    graph=onnx.load(raw)
    expected=[[1,64,s,s] for s in (160,80,40)]+[[1,4,s,s] for s in (160,80,40)]
    if [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in graph.graph.output]!=expected:raise ValueError('raw head shapes differ')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    f=ort.InferenceSession(str(fused),sess_options=options,providers=['CPUExecutionProvider']);r=ort.InferenceSession(str(raw),sess_options=options,providers=['CPUExecutionProvider'])
    import cv2
    manifest=json.loads((args.pool/'manifest.json').read_text());samples=[x for x in manifest['images'] if x['variant']=='base']
    representatives=[];seen=set()
    parent_states={p['id']:p['state'] for p in manifest['parents']}
    for x in samples:
        state=parent_states[x['parent_id']]
        if state not in seen:representatives.append(x);seen.add(state)
    parity=[]
    for x in representatives:
        image=cv2.imread(str(args.pool/x['image']));tensor=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        a=f.run(None,{f.get_inputs()[0].name:tensor})[0];b=decode_heads(r.run(None,{r.get_inputs()[0].name:tensor}))
        box=float(np.abs(a[:,:4]-b[:,:4]).max());score=float(np.abs(a[:,4:]-b[:,4:]).max())
        if box>.001 or score>1e-5:raise ValueError('raw/fused FP32 export parity failed')
        parity.append({'id':x['id'],'max_box_error':box,'max_score_error':score})
    calibration=args.output/'calibration';calibration.mkdir()
    for x in samples:shutil.copyfile(args.pool/x['image'],calibration/Path(x['image']).name)
    metadata={'checkpoint_sha256':digest(checkpoint),'fused_sha256':digest(fused),'raw_sha256':digest(raw),
              'output_contract':expected,'raw_fused_parity':parity,'calibration_base_images':len(samples),
              'calibration':'90 real preprocessed BASE views only; no random augmented copies or val/test images'}
    (out/'export-lineage.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    status(args.output,'export_complete',**metadata)
    if args.no_compile:return
    status(args.output,'compiling',export=str(out),calibration=str(calibration))
    command=[sys.executable,str(REPO/'tools/compile_ic_terminal_on_admin.py'),'--export',str(out.resolve()),'--graph',raw.name,
             '--calibration',str(calibration.resolve()),'--input-side','1280','--timeout-hours','3']
    with (args.output/'compile-controller.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3.25*3600,cwd=REPO)
    if result.returncode:raise RuntimeError(f'compiler failed exit {result.returncode}; inspect compile-controller.log')
    if not (out/'compilation.json').is_file():raise RuntimeError('compile returned without verified compilation metadata')
    status(args.output,'compiled',export=str(out),compilation_sha256=digest(out/'compilation.json'),deployed=False,validated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pool','init','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--no-compile',action='store_true');args=p.parse_args()
    if args.output.exists():p.error('output already exists; refusing to alter an existing candidate')
    try:
        checkpoint=train(args)
        export_and_compile(args,checkpoint)
    except Exception:
        traceback.print_exc()
        if args.output.exists():status(args.output,'failed',error=traceback.format_exc())
        raise


if __name__=='__main__':main()
