"""Fixed-settings FP32 comparison on reviewed historical VAL crops; no tuning/board use."""
from __future__ import annotations
import argparse
import ast
import html
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import prepare_augmented_training_pool as prep
from portable_tensor_parity import LABELS,digest,detection_rows
from portable_labelled_eval import metrics,box_iou,match


def ground_truth(boxes):
    return [{'class_id':b['class_id'],'xywh':[b['xyxy'][0],b['xyxy'][1],b['xyxy'][2]-b['xyxy'][0],b['xyxy'][3]-b['xyxy'][1]]} for b in boxes]


def predict(session,image):
    tensor=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
    raw=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    rows=detection_rows(raw,threshold=.05,iou=.45)[:100]
    return raw,[{'class_id':r['class_id'],'score':r['score'],'xywh':list(r['model_xywh'])} for r in rows]


def failure_details(gt,pred):
    pairs,fp,fn=match(gt,pred)
    details=[]
    for p in fp:
        overlaps=[(box_iou(pred[p]['xywh'],g['xywh']),i) for i,g in enumerate(gt)]
        overlap,g=max(overlaps,default=(0.,-1))
        details.append({'prediction_index':p,'best_gt_index':g,'best_gt_iou':overlap,
                        'best_gt_class':gt[g]['class_id'] if g>=0 else None})
    return {'matches':pairs,'false_positive':fp,'false_negative':fn,'unmatched_prediction_overlaps':details}


def run(args):
    draft=json.loads((args.pack/'annotations.draft.json').read_text(encoding='utf-8'))
    review=json.loads(args.review.read_text(encoding='utf-8'));prep.validate_review(draft,review)
    selection=json.loads((args.pack/'selection.json').read_text(encoding='utf-8'))
    if digest(args.pack/'selection.json')!=review['hybrid_results_sha256']:raise ValueError('selection fingerprint differs')
    original={r['id']:r for r in draft['annotations']}
    for r in review['annotations']:
        for k in ('source_to_crop','source_ancestor_sha256','original_split'):
            if r[k]!=original[r['id']][k]:raise ValueError('review lineage differs')
        if r['split']!='val' or digest(args.pack/r['image'])!=r['image_sha256']:raise ValueError('validation split/image differs')
    if args.output.exists():raise FileExistsError('refusing overwrite')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    models={}
    for name,path in [('old_on_same_crops',args.old),('new_crop_candidate',args.new)]:
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        if ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])!=dict(enumerate(LABELS)):raise ValueError('model class order differs')
        if session.get_inputs()[0].shape!=[1,3,1280,1280]:raise ValueError('model input differs')
        models[name]=session
    args.output.mkdir(parents=True);(args.output/'comparisons').mkdir();raws={};rows=[]
    for r in review['annotations']:
        crop=cv2.imread(str(args.pack/r['image']));image,matrix=prep.letterbox(crop)
        gt=ground_truth(prep.transform_boxes(r['boxes'],matrix));row={'id':r['id'],'state':r['state'],'ground_truth':gt,'image_sha256':r['image_sha256'],'crop_to_model':matrix.tolist()}
        for name,s in models.items():
            raw,pred=predict(s,image);raws[r['id']+'__'+name]=raw;row[name]=pred;row[name+'_errors']=failure_details(gt,pred)
        panels=[]
        for title,boxes in [('Reviewed labels',gt)]+[(name,row[name]) for name in models]:
            panel=image.copy()
            for b in boxes:
                x,y,w,h=b['xywh'];x,y,x2,y2=map(round,(x,y,x+w,y+h));color=(0,190,0) if 'score' not in b else (0,110,255)
                cv2.rectangle(panel,(x,y),(x2,y2),color,3);text=LABELS[b['class_id']]+(f" {b['score']:.3f}" if 'score' in b else '')
                cv2.putText(panel,text,(max(0,x),max(24,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            cv2.putText(panel,title,(15,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2);panels.append(cv2.resize(panel,(640,640)))
        cv2.imwrite(str(args.output/'comparisons'/f"{r['id']}.jpg"),np.hstack(panels));rows.append(row)
        print(r['id'],*[f'{name}:{len(row[name])}' for name in models],flush=True)
    np.savez_compressed(args.output/'raw-predictions.npz',**raws)
    result={'scope':'reviewed HISTORICAL VAL crops; NOT independent new-part/session acceptance',
            'confidence':.05,'nms_iou':.45,'matching_iou':.5,'topk':100,'settings':'fixed diagnostic settings, no confidence tuning; NOT mAP',
            'preprocessing':'same frozen native hybrid crops; deterministic1280 letterbox pad114; RGB/255 once; no augmentation',
            'review_sha256':digest(args.review),'pack_selection_sha256':digest(args.pack/'selection.json'),
            'models':{name:{'path':str(path.resolve()),'sha256':digest(path)} for name,path in [('old_on_same_crops',args.old),('new_crop_candidate',args.new)]},
            'coverage':{'reserved_sources':len(selection['sources']),'reviewed_located':len(rows),'quarantined':[r['id'] for r in selection['sources'] if r['crop_status']!='located'],
                        'fraction':len(rows)/len(selection['sources']),'unknowns_not_silently_counted_as_success':True},
            'metrics':{name:metrics(rows,name) for name in models},'samples':rows,
            'limitations':['Historical validation shares source recordings with training; correlated data.',
                           'Old baseline is OLD MODEL ON THESE SAME CROPS, not old full-frame runtime performance.',
                           'Localization-or-wrong-object FP category does not prove semantic object identity.',
                           'No compiled/native inference, fresh camera data, no-board negatives or production threshold qualification.',
                           'Crop unknown excluded from reviewed box metrics but explicitly included in crop coverage denominator.']}
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    shutil.copyfile(args.review,args.output/'reviewed-labels.snapshot.json');shutil.copyfile(__file__,args.output/'evaluator-source.py')
    text=['# Historical reviewed validation: FP32 crop comparison','', 'Diagnostic confidence0.05, class-aware NMS0.45, match IoU0.50; no tuning.',
          f"Crop coverage: {len(rows)}/{len(selection['sources'])}. Unknowns are NOT inspection passes.",'', '| Model | TP | FP | FN | Precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for name,values in result['metrics'].items():
        o=values['overall'];text.append(f"| {name} | {o['tp']} | {o['fp']} | {o['fn']} | {o['precision'] or 0:.3f} | {o['recall'] or 0:.3f} |")
    text+=['','Historical correlated frames, not production accuracy. Old model tested on same crops, not its full-frame serving baseline.','See evaluation.json for per-class results, confusion and per-frame failures.']
    (args.output/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    page='<html><meta charset="utf-8"><title>Reviewed validation FP32 comparison</title><h1>Historical validation: labels / old crop baseline / new model</h1><p>Fixed .05 confidence/.45 NMS/.50 matching. Correlated recordings, NOT production accuracy. Crop coverage31/32; one unknown excluded from box metrics.</p><pre>'+html.escape('\n'.join(text))+'</pre>'
    for r in rows:page+=f'<h2>{html.escape(r["id"])}</h2><img style="max-width:100%" src="comparisons/{r["id"]}.jpg">'
    (args.output/'index.html').write_text(page+'</html>',encoding='utf-8')
    print(json.dumps({'coverage':result['coverage'],'metrics':result['metrics']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('pack','review','old','new','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
