"""Score isolated board crop replay against reviewed labels and matched FP32."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import cv2
import numpy as np
from portable_tensor_parity import LABELS,digest,decode_heads,detection_rows
from portable_labelled_eval import metrics,match


def run(args):
    reference=json.loads((args.reference/'evaluation.json').read_text(encoding='utf-8'))
    board=json.loads((args.replay/'board/board.json').read_text(encoding='utf-8'))
    inputs=json.loads((args.replay/'inputs/inputs.json').read_text(encoding='utf-8'))
    compilation=json.loads(args.compilation.read_text(encoding='utf-8'))
    if board['elf_sha256']!=compilation['artifacts']['yolov8_stage1_mla.elf'] or board['mpk_sha256']!=compilation['artifacts']['yolov8_mpk.json']:raise ValueError('compiled lineage differs')
    if digest(args.replay/'board/outputs.npz')!=board['outputs_sha256'] or digest(args.replay/'inputs/inputs.npz')!=inputs['inputs_sha256'] or inputs['inputs_sha256']!=board['inputs_sha256']:raise ValueError('input/output fingerprint mismatch')
    if len(reference['samples'])!=len(inputs['images']) or len(board['samples'])!=len(inputs['images']):raise ValueError('sample counts differ')
    if board['labels']!=LABELS or board['transforms']!='software':raise ValueError('unexpected decoder contract')
    if args.output.exists():raise FileExistsError('refusing overwrite')
    args.output.mkdir();(args.output/'comparisons').mkdir()
    rows=[];parity=[]
    with np.load(args.replay/'board/outputs.npz',allow_pickle=False) as arrays:
        for r,item,device in zip(reference['samples'],inputs['images'],board['samples']):
            if Path(item['image']).stem!=r['id'] or item['key']!=device['key']:raise ValueError('sample order differs')
            image_path=Path(item['image'])
            if digest(image_path)!=item['image_sha256']:raise ValueError('prepared image changed')
            fused=decode_heads([arrays[f"{item['key']}_head_{j}"] for j in range(6)],layout='NHWC')
            predictions=[{'class_id':p['class_id'],'score':p['score'],'xywh':p['model_xywh']} for p in detection_rows(fused,threshold=.05,iou=.45)[:100]]
            row={**r,'compiled':predictions};rows.append(row)
            pairs,fp,fn=match(r['new_crop_candidate'],predictions)
            parity.append({'id':r['id'],'matched':len(pairs),'compiled_unmatched':fp,'fp32_unmatched':fn,
                'pairs':[{'fp32_index':g,'compiled_index':p,'iou':iou,'score_delta':predictions[p]['score']-r['new_crop_candidate'][g]['score'],
                          'max_xywh_error_px':float(np.abs(np.array(predictions[p]['xywh'])-r['new_crop_candidate'][g]['xywh']).max())} for g,p,iou in pairs]})
            panels=[];image=cv2.imread(str(image_path))
            for title,boxes in [('Reviewed labels',r['ground_truth']),('FP32',r['new_crop_candidate']),('Compiled MLA',predictions)]:
                panel=image.copy()
                for b in boxes:
                    x,y,w,h=b['xywh'];x,y,x2,y2=map(round,(x,y,x+w,y+h));color=(0,190,0) if 'score' not in b else (0,110,255)
                    cv2.rectangle(panel,(x,y),(x2,y2),color,3);text=LABELS[b['class_id']]+(f" {b['score']:.3f}" if 'score' in b else '')
                    cv2.putText(panel,text,(max(0,x),max(24,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
                cv2.putText(panel,title,(15,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2);panels.append(cv2.resize(panel,(640,640)))
            cv2.imwrite(str(args.output/'comparisons'/f"{r['id']}.jpg"),np.hstack(panels))
    pairs=[p for r in parity for p in r['pairs']]
    result={'scope':'HISTORICAL VAL, actual MLA with SDK software transforms; NOT native camera/stream acceptance',
            'confidence':.05,'nms_iou':.45,'matching_iou':.5,'topk':100,'coverage':reference['coverage'],
            'metrics':{'fp32':metrics(rows,'new_crop_candidate'),'compiled':metrics(rows,'compiled')},
            'reference_sha256':digest(args.reference/'evaluation.json'),'compilation_sha256':digest(args.compilation),
            'board_report_sha256':digest(args.replay/'board/board.json'),'parity':parity,
            'retained_detection_agreement':{'fp32_boxes':sum(len(r['new_crop_candidate']) for r in rows),'compiled_boxes':sum(len(r['compiled']) for r in rows),
                'class_aware_iou50_matched':len(pairs),'min_pair_iou':min(p['iou'] for p in pairs) if pairs else None,
                'max_abs_score_delta':max(abs(p['score_delta']) for p in pairs) if pairs else None,
                'max_xywh_error_px':max(p['max_xywh_error_px'] for p in pairs) if pairs else None},
            'samples':rows,'limitations':['Historical correlated recordings; not independent production accuracy.',
                'Software transform times are NOT the eventual native streaming throughput.',
                'One crop unknown remains in coverage denominator; not an inspection pass.',
                'Native crop/pad114 integration and native-decoder parity remain required before end-to-end streaming.']}
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary=['# Compiled crop replay','', '| Model | TP | FP | FN | Precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for name,m in result['metrics'].items():
        o=m['overall'];summary.append(f"| {name} | {o['tp']} | {o['fp']} | {o['fn']} | {o['precision']:.4f} | {o['recall']:.4f} |")
    summary+=['','Coverage31/32 historical VAL sources. Actual board MLA, software transforms, not native streaming. No threshold tuning.','',json.dumps(result['retained_detection_agreement'],indent=2)]
    (args.output/'SUMMARY.md').write_text('\n'.join(summary),encoding='utf-8')
    page='<html><meta charset="utf-8"><h1>Reviewed labels / FP32 / actual compiled MLA</h1><pre>'+html.escape('\n'.join(summary))+'</pre>'
    for r in rows:page+=f'<h2>{html.escape(r["id"])}</h2><img style="max-width:100%" src="comparisons/{r["id"]}.jpg">'
    (args.output/'index.html').write_text(page+'</html>',encoding='utf-8')
    print(json.dumps({'metrics':result['metrics'],'agreement':result['retained_detection_agreement']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','replay','compilation','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
