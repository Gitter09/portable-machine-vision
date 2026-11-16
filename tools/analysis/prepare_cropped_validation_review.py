"""Prepare reserved historical VAL only, unreviewed; no test, training or board use."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import prepare_next_training_batch as batch
import prepare_augmented_training_pool as augmentation

labels=batch.labels
base=batch.base
STATES={'present':('both_present',{1,2}),'absent':('ic_missing_terminal_present',{0,1}),'terminal_absent':('both_missing',{0,3})}


def reserved(rows):
    others=[r for r in rows if r['split'] in ('train','test')]
    hashes={r[k] for r in others for k in ('source_sha256','model_sha256')}
    result=[r for r in rows if r['split']=='val']
    for r in result:
        if any(r[k] in hashes for k in ('source_sha256','model_sha256')):raise ValueError('validation source overlaps train/test hashes')
    if not result:raise ValueError('no reserved validation images')
    return result


def run(args):
    if args.output.exists():raise FileExistsError('refusing output overwrite')
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    report=json.loads((args.dataset/'report.json').read_text(encoding='utf-8'))
    if report['classes']!=labels.CLASSES:raise ValueError('class contract differs')
    sources=reserved(manifest)
    args.output.mkdir(parents=True)
    for sub in ('images','review-images','review-sheets','quarantine','model-inputs','draft-labels-DO-NOT-EVALUATE','source-snapshot'):(args.output/sub).mkdir()
    audit=[];rows=[];tiles=[]
    for r in sources:
        path=args.dataset/r['file'];ident='val-'+path.stem
        if base.digest(path)!=r['model_sha256']:raise ValueError('historical image hash differs')
        native=batch.unpad(cv2.imread(str(path)),report)
        source_label=args.dataset/'labels/val'/(path.stem+'.txt')
        boxes=batch.source_boxes(source_label)
        state,expected=STATES[r['run']]
        if len(boxes)!=2 or {b['class_id'] for b in boxes}!=expected:raise ValueError('class/state mismatch')
        found=batch.hybrid.select_all(native)[0]['colour_seed_canny']
        entry={'id':ident,'original_split':'val','source_image':r['file'],'source_ancestor_sha256':r['source_sha256'],
               'source_image_sha256':base.digest(path),'source_label_sha256':base.digest(source_label),
               'capture_group':'historical-source-run-'+r['run'],'frame':r['frame'],'crop_status':found['status'],'reason':found.get('reason')}
        audit.append(entry)
        if found['status']!='located':
            cv2.imwrite(str(args.output/'quarantine'/f'{ident}.png'),native);continue
        crop,_,matrix=base.crop_pixels(native,found['crop_quad']);h,w=crop.shape[:2]
        try:
            proposed=[{'target':'IC' if b['class_id'] in (0,2) else 'terminal','class_id':b['class_id'],'label':labels.CLASSES[b['class_id']],
                       'xyxy':labels.projected_box(b['polygon'],matrix,w,h),'proposal_source':'historical VAL label transfer; UNREVIEWED'}
                      for b in sorted(boxes,key=lambda b:0 if b['class_id'] in (0,2) else 1)]
        except ValueError as exc:
            entry.update(crop_status='quarantined_target_clipped',reason=str(exc));cv2.imwrite(str(args.output/'quarantine'/f'{ident}.png'),native);continue
        dest=args.output/'images'/f'{ident}.png';cv2.imwrite(str(dest),crop)
        row={'id':ident,'image':f'images/{ident}.png','image_sha256':base.digest(dest),'width':w,'height':h,'state':state,
             'capture_group':entry['capture_group'],'source_ancestor_sha256':r['source_sha256'],'source_to_crop':matrix.tolist(),
             'split':'val','original_split':'val','boxes':proposed,'review_status':'pending','reviewer':None,'review_notes':'',
             'training_eligible':False,'evaluation_eligible':False,'historical_camera':True,'review_priority':'normal'}
        model_image,model_matrix=augmentation.letterbox(crop);model_boxes=augmentation.transform_boxes(proposed,model_matrix)
        model_dest=args.output/'model-inputs'/f'{ident}.png';cv2.imwrite(str(model_dest),model_image)
        label_dest=args.output/'draft-labels-DO-NOT-EVALUATE'/f'{ident}.txt'
        label_dest.write_text('\n'.join(labels.yolo_line(b,1280,1280) for b in model_boxes)+'\n',encoding='utf-8')
        entry.update(crop_sha256=base.digest(dest),crop_quad=found['crop_quad'],crop_to_model=model_matrix.tolist(),
                     model_image_sha256=base.digest(model_dest),draft_model_label_sha256=base.digest(label_dest))
        tile=labels.render(crop,row);cv2.imwrite(str(args.output/'review-images'/f'{ident}.jpg'),tile);tiles.append(tile);rows.append(row)
    for i in range(0,len(tiles),6):cv2.imwrite(str(args.output/'review-sheets'/f'page-{i//6+1:02}.jpg'),np.vstack(tiles[i:i+6]))
    audit_data={'source_manifest_sha256':base.digest(args.dataset/'manifest.json'),'source_report_sha256':base.digest(args.dataset/'report.json'),
                'policy':batch.hybrid.POLICY,'split':'reserved historical val ONLY','test_images_consumed':0,
                'selection':'ALL reserved val; no model-based or label-based crop selection. Unknown/clipped sources retained in coverage denominator.',
                'limitations':'Historical recordings overlap TRAIN capture groups; NOT independent-part/session production accuracy.',
                'preprocessing':'frozen colour-seed/local-Canny native crop; deterministic resize/pad114 to1280; RGB/255 once at inference; NO augmentation',
                'sources':audit}
    (args.output/'selection.json').write_text(json.dumps(audit_data,indent=2),encoding='utf-8')
    data={'schema':'portable-cropped-validation-boxes/v1','status':'UNREVIEWED reserved VAL proposals; no evaluation approval',
          'classes':labels.CLASSES,'source_review_sha256':None,'hybrid_results_sha256':base.digest(args.output/'selection.json'),
          'training_eligible':False,'evaluation_eligible':False,'approved_images':0,'annotations':rows,
          'class_counts':{name:sum(b['class_id']==i for row in rows for b in row['boxes']) for i,name in enumerate(labels.CLASSES)}}
    (args.output/'annotations.draft.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    page=Path(labels.__file__).with_name('cropped_training_review.html').read_text(encoding='utf-8')
    page=page.replace('/*ANNOTATIONS*/null',json.dumps(data).replace('<','\\u003c')).replace('cropped-training.review-export.json','cropped-validation.review-export.json')
    page=page.replace('training-box','validation-box').replace('training boxes','validation boxes').replace('Training-box human review export','Validation-box human review export')
    page=page.replace('This pack is an unsplit development pool, not a final training/validation set.','Reserved historical VAL only. Shared recordings with TRAIN: diagnostic evaluation, not independent production accuracy.')
    (args.output/'review.html').write_text(page,encoding='utf-8')
    (args.output/'index.html').write_text('<meta http-equiv="refresh" content="0;url=review.html">',encoding='utf-8')
    for script in [Path(__file__),Path(batch.__file__),Path(labels.__file__),Path(batch.hybrid.__file__),Path(augmentation.__file__)]:shutil.copyfile(script,args.output/'source-snapshot'/script.name)
    summary={'reserved_sources':len(sources),'review_images':len(rows),'boxes':len(rows)*2,'quarantined':len(sources)-len(rows),
             'class_counts':data['class_counts'],'test_images_consumed':0,'augmentation':False,'evaluation_ready':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'START-HERE.md').write_text('# Reserved historical validation review\n\nOpen review.html. Inspect/correct both boxes on each image, enter your name, mark reviewed, then EXPORT cropped-validation.review-export.json. No autosave: resume by importing your export.\n\nPresent IC: body and own pins. Terminal: housing, exclude wires. Missing targets: footprint plus pads/holes.\n\nHistorical VAL only, no augmentation or test use. Unknown/clipped sources in quarantine remain part of crop coverage. Model-inputs and draft labels are proposals only: human corrections must be propagated before scoring. No evaluation-ready dataset/YAML created. Shared source recordings with TRAIN limit independence.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
