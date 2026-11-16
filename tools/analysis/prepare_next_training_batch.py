"""Historical TRAIN-only crop label batch. Never consumes validation/test images.

Select by class deficit and angular diversity among frozen crop-accepted sources.
All transferred historical boxes are new UNREVIEWED proposals, not predictions.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def helper(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

labels=helper('training_label_helpers','prepare_cropped_training_review.py')
hybrid=helper('hybrid_batch_helpers','portable_hybrid_crop.py')
base=labels.base


def unpad(image,report):
    t=report['transform']
    if t['mode']!='letterbox' or t['scale']!=1 or t['crop_size']!=1280 or image.shape!=(1280,1280,3):raise ValueError('unsupported historical transform')
    if not np.all(image[:280]==t['pad_value']) or not np.all(image[1000:]==t['pad_value']):raise ValueError('historical padding differs')
    return image[280:1000].copy()


def source_boxes(path):
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        cid,x,y,w,h=map(float,line.split())
        if cid!=int(cid) or int(cid) not in range(4):raise ValueError('invalid historical class')
        x*=1280;y=y*1280-280;w*=1280;h*=1280
        q=[[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2]]
        if not np.isfinite(q).all() or w<=0 or h<=0 or np.min(q)<0 or np.max(np.array(q)[:,0])>1280 or np.max(np.array(q)[:,1])>720:raise ValueError('historical label outside native content')
        result.append({'class_id':int(cid),'polygon':q})
    return result


def angle_distance(a,b):return abs((a-b+180)%360-180)


def diverse_select(rows,count):
    if len(rows)<count:raise ValueError('insufficient crop-accepted examples')
    remaining=sorted(rows,key=lambda r:r['id']);chosen=[]
    first=min(remaining,key=lambda r:(angle_distance(r['angle'],0),r['id']));chosen.append(first);remaining.remove(first)
    while len(chosen)<count:
        best=min(remaining,key=lambda r:(-min(angle_distance(r['angle'],s['angle']) for s in chosen),r['id']))
        chosen.append(best);remaining.remove(best)
    return sorted(chosen,key=lambda r:(r['angle'],r['id']))


def guard_manifest(rows):
    holdout_source={r['source_sha256'] for r in rows if r['split'] in ('val','test')}
    holdout_model={r['model_sha256'] for r in rows if r['split'] in ('val','test')}
    train=[r for r in rows if r['split']=='train' and r['run'] in ('present','absent')]
    for r in train:
        if r['source_sha256'] in holdout_source or r['model_sha256'] in holdout_model:raise ValueError('training candidate overlaps historical holdout hash')
    return train


def run(args):
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'));report=json.loads((args.dataset/'report.json').read_text(encoding='utf-8'))
    if report['classes']!=labels.CLASSES:raise ValueError('class contract differs')
    candidates=guard_manifest(manifest);pools={'present':[],'absent':[]};audit=[]
    prior=json.loads(args.first_batch.read_text(encoding='utf-8'));prior_hashes={r['image_sha256'] for r in prior['annotations']}
    for r in candidates:
        path=args.dataset/r['file']
        if base.digest(path)!=r['model_sha256']:raise ValueError('historical image changed')
        native=unpad(cv2.imread(str(path)),report);label_path=args.dataset/'labels/train'/(path.stem+'.txt');boxes=source_boxes(label_path)
        expected={1,2} if r['run']=='present' else {0,1}
        if len(boxes)!=2 or {b['class_id'] for b in boxes}!=expected:raise ValueError('historical run class/presence differs')
        found=hybrid.select_all(native)[0]['colour_seed_canny'];ident='hist-'+path.stem
        entry={'id':ident,'original_split':'train','run':r['run'],'frame':r['frame'],'historical_image':r['file'],
               'source_ancestor_sha256':r['source_sha256'],'historical_image_sha256':r['model_sha256'],
               'historical_label_sha256':base.digest(label_path),'crop_status':found['status'],'reason':found.get('reason')}
        audit.append(entry)
        if found['status']!='located':continue
        crop,_,matrix=base.crop_pixels(native,found['crop_quad'])
        means=np.mean([np.mean(b['polygon'],axis=0) for b in boxes],axis=0);cx,cy=report['transform']['centre'];angle=float(np.degrees(np.arctan2(means[1]-cy,means[0]-cx))%360)
        pools[r['run']].append({**entry,'angle':angle,'source':r,'native':native,'crop':crop,'matrix':matrix,'boxes':boxes,'found':found})
    selected=diverse_select(pools['present'],32)+diverse_select(pools['absent'],16)
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('images','draft-yolo-DO-NOT-TRAIN','review-images','review-sheets','quarantine'):(args.output/sub).mkdir()
    rows=[];tiles=[];selection=[];hashes=set()
    for candidate in selected:
        c=candidate;image=c['crop'];h,w=image.shape[:2];ident=c['id'];destination=args.output/'images'/f'{ident}.png';cv2.imwrite(str(destination),image)
        sha=base.digest(destination)
        if sha in hashes or sha in prior_hashes:raise ValueError('duplicate crop across batch/first batch')
        hashes.add(sha);bs=[]
        for b in sorted(c['boxes'],key=lambda b:0 if b['class_id'] in (0,2) else 1):
            cid=b['class_id'];bs.append({'target':'IC' if cid in (0,2) else 'terminal','class_id':cid,'label':labels.CLASSES[cid],
                'xyxy':labels.projected_box(b['polygon'],c['matrix'],w,h),
                'proposal_source':'historical TRAIN YOLO box projected to frozen hybrid crop; new manual review required'})
        row={'id':ident,'image':f'images/{ident}.png','image_sha256':sha,'width':w,'height':h,
             'state':'both_present' if c['run']=='present' else 'ic_missing_terminal_present',
             'capture_group':'historical-source-run-'+c['run'],'source_ancestor_sha256':c['source']['source_sha256'],
             'source_to_crop':c['matrix'].tolist(),'split':'historical_train_candidate_pool_unsplit',
             'original_split':'train','boxes':bs,'review_status':'pending','reviewer':None,'review_notes':'',
             'training_eligible':False,'review_priority':'normal','historical_camera':True}
        (args.output/'draft-yolo-DO-NOT-TRAIN'/f'{ident}.txt').write_text('\n'.join(labels.yolo_line(b,w,h) for b in bs)+'\n',encoding='utf-8')
        tile=labels.render(image,row);cv2.imwrite(str(args.output/'review-images'/f'{ident}.jpg'),tile);tiles.append(tile);rows.append(row)
        selection.append({k:v for k,v in c.items() if k not in ('native','crop','matrix','boxes','found','source')})
    for start in range(0,len(tiles),6):cv2.imwrite(str(args.output/'review-sheets'/f'page-{start//6+1:02}.jpg'),np.vstack(tiles[start:start+6]))
    selected_ids={r['id'] for r in rows}
    for a in audit:a['selected']=a['id'] in selected_ids
    selection_report={'source_manifest_sha256':base.digest(args.dataset/'manifest.json'),'source_report_sha256':base.digest(args.dataset/'report.json'),
                      'selection':'TRAIN ONLY; crop-status accepted; class quota present32/absent16; farthest-angle selection; no GT-based crop choice',
                      'policy':hybrid.POLICY,'candidates':audit,'selected':selection,'holdout_images_consumed':0,
                      'limitations':'Historical camera; prior train/val/test share source runs. Original holdouts untouched, not an independent new-part validation set.'}
    (args.output/'selection.json').write_text(json.dumps(selection_report,indent=2),encoding='utf-8')
    # Preserve failed-source visual evidence, not trainable substitute crops.
    for a in audit:
        if a['crop_status']=='unknown':
            im=cv2.imread(str(args.dataset/a['historical_image']));cv2.imwrite(str(args.output/'quarantine'/f"{a['id']}.jpg"),im)
    counts={label:sum(b['class_id']==cid for r in rows for b in r['boxes']) for cid,label in enumerate(labels.CLASSES)}
    data={'schema':'portable-cropped-training-boxes/v1','status':'Batch02 UNREVIEWED; no training', 'classes':labels.CLASSES,
          'coordinate_contract':'xyxy native crop pixels; axis-aligned boxes','source_review_sha256':None,
          'upstream_annotation_kind':'historical train labels, not fresh human source-site approval',
          'hybrid_results_sha256':base.digest(args.output/'selection.json'),'builder_sha256':base.digest(__file__),
          'hybrid_helper_sha256':base.digest(Path(hybrid.__file__)),'box_helper_sha256':base.digest(Path(labels.__file__)),
          'split_policy':'Original train-only membership preserved; new dataset unsplit, group by source recording/capture, not adjacent frames',
          'class_counts':counts,'approved_images':0,'training_eligible':False,'annotations':rows}
    (args.output/'annotations.draft.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    template=Path(labels.__file__).with_name('cropped_training_review.html').read_text(encoding='utf-8')
    page=template.replace('/*ANNOTATIONS*/null',json.dumps(data).replace('<','\\u003c')).replace('cropped-training.review-export.json','cropped-training-batch02.review-export.json')
    (args.output/'review.html').write_text(page,encoding='utf-8')
    (args.output/'classes.txt').write_text('\n'.join(labels.CLASSES)+'\n',encoding='utf-8')
    (args.output/'DO-NOT-TRAIN.txt').write_text('UNREVIEWED historical training-label transfer. No final split/YAML, training not started. Eight unknown crop sources quarantined. New camera/session validation still required.\n',encoding='utf-8')
    approval={'scope':'first batch only, unchanged draft hash','draft_sha256':base.digest(args.first_batch),
              'review_basis':'Owner conversation: done; all of them were accurate. No new cropped review export found in repo root.',
              'approval_type':'owner verbal approval of unchanged first-batch proposals; NOT a fabricated browser export',
              'images':len(prior['annotations']),'boxes':sum(len(r['boxes']) for r in prior['annotations']),
              'training_eligible':False,'note':'If boxes were edited, obtain actual exported JSON before finalizing; original draft remains unchanged.'}
    (args.output/'first-batch-owner-confirmation.json').write_text(json.dumps(approval,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(rows),'boxes':sum(counts.values()),'class_counts':counts,'candidate_sources':len(audit),
                      'crop_accepted':sum(a['crop_status']=='located' for a in audit),'quarantined':sum(a['crop_status']=='unknown' for a in audit),
                      'combined_first_and_second_proposed_counts':{k:prior['class_counts'][k]+counts[k] for k in counts}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','first-batch','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
