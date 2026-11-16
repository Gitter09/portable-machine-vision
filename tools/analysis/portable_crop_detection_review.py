"""Offline original-versus-hybrid-crop FP32 detection review. No board access.

No site filtering, threshold tuning, model changes or inspection verdicts.
"""
from __future__ import annotations
import argparse
import ast
import html
import importlib.util
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def load_helper(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

base=load_helper('canny_review_helpers','portable_canny_crop.py')
parity=load_helper('tensor_review_helpers','portable_tensor_parity.py')
MODEL_SHA256='b59eaab038d54ee4a623906366087544900c3570381e2abcf6b5cee7323290d6'
CONFIDENCE=.05
NMS_IOU=.45
TOPK=100
COLOURS=[(30,30,230),(30,190,30),(220,170,0),(200,30,200)]


def prepare(image,side=1280):
    h,w=image.shape[:2];scale=min(side/w,side/h)
    nw,nh=round(w*scale),round(h*scale)
    left=round((side-nw)/2-.1);top=round((side-nh)/2-.1)
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,top,side-nh-top,left,side-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
    return tensor,{'original_size':[w,h],'resized_size':[nw,nh],'scale_xy':[nw/w,nh/h],'padding_left_top':[left,top],
                   'tensor_sha256':__import__('hashlib').sha256(tensor.tobytes()).hexdigest()}


def image_boxes(prediction,prep):
    sx,sy=prep['scale_xy'];left,top=prep['padding_left_top'];width,height=prep['original_size'];out=[]
    for row in parity.detection_rows(prediction,CONFIDENCE,NMS_IOU)[:TOPK]:
        x,y,w,h=row['model_xywh'];a=max(0.,(x-left)/sx);b=max(0.,(y-top)/sy)
        c=min(float(width),(x+w-left)/sx);d=min(float(height),(y+h-top)/sy)
        if c>a and d>b:out.append({'class_id':row['class_id'],'label':row['label'],'score':row['score'],'xywh':[a,b,c-a,d-b]})
    return out


def box_polygon(box):
    x,y,w,h=box['xywh'];return np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])


def transform_polygons(boxes,matrix):
    return [{**r,'polygon':cv2.perspectiveTransform(box_polygon(r)[None],np.float64(matrix))[0].tolist()} for r in boxes]


def draw(image,boxes,title,matrix=None,size=None):
    if matrix is None:matrix=np.eye(3)
    canvas=image.copy();scale_matrix=np.eye(3)
    if size:
        w,h=size;ih,iw=image.shape[:2];s=min(w/iw,h/ih);nw,nh=round(iw*s),round(ih*s);x=(w-nw)//2;y=(h-nh)//2
        canvas=np.full((h,w,3),114,np.uint8);canvas[y:y+nh,x:x+nw]=cv2.resize(image,(nw,nh))
        scale_matrix=np.float64([[nw/iw,0,x],[0,nh/ih,y],[0,0,1]])
    for r in transform_polygons(boxes,scale_matrix@matrix):
        points=np.array(r['polygon']);h,w=canvas.shape[:2]
        if points[:,0].max()<0 or points[:,1].max()<0 or points[:,0].min()>=w or points[:,1].min()>=h:continue
        colour=COLOURS[r['class_id']];cv2.polylines(canvas,[np.int32(points)],True,colour,2)
        text=f"{r['label']} {r['score']:.3f}";font=.5
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,font,1)
        x=max(0,min(w-tw-4,round(float(points[:,0].min()))));y=max(th+3,min(h-3,round(float(points[:,1].min()))-4))
        cv2.rectangle(canvas,(x,y-th-3),(x+tw+4,y+2),colour,-1)
        cv2.putText(canvas,text,(x+2,y),cv2.FONT_HERSHEY_SIMPLEX,font,(255,255,255),1,cv2.LINE_AA)
    canvas=cv2.copyMakeBorder(canvas,42,0,0,0,cv2.BORDER_CONSTANT,value=(240,240,240))
    cv2.putText(canvas,title,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.63,(0,0,0),1,cv2.LINE_AA)
    return canvas


def run(args):
    import onnxruntime as ort
    review=json.loads(args.review.read_text(encoding='utf-8'));original=json.loads(args.original.read_text(encoding='utf-8'))
    validation=base.validate_review(review,original,args.images)
    hybrid=json.loads((args.hybrid/'results.json').read_text(encoding='utf-8'))
    if hybrid['review_sha256']!=base.digest(args.review):raise ValueError('crop/review version differs')
    if base.digest(args.model)!=MODEL_SHA256:raise ValueError('FP32 model hash differs from pinned reference')
    found={r['id']:r for r in hybrid['results']['colour_seed_canny']['frames']}
    inputs=[]
    for row in review['annotations']:
        item=found[row['id']]
        if item['status']!='located':raise ValueError('crop unavailable; refusing selective review subset')
        image=cv2.imread(str(args.images/row['image']));path=args.hybrid/'colour_seed_canny'/f"crop-{row['id']}.png"
        crop=cv2.imread(str(path));regenerated,_,matrix=base.crop_pixels(image,item['crop_quad'])
        if crop is None or not np.array_equal(crop,regenerated):raise ValueError('saved crop pixels differ from original+recorded transform')
        if not np.allclose(matrix,item['source_to_crop'],atol=1e-8,rtol=0):raise ValueError('saved crop transform differs')
        inputs.append((row,image,crop,path,matrix))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape!=[1,3,1280,1280] or ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])!=dict(enumerate(parity.LABELS)):
        raise ValueError('model input/class contract differs')
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('comparison','original-full-frame','original-roi-zoom','cropped','contact-sheets'):(args.output/sub).mkdir()
    records=[];arrays={};thumbs=[];gallery=[]
    for i,(row,image,crop,path,matrix) in enumerate(inputs):
        result={'id':row['id'],'state':row['state'],'source_image_sha256':row['source_image_sha256'],
                'crop_image_sha256':base.digest(path),'crop_path':str(path.resolve()),'source_to_crop':matrix.tolist()}
        for branch,pixels in [('original',image),('cropped',crop)]:
            tensor,prep=prepare(pixels);prediction=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            if prediction.shape!=(1,8,33600) or not np.isfinite(prediction).all():raise ValueError('FP32 output contract differs')
            arrays[f'{i:02d}_{branch}']=prediction
            result[branch]={'preprocessing':prep,'detections':image_boxes(prediction,prep)}
        orig=result['original']['detections'];cropped=result['cropped']['detections']
        result['cropped']['detections_source_polygons']=transform_polygons(cropped,np.linalg.inv(matrix))
        name=row['id'];state=row['state'].replace('_',' ')
        full=draw(image,orig,f'{name} | {state} | FP32 ORIGINAL, all {len(orig)} detections')
        zoom=draw(crop,orig,f'{name} | FP32 ORIGINAL predictions, crop-region zoom',matrix=matrix,size=(1280,640))
        annotated=draw(crop,cropped,f'{name} | FP32 CROPPED, all {len(cropped)} detections',size=(1280,640))
        cv2.imwrite(str(args.output/'original-full-frame'/f'{name}.jpg'),full)
        cv2.imwrite(str(args.output/'original-roi-zoom'/f'{name}.jpg'),zoom)
        cv2.imwrite(str(args.output/'cropped'/f'{name}.jpg'),annotated)
        comparison=np.hstack([zoom,annotated]);cv2.imwrite(str(args.output/'comparison'/f'{name}.jpg'),comparison)
        thumbs.append(cv2.resize(comparison,(1280,341)))
        gallery.append(f'<section><h2>{html.escape(name)} — {html.escape(state)}</h2><p><a href="original-full-frame/{name}.jpg">Full original frame ({len(orig)} boxes)</a> · <a href="cropped/{name}.jpg">Cropped input ({len(cropped)} boxes)</a> · <a href="comparison/{name}.jpg">Open comparison full size</a></p><a href="comparison/{name}.jpg"><img loading="lazy" src="comparison/{name}.jpg" alt="{name} original predictions left; cropped predictions right"></a></section>')
        records.append(result);print(name,'original',len(orig),'cropped',len(cropped),flush=True)
    np.savez_compressed(args.output/'raw-predictions.npz',**arrays)
    for start in range(0,len(thumbs),6):cv2.imwrite(str(args.output/'contact-sheets'/f'page-{start//6+1:02}.jpg'),np.vstack(thumbs[start:start+6]))
    summary={branch:{'boxes':sum(len(r[branch]['detections']) for r in records),
                     'class_counts':{label:sum(d['class_id']==cid for r in records for d in r[branch]['detections']) for cid,label in enumerate(parity.LABELS)}} for branch in ('original','cropped')}
    report={'scope':'FP32 CPU original-versus-crop visual DEVELOPMENT review, not native compiled detections or labelled precision/recall',
            'model_sha256':MODEL_SHA256,'confidence':CONFIDENCE,'nms_iou':NMS_IOU,'topk':TOPK,'nms':'class-aware OpenCV NMSBoxesBatched',
            'input_preprocessing':'BGR to RGB float32 /255; bilinear resize; centered 1280x1280 padding 114. Crop uses native-size RGB pixels, not 512x256 preview.',
            'coordinate_mapping':'Undo actual rounded resize x/y scales and padding; clip to image extent. Crop/source mapping retains all four corners.',
            'limitations':['FP32 CPU only; quantized/native board output may differ.','Different image scale/orientation; no weights/calibration changes.','Original ROI zoom is for display and omits off-ROI boxes; full-frame images and JSON preserve all.','No site gate, enhancement, confidence tuning or board verdict.','Development data: 40 missing, 2 adjacent present, third state absent; not held-out accuracy.'],
            'review_validation':validation,'review_sha256':base.digest(args.review),'hybrid_results_sha256':base.digest(args.hybrid/'results.json'),
            'script_sha256':base.digest(__file__),'canny_helper_sha256':base.digest(Path(base.__file__)),
            'decoder_helper_sha256':base.digest(Path(parity.__file__)),'onnxruntime_version':ort.__version__,
            'raw_predictions_sha256':base.digest(args.output/'raw-predictions.npz'),'summary':summary,'frames':records}
    (args.output/'detections.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    shutil.copyfile(args.review,args.output/'annotations.review-export.snapshot.json')
    shutil.copyfile(args.hybrid/'results.json',args.output/'hybrid-results.snapshot.json')
    intro='''<!doctype html><meta charset="utf-8"><title>Original vs cropped — FP32 detection review</title><style>body{font:16px system-ui;margin:24px;background:#20252c;color:#eee}a{color:#91d4ff}img{width:100%;height:auto}section{margin:32px 0;border-top:1px solid #777}p{max-width:1200px}</style><h1>Original vs cropped: FP32 detection review</h1><p>42 images. LEFT: original-frame predictions projected into the same crop for close inspection. RIGHT: predictions from feeding the cropped image to the model. Full-original links retain background detections outside this zoom.</p><p>Same unchanged FP32 model, confidence 0.05, class-aware NMS IoU 0.45, top 100. All surviving detections shown; no site filtering or automatic verdict. These are CPU FP32 results, NOT quantized/native board output. Enlarging an image adds no camera detail. Labels mean missing/present components only.</p><p>IC_Fail = missing IC (red); Terminal_pass = present terminal (green); IC_Pass = present IC (cyan); Terminal_Fail = missing terminal (magenta). JSON and raw predictions retained for inspection. Development data, not independent accuracy.</p>'''
    (args.output/'index.html').write_text(intro+'\n'.join(gallery),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','original','images','hybrid','model','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
