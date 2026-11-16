"""Smooth unannotated camera passthrough + asynchronous latest-frame MLA results.
Result boxes appear ONLY on their own saved inspection frame, never live video.
"""
from __future__ import annotations
import argparse,json,os,signal,threading,time
from pathlib import Path
import cv2
import numpy as np
import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst
from mpk_parser import Parser
from sima.operations import SimaModelRunner
from ev_transforms import transforms
import portable_hybrid_crop as hybrid
from portable_tensor_parity import SoftwareTransforms,decode_heads,digest,LABELS
from crop_geometry import detection_rows,letterbox,source_quad
from latest_frame_graph import graph
from pcb_pose_tracker import PoseTracker
from fast_crop_inference import rgb_tensor,decode_candidates,FastTransforms,select_crop


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def run(args):
    if not 15<=args.seconds<=300:raise ValueError('bounded15..300seconds required')
    if args.track_pose and not args.coordinates_only:raise ValueError('pose candidate requires coordinate-only mode')
    description=graph(args.host,args.port)
    if args.feature_roi and not args.track_pose:raise ValueError('feature ROI requires pose tracking')
    tracker=PoseTracker(feature_roi=args.feature_roi) if args.track_pose else None
    publisher=None
    assets=(args.release/'assets').resolve();inventory=json.loads((args.release/'release.json').read_text())
    for name,sha in inventory['assets'].items():
        path=(assets/name).resolve()
        if not path.is_relative_to(assets) or digest(path)!=sha:raise ValueError('asset hash/path differs')
    mpk=assets/'provenance/mpk.json';software=SoftwareTransforms(json.loads(mpk.read_text()),transforms);software.check_output_layouts()
    fast=None
    if args.fast_transforms:
        fast=FastTransforms(software.contract);preflight=fast.preflight(software)
    cpp=None
    if args.cpp_library:
        if fast is None or not args.cpp_sha256 or digest(args.cpp_library)!=args.cpp_sha256:raise ValueError('C++ library requires fast path and matching SHA256')
        from fused_input import FusedInputPacker
        cpp=FusedInputPacker(args.cpp_library,fast,args.cpp_kernel)
        preflight['cpp_input']=cpp.preflight(fast);preflight['cpp_input']['library_sha256']=args.cpp_sha256
    runner=SimaModelRunner(Parser(str(mpk)));runner.configure({'model_path':str(assets/'share/processmla/yolov8_stage1_mla.elf')})
    args.output.mkdir(parents=True,exist_ok=False)
    if args.coordinates_only:
        from coordinate_publisher import CoordinatePublisher
        publisher=CoordinatePublisher(args.host,args.port+2,args.output/'published.jsonl')
    if fast:atomic_json(args.output/'transform-preflight.json',preflight)
    Gst.init(None)
    pipeline=Gst.parse_launch(description);pipeline.get_by_name('camera').set_property('location',args.rtsp)
    stopping=threading.Event()
    def request_stop(signum,frame):stopping.set()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    sink=pipeline.get_by_name('input');start=time.monotonic();count=0;located=0;errors=[];video_buffers=None
    try:
        if pipeline.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:raise RuntimeError('pipeline refused PLAYING')
        with (args.output/'frames.jsonl').open('w') as log:
            while time.monotonic()-start<args.seconds and not stopping.is_set():
                message=pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR|Gst.MessageType.EOS)
                if message:raise RuntimeError(str(message.parse_error()) if message.type==Gst.MessageType.ERROR else 'unexpected EOS')
                stages={};checkpoint=time.perf_counter()
                def mark(name):
                    nonlocal checkpoint
                    now=time.perf_counter();stages[name]=1000*(now-checkpoint);checkpoint=now
                sample=sink.emit('try-pull-sample',Gst.SECOND)
                mark('sample_wait')
                if sample is None:continue
                began=time.monotonic();taken=time.time();buffer=sample.get_buffer();raw=buffer.extract_dup(0,buffer.get_size())
                if len(raw)!=1280*720*3//2:raise ValueError('NV12 size differs')
                fingerprint=None  # Off-thread publisher computes the exact NV12 fingerprint.
                mark('extract_nv12')
                image=cv2.cvtColor(np.frombuffer(raw,np.uint8).reshape(1080,1280),cv2.COLOR_YUV2BGR_NV12)
                mark('nv12_to_bgr')
                if tracker:found,pose=tracker.update(image,time.monotonic())
                else:
                    found=select_crop(image) if args.optimized else hybrid.select_all(image)[0]['colour_seed_canny'];pose={'source':'full' if found['status']=='located' else 'unknown'}
                detections=[]
                mark('crop_select')
                if found['status']=='located':
                    crop,_,matrix=hybrid.base.crop_pixels(image,found['crop_quad']);mark('crop_warp')
                    model,lm=letterbox(crop);inverse=np.linalg.inv(lm@matrix);mark('letterbox_inverse')
                    tensor=None if fast else (rgb_tensor(model) if args.optimized else np.ascontiguousarray(model[:,:,::-1],dtype=np.float32)/np.float32(255));mark('rgb_normalize')
                    packed=cpp.encode_bgr(model) if cpp else (fast.encode_bgr(model) if fast else software.encode(tensor));mark('quantize_tessellate')
                    raw_heads=runner.run(packed);mark('mla')
                    heads=None if fast else software.decode(raw_heads);mark('unpack_dequantize')
                    decoded=fast.predictions(raw_heads,.05) if fast else (decode_candidates(heads,1280,.05) if args.optimized else decode_heads(heads,1280,'NHWC'));mark('dfl_decode')
                    detections=detection_rows(decoded,threshold=.05,iou=.45)[:100];mark('nms')
                    for d in detections:
                        quad=source_quad(d['model_xywh'],inverse);d['source_quad']=quad.tolist();d['label']=LABELS[d['class_id']]
                        color=(0,0,255) if d['class_id'] in (0,3) else (0,220,0)
                        if not args.coordinates_only:
                            cv2.polylines(image,[np.int32(np.round(quad))],True,color,2);x,y=np.int32(quad.min(axis=0))
                            cv2.putText(image,f"{d['label']} {d['score']:.2f}",(max(0,x),max(25,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
                    located+=1
                elif not args.coordinates_only:cv2.putText(image,'UNKNOWN - crop not located',(20,90),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2)
                if not args.coordinates_only:cv2.putText(image,f'INSPECTION SNAPSHOT #{count} - NOT LIVE VIDEO',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
                mark('overlay')
                filename=None if args.coordinates_only else f'inspection-{count:06}.jpg'
                if filename and not cv2.imwrite(str(args.output/filename),image,[cv2.IMWRITE_JPEG_QUALITY,85]):raise RuntimeError('snapshot write failed')
                mark('jpeg_write')
                result={'frame_sha256':fingerprint,'pose':pose,'pose_crop_quad':found.get('crop_quad'),'pose_seed_quad':found.get('seed_quad'),'stages_ms':stages,'sequence':count,'image':filename,'input_taken_unix':taken,'completed_unix':time.time(),'source_pts_ns':None if buffer.pts==Gst.CLOCK_TIME_NONE else int(buffer.pts),
                        'processing_ms':1000*(time.monotonic()-began),'crop_status':found['status'],'crop_reason':found.get('reason'),
                        'detections':detections,'result_scope':'inspection snapshot ONLY; never registered to current live video'}
                if publisher:publisher.submit(result,raw)
                log.write(json.dumps(result)+'\n');log.flush();atomic_json(args.output/'latest.json',result);count+=1
    except Exception as e:errors.append(repr(e));raise
    finally:
        try:video_buffers=int(pipeline.get_by_name('video_count').get_property('stats').get_value('num-buffers'))
        except (TypeError,AttributeError):pass
        pipeline.set_state(Gst.State.NULL)
        if publisher:publisher.close();errors.extend(publisher.errors)
        elapsed=time.monotonic()-start
        atomic_json(args.output/'worker.json',{'state':'error' if errors else ('stopped' if stopping.is_set() else ('completed' if count else 'error')),'stop_requested':stopping.is_set(),'inference_frames':count,'located':located,'unknown':count-located,
            'published_results':publisher.sent if publisher else None,'publication_replaced':publisher.replaced if publisher else None,
            'feature_roi':args.feature_roi,'cpp_input_kernel':args.cpp_kernel if cpp else None,'cpp_library_sha256':args.cpp_sha256 if cpp else None,
            'coordinates_only':args.coordinates_only,'track_pose':args.track_pose,
            'pose_full':tracker.full if tracker else None,'pose_tracked':tracker.tracked if tracker else None,'pose_failures':tracker.failures if tracker else None,
            'fast_transforms':args.fast_transforms,'optimized':args.optimized,'elapsed_seconds':elapsed,'inference_fps':count/elapsed,'video_access_units':video_buffers,
            'video_fps':video_buffers/elapsed if video_buffers is not None else None,'errors':errors,
            'live_video':'unannotated original camera H264 passthrough; no stale result overlays',
            'queue_policy':'appsink max-buffers1/drop=true; no accumulated inference queue',
            'elf_sha256':digest(assets/'share/processmla/yolov8_stage1_mla.elf'),'mpk_sha256':digest(mpk)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('release','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rtsp',required=True);p.add_argument('--host',default='172.21.30.230');p.add_argument('--port',type=int,default=5010)
    p.add_argument('--cpp-library',type=Path);p.add_argument('--cpp-sha256');p.add_argument('--cpp-kernel',choices=('scalar','neon','parallel2','parallel4','parallel8','parallel16'),default='parallel8')
    p.add_argument('--feature-roi',action='store_true');p.add_argument('--coordinates-only',action='store_true');p.add_argument('--track-pose',action='store_true');p.add_argument('--fast-transforms',action='store_true');p.add_argument('--optimized',action='store_true');p.add_argument('--seconds',type=int,default=60);run(p.parse_args())
