"""Bounded crop-aware MLA streaming DIAGNOSTIC (software transforms/encoder).
Not the native 30fps graph. One hardware decoder; no modification of r33.
"""
from __future__ import annotations
import argparse,json,time
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
from crop_geometry import detection_rows,letterbox,source_quad,approved_receiver


def check_bus(pipeline):
    message=pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR|Gst.MessageType.EOS)
    if message:
        if message.type==Gst.MessageType.ERROR:raise RuntimeError(str(message.parse_error()))
        raise RuntimeError('unexpected EOS')


def run(args):
    if not approved_receiver(args.host,args.port):raise ValueError('receiver not owner-approved')
    if not 0<args.seconds<=300:raise ValueError('diagnostic must be bounded to <=300 seconds')
    inventory=json.loads((args.release/'release.json').read_text())
    for name,sha in inventory['assets'].items():
        path=(args.release/'assets'/name).resolve()
        if not path.is_relative_to((args.release/'assets').resolve()) or digest(path)!=sha:raise ValueError('asset hash/path differs')
    mpk=args.release/'assets/provenance/mpk.json';contract=SoftwareTransforms(json.loads(mpk.read_text()),transforms);contract.check_output_layouts()
    runner=SimaModelRunner(Parser(str(mpk)));runner.configure({'model_path':str(args.release/'assets/share/processmla/yolov8_stage1_mla.elf')})
    args.output.mkdir(parents=True,exist_ok=False)
    Gst.init(None)
    capture=Gst.parse_launch('rtspsrc name=camera protocols=tcp latency=100 ! rtph264depay wait-for-keyframe=true ! h264parse ! video/x-h264,parsed=true,stream-format=byte-stream,alignment=au ! simaaidecoder name=decoder sima-allocator-type=2 next-element=CVU ! video/x-raw,format=NV12,width=1280,height=720 ! appsink name=input sync=false max-buffers=1 drop=true')
    capture.get_by_name('camera').set_property('location',args.rtsp)
    transmit=Gst.parse_launch(f'appsrc name=output is-live=true format=time do-timestamp=true block=false caps="video/x-raw,format=BGR,width=1280,height=720,framerate=5/1" ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! video/x-raw,format=I420 ! openh264enc bitrate=4000000 gop-size=10 complexity=low ! h264parse name=encoded ! tee name=encodedtee ! queue ! rtph264pay name=pay pt=96 config-interval=1 ! udpsink host={args.host} port={args.port} sync=false async=false encodedtee. ! queue ! filesink location="{args.output / "encoded.h264"}" sync=false')
    counters={'encoded_buffers':0,'rtp_packets':0}
    def count_buffer(pad,info,key):
        batch=info.get_buffer_list() if info.type & Gst.PadProbeType.BUFFER_LIST else None
        counters[key]+=batch.length() if batch is not None else 1
        return Gst.PadProbeReturn.OK
    transmit.get_by_name('encoded').get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER,count_buffer,'encoded_buffers')
    transmit.get_by_name('pay').get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER|Gst.PadProbeType.BUFFER_LIST,count_buffer,'rtp_packets')
    sink=capture.get_by_name('input');source=transmit.get_by_name('output');start=time.monotonic();frames=0;located=0;snapshots=0;errors=[]
    try:
        for pipeline in (transmit,capture):
            if pipeline.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:raise RuntimeError('pipeline refused PLAYING')
        with (args.output/'frames.jsonl').open('w') as records:
            while time.monotonic()-start<args.seconds:
                check_bus(capture);check_bus(transmit)
                sample=sink.emit('try-pull-sample',Gst.SECOND)
                if sample is None:continue
                began=time.monotonic();buffer=sample.get_buffer();raw=buffer.extract_dup(0,buffer.get_size())
                if len(raw)!=1280*720*3//2:raise ValueError('NV12 byte count differs')
                image=cv2.cvtColor(np.frombuffer(raw,np.uint8).reshape(1080,1280),cv2.COLOR_YUV2BGR_NV12)
                original=image.copy();found=hybrid.select_all(image)[0]['colour_seed_canny'];detections=[];timings={}
                if found['status']=='located':
                    crop,_,matrix=hybrid.base.crop_pixels(image,found['crop_quad']);model,lm=letterbox(crop);inverse=np.linalg.inv(lm@matrix)
                    tensor=np.ascontiguousarray(model[:,:,::-1],dtype=np.float32)/np.float32(255)
                    t=time.monotonic();packed=contract.encode(tensor);q=time.monotonic();out=runner.run(packed);r=time.monotonic();heads=contract.decode(out)
                    timings={'quantize_ms':1000*(q-t),'mla_ms':1000*(r-q),'unpack_ms':1000*(time.monotonic()-r)}
                    detections=detection_rows(decode_heads(heads,1280,'NHWC'),threshold=.05,iou=.45)[:100]
                    for det in detections:
                        quad=source_quad(det['model_xywh'],inverse);det['source_quad']=quad.tolist();color=(0,0,255) if det['class_id'] in (0,3) else (0,220,0)
                        cv2.polylines(image,[np.int32(np.round(quad))],True,color,2);x,y=np.int32(quad.min(axis=0));cv2.putText(image,f"{LABELS[det['class_id']]} {det['score']:.2f}",(max(0,x),max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
                    if snapshots<3:
                        stem=args.output/f'sample-{snapshots}';cv2.imwrite(str(stem.with_suffix('.source.png')),original);cv2.imwrite(str(stem.with_suffix('.model.png')),model)
                        np.savez_compressed(stem.with_suffix('.heads.npz'),**{f'head_{i}':h for i,h in enumerate(heads)})
                        (stem.with_suffix('.json')).write_text(json.dumps({'frame':frames,'source_to_model':(lm@matrix).tolist(),'detections':detections,'crop_quad':found['crop_quad'],'model_sha256':digest(stem.with_suffix('.model.png'))},indent=2));snapshots+=1
                    located+=1
                else:cv2.putText(image,'UNKNOWN - PCB crop not located',(25,90),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,220,255),2)
                cv2.putText(image,'CROP MLA DIAGNOSTIC - software path, not 30fps',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
                data=image.tobytes();output=Gst.Buffer.new_allocate(None,len(data),None);output.fill(0,data);output.duration=Gst.SECOND//5
                if source.emit('push-buffer',output)!=Gst.FlowReturn.OK:raise RuntimeError('stream push-buffer failed')
                record={'frame':frames,'elapsed_seconds':time.monotonic()-start,'source_pts_ns':int(buffer.pts),'crop_status':found['status'],
                        'crop_reason':found.get('reason'),'detections':detections,'timings':timings,'processing_ms':1000*(time.monotonic()-began)}
                records.write(json.dumps(record)+'\n');records.flush();frames+=1
    except Exception as e:errors.append(repr(e));raise
    finally:
        source.emit('end-of-stream');time.sleep(.3)
        for pipeline in (capture,transmit):pipeline.set_state(Gst.State.NULL)
        elapsed=time.monotonic()-start
        (args.output/'worker.json').write_text(json.dumps({'state':'error' if errors or not frames else 'completed','frames':frames,'located':located,'unknown':frames-located,
            'elapsed_seconds':elapsed,'processed_fps':frames/elapsed,'errors':errors,**counters,'software_diagnostic':True,
            'elf_sha256':digest(args.release/'assets/share/processmla/yolov8_stage1_mla.elf'),'mpk_sha256':digest(mpk)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rtsp',required=True)
    p.add_argument('--host',default='172.21.30.13');p.add_argument('--port',type=int,default=5010);p.add_argument('--seconds',type=float,default=60)
    run(p.parse_args())
