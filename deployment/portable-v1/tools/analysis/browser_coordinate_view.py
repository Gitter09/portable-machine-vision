"""Host RTP decode and bounded exact-frame coordinate matching. No board JPEGs.
The second browser pane draws on a hash-matched inspection frame, NOT stale live.
"""
import collections,hashlib,json,socket,threading,time
import cv2
import numpy as np


class RTPAssembler:
    def __init__(self):self.timestamp=None;self.sequence=None;self.data=bytearray();self.bad=False;self.fu=False;self.gaps=0
    def push(self,packet):
        if len(packet)<12 or packet[0]>>6!=2:return None
        seq=int.from_bytes(packet[2:4],'big');ts=int.from_bytes(packet[4:8],'big');offset=12+4*(packet[0]&15)
        if packet[0]&16:
            if len(packet)<offset+4:return None
            offset+=4+4*int.from_bytes(packet[offset+2:offset+4],'big')
        end=len(packet)-(packet[-1] if packet[0]&32 else 0)
        payload=packet[offset:end]
        if not payload:return None
        gap=self.sequence is not None and seq!=(self.sequence+1)&65535
        self.sequence=seq
        if self.timestamp!=ts:self.timestamp=ts;self.data=bytearray();self.bad=False;self.fu=False
        if gap:self.gaps+=1;self.bad=True
        typ=payload[0]&31
        if 1<=typ<=23:self.data+=b'\0\0\0\1'+payload
        elif typ==24:
            n=1
            while n+2<=len(payload):
                size=int.from_bytes(payload[n:n+2],'big');n+=2
                if not size or n+size>len(payload):self.bad=True;break
                self.data+=b'\0\0\0\1'+payload[n:n+size];n+=size
        elif typ==28 and len(payload)>=2:
            if payload[1]&128:self.fu=True;self.data+=b'\0\0\0\1'+bytes([(payload[0]&224)|(payload[1]&31)])+payload[2:]
            elif self.fu:self.data+=payload[2:]
            else:self.bad=True
            if payload[1]&64:self.fu=False
        else:self.bad=True
        if packet[1]&128:
            result=bytes(self.data) if not self.bad and not self.fu else None
            self.data=bytearray();return result
        return None


class BrowserReceiver:
    def __init__(self,host,port=5010,audit_dir=None):
        import av
        self.av=av;self.video=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.video.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024);self.video.bind((host,port));self.video.settimeout(.2)
        self.meta=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.meta.bind((host,port+2));self.meta.settimeout(.2)
        self.audit_dir=audit_dir;self.raw_frames=collections.OrderedDict()
        if audit_dir:audit_dir.mkdir(parents=True,exist_ok=False)
        self.stop=threading.Event();self.lock=threading.Lock();self.frames=collections.OrderedDict();self.latest=None;self.result=None
        self.assembler=RTPAssembler();self.decoded=0;self.results=0;self.matched=0;self.errors=[];self.last_arrival=None;self.first_arrival=None;self.gaps=[]
        self.threads=[];self.started=time.monotonic()
    def start(self):
        for target in (self.read_video,self.read_results):
            t=threading.Thread(target=target,daemon=True);t.start();self.threads.append(t)
    def read_video(self):
        codec=self.av.CodecContext.create('h264','r');codec.thread_count=1
        try:
            while not self.stop.is_set():
                try:packet,_=self.video.recvfrom(65536)
                except socket.timeout:continue
                access=self.assembler.push(packet)
                if access is None:continue
                try:frames=codec.decode(self.av.Packet(access))
                except self.av.error.InvalidDataError:continue
                for frame in frames:
                    nv12=frame.reformat(format='nv12').to_ndarray();key=hashlib.sha256(nv12.tobytes()).hexdigest()
                    image=cv2.cvtColor(nv12,cv2.COLOR_YUV2BGR_NV12);ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])
                    if not ok:continue
                    now=time.monotonic()
                    with self.lock:
                        if self.last_arrival is not None:self.gaps.append(now-self.last_arrival)
                        self.last_arrival=now
                        if self.first_arrival is None:self.first_arrival=now
                        self.frames[key]=jpg.tobytes();self.frames.move_to_end(key)
                        while len(self.frames)>90:self.frames.popitem(last=False)
                        self.latest=key;self.decoded+=1
                        if self.audit_dir:
                            self.raw_frames[key]=nv12.tobytes();self.raw_frames.move_to_end(key)
                            while len(self.raw_frames)>30:self.raw_frames.popitem(last=False)
        except Exception as e:self.errors.append(repr(e));self.stop.set()
    def read_results(self):
        try:
            while not self.stop.is_set():
                try:data,_=self.meta.recvfrom(65536)
                except socket.timeout:continue
                result=json.loads(data);key=result.get('frame_sha256')
                if not isinstance(key,str) or len(key)!=64:continue
                with self.lock:
                    result['received_unix']=time.time();self.result=result;self.results+=1
                    if key in self.frames:self.matched+=1
                    audit_raw=self.raw_frames.get(key) if self.audit_dir and result['sequence']%25==0 else None
                if audit_raw is not None:
                    (self.audit_dir/f"{result['sequence']:06}.nv12").write_bytes(audit_raw)
                    (self.audit_dir/f"{result['sequence']:06}.json").write_text(json.dumps(result,indent=2))
        except Exception as e:self.errors.append(repr(e));self.stop.set()
    def current(self):
        with self.lock:
            if self.result is None:return {'state':'waiting','detections':[]}
            r=dict(self.result);r['matched']=r['frame_sha256'] in self.frames
            r['state']='stale' if time.time()-r['received_unix']>.5 else ('matched' if r['matched'] else 'unmatched')
            r['image_url']='/frames/'+r['frame_sha256']+'.jpg' if r['matched'] else None
            return r
    def image(self,key=None):
        with self.lock:return self.frames.get(key or self.latest)
    def summary(self):
        with self.lock:return {'decoded_frames':self.decoded,'metadata_results':self.results,'matched_at_arrival':self.matched,
            'rtp_sequence_gaps':self.assembler.gaps,'host_errors':self.errors,'buffered_frames':len(self.frames),
            'max_host_frame_gap_seconds':max(self.gaps,default=None),
            'decoded_arrival_fps':(self.decoded-1)/(self.last_arrival-self.first_arrival) if self.decoded>1 else None,
            'p99_host_frame_gap_seconds':float(np.percentile(self.gaps,99)) if self.gaps else None,'elapsed_seconds':time.monotonic()-self.started}
    def close(self):
        self.stop.set()
        for t in self.threads:t.join(timeout=2)
        self.video.close();self.meta.close()
