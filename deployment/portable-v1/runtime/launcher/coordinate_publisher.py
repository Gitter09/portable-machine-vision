"""Latest-only metadata publisher. Hashing/network I/O is off inference thread."""
import hashlib,json,queue,socket,threading,time


class CoordinatePublisher:
    def __init__(self,host,port,log_path):
        self.queue=queue.Queue(maxsize=1);self.sent=0;self.replaced=0;self.errors=[];self.host=host;self.port=port;self.log_path=log_path
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def submit(self,result,raw):
        item=(dict(result),raw)
        try:self.queue.put_nowait(item)
        except queue.Full:
            try:self.queue.get_nowait();self.replaced+=1
            except queue.Empty:pass
            self.queue.put_nowait(item)
    def run(self):
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock,self.log_path.open('w') as log:
                while True:
                    item=self.queue.get()
                    if item is None:break
                    result,raw=item;start=time.monotonic();result['frame_sha256']=hashlib.sha256(raw).hexdigest()
                    result['publication_ms']=1000*(time.monotonic()-start);result['published_unix']=time.time()
                    data=json.dumps(result,separators=(',',':')).encode()
                    if len(data)>60000:raise ValueError('coordinate payload too large')
                    sock.sendto(data,(self.host,self.port));log.write(json.dumps(result)+'\n');log.flush();self.sent+=1
        except Exception as e:self.errors.append(repr(e))
    def close(self):
        if self.thread.is_alive():
            try:self.queue.put(None,timeout=1)
            except queue.Full:self.errors.append('publisher drain timeout')
            self.thread.join(timeout=2)
            if self.thread.is_alive():self.errors.append('publisher still alive')
