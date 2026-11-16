import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/launcher'));sys.path.insert(0,str(ROOT/'tools/analysis'))
from pcb_pose_tracker import PoseTracker
from browser_coordinate_view import RTPAssembler


def board():
    image=np.full((720,1280,3),190,np.uint8)
    cv2.rectangle(image,(480,300),(700,370),(180,50,20),-1)
    for x in range(500,680,15):cv2.rectangle(image,(x,320),(x+5,345),(220,220,220),-1)
    return image


def test_tracking_known_translation_and_loss():
    tracker=PoseTracker();image=board();first,meta=tracker.update(image,1.)
    assert first['status']=='located'
    matrix=np.float32([[1,0,3],[0,1,2]]);moved=cv2.warpAffine(image,matrix,(1280,720),borderValue=(190,190,190))
    found,meta=tracker.update(moved,1.05)
    assert meta['source']=='tracked'
    expected=cv2.transform(np.float32([first['crop_quad']]),matrix)[0]
    assert np.max(np.abs(expected-found['crop_quad']))<1.
    absent,meta=tracker.update(np.full_like(image,190),1.1)
    assert absent['status']=='unknown' and tracker.found is None


def test_tracking_known_small_rotation():
    t=PoseTracker();im=board();initial,_=t.update(im,1.)
    matrix=cv2.getRotationMatrix2D((590,335),2.,1.)
    moved=cv2.warpAffine(im,matrix,(1280,720),borderValue=(190,190,190))
    found,meta=t.update(moved,1.05)
    assert meta['source']=='tracked'
    assert np.max(np.abs(cv2.transform(np.float32([initial['crop_quad']]),matrix)[0]-found['crop_quad']))<1.5


def test_large_jump_forces_relocalization():
    t=PoseTracker();im=board();t.update(im,1.)
    moved=cv2.warpAffine(im,np.float32([[1,0,120],[0,1,0]]),(1280,720),borderValue=(190,190,190))
    found,meta=t.update(moved,1.05)
    assert found['status']=='located' and meta['source']=='full'


def test_stale_pose_forces_relocalization():
    t=PoseTracker();t.update(board(),1.);_,meta=t.update(board(),1.3)
    assert meta['source']=='full'


def test_periodic_refresh():
    t=PoseTracker(refresh_seconds=.5);t.update(board(),1.);_,meta=t.update(board(),1.6)
    assert meta['source']=='full'


def packet(seq,payload,marker=True,ts=42):
    return bytes([0x80,96|(128 if marker else 0)])+seq.to_bytes(2,'big')+ts.to_bytes(4,'big')+b'\0\0\0\1'+payload


def test_single_rtp_nal():
    assert RTPAssembler().push(packet(1,b'\x65abc'))==b'\0\0\0\1\x65abc'


def test_fragmented_rtp_nal():
    a=RTPAssembler();assert a.push(packet(1,b'\x7c\x85abc',False)) is None
    assert a.push(packet(2,b'\x7c\x45def'))==b'\0\0\0\1\x65abcdef'


def test_packet_gap_fails_closed():
    a=RTPAssembler();a.push(packet(1,b'\x7c\x85abc',False));assert a.push(packet(3,b'\x7c\x45def')) is None
    assert a.gaps==1


def test_exact_frame_required_and_result_expires():
    import threading,time
    from browser_coordinate_view import BrowserReceiver
    hub=object.__new__(BrowserReceiver);hub.lock=threading.Lock();hub.result={'frame_sha256':'a'*64,'received_unix':time.time()};hub.frames={}
    assert hub.current()['state']=='unmatched' and hub.current()['image_url'] is None
    hub.frames['a'*64]=b'jpeg';assert hub.current()['state']=='matched'
    hub.result['received_unix']=time.time()-1;assert hub.current()['state']=='stale'


def test_coordinate_publication_hashes_matching_raw(tmp_path):
    import socket,json,hashlib
    from coordinate_publisher import CoordinatePublisher
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1',0));receiver.settimeout(2)
        publisher=CoordinatePublisher('127.0.0.1',receiver.getsockname()[1],tmp_path/'published.jsonl')
        try:
            publisher.submit({'sequence':3},b'matching-raw-pixels');result=json.loads(receiver.recvfrom(65536)[0])
            assert result['sequence']==3 and result['frame_sha256']==hashlib.sha256(b'matching-raw-pixels').hexdigest()
        finally:publisher.close()
        assert not publisher.errors


def test_coordinate_publisher_is_latest_only():
    import queue
    from coordinate_publisher import CoordinatePublisher
    p=object.__new__(CoordinatePublisher);p.queue=queue.Queue(maxsize=1);p.replaced=0
    p.submit({'sequence':1},b'old');p.submit({'sequence':2},b'new')
    assert p.queue.qsize()==1 and p.replaced==1 and p.queue.get()[0]['sequence']==2


def test_browser_clears_unmatched_overlays():
    text=(ROOT/'tools/analysis/coordinate_browser.html').read_text(encoding='utf-8')
    assert "r.state==='matched'" in text and 'clearRect' in text
    assert 'not stale boxes on moving video' in text
