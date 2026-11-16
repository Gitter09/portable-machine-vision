"""Pure crop-stream geometry and OpenCV4.6-compatible class-aware NMS."""
import ipaddress
import cv2
import numpy as np


def approved_receiver(host,port):
    """Destination explicitly supplied by the authenticated route-selecting host."""
    try:address=ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:return False
    networks=('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16','169.254.0.0/16')
    return port==5010 and any(address in ipaddress.ip_network(n) for n in networks) and not address.is_loopback and not address.is_multicast and not address.is_unspecified


def letterbox(image):
    h,w=image.shape[:2];scale=min(1280/w,1280/h);nw,nh=round(w*scale),round(h*scale)
    left=round((1280-nw)/2-.1);top=round((1280-nh)/2-.1)
    out=cv2.copyMakeBorder(cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR),top,1280-nh-top,left,1280-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    return out,np.float64([[nw/w,0,left],[0,nh/h,top],[0,0,1]])


def source_quad(xywh,inverse):
    x,y,w,h=xywh
    return cv2.perspectiveTransform(np.float32([[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]]),inverse)[0]


def detection_rows(prediction,threshold=.05,iou=.45):
    values=prediction[0].T
    ids=values[:,4:].argmax(axis=1);scores=values[np.arange(len(values)),ids+4];selected=scores>=threshold
    boxes=values[selected,:4].copy();boxes[:,:2]-=boxes[:,2:]/2;scores=scores[selected];ids=ids[selected]
    keep=[]
    for cid in range(4):
        indexes=np.flatnonzero(ids==cid)
        if len(indexes):
            kept=np.asarray(cv2.dnn.NMSBoxes(boxes[indexes].tolist(),scores[indexes].tolist(),threshold,iou),dtype=int).reshape(-1)
            keep.extend(indexes[kept].tolist())
    keep=sorted(keep,key=lambda j:(-float(scores[j]),j))[:300]
    return [{'class_id':int(ids[j]),'score':float(scores[j]),'model_xywh':boxes[j].tolist()} for j in keep]
