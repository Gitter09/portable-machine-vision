"""Independent compressed-video branch; bounded latest-only decoded AI input."""
from crop_geometry import approved_receiver


def graph(host,port):
    if not approved_receiver(host,port):raise ValueError('receiver not owner-approved')
    return (
        'rtspsrc name=camera protocols=tcp latency=100 ! '
        'rtph264depay wait-for-keyframe=true ! h264parse ! '
        'video/x-h264,parsed=true,stream-format=byte-stream,alignment=au ! tee name=compressed '
        'compressed. ! queue max-size-buffers=8 ! identity name=video_count silent=true ! '
        f'rtph264pay pt=96 config-interval=1 ! udpsink host={host} port={port} sync=false async=false '
        'compressed. ! queue max-size-buffers=4 ! '
        'simaaidecoder name=decoder sima-allocator-type=2 next-element=CVU ! '
        'video/x-raw,format=NV12,width=1280,height=720 ! '
        'appsink name=input sync=false max-buffers=1 drop=true')
