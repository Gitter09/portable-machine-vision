"""Demand-only Aravis 0.8 capture; no continuous GStreamer frame queue.

Uses the installed C ABI because production images need not contain GI typelibs.
Device timestamps use the camera clock; trigger/receipt timestamps use board UTC.
"""
import ctypes as C
import time
from datetime import datetime, timezone

import cv2
import numpy as np


class CameraFreshnessError(RuntimeError):
    pass


class GError(C.Structure):
    _fields_ = [("domain", C.c_uint), ("code", C.c_int), ("message", C.c_char_p)]


class Aravis:
    def __init__(self):
        self.lib = C.CDLL("libaravis-0.8.so.0")
        self.gobject = C.CDLL("libgobject-2.0.so.0")
        self.glib = C.CDLL("libglib-2.0.so.0")
        self.gobject.g_object_unref.argtypes = [C.c_void_p]
        self.gobject.g_object_unref.restype = None
        self.glib.g_error_free.argtypes = [C.c_void_p]
        self.glib.g_error_free.restype = None
        P, S, I, U, Q = C.c_void_p, C.c_char_p, C.c_int, C.c_uint, C.c_uint64
        self._errors = set()
        # All camera functions below take GError** as their final argument.
        for name, ret, args in [
            ("camera_new", P, [S]), ("camera_stop_acquisition", None, [P]),
            ("camera_start_acquisition", None, [P]),
            ("camera_is_software_trigger_supported", I, [P]),
            ("camera_clear_triggers", None, [P]), ("camera_set_trigger", None, [P,S]),
            ("camera_get_trigger_source", S, [P]),
            ("camera_get_string", S, [P,S]),
            ("camera_set_pixel_format_from_string", None, [P,S]),
            ("camera_set_acquisition_mode", None, [P,I]),
            ("camera_set_frame_rate", None, [P,C.c_double]),
            ("camera_get_frame_rate", C.c_double, [P]),
            ("camera_get_exposure_time", C.c_double, [P]),
            ("camera_gv_auto_packet_size", U, [P]),
            ("camera_gv_set_packet_size", None, [P,I]),
            ("camera_get_payload", U, [P]),
            ("camera_create_stream", P, [P,P,P]),
            ("camera_software_trigger", None, [P]),
        ]:
            fn = getattr(self.lib, "arv_"+name)
            fn.restype, fn.argtypes = ret, args+[C.POINTER(C.c_void_p)]
            self._errors.add(name)
        for name, ret, args in [
            ("buffer_new_allocate", P, [C.c_size_t]),
            ("stream_push_buffer", None, [P,P]),
            ("stream_try_pop_buffer", P, [P]),
            ("stream_timeout_pop_buffer", P, [P,Q]),
            ("buffer_get_status", I, [P]), ("buffer_get_frame_id", Q, [P]),
            ("buffer_get_timestamp", Q, [P]), ("buffer_get_system_timestamp", Q, [P]),
            ("buffer_get_image_width", I, [P]), ("buffer_get_image_height", I, [P]),
            ("buffer_get_image_pixel_format", U, [P]),
            ("buffer_get_data", P, [P,C.POINTER(C.c_size_t)]),
        ]:
            fn = getattr(self.lib, "arv_"+name)
            fn.restype, fn.argtypes = ret, args

    def call(self, name, *args):
        fn = getattr(self.lib, "arv_"+name)
        if name not in self._errors:
            return fn(*args)
        error = C.c_void_p()
        result = fn(*args, C.byref(error))
        if error.value:
            message = C.cast(error, C.POINTER(GError)).contents.message.decode(errors="replace")
            self.glib.g_error_free(error)
            raise RuntimeError(name+": "+message)
        return result

    def unref(self, obj):
        if obj:
            self.gobject.g_object_unref(obj)


class TriggeredCamera:
    freshness_version = 1

    def __init__(self, camera_ip, mode="5fps", api=None):
        if mode not in {"5fps", "10fps-jumbo", "10fps-jumbo-trigger"}:
            raise ValueError("unsupported camera mode "+mode)
        self.camera_ip = camera_ip
        self.api = api or Aravis()
        self.mode = mode
        self.camera = self.stream = None
        self.poisoned = False
        self.last_frame_id = self.last_device_ns = None
        self.last_evidence = {}
        try:
            self.camera = self.api.call("camera_new", camera_ip.encode())
            if not self.camera:
                raise RuntimeError("camera not found")
            self.api.call("camera_stop_acquisition", self.camera)
            if not self.api.call("camera_is_software_trigger_supported", self.camera):
                raise RuntimeError("camera does not support software triggering; inspection disabled")
            self.api.call("camera_clear_triggers", self.camera)
            self.api.call("camera_set_pixel_format_from_string", self.camera, b"BayerRG8")
            self.api.call("camera_set_acquisition_mode", self.camera, 0)  # CONTINUOUS, trigger gated
            self.api.call("camera_set_frame_rate", self.camera, 5.0 if mode=="5fps" else 10.0)
            self.api.call("camera_set_trigger", self.camera, b"Software")
            if self.api.call("camera_get_trigger_source", self.camera) != b"Software":
                raise RuntimeError("software trigger source did not apply")
            if self.api.call("camera_get_string", self.camera, b"TriggerMode") != b"On":
                raise RuntimeError("trigger mode did not apply")
            if mode=="5fps":
                self.api.call("camera_gv_auto_packet_size", self.camera)
            else:
                self.api.call("camera_gv_set_packet_size", self.camera, 9000)
            self.settings = {"trigger_mode":"On", "trigger_source":"Software",
                             "configured_frame_rate_hz": self.api.call("camera_get_frame_rate",self.camera),
                             "exposure_time_us":self.api.call("camera_get_exposure_time",self.camera)}
            self.stream = self.api.call("camera_create_stream", self.camera, None, None)
            if not self.stream:
                raise RuntimeError("camera stream creation failed")
            size = self.api.call("camera_get_payload", self.camera)
            for _ in range(3):
                buffer = self.api.call("buffer_new_allocate", size)
                if not buffer:
                    raise RuntimeError("camera buffer allocation failed")
                self.api.call("stream_push_buffer", self.stream, buffer)
            self.api.call("camera_start_acquisition", self.camera)
            # Any unsolicited frame would invalidate trigger ownership.
            unexpected = self.api.call("stream_timeout_pop_buffer", self.stream, 300000)
            if unexpected:
                self.api.call("stream_push_buffer", self.stream, unexpected)
                raise CameraFreshnessError("camera produced a frame without a software trigger")
        except Exception:
            self.close()
            raise

    def drain(self):
        if self.poisoned or not self.stream:
            raise CameraFreshnessError("camera freshness is uncertain; restart the board worker")
        count = 0
        while True:
            buffer = self.api.call("stream_try_pop_buffer", self.stream)
            if not buffer:
                return count
            self.api.call("stream_push_buffer", self.stream, buffer)
            count += 1
            if count > 3:
                self.poisoned = True
                raise CameraFreshnessError("unexpected continuous frames in triggered camera mode")

    def capture_timed(self, timeout_seconds=2.0):
        started = time.perf_counter()
        discarded = self.drain()
        # A frame arriving between requests is not part of a known trigger.
        if discarded:
            self.poisoned = True
            raise CameraFreshnessError("unsolicited pending frame; restart the board worker")
        trigger_ns = time.time_ns()
        self.last_evidence = {"freshness_version":1, "acquisition_mode":"software_trigger",
                              "board_trigger_utc":datetime.now(timezone.utc).isoformat(),
                              "board_trigger_system_ns":trigger_ns, "discarded_pending_frames":discarded,
                              **self.settings}
        buffer = None
        try:
            self.api.call("camera_software_trigger", self.camera)
            triggered = time.perf_counter()
            buffer = self.api.call("stream_timeout_pop_buffer", self.stream, int(timeout_seconds*1000000))
            received = time.perf_counter()
            if not buffer:
                raise CameraFreshnessError("software-triggered frame timed out")
            if self.api.call("buffer_get_status", buffer) != 0:
                raise CameraFreshnessError("software-triggered frame is incomplete")
            frame_id = self.api.call("buffer_get_frame_id",buffer)
            device_ns = self.api.call("buffer_get_timestamp",buffer)
            system_ns = self.api.call("buffer_get_system_timestamp",buffer)
            self.last_evidence.update(camera_frame_id=frame_id,camera_timestamp_ns=device_ns,
                                      board_receive_system_ns=system_ns,
                                      board_received_utc=datetime.now(timezone.utc).isoformat())
            if not device_ns or system_ns < trigger_ns or frame_id==self.last_frame_id:
                raise CameraFreshnessError("stale or unidentified software-triggered frame")
            if self.last_device_ns is not None and device_ns <= self.last_device_ns:
                raise CameraFreshnessError("camera acquisition timestamp did not advance")
            width,height=(self.api.call("buffer_get_image_"+n,buffer) for n in ("width","height"))
            if (width,height)!=(3200,2256) or self.api.call("buffer_get_image_pixel_format",buffer)!=0x01080009:
                raise CameraFreshnessError("unexpected triggered frame geometry or pixel format")
            size = C.c_size_t()
            data = self.api.call("buffer_get_data",buffer,C.byref(size))
            if not data or size.value != width*height:
                raise CameraFreshnessError("unexpected triggered frame payload size")
            raw=np.frombuffer(C.string_at(data,size.value),dtype=np.uint8).reshape(height,width)
            # OpenCV's RGGB naming is reversed relative to its short Bayer names.
            frame=cv2.cvtColor(raw,cv2.COLOR_BayerRGGB2BGR)
            self.last_frame_id,self.last_device_ns=frame_id,device_ns
            self.last_evidence["freshness_verified"]=True
            return frame, {"software_trigger_transmission":(triggered-started)*1000,
                           "exposure_frame_wait_and_camera_transfer":(received-triggered)*1000,
                           "buffer_mapping_and_memory_copy":(time.perf_counter()-received)*1000}
        except Exception:
            # A late reply to a timed-out trigger must never satisfy another request.
            self.poisoned=True
            self.last_evidence["freshness_verified"]=False
            raise
        finally:
            if buffer:
                self.api.call("stream_push_buffer",self.stream,buffer)

    def capture(self, timeout_seconds=2.0):
        return self.capture_timed(timeout_seconds)[0]

    def close(self):
        if self.camera:
            try:
                self.api.call("camera_stop_acquisition",self.camera)
            except Exception:
                pass
        self.api.unref(self.stream)
        self.api.unref(self.camera)
        self.stream=self.camera=None
