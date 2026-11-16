from vision_sdk.board_runtime.triggered_camera import TriggeredCamera


class FakeAravis:
    def call(self, name, *args):
        return {
            "camera_new": 1,
            "camera_is_software_trigger_supported": 1,
            "camera_get_trigger_source": b"Software",
            "camera_get_string": b"On",
            "camera_get_frame_rate": 5.0,
            "camera_get_exposure_time": 100000.0,
            "camera_get_payload": 1024,
            "camera_create_stream": 2,
            "buffer_new_allocate": 3,
        }.get(name)

    def unref(self, value):
        pass


def test_triggered_camera_retains_configured_ip_for_ready_message():
    camera = TriggeredCamera("192.168.1.10", api=FakeAravis())

    assert camera.camera_ip == "192.168.1.10"
    assert camera.mode == "5fps"

    camera.close()
