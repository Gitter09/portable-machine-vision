from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.ic_terminal_dataset import (
    CLASS_NAMES,
    Frame,
    Run,
    Transform,
    assign_splits,
    cover_centre,
    fit_rotation_axis,
    sector_of,
)


def frame(path: Path | None, index: int, boxes: list[tuple[int, float, float, float, float]],
          run: str = "r", size: tuple[int, int] = (1280, 720)) -> Frame:
    return Frame(run, index, path, size[0], size[1], boxes)  # type: ignore[arg-type]


def solid(path: Path, width: int, height: int, value: int = 7) -> Path:
    cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))
    return path


def test_letterbox_puts_boxes_in_the_padded_canvas(tmp_path):
    image = solid(tmp_path / "frame.png", 1280, 720)
    transform = Transform("letterbox", 640, (640.0, 360.0), 1280, 720)
    boxes = [(2, 100.0, 100.0, 20.0, 20.0)]
    canvas, mapped, dropped = transform.apply(frame(image, 1, boxes))
    assert canvas.shape == (640, 640, 3)
    assert dropped == 0
    class_id, x, y, width, height = mapped[0]
    assert class_id == 2
    assert (round(x, 3), round(y, 3)) == (50.0, 190.0)
    assert (round(width, 3), round(height, 3)) == (10.0, 10.0)


def test_letterbox_pads_with_the_documented_grey(tmp_path):
    image = solid(tmp_path / "frame.png", 1280, 720, value=0)
    transform = Transform("letterbox", 640, (640.0, 360.0), 1280, 720)
    canvas, _, _ = transform.apply(frame(image, 1, []))
    assert int(canvas[0, 320, 0]) == 128
    assert int(canvas[-1, 320, 0]) == 128
    assert int(canvas[320, 320, 0]) == 0


def test_crop_maps_boxes_through_the_window(tmp_path):
    image = solid(tmp_path / "frame.png", 200, 200)
    transform = Transform("crop", 100, (100.0, 100.0), 200, 200)
    assert transform.window() == (50, 50, 100)
    assert transform.project((100.0, 100.0)) == (320.0, 320.0)
    boxes = [(2, 60.0, 60.0, 10.0, 10.0)]
    _, mapped, dropped = transform.apply(frame(image, 1, boxes, size=(200, 200)))
    assert dropped == 0
    _, x, y, width, height = mapped[0]
    assert (round(x, 2), round(y, 2)) == (64.0, 64.0)
    assert (round(width, 2), round(height, 2)) == (64.0, 64.0)


def test_crop_drops_a_box_that_leaves_the_window(tmp_path):
    image = solid(tmp_path / "frame.png", 200, 200)
    transform = Transform("crop", 100, (100.0, 100.0), 200, 200)
    boxes = [(2, 190.0, 60.0, 10.0, 10.0)]
    _, mapped, dropped = transform.apply(frame(image, 1, boxes, size=(200, 200)))
    assert dropped == 1
    assert mapped == []


def test_crop_keeps_no_padding(tmp_path):
    image = solid(tmp_path / "frame.png", 200, 200, value=7)
    transform = Transform("crop", 100, (100.0, 100.0), 200, 200)
    canvas, _, _ = transform.apply(frame(image, 1, [], size=(200, 200)))
    assert canvas.shape == (640, 640, 3)
    assert int(canvas.min()) == 7 and int(canvas.max()) == 7


def test_crop_window_is_clamped_inside_the_frame():
    transform = Transform("crop", 400, (10.0, 10.0), 1280, 720)
    x0, y0, size = transform.window()
    assert x0 == 0 and y0 == 0 and size == 400


def test_crop_larger_than_the_frame_is_rejected(tmp_path):
    image = solid(tmp_path / "frame.png", 200, 200)
    transform = Transform("crop", 400, (100.0, 100.0), 200, 200)
    with pytest.raises(SystemExit):
        transform.apply(frame(image, 1, [], size=(200, 200)))


def test_fit_rotation_axis_recovers_a_known_centre():
    centre_x, centre_y, radius = 640.0, 360.0, 200.0
    frames = []
    for step in range(12):
        angle = math.radians(step * 30)
        x = centre_x + radius * math.cos(angle)
        y = centre_y + radius * math.sin(angle)
        frames.append(frame(None, step, [(2, x - 10, y - 10, 20, 20)]))
    found_x, found_y = fit_rotation_axis(frames)
    assert abs(found_x - centre_x) < 0.5
    assert abs(found_y - centre_y) < 0.5


def test_every_split_receives_blocks_from_every_run():
    runs = [
        Run("present", None, [frame(None, index, [(2, 100.0, 100.0, 20.0, 20.0)], "present")
                              for index in range(60)]),  # type: ignore[arg-type]
        Run("absent", None, [frame(None, index, [(0, 100.0, 100.0, 20.0, 20.0)], "absent")
                             for index in range(60)]),  # type: ignore[arg-type]
    ]
    splits = assign_splits(runs, blocks=6, guard=0, fractions=(0.7, 0.15, 0.15))
    for split in ("train", "val", "test"):
        members = [key for key, value in splits.items() if value == split]
        assert members, f"{split} is empty"
        assert {key.split(":")[0] for key in members} == {"present", "absent"}, \
            f"{split} is missing a whole run, and therefore whole classes"


def test_guard_band_discards_frames_near_a_boundary():
    run = Run("present", None, [frame(None, index, [(2, 100.0, 100.0, 20.0, 20.0)], "present")
                                for index in range(60)])  # type: ignore[arg-type]
    unguarded = assign_splits([run], blocks=6, guard=0, fractions=(0.7, 0.15, 0.15))
    guarded = assign_splits([run], blocks=6, guard=3, fractions=(0.7, 0.15, 0.15))
    assert len(guarded) < len(unguarded)
    assert len(unguarded) == 60


def test_sector_covers_the_full_revolution():
    transform = Transform("crop", 100, (100.0, 100.0), 200, 200)
    axis = transform.project((100.0, 100.0))
    seen = set()
    for step in range(36):
        angle = math.radians(step * 10 + 5)
        x = axis[0] + 40 * math.cos(angle)
        y = axis[1] + 40 * math.sin(angle)
        seen.add(sector_of(axis, (2, x - 5, y - 5, 10, 10)))
    assert seen == set(range(0, 360, 10))


def test_cover_centre_balances_runs_that_do_not_share_an_axis():
    # Two runs at one axis, one nudged 60 px away: the centre must move toward the
    # outlier, not sit on the majority.
    items = [(658.5, 321.9, 235.0), (658.5, 321.9, 235.0), (606.2, 347.3, 266.0)]
    centre_x, centre_y = cover_centre(items)
    worst = max(math.hypot(centre_x - x, centre_y - y) + r for x, y, r in items)
    mean_x = sum(x for x, _, _ in items) / 3
    mean_y = sum(y for _, y, _ in items) / 3
    worst_mean = max(math.hypot(mean_x - x, mean_y - y) + r for x, y, r in items)
    assert worst <= worst_mean
    assert worst < 300


def test_cover_centre_is_the_axis_when_all_runs_agree():
    items = [(100.0, 200.0, 50.0), (100.0, 200.0, 60.0)]
    assert cover_centre(items) == (100.0, 200.0)


def test_class_order_is_the_documented_contract():
    assert CLASS_NAMES == ["IC_Fail", "Terminal_pass", "IC_Pass", "Terminal_Fail"]
