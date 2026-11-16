"""Product-agnostic preprocessor.

Serves every product the model must detect, rather than one product on one rig.
The fixed-reference pipeline assumed a golden image of a powerboard, a hard-coded
crop calibrated to that board in its metal fixture, and enhancement tuned against
one green PCB under one light level. Measured on the supplied raw frames, that
reference matches 289-314 ORB features on the powerboard and 0-26 on every other
family, so new products are rejected before the model is reached.

The rig, unlike the part, is common. The part is the bright pool; the surround is
near-black; the LED rail is a bright band touching the top edge. Locating the part
is therefore a photometric problem, not a product-recognition problem, and it
generalises.

Stages, each independently switchable so the pipeline can be ablated:

    S0  unbar          strip fully-black pillarbox bars
    S1  downscale      build the localisation domain only
    S2  locate         rail mask -> Otsu -> morphology -> largest central region
    S3  flat-field     divide by a stored empty-fixture illumination map
    S4  tone           gain so p99 lands on a fixed target, leaving CLAHE headroom
    S5  clahe          LAB L only, after the geometric downscale
    S6  letterbox      pad to the model size
    S7  tensor         BGR -> RGB, /255, NCHW float32

Decisions behind this file are recorded in docs/decisions/log.md.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BasePreprocessor, Box

MODEL_SIZE = 640

# The supplied raw frames are 1280x720 with a 129 px pure-black bar on each side,
# leaving 1021x720 of content. Localising in that domain is ~8x cheaper than at
# 3200x2256 and loses nothing: the located box is rescaled back to source pixels
# and the crop is taken from the full-resolution frame.
DEFAULT_LOCATE_WIDTH = 1021
# Tuned against the operator's own capture of 60 frames. Lower values admit the
# deck's woven texture, which bridges neighbouring parts into one blob; higher
# values start losing genuinely dark parts. 18 gave the best separation.
DEFAULT_INSTANCE_THRESHOLD = 18.0
# An instance this much larger than the median is reported as possibly merged.
MERGED_AREA_FACTOR = 4.0
# A lone instance this much wider than tall is more likely a run of touching
# parts than a genuine part, since parts are roughly square.
MERGED_ASPECT_FACTOR = 3.0


class PartNotFoundError(RuntimeError):
    """No plausible part was found and no fallback window is configured.

    Raised rather than returned, so a misconfigured or unlit scene becomes a
    rejected part instead of a confident detection from the wrong pixels.
    """


@dataclass(frozen=True)
class ProductProfile:
    """Preprocessing configuration.

    Product identity comes from the model, not from upstream, so there is no
    per-product profile. One profile serves the general path; `crop_mode` is
    "fixed" only for a release pinned to a physically fixed rig.
    """

    name: str = "general"
    crop_mode: str = "auto"
    fixed_roi: tuple[int, int, int, int] | None = None
    fallback_roi: tuple[int, int, int, int] | None = None
    margin_frac: float = 0.06

    # S0/S2 localisation
    bar_max_fraction: float = 0.25
    bar_value: int = 2
    bar_column_fraction: float = 0.999
    locate_width: int = DEFAULT_LOCATE_WIDTH
    min_part_area_frac: float = 0.03
    rail_mask_rows: float = 0.30
    rail_percentile: float = 92.0
    rail_dilate: int = 5
    locate_floor: int = 28
    close_kernel: int = 21
    open_kernel: int = 11
    centre_y_range: tuple[float, float] = (0.28, 0.92)

    # Per-instance locating, used when several parts arrive at once. These are
    # deliberately relative to the frame rather than absolute: an absolute floor
    # is an undeclared assumption that parts are large, and a bottle cap is 0.64
    # percent of the frame, which such a floor refuses outright.
    instance_threshold: float = DEFAULT_INSTANCE_THRESHOLD
    min_instance_area_frac: float = 0.0004
    instance_open_kernel: int = 5
    instance_close_kernel: int = 9

    # S3 flat field
    flatfield_lut: str | None = None
    flatfield_gain_cap: float = 3.3

    # S3b presence. Closes the gap the locator cannot: an empty fixture is itself
    # a lit region, so the locator returns a confident box when no part is there.
    # None defers to the stored map's calibrated threshold; a value overrides it.
    stage_presence: bool = True
    presence_threshold: float | None = None

    # S4 tone
    tone_target_p99: int = 140

    # S5 CLAHE
    clahe_clip: float = 1.6
    clahe_tiles: tuple[int, int] = (8, 8)

    # S6 letterbox
    pad_colour: tuple[int, int, int] = (114, 114, 114)

    # Stage switches, for ablation
    stage_unbar: bool = True
    stage_locate: bool = True
    stage_flatfield: bool = True
    stage_tone: bool = True
    stage_clahe: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProductProfile":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown preprocessing profile fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = dict(value)
        for name in ("fixed_roi", "fallback_roi"):
            if kwargs.get(name) is not None:
                kwargs[name] = _as_roi(kwargs[name], name)
        for name in ("centre_y_range", "clahe_tiles", "pad_colour"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        if kwargs.get("crop_mode", "auto") not in {"auto", "fixed", "instances"}:
            raise ValueError("crop_mode must be 'auto', 'fixed' or 'instances'")
        if kwargs.get("crop_mode") == "fixed" and kwargs.get("fixed_roi") is None:
            raise ValueError("crop_mode 'fixed' requires fixed_roi")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "ProductProfile":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _as_roi(value: Any, name: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in value)
    if len(parts) != 4:
        raise ValueError(f"{name} must be [x, y, width, height]")
    x, y, width, height = parts
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{name} must have x, y >= 0 and width, height > 0")
    return parts


# --------------------------------------------------------------------------
# Flat-field map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatField:
    """A stored illumination field derived from an empty-fixture capture.

    The map is a small low-pass of the empty fixture's luminance, normalised to
    unity at its own bright end, so it is a pure multiplicative field. The gain
    it implies is capped: measured uncapped, the field reached a 9.4x gain which
    lifted the surround into the part and destroyed the threshold separation the
    locator depends on. See docs/decisions/log.md D-14.
    """

    lut: np.ndarray
    frame_width: int
    frame_height: int
    gain_cap: float
    created_utc: str
    #: LAB L of the empty fixture, downsampled. None in a map written before the
    #: presence check existed, in which case presence cannot be verified.
    empty_fixture: np.ndarray | None = None
    #: Similarity below which a window is judged to hold no part.
    presence_threshold: float = 0.18


#: Longest edge the stored empty fixture is kept at. Small enough to keep the map
#: compact, large enough that a part still differs from bare fixture.
EMPTY_FIXTURE_WIDTH = 512

#: Both the live window and the stored fixture are reduced to this before
#: comparison, so the cost does not depend on the crop size.
PRESENCE_PROBE = (64, 48)


def build_flatfield(
    frames: list[np.ndarray],
    gain_cap: float = 3.3,
    presence_threshold: float = 0.18,
) -> FlatField:
    """Derive an illumination field and an empty-fixture reference from frames.

    Averaging suppresses sensor noise; a heavy low-pass then leaves the smooth
    illumination term. A polynomial fit was tried and rejected: it goes negative
    at the corners, implying a nonsensical centre-to-corner ratio.

    The median frame is also kept as the empty fixture, because it is what the
    presence check compares against. It is the same capture, so it cannot drift
    away from the field it was derived from.
    """
    if not frames:
        raise ValueError("Cannot build a flat-field map without frames")
    shape = frames[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frames):
        raise ValueError("Every empty-fixture frame must be the same size")
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    luminance = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    smooth = cv2.GaussianBlur(luminance, (0, 0), 40)
    reference = float(np.percentile(smooth, 97))
    if reference <= 1.0:
        raise ValueError("Empty-fixture frames are too dark to derive a flat field")
    lut = cv2.resize(
        smooth / reference, (128, 86), interpolation=cv2.INTER_AREA
    )
    lut = np.clip(lut, 1.0 / max(gain_cap, 1.0), None)
    fixture_width = min(EMPTY_FIXTURE_WIDTH, shape[1])
    fixture_height = max(1, int(round(shape[0] * fixture_width / shape[1])))
    return FlatField(
        lut=lut.astype(np.float32),
        frame_width=shape[1],
        frame_height=shape[0],
        gain_cap=gain_cap,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        empty_fixture=cv2.resize(
            luminance, (fixture_width, fixture_height), interpolation=cv2.INTER_AREA
        ).astype(np.uint8),
        presence_threshold=presence_threshold,
    )


def save_flatfield(flatfield: FlatField, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lut": flatfield.lut,
        "frame_width": np.array([flatfield.frame_width]),
        "frame_height": np.array([flatfield.frame_height]),
        "gain_cap": np.array([flatfield.gain_cap]),
        "created_utc": np.array([flatfield.created_utc]),
        "presence_threshold": np.array([flatfield.presence_threshold]),
    }
    if flatfield.empty_fixture is not None:
        payload["empty_fixture"] = flatfield.empty_fixture
    np.savez_compressed(path, **payload)
    return path


def load_flatfield(path: str | Path) -> FlatField:
    data = np.load(str(path), allow_pickle=False)
    return FlatField(
        lut=np.asarray(data["lut"], dtype=np.float32),
        frame_width=int(data["frame_width"][0]),
        frame_height=int(data["frame_height"][0]),
        gain_cap=float(data["gain_cap"][0]),
        created_utc=str(data["created_utc"][0]),
        empty_fixture=(
            np.asarray(data["empty_fixture"])
            if "empty_fixture" in data.files
            else None
        ),
        presence_threshold=(
            float(data["presence_threshold"][0])
            if "presence_threshold" in data.files
            else 0.18
        ),
    )


def normalised_mad(left: np.ndarray, right: np.ndarray) -> float:
    """Level-invariant mean absolute difference between two images.

    A correlation score was tried and rejected: measured over the supplied frames,
    whole-frame correlation of a frame that does contain a part reached 0.977,
    which is indistinguishable from a match. This measure separated cleanly, with
    the lowest value over 58 part-bearing frames at 0.286 against 0.0 for a frame
    compared against itself.
    """
    a = left.astype(np.float32) - float(left.mean())
    b = right.astype(np.float32) - float(right.mean())
    denominator = float(np.abs(b).mean()) + 1e-6
    return float(np.abs(a - b).mean() / denominator)


def presence_similarity(
    image_bgr: np.ndarray,
    window: tuple[int, int, int, int],
    flatfield: FlatField,
) -> float | None:
    """How much a window differs from the stored empty fixture.

    Returns None when no empty fixture is stored, so the caller can tell "not
    measurable" from "identical". A window with no part in it matches the fixture
    closely, because the fixture is what the bare rig looks like.

    The comparison ignores absolute level, so a drifting exposure does not by
    itself make an empty fixture look occupied.
    """
    if flatfield.empty_fixture is None:
        return None
    x, y, width, height = window
    source_height, source_width = image_bgr.shape[:2]
    fixture_height, fixture_width = flatfield.empty_fixture.shape[:2]
    scale_x = fixture_width / max(source_width, 1)
    scale_y = fixture_height / max(source_height, 1)

    fx0 = max(0, min(int(round(x * scale_x)), fixture_width - 2))
    fy0 = max(0, min(int(round(y * scale_y)), fixture_height - 2))
    fx1 = min(fx0 + max(1, int(round(width * scale_x))), fixture_width)
    fy1 = min(fy0 + max(1, int(round(height * scale_y))), fixture_height)
    x0 = max(0, min(int(x), source_width - 2))
    y0 = max(0, min(int(y), source_height - 2))
    x1 = min(x0 + max(1, int(width)), source_width)
    y1 = min(y0 + max(1, int(height)), source_height)
    if fx1 - fx0 < 8 or fy1 - fy0 < 8 or x1 - x0 < 8 or y1 - y0 < 8:
        return None

    # Downsample before converting to LAB. Only a 64x48 probe is compared, so
    # converting the full crop first would be work on pixels that are thrown away;
    # measured at 14 ms per frame against about 1 ms this way. The stored fixture is
    # already a single L channel, so only the live crop needs converting.
    live_crop = cv2.resize(
        image_bgr[y0:y1, x0:x1], PRESENCE_PROBE, interpolation=cv2.INTER_AREA
    )
    live = cv2.cvtColor(live_crop, cv2.COLOR_BGR2LAB)[:, :, 0]
    fixture = cv2.resize(
        flatfield.empty_fixture[fy0:fy1, fx0:fx1], PRESENCE_PROBE, interpolation=cv2.INTER_AREA
    )
    return normalised_mad(live, fixture)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def detect_bars(
    image_bgr: np.ndarray,
    bar_value: int = 2,
    bar_column_fraction: float = 0.999,
    max_fraction: float = 0.25,
    probe_width: int = 320,
) -> tuple[int, int]:
    """Return the (left, right) width of fully-black pillarbox bars.

    The supplied frames carry 129 px of pure black on each side, which changes
    the aspect ratio from 1.4181 to 1.778. That alone trips the aspect gate the
    fixed-reference pipeline applies, so every new frame is rejected before the
    model is reached. Stripping the bars restores the aspect to within 4 decimal
    places of the reference.

    Tested on a downscaled probe. A fully black column averages to black, and a
    column carrying any content stays above the threshold, so the test survives
    the downscale; a bar is 129 px wide and remains ~32 px at probe scale. Nearest
    neighbour is deliberate: area-averaging a 7-megapixel frame down to a probe
    measured 16.5 ms against 0.12 ms for nearest, and the bar test does not need
    the anti-aliasing.
    """
    height, width = image_bgr.shape[:2]
    if width > probe_width:
        scale = probe_width / float(width)
        probe = cv2.resize(
            image_bgr,
            (probe_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        probe = image_bgr

    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    fully_black = (gray <= bar_value).mean(axis=0)
    probe_columns = fully_black.shape[0]

    left = 0
    while left < probe_columns and fully_black[left] > bar_column_fraction:
        left += 1
    right = 0
    while right < probe_columns and fully_black[probe_columns - 1 - right] > bar_column_fraction:
        right += 1

    bar_left = int(round(left * width / probe.shape[1]))
    bar_right = int(round(right * width / probe.shape[1]))
    if bar_left + bar_right > max_fraction * width:
        # Not a pillarbox. A genuinely dark frame must not be cropped.
        return 0, 0
    return bar_left, bar_right


def rail_mask(
    luminance: np.ndarray,
    mask_rows: float = 0.30,
    rail_percentile: float = 92.0,
    dilate: int = 5,
) -> np.ndarray:
    """Mask the LED rail, a bright band touching the top edge.

    Without this the rail wins the threshold split and the locator returns the
    rail instead of the part.

    The rail's bottom edge is a step, so the band is found from the largest
    downward step in the row-brightness profile. A percentile of that profile is
    unreliable here: the rail occupies roughly half of the searched band, so the
    92nd percentile lands exactly on the rail level, and a strict comparison then
    masks nothing. Only the contiguous run from row 0 is masked, so a part that
    happens to be bright near the top is not cut away.
    """
    height = luminance.shape[0]
    limit = min(int(height * mask_rows), height)
    mask = np.zeros(luminance.shape, dtype=np.uint8)
    if limit < 3:
        return mask

    profile = cv2.blur(luminance[:limit], (1, 1)).mean(axis=1).astype(np.float64)
    steps = np.diff(profile)
    search_end = max(2, int(steps.size * 0.9))
    step_index = int(np.argmin(steps[:search_end]))
    step_drop = float(-steps[step_index])
    # The step must be large against the band's own brightness to be a real edge
    # rather than a smooth gradient.
    margin = max(6.0, 0.12 * float(np.percentile(profile, rail_percentile)))

    if step_drop < margin:
        return mask

    # Dilate the full-height mask. Dilating only the masked slice returns a
    # shorter array, which then fails to index the full frame.
    mask[: min(step_index + 2, limit), :] = 255
    return cv2.dilate(mask, np.ones((dilate, 1), np.uint8))


@dataclass(frozen=True)
class LocatedPart:
    box: tuple[int, int, int, int]
    area_frac: float
    centre_y: float


@dataclass(frozen=True)
class _ConditionedWindow:
    """One source crop carried all the way to a model input."""

    tensor: np.ndarray
    model_bgr: np.ndarray
    scale: float
    pad_left: int
    pad_top: int
    tone_gain: float


@dataclass(frozen=True)
class InstanceSet:
    """Every part in a frame, found without knowing what a part looks like.


    The frame is compared against the stored empty fixture rather than
    thresholded on brightness. A brightness threshold can only find parts that
    are lighter than the deck, so a dark part on a dark deck is invisible to it
    and the brightest thing in frame turns out to be the background instead. On
    the supplied captures that produced a box over bare background with every
    part outside it.

    Differencing against the fixture has no such assumption: a part is anything
    that is not the bare deck, whatever its colour, and a frame with no part
    differs from the fixture only by noise.
    """

    boxes: tuple[tuple[int, int, int, int], ...]
    mask: np.ndarray
    coverage: float
    possibly_merged: int = 0
    #: Parts found before the max_instances cap, so a truncated frame is visible.
    total: int = 0

    def __bool__(self) -> bool:
        return bool(self.boxes)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def truncated(self) -> bool:
        return self.total > self.count


def locate_instances(
    luminance: np.ndarray,
    empty_fixture: np.ndarray | None,
    threshold: float = DEFAULT_INSTANCE_THRESHOLD,
    min_area_frac: float = 0.0004,
    max_area_frac: float = 0.90,
    max_aspect: float = 25.0,
    open_kernel: int = 5,
    close_kernel: int = 9,
    max_instances: int = 64,
) -> InstanceSet:
    """Find every part by differencing against the stored bare fixture.

    ``empty_fixture`` is the single-channel L channel captured with nothing on the
    rig. It is resized to the working resolution, so a map captured at one size
    serves a frame at another.

    Returns an empty set when the frame matches the fixture, which is how a bare
    deck is distinguished from a frame that merely looks unusual.
    """
    if empty_fixture is None:
        return InstanceSet((), np.zeros(luminance.shape[:2], np.uint8), 0.0)
    height, width = luminance.shape[:2]
    fixture = empty_fixture
    if fixture.shape[:2] != (height, width):
        fixture = cv2.resize(fixture, (width, height), interpolation=cv2.INTER_AREA)
    if fixture.ndim == 3:
        fixture = fixture[:, :, 0]
    fixture = fixture.astype(np.int16)

    difference = cv2.medianBlur(cv2.absdiff(luminance.astype(np.int16), fixture).astype(np.uint8), 5)
    binary = (difference > float(threshold)).astype(np.uint8) * 255
    # Open first: it removes single-pixel speckle without bridging the gap
    # between two parts that happen to be close. Closing afterwards would do the
    # opposite and weld neighbouring instances into one blob.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))

    frame_area = float(height * width)

    def _usable(index: int) -> bool:
        x, y, box_width, box_height, area = stats[index]
        fraction = area / frame_area
        if fraction < min_area_frac or fraction > max_area_frac:
            return False
        # A part is compact. A band spanning the full width is rig structure, not
        # a part: the LED rail differs from the fixture once the fixture is
        # rescaled, and without this it arrives as a full-width instance.
        longer, shorter = max(box_width, box_height), max(1, min(box_width, box_height))
        return longer <= max_aspect * shorter

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = [i for i in range(1, count) if _usable(i)]
    boxes = [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])) for i in kept]
    # Largest first, so a caller that can only afford one instance takes the
    # part most likely to be the subject.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    mask = np.where(np.isin(labels, kept), 255, 0).astype(np.uint8)
    coverage = float((mask > 0).mean())
    # Parts that touch or overlap arrive as one blob, and a silently merged pair
    # is worse than a wrong count because nothing looks wrong. Two shapes of merge
    # are reported: one instance far larger than its peers, and a single instance
    # that is very wide relative to its height, which is what a run of touching
    # parts looks like. A lone instance is otherwise indistinguishable from a
    # genuine wide part, so this reports rather than tries to split.
    merged = 0
    if len(boxes) >= 2:
        areas = sorted(b[2] * b[3] for b in boxes)
        median_area = areas[len(areas) // 2]
        if median_area > 0:
            merged = sum(1 for a in areas if a > MERGED_AREA_FACTOR * median_area)
    elif len(boxes) == 1:
        bx, by, bw, bh = boxes[0]
        if max(bw, bh) > 0 and max(bw, bh) / max(1, min(bw, bh)) > MERGED_ASPECT_FACTOR:
            merged = 1
    total = len(boxes)
    return InstanceSet(tuple(boxes[:max_instances]), mask, coverage, merged, total)


def locate_part(
    luminance: np.ndarray,
    min_area_frac: float = 0.03,
    rail_mask_rows: float = 0.30,
    rail_percentile: float = 92.0,
    rail_dilate: int = 5,
    floor: int = 28,
    close_kernel: int = 21,
    open_kernel: int = 11,
    centre_y_range: tuple[float, float] = (0.28, 0.92),
) -> LocatedPart | None:
    """Find the part as the largest central bright region.

    Object-agnostic because the rig is common even when the part is not. Returns
    None when nothing plausible is present, so the caller can fall back or
    reject rather than inspect the wrong pixels.
    """
    height, width = luminance.shape[:2]
    frame_area = float(height * width)

    masked = luminance.copy()
    masked[rail_mask(luminance, rail_mask_rows, rail_percentile, rail_dilate) > 0] = 0
    otsu_value, _ = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = (masked > max(float(otsu_value), float(floor))).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return None

    best: LocatedPart | None = None
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if area < min_area_frac * frame_area:
            continue
        centre_y = (y + box_height / 2.0) / height
        if not centre_y_range[0] <= centre_y <= centre_y_range[1]:
            continue
        if best is None or area > best.area_frac * frame_area:
            best = LocatedPart(
                box=(int(x), int(y), int(box_width), int(box_height)),
                area_frac=float(area) / frame_area,
                centre_y=float(centre_y),
            )
    return best


def percentile_from_histogram(values: np.ndarray, percentile: float) -> float:
    """Exact 8-bit percentile via a 256-bin histogram.

    np.percentile sorts the whole array, which costs tens of milliseconds on a
    multi-megapixel crop. A histogram is O(N) with no sort.
    """
    if values.size == 0:
        return 0.0
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    histogram = cv2.calcHist([values], [0], None, [256], [0, 256]).reshape(-1)
    cumulative = np.cumsum(histogram)
    target = percentile / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(min(index, 255))


def apply_flatfield(image_bgr: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Divide out the stored illumination field, per channel."""
    if field.shape[:2] != image_bgr.shape[:2]:
        field = cv2.resize(field, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    divisor = np.maximum(field, 1e-3)
    three_channel = cv2.cvtColor(np.ascontiguousarray(divisor, dtype=np.float32), cv2.COLOR_GRAY2BGR)
    # The explicit output type keeps this in OpenCV's saturating 8-bit path
    # instead of the float round trip, measured 8.7 ms against 0.5 ms here.
    return cv2.multiply(image_bgr, three_channel, dtype=cv2.CV_8U)


def apply_tone(image_bgr: np.ndarray, target_p99: int) -> tuple[np.ndarray, float]:
    """Scale luminance so p99 lands on a fixed target.

    Targets p99 rather than the mean on purpose. Family means legitimately differ
    (a dark wood board against a white card), and flattening that would destroy a
    real tonal cue. What must be equalised is exposure level, dynamic range and
    clipping, and a percentile target does that while leaving the parts distinct.
    See docs/decisions/log.md D-15.

    The gain is applied to the LAB L channel, not to BGR. sRGB to Lab is a cube
    root, so scaling BGR by g does not scale L by g: measured on a flat frame the
    BGR route overshot a p99 target of 140 to 148. Scaling L makes the target exact
    and leaves a and b untouched, so a green cap stays green.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    measured = percentile_from_histogram(lab[:, :, 0], 99.0)
    gain = 1.0 if measured <= 1.0 else float(target_p99) / measured
    if abs(gain - 1.0) < 1e-3:
        return image_bgr, 1.0
    scaled = np.clip(lab[:, :, 0].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([scaled, lab[:, :, 1], lab[:, :, 2]]), cv2.COLOR_LAB2BGR), gain


def apply_clahe(image_bgr: np.ndarray, clip: float, tiles: tuple[int, int]) -> np.ndarray:
    """CLAHE on the LAB L channel only, leaving hue and saturation intact."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def letterbox(
    image_bgr: np.ndarray,
    target_width: int,
    target_height: int,
    pad_colour: tuple[int, int, int],
) -> tuple[np.ndarray, float, int, int]:
    """Scale to fit, then centre-pad. Returns the image, scale and pad offsets."""
    height, width = image_bgr.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=interpolation)
    pad_x = target_width - new_width
    pad_y = target_height - new_height
    pad_left = pad_x // 2
    pad_top = pad_y // 2
    return (
        cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_y - pad_top,
            pad_left,
            pad_x - pad_left,
            cv2.BORDER_CONSTANT,
            value=pad_colour,
        ),
        scale,
        pad_left,
        pad_top,
    )


def expand_box(
    box: tuple[int, int, int, int],
    margin_frac: float,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    """Grow a box by a fraction of its own size, clipped to the source frame."""
    x, y, width, height = box
    grow_x = int(round(width * margin_frac))
    grow_y = int(round(height * margin_frac))
    x = max(0, x - grow_x)
    y = max(0, y - grow_y)
    width = min(source_width - x, width + 2 * grow_x)
    height = min(source_height - y, height + 2 * grow_y)
    return x, y, max(1, width), max(1, height)


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------


def _serialised(method):
    """Run one preprocess call at a time.

    The preprocessor keeps per-call state (stage timings, instance counts) for
    its metadata. One instance can be shared by a capture thread and a benchmark
    thread, so the calls are serialised rather than letting the two frames
    interleave their state.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class ProductAgnosticPreprocessor(BasePreprocessor):
    """Locate the part, then condition it, for any product on the shared rig."""

    def __init__(
        self,
        profile: ProductProfile | None = None,
        flatfield_lut: str | Path | None = None,
    ):
        self.profile = profile or ProductProfile()
        self._flatfield: FlatField | None = None
        self._last_timings: dict[str, float] = {}
        # How many parts the last locate found, and whether any looked merged.
        self._last_instance_count: int = 0
        self._last_instance_total: int = 0
        self._last_possibly_merged: int = 0
        self._lock = threading.RLock()
        if flatfield_lut is not None:
            self.load_flatfield(flatfield_lut)

    @property
    def name(self) -> str:
        return "product_agnostic"

    @property
    def flatfield(self) -> FlatField | None:
        """The loaded flat-field map, or None when illumination correction is off.

        Read-only on purpose: the map is calibration, and swapping it under a
        running preprocessor would make a frame condition against one map and be
        judged against another.
        """
        return self._flatfield

    def with_profile(self, profile: ProductProfile) -> "ProductAgnosticPreprocessor":
        """Return a copy bound to a different profile, sharing the flat field."""
        clone = ProductAgnosticPreprocessor(profile=profile)
        clone._flatfield = self._flatfield
        return clone

    def load_flatfield(self, path: str | Path) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Flat-field map not found: {path}")
        self._flatfield = load_flatfield(path)

    def _presence_threshold(self) -> float:
        """The threshold the presence check enforces.

        A profile value overrides the map's calibrated value; with neither, the
        historical default applies. The effective value is what metadata reports
        and what calibration records, so the three can never disagree.
        """
        if self.profile.presence_threshold is not None:
            return float(self.profile.presence_threshold)
        if self._flatfield is not None:
            return float(self._flatfield.presence_threshold)
        return 0.18

    def forward_map(
        self,
        box: Box,
        transform_metadata: dict[str, Any],
        clip: bool = True,
    ) -> Box:
        """Map a source-frame box into model-input coordinates.

        The exact inverse of `inverse_map`, kept beside it so a label transform
        and a detection transform can never drift apart. `clip` bounds the box
        to the model input; callers that must know whether a box was cut use
        `clip=False` and compare.
        """
        scale = float(transform_metadata["scale"])
        pad_left = float(transform_metadata["pad_left"])
        pad_top = float(transform_metadata["pad_top"])
        roi = transform_metadata["roi"]

        x = (box.x - float(roi["x"])) * scale + pad_left
        y = (box.y - float(roi["y"])) * scale + pad_top
        width = box.width * scale
        height = box.height * scale
        if not clip:
            return Box(x, y, width, height)

        target_width = float(transform_metadata.get("target_width", MODEL_SIZE))
        target_height = float(transform_metadata.get("target_height", MODEL_SIZE))
        x0 = min(max(x, 0.0), target_width)
        y0 = min(max(y, 0.0), target_height)
        x1 = min(max(x + width, 0.0), target_width)
        y1 = min(max(y + height, 0.0), target_height)
        return Box(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


    def field_window(
        self, x: int, y: int, width: int, height: int, out_width: int, out_height: int
    ) -> np.ndarray | None:
        """Sample the field over a crop window, straight to the output size.

        Resampling the full-resolution field and then slicing would cost work
        proportional to the crop, which is several megapixels. The field is
        smooth by construction, so the small LUT region covering the window is
        sliced and resized directly, making the cost proportional to the output.
        """
        if self._flatfield is None:
            return None
        lut = self._flatfield.lut
        lut_height, lut_width = lut.shape[:2]
        frame_width = float(self._flatfield.frame_width)
        frame_height = float(self._flatfield.frame_height)

        x0 = x / frame_width * lut_width
        y0 = y / frame_height * lut_height
        x1 = (x + width) / frame_width * lut_width
        y1 = (y + height) / frame_height * lut_height
        col0, col1 = int(np.floor(x0)), int(np.ceil(x1))
        row0, row1 = int(np.floor(y0)), int(np.ceil(y1))
        col0 = max(0, min(col0, lut_width - 1))
        row0 = max(0, min(row0, lut_height - 1))
        col1 = max(col0 + 1, min(col1, lut_width))
        row1 = max(row0 + 1, min(row1, lut_height))

        region = lut[row0:row1, col0:col1]
        return np.ascontiguousarray(
            cv2.resize(region, (out_width, out_height), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )

    def _resolve_box(
        self, image_bgr: np.ndarray, view: np.ndarray, bar_left: int
    ) -> tuple[tuple[int, int, int, int], str, float]:
        """Return (box in source coordinates, method, part area fraction)."""
        profile = self.profile
        source_height, source_width = image_bgr.shape[:2]

        if profile.crop_mode == "fixed":
            assert profile.fixed_roi is not None
            x, y, width, height = profile.fixed_roi
            x = max(0, min(x, source_width - 1))
            y = max(0, min(y, source_height - 1))
            return (
                x,
                y,
                min(width, source_width - x),
                min(height, source_height - y),
            ), "fixed", 0.0

        profile = self.profile
        located: LocatedPart | None = None
        located_width = min(profile.locate_width, view.shape[1])
        located_height = max(1, int(round(located_width * view.shape[0] / view.shape[1])))
        small = cv2.resize(view, (located_width, located_height), interpolation=cv2.INTER_AREA)
        luminance = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]

        # With a stored empty fixture, locate by differencing against it. This is
        # preferred over brightness because it is the only rule that holds for a
        # dark part on a dark deck: on the supplied capture the brightness rule
        # returned a window over bare background with every part outside it, and
        # refused a bottle cap outright. Without a fixture there is nothing to
        # difference against, so the brightness rule is all that is available and
        # is kept as the fallback.
        fixture = self._flatfield.empty_fixture if self._flatfield is not None else None
        if profile.stage_locate and fixture is not None:
            instances = locate_instances(
                luminance,
                fixture,
                threshold=profile.instance_threshold,
                min_area_frac=profile.min_instance_area_frac,
                open_kernel=profile.instance_open_kernel,
                close_kernel=profile.instance_close_kernel,
            )
            self._last_instance_count = instances.count
            self._last_instance_total = instances.total
            self._last_possibly_merged = instances.possibly_merged
            eligible_boxes = [
                box for box in instances.boxes
                if profile.centre_y_range[0]
                <= (box[1] + box[3] / 2.0) / float(located_height)
                <= profile.centre_y_range[1]
            ]
            if eligible_boxes:
                x, y, box_width, box_height = eligible_boxes[0]
                located = LocatedPart(
                    box=(x, y, box_width, box_height),
                    area_frac=(box_width * box_height) / float(located_width * located_height),
                    centre_y=(y + box_height / 2.0) / float(located_height),
                )
        elif profile.stage_locate:
            located = locate_part(
                luminance,
                min_area_frac=profile.min_part_area_frac,
                rail_mask_rows=profile.rail_mask_rows,
                rail_percentile=profile.rail_percentile,
                rail_dilate=profile.rail_dilate,
                floor=profile.locate_floor,
                close_kernel=profile.close_kernel,
                open_kernel=profile.open_kernel,
                centre_y_range=profile.centre_y_range,
            )
        if located is not None:
            scale = view.shape[1] / float(located_width)
            x, y, width, height = located.box
            located = replace(
                located,
                box=(
                    int(round(x * scale)),
                    int(round(y * scale)),
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
            )

        if located is not None:
            box = expand_box(located.box, profile.margin_frac, view.shape[1], view.shape[0])
            return (box[0] + bar_left, box[1], box[2], box[3]), "located", located.area_frac

        if profile.fallback_roi is not None:
            x, y, width, height = profile.fallback_roi
            x = max(0, min(x, source_width - 1))
            y = max(0, min(y, source_height - 1))
            width = min(width, source_width - x)
            height = min(height, source_height - y)
            return (x, y, width, height), "fallback", 0.0

        if fixture is not None and profile.stage_locate:
            # Only speak for the locator when the locator actually ran. With
            # locating disabled the presence gate is the authority on whether a
            # part is there, and pre-empting it here would make that stage
            # unreachable and therefore untestable.
            raise PartNotFoundError(
                "no part differs from the stored empty fixture, so the fixture is bare; "
                "reject the frame rather than inspect it"
            )
        raise PartNotFoundError(
            "no part was found and no fallback window is configured; "
            "reject this part rather than inspect unrelated pixels"
        )

    @_serialised
    def preprocess(
        self,
        image_bgr: np.ndarray,
        target_width: int = MODEL_SIZE,
        target_height: int = MODEL_SIZE,
        require_alignment: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del require_alignment  # no reference alignment in this pipeline
        started = time.perf_counter()
        profile = self.profile
        if profile.crop_mode == "instances":
            raise ValueError(
                "crop_mode 'instances' returns one model input per part, so it cannot be "
                "served by the single-crop path; call preprocess_instances() and handle "
                "its list, rather than silently conditioning only the largest part"
            )
        self._last_timings = {}
        self._last_instance_count = 0
        self._last_instance_total = 0
        self._last_possibly_merged = 0
        source_height, source_width = image_bgr.shape[:2]

        stage = time.perf_counter()
        if profile.stage_unbar:
            bar_left, bar_right = detect_bars(
                image_bgr, profile.bar_value, profile.bar_column_fraction, profile.bar_max_fraction
            )
        else:
            bar_left = bar_right = 0
        # Either detected bar means the view must be stripped: a one-sided decode
        # artefact leaves a bar in the frame, and the box origin is offset by
        # bar_left, so an unstripped view would shift every crop.
        view = (
            np.ascontiguousarray(image_bgr[:, bar_left : source_width - bar_right])
            if bar_left or bar_right
            else image_bgr
        )
        self._last_timings["unbar"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        box, method, part_area_frac = self._resolve_box(image_bgr, view, bar_left)
        self._last_timings["locate"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        x, y, width, height = box
        # Always measure when a fixture is stored, and enforce only when the stage
        # is on. Keeping the two apart means the value is available in metadata for
        # diagnostics and threshold calibration even while enforcement is disabled.
        presence_value: float | None = None
        presence_enforced = profile.stage_presence and self._flatfield is not None
        threshold = self._presence_threshold()
        if self._flatfield is not None:
            presence_value = presence_similarity(image_bgr, box, self._flatfield)
            if presence_enforced and presence_value is not None:
                if presence_value < threshold:
                    self._last_timings["presence"] = (time.perf_counter() - stage) * 1000
                    raise PartNotFoundError(
                        "the resolved window matches the stored empty fixture "
                        f"(similarity {presence_value:.3f} < {threshold:.3f}); "
                        "no part is in view, so reject it rather than inspect the fixture"
                    )
        self._last_timings["presence"] = (time.perf_counter() - stage) * 1000

        crop = image_bgr[y : y + height, x : x + width]
        if crop.size == 0:
            raise PartNotFoundError(f"the resolved window {box} contains no pixels")
        window = self._condition_window(
            crop, x, y, width, height, target_width, target_height
        )

        final_luminance = cv2.cvtColor(window.model_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        metadata = {
            "preprocessor": self.name,
            "profile": profile.name,
            "source_width": source_width,
            "source_height": source_height,
            "target_width": target_width,
            "target_height": target_height,
            "scale": window.scale,
            "pad_left": window.pad_left,
            "pad_top": window.pad_top,
            "roi": {"x": x, "y": y, "width": width, "height": height},
            "window": method,
            "instance_count": self._last_instance_count,
            "instance_total": self._last_instance_total,
            "instances_truncated": self._last_instance_total > self._last_instance_count,
            "possibly_merged": self._last_possibly_merged,
            "part_area_frac": part_area_frac,
            "bar_left": bar_left,
            "bar_right": bar_right,
            "tone_gain": window.tone_gain,
            "presence_checked": presence_value is not None,
            "presence_enforced": presence_enforced,
            "presence_similarity": presence_value,
            "presence_threshold": threshold if self._flatfield is not None else None,
            "flatfield_loaded": self._flatfield is not None,
            "flatfield_created_utc": self._flatfield.created_utc if self._flatfield else None,
            "saturated_percent": float((final_luminance >= 250).mean() * 100.0),
            "roi_luminance_mean": float(final_luminance.mean()),
            "stages": {
                "unbar": profile.stage_unbar,
                "locate": profile.stage_locate,
                "flatfield": profile.stage_flatfield,
                "presence": profile.stage_presence,
                "tone": profile.stage_tone,
                "clahe": profile.stage_clahe,
            },
            "stage_timings_ms": {
                **self._last_timings,
                "preprocessing_total": (time.perf_counter() - started) * 1000,
            },
        }
        return window.tensor, window.model_bgr, metadata

    @_serialised
    def preprocess_instances(
        self,
        image_bgr: np.ndarray,
        target_width: int = MODEL_SIZE,
        target_height: int = MODEL_SIZE,
        max_instances: int = 64,
    ) -> list[dict[str, Any]]:
        """Condition every part in the frame, one model input each.

        For lines where several parts arrive at once. A cap hole is 24.5 px on a
        218 px cap, which is 4.9 px in a 640 letterbox of the whole frame and
        51.5 px cropped to a single cap. The first is not detectable, so
        multi-instance lines have to be given per-instance crops rather than a
        scene view; no model input can recover that resolution afterwards.

        The presence check runs per instance. An instance whose window matches
        the stored empty fixture is skipped rather than conditioned, so a shadow
        or a deck artefact cannot become a part; metadata carries the score and
        whether it was enforced. Returns a list, largest instance first. Empty
        means nothing was located, which for a configured fixture means a bare
        deck, so the caller can reject the frame rather than inspect the fixture.
        """
        started = time.perf_counter()
        profile = self.profile
        self._last_timings = {}
        self._last_instance_count = 0
        self._last_instance_total = 0
        self._last_possibly_merged = 0
        source_height, source_width = image_bgr.shape[:2]

        stage = time.perf_counter()
        if profile.stage_unbar:
            bar_left, bar_right = detect_bars(
                image_bgr, profile.bar_value, profile.bar_column_fraction, profile.bar_max_fraction
            )
        else:
            bar_left = bar_right = 0
        view = (
            np.ascontiguousarray(image_bgr[:, bar_left : source_width - bar_right])
            if bar_left or bar_right
            else image_bgr
        )
        self._last_timings["unbar"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        fixture = self._flatfield.empty_fixture if self._flatfield is not None else None
        if fixture is None:
            raise PartNotFoundError(
                "locating several instances at once needs the stored empty fixture, "
                "because there is no brightness rule that finds a dark part on a dark "
                "deck; load a flat-field map with a stored empty fixture"
            )
        located_width = min(profile.locate_width, view.shape[1])
        located_height = max(1, int(round(located_width * view.shape[0] / view.shape[1])))
        small = cv2.resize(view, (located_width, located_height), interpolation=cv2.INTER_AREA)
        luminance = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
        instances = locate_instances(
            luminance,
            fixture,
            threshold=profile.instance_threshold,
            min_area_frac=profile.min_instance_area_frac,
            open_kernel=profile.instance_open_kernel,
            close_kernel=profile.instance_close_kernel,
            max_instances=max_instances,
        )
        self._last_timings["locate"] = (time.perf_counter() - stage) * 1000
        if not instances.boxes:
            raise PartNotFoundError(
                "no part differs from the stored empty fixture, so the fixture is bare; "
                "reject the frame rather than inspect it"
            )

        scale_up = view.shape[1] / float(located_width)
        results: list[dict[str, Any]] = []
        skipped_presence: list[dict[str, Any]] = []
        presence_enforced = profile.stage_presence and self._flatfield is not None
        threshold = self._presence_threshold()
        for index, (bx, by, bw, bh) in enumerate(instances.boxes):
            bx, by = int(round(bx * scale_up)), int(round(by * scale_up))
            bw, bh = max(1, int(round(bw * scale_up))), max(1, int(round(bh * scale_up)))
            box = expand_box((bx, by, bw, bh), profile.margin_frac, view.shape[1], view.shape[0])
            x, y, width, height = box
            x += bar_left
            x = max(0, min(x, source_width - 1))
            y = max(0, min(y, source_height - 1))
            width = min(width, source_width - x)
            height = min(height, source_height - y)
            crop = image_bgr[y : y + height, x : x + width]
            if crop.size == 0:
                continue
            presence = None
            if self._flatfield is not None:
                presence = presence_similarity(image_bgr, (x, y, width, height), self._flatfield)
                if presence_enforced and presence is not None and presence < threshold:
                    skipped_presence.append({"index": index, "roi": {"x": x, "y": y, "width": width, "height": height}, "presence_similarity": float(presence)})
                    continue
            window = self._condition_window(
                crop, x, y, width, height, target_width, target_height
            )
            results.append(
                {
                    "index": index,
                    "tensor": window.tensor,
                    "model_bgr": window.model_bgr,
                    "box": {"x": x, "y": y, "width": width, "height": height},
                    "scale": window.scale,
                    "pad_left": window.pad_left,
                    "pad_top": window.pad_top,
                    "tone_gain": window.tone_gain,
                    "presence_similarity": presence,
                    "presence_checked": presence is not None,
                    "presence_enforced": presence_enforced,
                    "presence_threshold": threshold if self._flatfield is not None else None,
                    # Shaped so inverse_map works on it unchanged. A detection
                    # found in this crop has to come back to where the part
                    # actually is in the camera frame, and hand-maintaining a
                    # second mapping here is how the two would drift.
                    "metadata": {
                        "preprocessor": self.name,
                        "profile": profile.name,
                        "source_width": source_width,
                        "source_height": source_height,
                        "target_width": target_width,
                        "target_height": target_height,
                        "scale": window.scale,
                        "pad_left": window.pad_left,
                        "pad_top": window.pad_top,
                        "roi": {"x": x, "y": y, "width": width, "height": height},
                        "bar_left": bar_left,
                        "bar_right": bar_right,
                        "instance_index": index,
                    },
                }
            )
        if not results:
            raise PartNotFoundError(
                "every located window matches the stored empty fixture "
                f"(similarity below {threshold:.3f}); no part is in view, so reject "
                "the frame rather than inspect the fixture"
            )
        self._last_timings["preprocessing_total"] = (time.perf_counter() - started) * 1000
        for entry in results:
            # Carried on every entry so a caller that consumes one instance still
            # learns that the frame contained a merge or a truncation.
            entry["instance_count"] = len(results)
            entry["instance_total"] = instances.total
            entry["instances_truncated"] = instances.truncated
            entry["possibly_merged"] = instances.possibly_merged
            entry["instances_skipped_presence"] = len(skipped_presence)
        return results

    def _condition_window(
        self,
        crop: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
    ) -> "_ConditionedWindow":
        """Take one source crop all the way to a model input.

        Shared by the single-window and per-instance paths so the two cannot
        drift apart, which is how train and serve conditioning would silently
        diverge.
        """
        profile = self.profile

        # Geometric downscale first. Every later stage is photometric and runs on
        # the pixels the model actually receives, so cost is proportional to the
        # 640-pixel input rather than to the multi-megapixel crop. The production
        # fixed-reference path already conditions at model resolution.
        content_scale = min(target_width / width, target_height / height)
        content_width = max(1, int(round(width * content_scale)))
        content_height = max(1, int(round(height * content_scale)))
        interpolation = cv2.INTER_AREA if content_width < width else cv2.INTER_LINEAR

        stage = time.perf_counter()
        model_bgr = cv2.resize(crop, (content_width, content_height), interpolation=interpolation)
        self._last_timings["resize_to_model"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        if profile.stage_flatfield:
            field = self.field_window(x, y, width, height, content_width, content_height)
            if field is not None:
                model_bgr = apply_flatfield(model_bgr, field)
        self._last_timings["flatfield"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        tone_gain = 1.0
        if profile.stage_tone:
            model_bgr, tone_gain = apply_tone(model_bgr, profile.tone_target_p99)
        self._last_timings["tone"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        if profile.stage_clahe:
            model_bgr = apply_clahe(model_bgr, profile.clahe_clip, profile.clahe_tiles)
        self._last_timings["clahe"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        model_bgr, letterbox_scale, pad_left, pad_top = letterbox(
            model_bgr, target_width, target_height, profile.pad_colour
        )
        self._last_timings["letterbox"] = (time.perf_counter() - stage) * 1000
        # One scale from source pixels to model pixels, so inverse_map stays a
        # single un-letterbox plus un-crop.
        scale = letterbox_scale * content_scale

        stage = time.perf_counter()
        model_rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            model_rgb.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
        )
        tensor *= np.float32(1.0 / 255.0)
        self._last_timings["tensor"] = (time.perf_counter() - stage) * 1000
        return _ConditionedWindow(
            tensor, model_bgr, scale, pad_left, pad_top, tone_gain
        )

    def inverse_map(
        self,
        box: Box,
        transform_metadata: dict[str, Any],
        clip: bool = True,
    ) -> Box:
        scale = float(transform_metadata["scale"])
        pad_left = float(transform_metadata["pad_left"])
        pad_top = float(transform_metadata["pad_top"])
        roi = transform_metadata["roi"]

        crop_x = (box.x - pad_left) / scale
        crop_y = (box.y - pad_top) / scale
        origin_x = crop_x + float(roi["x"])
        origin_y = crop_y + float(roi["y"])
        width = box.width / scale
        height = box.height / scale

        if not clip:
            return Box(origin_x, origin_y, width, height)

        source_width = float(transform_metadata["source_width"])
        source_height = float(transform_metadata["source_height"])
        x1 = min(max(origin_x, 0.0), source_width)
        y1 = min(max(origin_y, 0.0), source_height)
        x2 = min(max(origin_x + width, 0.0), source_width)
        y2 = min(max(origin_y + height, 0.0), source_height)
        return Box(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))
