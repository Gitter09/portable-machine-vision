"""Build and audit the IC/terminal inspection dataset.

Two transforms are supported, because the choice between them is the accuracy
decision this project turns on:

* ``letterbox`` -- the documented runtime contract: 1280x720 resized to 640x360
  and padded to 640x640. It is what the board's CVU stage does today, and it
  halves the source resolution.
* ``crop`` -- a square window around the plate's rotation axis, resized to
  640x640 with no padding. The measured parts are ~21x15 px in the source frame,
  which letterboxing shrinks to ~11x8 px: one cell of the stride-8 grid. The same
  parts inside a ~560 px crop arrive at ~24x20 px instead.

The crop window is not guessed. The board rotates rigidly about a fixed axis, so
the tool fits that axis from the annotations and measures how far the board
actually extends from it, then sizes the window from the measurement.

Every stored image is produced by the transform that will run at inference, and
every stored box is the box the model must predict. Nothing is re-derived from
the framework's own letterboxing, so a padding or colour mismatch cannot appear
between training and serving without this file changing.

Splits are made from whole runs and contiguous blocks inside a run, never from
individual frames. A run is one continuous recording of one board state; frames
within it are adjacent views of the same object. Blocks are separated by a guard
band that drops frames near a boundary, so no near-duplicate pair straddles two
partitions. The report states the number of blocks, because that -- not the frame
count -- is the number of independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Class IDs, in the order the checkpoint, export, decoder and overlay must use.
CLASS_NAMES = ["IC_Fail", "Terminal_pass", "IC_Pass", "Terminal_Fail"]

#: COCO category names map by name, never by ID: the source files number 1..4.
COCO_NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

LETTERBOX_SIZE = 640
LETTERBOX_PAD_VALUE = 128


@dataclass
class Frame:
    run: str
    index: int
    image_path: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]]  # class_id, x, y, w, h


@dataclass
class Run:
    name: str
    path: Path
    frames: list[Frame]


def _frame_number(name: str) -> int:
    digits = "".join(character for character in name if character.isdigit())
    return int(digits) if digits else 0


def load_run(name: str, path: Path) -> Run:
    """Read one capture run: a folder holding images and one COCO JSON.

    Images may sit in an ``images/`` subfolder or directly in the run folder;
    exports from different tools use both layouts.
    """
    annotation_files = sorted((path / "annotations").glob("*.json"))
    if len(annotation_files) != 1:
        raise SystemExit(f"{path}: expected exactly one annotation file, found {len(annotation_files)}")
    images_dir = path / "images"
    if not images_dir.is_dir():
        images_dir = path
    coco = json.loads(annotation_files[0].read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in coco["categories"]}
    unknown = sorted(set(names.values()) - set(CLASS_NAMES))
    if unknown:
        raise SystemExit(f"{path}: unknown class names {unknown}")
    by_image: dict[int, list] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    frames = []
    for image in coco["images"]:
        image_path = images_dir / image["file_name"]
        if not image_path.is_file():
            raise SystemExit(f"missing image {image_path}")
        boxes = []
        for annotation in by_image.get(image["id"], []):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            boxes.append((COCO_NAME_TO_ID[names[annotation["category_id"]]], x, y, width, height))
        frames.append(Frame(name, _frame_number(image["file_name"]), image_path,
                            int(image["width"]), int(image["height"]), boxes))
    frames.sort(key=lambda frame: frame.index)
    return Run(name, path, frames)


def fit_rotation_axis(frames: list[Frame]) -> tuple[float, float]:
    """Least-squares circle through the annotation centres.

    Every annotated site sits at a nearly constant distance from the plate's axis,
    so the circle those centres trace is the axis. A rigid rotating board gives a
    residual well under a pixel; a large residual means the board was repositioned
    between captures and a fixed crop is not safe.
    """
    points = np.array([(x + w / 2, y + h / 2)
                       for frame in frames for _, x, y, w, h in frame.boxes], dtype=float)
    if len(points) < 3:
        raise SystemExit("not enough annotated frames to fit a rotation axis")
    matrix = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]
    rhs = (points ** 2).sum(axis=1)
    centre_x, centre_y, constant = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    radius = math.sqrt(max(0.0, constant + centre_x ** 2 + centre_y ** 2))
    residuals = np.abs(np.hypot(points[:, 0] - centre_x, points[:, 1] - centre_y) - radius)
    return float(centre_x), float(centre_y)


def measure_board_radius(run: Run, centre: tuple[float, float], sample: int = 5) -> tuple[float, float]:
    """How far the board's coloured body actually reaches from the rotation axis.

    The annotations cover only the inspection sites, not the whole PCB, so the
    window cannot be sized from them alone. The board is the only saturated
    object on the white plate, which makes a saturation mask faithful -- but only
    if the component picked is the board. A frame where the board's silhouette
    merges with a cable or a reflection reports several times the true radius, so
    frames whose component area is far from the run's median area are discarded
    rather than allowed to inflate the window.

    Returns the run's median and worst plausible radius, the latter already scaled
    by ``margin``.
    """
    centre_x, centre_y = centre
    measured: list[tuple[float, float]] = []
    for frame in run.frames[::sample]:
        if not frame.boxes:
            continue
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        _, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        mask = ((saturation > 70) & (value > 25)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count < 2:
            continue
        board_x = float(np.mean([x + width / 2 for _, x, y, width, height in frame.boxes]))
        board_y = float(np.mean([y + height / 2 for _, x, y, width, height in frame.boxes]))
        nearest, nearest_distance = None, math.inf
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 800:
                continue
            distance = math.hypot(centroids[index][0] - board_x, centroids[index][1] - board_y)
            if distance < nearest_distance:
                nearest, nearest_distance = index, distance
        if nearest is None:
            continue
        rows, columns = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        distance = np.hypot(columns - centre_x, rows - centre_y)[labels == nearest]
        measured.append((float(distance.max()), float(stats[nearest, cv2.CC_STAT_AREA])))
    if not measured:
        raise SystemExit(f"{run.name}: could not measure the board silhouette")
    areas = [area for _, area in measured]
    typical = float(np.median(areas))
    kept = [radius for radius, area in measured if 0.4 * typical <= area <= 2.5 * typical]
    if not kept:
        kept = [radius for radius, _ in measured]
    # When the board's silhouette merges with a cable the reported radius jumps
    # by more than double, so merging can only ever inflate the measurement. The
    # median therefore describes the board; the maximum would describe the worst
    # accident.
    return typical, float(np.median(kept))


def cover_centre(items: list[tuple[float, float, float]]) -> tuple[float, float]:
    """The point that puts the smallest circle around every run's swept board.

    Runs are not guaranteed to share a plate axis -- a fixture that was nudged
    between sessions moves it. Centring on the mean axis would then waste margin
    on the side that already fits and starve the side that does not.
    """
    centre_x = float(np.mean([item[0] for item in items]))
    centre_y = float(np.mean([item[1] for item in items]))

    def radius_at(point_x: float, point_y: float) -> float:
        return max(math.hypot(point_x - item[0], point_y - item[1]) + item[2] for item in items)

    best = radius_at(centre_x, centre_y)
    for step in (8.0, 2.0, 0.5, 0.1):
        improved = True
        while improved:
            improved = False
            for offset_x, offset_y in ((step, 0), (-step, 0), (0, step), (0, -step),
                                       (step, step), (step, -step), (-step, step), (-step, -step)):
                candidate_x, candidate_y = centre_x + offset_x, centre_y + offset_y
                candidate = radius_at(candidate_x, candidate_y)
                if candidate < best - 1e-6:
                    best, centre_x, centre_y, improved = candidate, candidate_x, candidate_y, True
    return centre_x, centre_y


@dataclass
class Transform:
    """Source-frame coordinates to 640x640 model coordinates."""

    mode: str
    size: int
    centre: tuple[float, float]
    source_width: int
    source_height: int

    @property
    def side(self) -> int:
        return self.size

    @property
    def output_size(self) -> int:
        """Side of the square model image this transform produces."""
        return self.size if self.mode == "letterbox" else LETTERBOX_SIZE

    def window(self) -> tuple[int, int, int] | None:
        """The source-pixel square this transform reads. None for letterbox."""
        if self.mode != "crop":
            return None
        half = self.size / 2
        x0 = int(round(self.centre[0] - half))
        y0 = int(round(self.centre[1] - half))
        x0 = max(0, min(x0, self.source_width - self.size))
        y0 = max(0, min(y0, self.source_height - self.size))
        return x0, y0, self.size

    def project(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map a source-frame point into 640x640 model coordinates."""
        window = self.window()
        if window is None:
            scale = self.size / self.source_width
            height = self.source_height * scale
            return point[0] * scale, point[1] * scale + (self.size - height) / 2
        x0, y0, size = window
        scale = LETTERBOX_SIZE / size
        return (point[0] - x0) * scale, (point[1] - y0) * scale

    def apply(self, frame: Frame) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]], int]:
        """Return the model image, its boxes, and how many boxes were dropped."""
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read {frame.image_path}")
        boxes = []
        dropped = 0
        if self.mode == "letterbox":
            scale = self.size / frame.width
            height = int(round(frame.height * scale))
            resized = cv2.resize(image, (self.size, height), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.size, self.size, 3), LETTERBOX_PAD_VALUE, np.uint8)
            top = (self.size - height) // 2
            canvas[top:top + height] = resized
            for class_id, x, y, width, height_px in frame.boxes:
                boxes.append((class_id, x * scale, y * scale + top, width * scale, height_px * scale))
            return canvas, boxes, dropped

        x0, y0, size = self.window()
        crop = image[y0:y0 + size, x0:x0 + size]
        if crop.shape[0] != size or crop.shape[1] != size:
            raise SystemExit(f"{frame.image_path}: crop is {(crop.shape[1], crop.shape[0])}, not {size} square")
        model = cv2.resize(crop, (LETTERBOX_SIZE, LETTERBOX_SIZE), interpolation=cv2.INTER_LINEAR)
        scale = LETTERBOX_SIZE / size
        for class_id, x, y, width, height_px in frame.boxes:
            mapped_x = (x - x0) * scale
            mapped_y = (y - y0) * scale
            mapped_width = width * scale
            mapped_height = height_px * scale
            inside = (mapped_x >= 0 and mapped_y >= 0
                      and mapped_x + mapped_width <= LETTERBOX_SIZE
                      and mapped_y + mapped_height <= LETTERBOX_SIZE)
            if not inside:
                dropped += 1
                continue
            boxes.append((class_id, mapped_x, mapped_y, mapped_width, mapped_height))
        return model, boxes, dropped


def sector_of(centre: tuple[float, float], box: tuple[int, float, float, float, float]) -> int:
    _, x, y, width, height = box
    angle = math.degrees(math.atan2(y + height / 2 - centre[1], x + width / 2 - centre[0])) % 360.0
    return int(angle // 10) * 10


def assign_splits(runs: list[Run], blocks: int, guard: int, fractions: tuple[float, float, float]) -> dict[str, str]:
    """Map frame key to split using contiguous blocks with a discarded guard band.

    Blocks are dealt out per run, not across the pooled set. A run holds only the
    states it was captured in, so a partition that receives blocks from one run
    and not another is missing whole classes and the coverage check will reject
    it. Each run therefore contributes its own train/val/test blocks, and the
    shuffled order is rotated per run so two runs do not put their rare
    validation block on the same side of the cycle.
    """
    import random

    splits = {}
    for run_index, run in enumerate(runs):
        count = len(run.frames)
        edges = np.linspace(0, count, blocks + 1).round().astype(int)
        members_by_block = []
        for index in range(blocks):
            members = run.frames[edges[index]:edges[index + 1]]
            if guard:
                members = members[guard:len(members) - guard]
            if len(members) >= 2:
                members_by_block.append(members)
        if not members_by_block:
            raise SystemExit(f"{run.name}: no block survives a guard band of {guard} frames")
        random.Random(42).shuffle(members_by_block)
        shift = run_index % len(members_by_block)
        members_by_block = members_by_block[shift:] + members_by_block[:shift]

        total = len(members_by_block)
        train_count = max(1, round(fractions[0] * total))
        val_count = max(1, round(fractions[1] * total))
        test_count = max(1, total - train_count - val_count)
        while train_count + val_count + test_count > total:
            train_count -= 1
        order = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        order += ["train"] * (total - len(order))
        for split, members in zip(order, members_by_block):
            for frame in members:
                splits[f"{run.name}:{frame.index}"] = split
    return splits


def build(arguments: argparse.Namespace) -> dict:
    runs = [load_run(name, path) for name, path in arguments.run]
    quarantined = set(arguments.quarantine or [])
    for run in runs:
        before = len(run.frames)
        run.frames = [frame for frame in run.frames if f"{run.name}:{frame.index}" not in quarantined]
        if len(run.frames) != before:
            print(f"quarantined {before - len(run.frames)} frame(s) from {run.name}")

    all_frames = [frame for run in runs for frame in run.frames]
    axes = [fit_rotation_axis(run.frames) for run in runs]
    swept = []
    for run, axis in zip(runs, axes):
        _, extent = measure_board_radius(run, axis)
        swept.append((axis[0], axis[1], extent))
    centre = cover_centre(swept)
    if arguments.crop_centre:
        supplied_x, supplied_y = (float(value) for value in arguments.crop_centre.split(","))
        centre = (supplied_x, supplied_y)
        print(f"window centre pinned to ({centre[0]:.1f}, {centre[1]:.1f})")
    for run, axis, (_, _, extent) in zip(runs, axes, swept):
        offset = math.hypot(axis[0] - centre[0], axis[1] - centre[1])
        print(f"{run.name}: board reaches {extent:.0f} px from its own axis "
              f"({axis[0]:.1f}, {axis[1]:.1f}), {offset:.0f} px from the common centre")
        if offset > 25:
            print(f"NOTE: {run.name} was captured with the fixture in a different position; "
                  f"the common window has to absorb that {offset:.0f} px shift")
    if arguments.mode == "crop":
        needed = max(math.hypot(centre[0] - axis[0], centre[1] - axis[1]) + extent
                     for axis, (_, _, extent) in zip(axes, swept))
        size = arguments.crop_size or int(2 * math.ceil(needed * arguments.crop_margin))
        limit = 2 * int(min(centre[1], all_frames[0].height - centre[1],
                            centre[0], all_frames[0].width - centre[0]))
        if size > limit:
            raise SystemExit(f"crop {size} px does not fit the frame around {centre} (limit {limit})")
        transform = Transform("crop", size, centre, all_frames[0].width, all_frames[0].height)
        print(f"rotation centre ({centre[0]:.1f}, {centre[1]:.1f}); crop {size}x{size} "
              f"(scale {LETTERBOX_SIZE / size:.3f}x)")
    else:
        transform = Transform("letterbox", arguments.letterbox_size, centre, all_frames[0].width, all_frames[0].height)
        print(f"letterbox contract: {arguments.letterbox_size} square, scale {arguments.letterbox_size / all_frames[0].width:.3f}x, pad {LETTERBOX_PAD_VALUE}")

    splits = assign_splits(runs, arguments.blocks_per_run, arguments.guard, (0.7, 0.15, 0.15))
    output = arguments.output
    if output.exists() and not arguments.force:
        raise SystemExit(f"{output} exists; pass --force to replace it")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: [0] * len(CLASS_NAMES) for split in ("train", "val", "test")}
    sectors = {split: defaultdict(int) for split in ("train", "val", "test")}
    sizes = defaultdict(list)
    manifest = []
    dropped = 0
    usable = 0
    model_axis = transform.project(centre)
    for frame in all_frames:
        split = splits.get(f"{frame.run}:{frame.index}")
        if split is None or not frame.boxes:
            continue
        model_image, boxes, lost = transform.apply(frame)
        dropped += lost
        if not boxes:
            continue
        stem = f"{frame.run}_{frame.index:06d}"
        relative = Path("images") / split / f"{stem}.png"
        cv2.imwrite(str(output / relative), model_image)
        lines = []
        for class_id, x, y, width, height in boxes:
            lines.append(f"{class_id} {(x + width / 2) / transform.output_size:.6f} "
                         f"{(y + height / 2) / transform.output_size:.6f} "
                         f"{width / transform.output_size:.6f} {height / transform.output_size:.6f}")
            counts[split][class_id] += 1
            sizes[CLASS_NAMES[class_id]].append((width, height))
            sectors[split][sector_of(model_axis, (class_id, x, y, width, height))] += 1
        (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({"file": str(relative).replace("\\", "/"), "run": frame.run,
                         "frame": frame.index, "split": split,
                         "source_sha256": hashlib.sha256(frame.image_path.read_bytes()).hexdigest(),
                         "model_sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                         "boxes": len(boxes)})
        usable += 1

    missing = [name for index, name in enumerate(CLASS_NAMES)
               if counts["train"][index] == 0 or counts["val"][index] == 0]
    report = {
        "classes": CLASS_NAMES,
        "transform": {"mode": transform.mode, "crop_size": transform.size,
                      "scale": (LETTERBOX_SIZE / transform.size) if transform.mode == "crop"
                      else transform.size / all_frames[0].width,
                      "centre": list(centre), "pad_value": LETTERBOX_PAD_VALUE},
        "counts": counts,
        "sectors": {split: dict(sorted(value.items())) for split, value in sectors.items()},
        "box_px": {name: {"median": [round(float(np.median([w for w, _ in items])), 1),
                                     round(float(np.median([h for _, h in items])), 1)],
                          "min": [round(min(w for w, _ in items), 1), round(min(h for _, h in items), 1)]}
                   for name, items in sizes.items()},
        "images": usable, "dropped_boxes": dropped,
        "missing_classes": missing,
        "runs": {run.name: len(run.frames) for run in runs},
        "blocks_per_run": arguments.blocks_per_run,
        "guard_frames": arguments.guard,
        "independent_blocks": arguments.blocks_per_run * len(runs),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                        help="capture run name and folder holding images/ and annotations/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("crop", "letterbox"), default="crop")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="square window in source pixels; default is measured from the board")
    parser.add_argument("--crop-centre", type=str, default=None, metavar="X,Y",
                        help="window centre in source pixels; default covers every run's axis")
    parser.add_argument("--letterbox-size", type=int, default=640,
                        help="square side of the letterbox model input")
    parser.add_argument("--crop-margin", type=float, default=1.10,
                        help="fraction added around the measured board extent")
    parser.add_argument("--quarantine", action="append", default=[], metavar="RUN:FRAME",
                        help="frame to hold out of the dataset entirely")
    parser.add_argument("--blocks-per-run", type=int, default=6)
    parser.add_argument("--guard", type=int, default=3,
                        help="frames discarded either side of a block boundary")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    arguments.run = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in arguments.run]
    report = build(arguments)
    print(json.dumps(report, indent=2))
    if report["missing_classes"]:
        print(f"\nWARNING: no examples in train and val for {report['missing_classes']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
