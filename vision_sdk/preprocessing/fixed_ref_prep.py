"""Fixed-Reference Universal PCB Preprocessor conforming to BasePreprocessor.

Implements the exact 6-stage pipeline from fixed_reference_preprocessor.py:
1. ORB alignment to fixed reference (058e928d2154443e9aac380c7b11d3d5.png)
2. Fixed PCB + metal-frame ROI crop (x=530, y=674, w=1780, h=1170)
3. CLAHE enhancement
4. Conservative PCB-only reflection suppression
5. Illumination normalization
6. Letterbox padding to 640x640 with (114, 114, 114) border
+ Compound inverse mapping through un-letterbox, un-crop, and inverse affine warp.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from .base import BasePreprocessor, Box


def _resolve_default_reference() -> Path:
    """Find the fixed reference image.

    Three places, in order: an explicit override for a rig with its own fixture,
    the copy that ships inside the package, and the seed folder that the frozen
    desktop build keeps beside the executable. The package copy is what a pip
    install and the board bundle use. The seed copy is the fallback, because a
    frozen build does not always carry package data directories.
    """
    override = os.environ.get("DEMETER_FIXED_REFERENCE")
    if override:
        return Path(override)
    packaged = (
        Path(__file__).resolve().parent
        / "Preprocessing_Fixed_Reference"
        / "058e928d2154443e9aac380c7b11d3d5.png"
    )
    if packaged.is_file():
        return packaged
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
    roots.extend(Path(__file__).resolve().parents)
    for root in roots:
        matches = sorted(root.glob("seed/*/reference/reference.png"))
        if matches:
            return matches[0]
    return packaged


DEFAULT_REF_PATH = _resolve_default_reference()

# Calibrated ROI in reference 3200x2256 coordinates
ROI_X = 530
ROI_Y = 674
ROI_W = 1780
ROI_H = 1170
TARGET_SIZE = 640

# ORB settings
ORB_FEATURES = 3000
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8
ORB_FAST_THRESHOLD = 10
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.99
RANSAC_REFINE_ITERS = 20

# Alignment quality gate. A fixed-reference ROI is only meaningful once the frame
# is in reference space, so a weak or missing transform must fail closed rather
# than silently crop the conveyor. Thresholds mirror the signed production
# service (production_runtime.validate_alignment) and can be tuned per release.
ALIGNMENT_MIN_MATCHES = int(os.environ.get("ALIGNMENT_MIN_MATCHES", "8"))
ALIGNMENT_MIN_INLIERS = int(os.environ.get("ALIGNMENT_MIN_INLIERS", "6"))
ALIGNMENT_MIN_INLIER_RATIO = float(os.environ.get("ALIGNMENT_MIN_INLIER_RATIO", "0.10"))
ALIGNMENT_MIN_SCALE_RATIO = float(os.environ.get("ALIGNMENT_MIN_SCALE_RATIO", "0.80"))
ALIGNMENT_MAX_SCALE_RATIO = float(os.environ.get("ALIGNMENT_MAX_SCALE_RATIO", "1.20"))


class AlignmentError(RuntimeError):
    """The fixed reference could not be aligned to the frame, or the transform is unusable."""

    def __init__(self, message: str, *, retryable: bool = False,
                 diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.diagnostics = diagnostics or {}


def clahe_image(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def suppress_pcb_reflections(img_bgr: np.ndarray) -> np.ndarray:
    """Conservatively suppress bright reflections on green PCB only."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    green = (h >= 35) & (h <= 100) & (s >= 55) & (v >= 35)
    reflection = green & (v >= 185) & (s <= 145)

    mask = reflection.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    soft_mask = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32) / 255.0

    vf = v.astype(np.float32)
    excess = np.maximum(vf - 170.0, 0.0)
    new_v = vf - 0.38 * soft_mask * excess
    new_v = np.clip(new_v, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, new_v]), cv2.COLOR_HSV2BGR)


def illumination_normalize(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    background = cv2.GaussianBlur(lf, (0, 0), 35)
    background = np.maximum(background, 1.0)
    corrected = lf / background
    corrected *= float(np.mean(background))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([corrected, a, b]), cv2.COLOR_LAB2BGR)


class FixedReferencePreprocessor(BasePreprocessor):
    """Calibrated fixed-reference PCB preprocessor with full 6-stage pipeline."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        roi: tuple[int, int, int, int] = (ROI_X, ROI_Y, ROI_W, ROI_H),
        alignment_max_width: int | None = None,
        enhance_at_model_resolution: bool = False,
        alignment_enabled: bool = True,
        direct_roi_warp: bool = False,
        direct_model_warp: bool = False,
        enhancement_enabled: bool = True,
        enhancement_steps: tuple[str, ...] = ("clahe", "reflection", "illumination"),
        feature_resize_interpolation: int = cv2.INTER_AREA,
        feature_subsample: bool = False,
        orb_features: int = ORB_FEATURES,
    ):
        self.reference_path = Path(reference_path or DEFAULT_REF_PATH)
        self.roi = roi  # (x, y, w, h)
        self.reference: np.ndarray | None = None
        self.kp_ref = None
        self.des_ref = None
        self.alignment_max_width = alignment_max_width
        self.enhance_at_model_resolution = enhance_at_model_resolution
        self.alignment_enabled = alignment_enabled
        self.direct_roi_warp = direct_roi_warp
        self.direct_model_warp = direct_model_warp
        self.enhancement_enabled = enhancement_enabled
        self.enhancement_steps = enhancement_steps if enhancement_enabled else ()
        self.feature_resize_interpolation = feature_resize_interpolation
        self.feature_subsample = feature_subsample
        self.orb_features = orb_features
        self.reference_feature_scale = 1.0
        self._last_alignment_timings: dict[str, float] = {}
        self._last_alignment_diagnostics: dict[str, Any] = {}

        self.orb = cv2.ORB_create(
            nfeatures=self.orb_features,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_LEVELS,
            fastThreshold=ORB_FAST_THRESHOLD,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if self.alignment_enabled:
            self._init_reference()

    def _feature_frame(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        scale = 1.0
        if self.alignment_max_width and image.shape[1] > self.alignment_max_width:
            if self.feature_subsample:
                step = max(1, round(image.shape[1] / self.alignment_max_width))
                image = np.ascontiguousarray(image[::step, ::step])
                scale = 1.0 / step
            else:
                scale = self.alignment_max_width / image.shape[1]
                image = cv2.resize(
                    image,
                    (self.alignment_max_width, max(1, round(image.shape[0] * scale))),
                    interpolation=self.feature_resize_interpolation,
                )
        return clahe_image(image), scale

    def _init_reference(self) -> None:
        if not self.reference_path.is_file():
            print(f"[WARN] Fixed reference image not found at: {self.reference_path}")
            return
        self.reference = cv2.imread(str(self.reference_path), cv2.IMREAD_COLOR)
        if self.reference is None:
            print(f"[WARN] Failed to read reference image at: {self.reference_path}")
            return
        feature, self.reference_feature_scale = self._feature_frame(self.reference)
        gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, None)
        print(f"[INFO] FixedReferencePreprocessor initialized: {len(self.kp_ref)} reference features")

    @property
    def name(self) -> str:
        return "fixed_reference"

    def align(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, int, int]:
        """Aligns source image to fixed reference. Returns (aligned_bgr, affine_M, good, inliers)."""
        started = time.perf_counter()
        self._last_alignment_timings = {}
        self._last_alignment_diagnostics = {}
        if not self.alignment_enabled or self.reference is None or self.des_ref is None:
            return source, None, 0, 0

        source_ratio = source.shape[1] / max(source.shape[0], 1)
        reference_ratio = self.reference.shape[1] / max(self.reference.shape[0], 1)
        if abs(source_ratio - reference_ratio) > 0.05:
            return source, None, 0, 0

        stage = time.perf_counter()
        feature, source_feature_scale = self._feature_frame(source)
        gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        self._last_alignment_timings["feature_frame"] = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        kp_src, des_src = self.orb.detectAndCompute(gray, None)
        self._last_alignment_timings["orb_extract"] = (time.perf_counter() - stage) * 1000

        if des_src is None or len(kp_src) < 20:
            return source, None, 0, 0

        stage = time.perf_counter()
        matches = self.matcher.knnMatch(des_src, self.des_ref, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < LOWE_RATIO * n.distance]
        self._last_alignment_timings["feature_match"] = (time.perf_counter() - stage) * 1000

        if len(good) < 8:
            return source, None, len(good), 0

        src_pts = (
            np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            / source_feature_scale
        )
        ref_pts = (
            np.float32([self.kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            / self.reference_feature_scale
        )

        stage = time.perf_counter()
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
            refineIters=RANSAC_REFINE_ITERS,
        )
        self._last_alignment_timings["affine_estimation"] = (time.perf_counter() - stage) * 1000

        if M is None:
            return source, None, len(good), 0

        if inliers is not None and np.all(np.isfinite(M)):
            mask = inliers.reshape(-1).astype(bool)
            source_inliers = src_pts.reshape(-1, 2)[mask]
            reference_inliers = ref_pts.reshape(-1, 2)[mask]
            if len(source_inliers):
                projected = cv2.transform(source_inliers.reshape(-1, 1, 2), M).reshape(-1, 2)
                residuals = np.linalg.norm(projected - reference_inliers, axis=1)
                self._last_alignment_diagnostics = {
                    "inlier_residual_mean_px": float(np.mean(residuals)),
                    "inlier_residual_max_px": float(np.max(residuals)),
                    "inlier_source_extent_px": np.ptp(source_inliers, axis=0).tolist(),
                    "inlier_reference_extent_px": np.ptp(reference_inliers, axis=0).tolist(),
                    "inlier_source_points": source_inliers[:100].tolist(),
                    "inlier_reference_points": reference_inliers[:100].tolist(),
                    "inlier_points_truncated": len(source_inliers) > 100,
                }

        stage = time.perf_counter()
        if self.direct_model_warp:
            rx, ry, rw, rh = self.roi
            scale = min(TARGET_SIZE / rw, TARGET_SIZE / rh)
            new_w = round(rw * scale)
            new_h = round(rh * scale)
            model_matrix = M.copy()
            model_matrix[0, 2] -= rx
            model_matrix[1, 2] -= ry
            model_matrix *= scale
            aligned = cv2.warpAffine(
                source,
                model_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        elif self.direct_roi_warp:
            rx, ry, rw, rh = self.roi
            crop_matrix = M.copy()
            crop_matrix[0, 2] -= rx
            crop_matrix[1, 2] -= ry
            aligned = cv2.warpAffine(
                source,
                crop_matrix,
                (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        else:
            aligned = cv2.warpAffine(
                source,
                M,
                (self.reference.shape[1], self.reference.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        self._last_alignment_timings["image_warp"] = (time.perf_counter() - stage) * 1000
        self._last_alignment_timings["alignment_total"] = (time.perf_counter() - started) * 1000
        return aligned, M, len(good), inlier_count

    def _validate_alignment(
        self,
        matrix: np.ndarray,
        matches: int,
        inliers: int,
        source_width: int,
    ) -> None:
        """Reject a transform that is too weak, degenerate, or the wrong scale.

        A frame can produce a transform that RANSAC accepts but that is not a real
        alignment to the reference (repetitive conveyor texture, low contrast, or a
        part that is not present). The reference ROI must not be applied through
        such a transform.
        """
        if (
            matches < ALIGNMENT_MIN_MATCHES
            or inliers < ALIGNMENT_MIN_INLIERS
            or inliers / max(matches, 1) < ALIGNMENT_MIN_INLIER_RATIO
        ):
            raise AlignmentError(
                "alignment quality is below the limit "
                f"(matches={matches}, inliers={inliers}, "
                f"ratio={inliers / max(matches, 1):.3f})", retryable=True,
            )
        values = np.asarray(matrix, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise AlignmentError("alignment transform contains a non-finite value")
        determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
        if determinant <= 0.0:
            raise AlignmentError("alignment transform mirrors the frame (negative determinant)")
        reference_width = self.reference.shape[1] if self.reference is not None else source_width
        expected_scale = reference_width / max(source_width, 1)
        measured_scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
        if not (
            ALIGNMENT_MIN_SCALE_RATIO
            <= measured_scale / expected_scale
            <= ALIGNMENT_MAX_SCALE_RATIO
        ):
            raise AlignmentError(
                "alignment scale is outside the limit "
                f"(measured={measured_scale:.3f}, expected={expected_scale:.3f})"
            )

    def preprocess(
        self,
        image_bgr: np.ndarray,
        target_width: int = TARGET_SIZE,
        target_height: int = TARGET_SIZE,
        require_alignment: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        preprocess_started = time.perf_counter()
        src_h, src_w = image_bgr.shape[:2]

        # Stage 1: ORB Alignment
        aligned, affine_M, good_matches, inliers = self.align(image_bgr)
        diagnostics = {
            **self._last_alignment_diagnostics,
            "orb_matches": good_matches, "orb_inliers": inliers,
            "inlier_ratio": inliers / max(good_matches, 1),
            "source_width": src_w, "source_height": src_h,
            "reference_path": str(self.reference_path), "roi": list(self.roi),
            "thresholds": {
                "min_matches": ALIGNMENT_MIN_MATCHES, "min_inliers": ALIGNMENT_MIN_INLIERS,
                "min_inlier_ratio": ALIGNMENT_MIN_INLIER_RATIO,
                "min_scale_ratio": ALIGNMENT_MIN_SCALE_RATIO,
                "max_scale_ratio": ALIGNMENT_MAX_SCALE_RATIO,
            },
        }
        if affine_M is not None:
            diagnostics["affine_matrix"] = [
                [float(v) if np.isfinite(v) else None for v in row] for row in affine_M
            ]
            if np.all(np.isfinite(affine_M)):
                scale = math.hypot(float(affine_M[0, 0]), float(affine_M[0, 1]))
                expected = self.reference.shape[1] / max(src_w, 1) if self.reference is not None else 1.0
                diagnostics["scale_ratio"] = scale / expected
                diagnostics["rotation_deg"] = math.degrees(math.atan2(float(affine_M[1, 0]), float(affine_M[0, 0])))
                corners = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
                transformed = cv2.transform(corners.reshape(-1, 1, 2), affine_M).reshape(-1, 2)
                rx, ry, rw, rh = self.roi
                roi_corners = np.float32([[rx, ry], [rx+rw, ry], [rx+rw, ry+rh], [rx, ry+rh]])
                area, _ = cv2.intersectConvexConvex(transformed, roi_corners)
                diagnostics["roi_source_coverage"] = max(0.0, min(1.0, float(area) / max(rw*rh, 1)))

        alignment_available = (
            self.alignment_enabled and self.reference is not None and self.des_ref is not None
        )
        if require_alignment and alignment_available and affine_M is None:
            # The fixed ROI is defined in reference coordinates. Without a transform
            # the frame is not in reference space, so cropping it would inspect the
            # conveyor instead of the part. Fail closed instead of returning a
            # plausible-looking result from the wrong pixels. Callers that only
            # warm the model up (no part in view yet) pass require_alignment=False.
            raise AlignmentError(
                "the fixed reference could not be aligned to this frame "
                f"(matches={good_matches}, inliers={inliers}); refusing to crop the "
                "reference ROI from an unaligned frame", retryable=True, diagnostics=diagnostics,
            )
        if require_alignment and affine_M is not None:
            try:
                self._validate_alignment(affine_M, good_matches, inliers, src_w)
            except AlignmentError as error:
                error.diagnostics = diagnostics
                raise

        # Stage 2: Fixed ROI Crop. Apply the reference ROI only when the frame is in
        # reference space (aligned), or when alignment is deliberately disabled and
        # the frame is already reference-sized. A camera-sized frame is not proof of
        # alignment: width alone must never trigger the crop.
        reference_width = self.reference.shape[1] if self.reference is not None else 3200
        use_reference_roi = affine_M is not None or (
            not self.alignment_enabled and abs(src_w - reference_width) < 400
        )
        if affine_M is not None and (self.direct_roi_warp or self.direct_model_warp):
            rx, ry, rw, rh = self.roi
            cropped = aligned
        elif use_reference_roi:
            rx, ry, rw, rh = self.roi
            rx = max(0, min(aligned.shape[1] - 1, rx))
            ry = max(0, min(aligned.shape[0] - 1, ry))
            rw = min(aligned.shape[1] - rx, max(1, rw))
            rh = min(aligned.shape[0] - ry, max(1, rh))
            cropped = aligned[ry : ry + rh, rx : rx + rw]
        else:
            # Fallback for smaller/different aspect inputs (e.g. 1280x720)
            rx, ry, rw, rh = 0, 0, aligned.shape[1], aligned.shape[0]
            cropped = aligned

        if cropped.size == 0:
            cropped = aligned
            rx, ry, rw, rh = 0, 0, aligned.shape[1], aligned.shape[0]

        # Compute the model-space content size from the unmodified crop. This
        # geometry is shared by exact and fast enhancement modes.
        ch, cw = cropped.shape[:2]
        if affine_M is not None and self.direct_model_warp:
            scale = min(target_width / rw, target_height / rh)
            new_w = int(round(rw * scale))
            new_h = int(round(rh * scale))
        else:
            scale = min(target_width / cw, target_height / ch)
            new_w = int(round(cw * scale))
            new_h = int(round(ch * scale))

        enhancement_started = time.perf_counter()
        if not self.enhancement_enabled:
            if cropped.shape[1] == new_w and cropped.shape[0] == new_h:
                resized = cropped
            else:
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif self.enhance_at_model_resolution:
            # These operations only feed a 640x640 model. Running them after
            # the geometric downscale avoids processing discarded pixels.
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if "clahe" in self.enhancement_steps:
                resized = clahe_image(resized)
            if "reflection" in self.enhancement_steps:
                resized = suppress_pcb_reflections(resized)
            if "illumination" in self.enhancement_steps:
                resized = illumination_normalize(resized)
        else:
            # Exact legacy order retained for the compatibility mode.
            enhanced = clahe_image(cropped)
            enhanced = suppress_pcb_reflections(enhanced)
            enhanced = illumination_normalize(enhanced)
            resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        enhancement_ms = (time.perf_counter() - enhancement_started) * 1000
        tensor_started = time.perf_counter()
        pad_x = target_width - new_w
        pad_y = target_height - new_h
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

        model_bgr = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Normalize tensor to [0.0, 1.0] RGB float32
        model_rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            model_rgb.transpose(2, 0, 1)[np.newaxis, ...],
            dtype=np.float32,
        ) / np.float32(255.0)
        tensor_ms = (time.perf_counter() - tensor_started) * 1000

        metadata = {
            "preprocessor": self.name,
            "source_width": src_w,
            "source_height": src_h,
            "target_width": target_width,
            "target_height": target_height,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "pad_right": pad_right,
            "pad_bottom": pad_bottom,
            "roi": {"x": rx, "y": ry, "width": rw, "height": rh},
            "aligned": affine_M is not None,
            "affine_M": affine_M.tolist() if affine_M is not None else None,
            "orb_matches": good_matches,
            "orb_inliers": inliers,
            "alignment_max_width": self.alignment_max_width,
            "alignment_enabled": self.alignment_enabled,
            "require_alignment": require_alignment,
            "direct_roi_warp": self.direct_roi_warp,
            "direct_model_warp": self.direct_model_warp,
            "enhancement_enabled": self.enhancement_enabled,
            "enhancement_steps": list(self.enhancement_steps),
            "feature_subsample": self.feature_subsample,
            "orb_features": self.orb_features,
            "enhancement_resolution": "model" if self.enhance_at_model_resolution else "source",
            "stage_timings_ms": {
                **self._last_alignment_timings,
                "enhancement_and_resize": enhancement_ms,
                "letterbox_color_tensor": tensor_ms,
                "preprocessing_total": (time.perf_counter() - preprocess_started) * 1000,
            },
        }
        return tensor, model_bgr, metadata

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
        rx = float(roi["x"])
        ry = float(roi["y"])
        src_w = float(transform_metadata["source_width"])
        src_h = float(transform_metadata["source_height"])
        coordinate_scale_x = float(transform_metadata.get("source_coordinate_scale_x", 1.0))
        coordinate_scale_y = float(transform_metadata.get("source_coordinate_scale_y", 1.0))

        # 1. Un-letterbox: Model space -> Crop space
        crop_x = (box.x - pad_left) / scale
        crop_y = (box.y - pad_top) / scale
        crop_w = box.width / scale
        crop_h = box.height / scale

        # 2. Un-crop: Crop space -> Aligned space
        align_x1 = crop_x + rx
        align_y1 = crop_y + ry
        align_x2 = align_x1 + crop_w
        align_y2 = align_y1 + crop_h

        # 3. Un-align: Aligned space -> Original Source space
        affine_M_list = transform_metadata.get("affine_M")
        if affine_M_list is not None:
            M = np.array(affine_M_list, dtype=np.float64)
            M_inv = cv2.invertAffineTransform(M)
            # Transform the 4 corner points
            corners = np.array([
                [align_x1, align_y1],
                [align_x2, align_y1],
                [align_x2, align_y2],
                [align_x1, align_y2],
            ], dtype=np.float64).reshape(-1, 1, 2)
            pts_src = cv2.transform(corners, M_inv).reshape(-1, 2)
            orig_x1 = float(pts_src[:, 0].min())
            orig_y1 = float(pts_src[:, 1].min())
            orig_x2 = float(pts_src[:, 0].max())
            orig_y2 = float(pts_src[:, 1].max())
        else:
            orig_x1 = align_x1
            orig_y1 = align_y1
            orig_x2 = align_x2
            orig_y2 = align_y2

        orig_x1 *= coordinate_scale_x
        orig_x2 *= coordinate_scale_x
        orig_y1 *= coordinate_scale_y
        orig_y2 *= coordinate_scale_y
        orig_w = max(0.0, orig_x2 - orig_x1)
        orig_h = max(0.0, orig_y2 - orig_y1)

        if clip:
            orig_x1 = min(max(orig_x1, 0.0), src_w)
            orig_y1 = min(max(orig_y1, 0.0), src_h)
            orig_x2 = min(max(orig_x2, 0.0), src_w)
            orig_y2 = min(max(orig_y2, 0.0), src_h)
            return Box(orig_x1, orig_y1, max(0.0, orig_x2 - orig_x1), max(0.0, orig_y2 - orig_y1))

        return Box(orig_x1, orig_y1, orig_w, orig_h)


class FastFixedReferencePreprocessor(FixedReferencePreprocessor):
    """Validated low-latency variant that preserves the same geometry."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        roi: tuple[int, int, int, int] = (ROI_X, ROI_Y, ROI_W, ROI_H),
    ):
        super().__init__(
            reference_path=reference_path,
            roi=roi,
            alignment_max_width=1600,
            enhance_at_model_resolution=True,
        )

    @property
    def name(self) -> str:
        return "fixed_reference_fast"


class RealtimeFixedReferencePreprocessor(FixedReferencePreprocessor):
    """Validated sub-200 ms path using direct model warping and CLAHE only."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        roi: tuple[int, int, int, int] = (ROI_X, ROI_Y, ROI_W, ROI_H),
    ):
        super().__init__(
            reference_path=reference_path,
            roi=roi,
            alignment_max_width=320,
            enhance_at_model_resolution=True,
            direct_model_warp=True,
            enhancement_steps=("clahe",),
            feature_resize_interpolation=cv2.INTER_LINEAR,
            feature_subsample=True,
            orb_features=ORB_FEATURES,
        )

    @property
    def name(self) -> str:
        return "fixed_reference_realtime"
