"""Build a preprocessor from release configuration.

Training, the host service and the board must all construct the same
preprocessor from the same settings. Anything else is a train/serve skew, and a
skew shows up as an unexplained accuracy loss rather than an error.

Keep this free of any dependency on VisionConfig so that the board runtime, which
does not ship the dataset code, can use it too.
"""

from __future__ import annotations

from pathlib import Path

from .agnostic_prep import ProductAgnosticPreprocessor, ProductProfile
from .base import BasePreprocessor
from .fixed_ref_prep import (
    FastFixedReferencePreprocessor,
    FixedReferencePreprocessor,
    RealtimeFixedReferencePreprocessor,
)

#: Preprocessors that take a reference image and a fixed ROI, not a profile.
_REFERENCE_PREPROCESSORS = {
    "fixed_reference": FixedReferencePreprocessor,
    "fixed_reference_fast": FastFixedReferencePreprocessor,
    "fixed_reference_realtime": RealtimeFixedReferencePreprocessor,
}

#: Kept for compatibility with the existing "universal" alias.
_ALIASES = {"universal": "fixed_reference"}


def resolve_preprocessor(
    name: str = "fixed_reference_realtime",
    *,
    reference_image: str | Path | None = None,
    roi: tuple[int, int, int, int] | None = None,
    profile_path: str | Path | None = None,
    flatfield_lut: str | Path | None = None,
) -> BasePreprocessor:
    """Instantiate the named preprocessor.

    `product_agnostic` reads a ProductProfile and a flat-field map. The
    reference-image preprocessors read a golden image and a fixed ROI. The two
    families are deliberately separate: the reference preprocessors are pinned to
    the screw rig and must not be silently re-pointed at another product.
    """
    key = _ALIASES.get(name.lower(), name.lower())

    if key == "product_agnostic":
        profile = ProductProfile.load(profile_path) if profile_path else ProductProfile()
        if flatfield_lut is None and profile.flatfield_lut:
            flatfield_lut = profile.flatfield_lut
        return ProductAgnosticPreprocessor(profile=profile, flatfield_lut=flatfield_lut)

    if key in _REFERENCE_PREPROCESSORS:
        kwargs: dict[str, object] = {}
        if reference_image:
            kwargs["reference_path"] = reference_image
        if roi is not None:
            kwargs["roi"] = tuple(roi)
        return _REFERENCE_PREPROCESSORS[key](**kwargs)

    raise ValueError(f"Unknown preprocessor: '{name}'")


def preprocessing_settings(
    name: str,
    profile_path: str | Path | None = None,
    flatfield_lut: str | Path | None = None,
) -> dict:
    """A serialisable summary of the preprocessing a release was built with.

    Recorded in the release manifest so a deployed model can be traced back to the
    exact conditioning that produced it, and so a mismatch between calibration and
    serving is detectable rather than silent.
    """
    settings: dict = {"preprocessor": name}
    if _ALIASES.get(name.lower(), name.lower()) != "product_agnostic":
        return settings
    profile = ProductProfile.load(profile_path) if profile_path else ProductProfile()
    settings["profile"] = profile.name
    settings["crop_mode"] = profile.crop_mode
    settings["clahe"] = {"clip": profile.clahe_clip, "tiles": list(profile.clahe_tiles)}
    settings["tone_target_p99"] = profile.tone_target_p99
    settings["pad_colour"] = list(profile.pad_colour)
    if profile.presence_threshold is not None:
        settings["presence_threshold"] = profile.presence_threshold
    settings["stages"] = {
        "unbar": profile.stage_unbar,
        "locate": profile.stage_locate,
        "flatfield": profile.stage_flatfield,
        "tone": profile.stage_tone,
        "clahe": profile.stage_clahe,
    }
    if flatfield_lut:
        settings["flatfield_lut"] = str(flatfield_lut)
    return settings
