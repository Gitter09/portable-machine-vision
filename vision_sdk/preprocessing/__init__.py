"""Preprocessing for the vision SDK.

`product_agnostic` is the general path. The `fixed_reference*` family is retained
for the screw-on-powerboard release, which is pinned to a fixed rig and a fixed
crop window; see docs/decisions/log.md D-10.
"""

from __future__ import annotations

from .agnostic_prep import (
    FlatField,
    PartNotFoundError,
    ProductAgnosticPreprocessor,
    ProductProfile,
    build_flatfield,
    load_flatfield,
    save_flatfield,
)
from .base import BasePreprocessor, Box
from .factory import preprocessing_settings, resolve_preprocessor
from .fixed_ref_prep import (
    FastFixedReferencePreprocessor,
    FixedReferencePreprocessor,
    RealtimeFixedReferencePreprocessor,
)

__all__ = [
    "BasePreprocessor",
    "Box",
    "FixedReferencePreprocessor",
    "FastFixedReferencePreprocessor",
    "RealtimeFixedReferencePreprocessor",
    "ProductAgnosticPreprocessor",
    "ProductProfile",
    "PartNotFoundError",
    "FlatField",
    "build_flatfield",
    "load_flatfield",
    "save_flatfield",
    "resolve_preprocessor",
    "preprocessing_settings",
]
