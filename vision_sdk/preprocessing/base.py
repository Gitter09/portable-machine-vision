"""Abstract BasePreprocessor and geometric structures for the preprocessing pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def as_dict(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "width": round(float(self.width), 2),
            "height": round(float(self.height), 2),
        }


class BasePreprocessor(ABC):
    """Abstract interface for all image preprocessing implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifying this preprocessor."""
        ...

    @abstractmethod
    def preprocess(
        self,
        image_bgr: np.ndarray,
        target_width: int = 640,
        target_height: int = 640,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Preprocesses input image into model tensor.

        Args:
            image_bgr: Source image array in OpenCV BGR format (H, W, 3).
            target_width: Model expected width (default: 640).
            target_height: Model expected height (default: 640).

        Returns:
            tensor: Preprocessed tensor (1, 3, target_height, target_width), float32 in [0.0, 1.0].
            model_vis_bgr: Preprocessed visualization image in BGR format (target_height, target_width, 3).
            transform_metadata: Dictionary of parameters needed for exact coordinate inversion.
        """
        ...

    @abstractmethod
    def inverse_map(
        self,
        box: Box,
        transform_metadata: dict[str, Any],
        clip: bool = True,
    ) -> Box:
        """Maps a model-space bounding box back to original source image coordinates.

        Args:
            box: Bounding box in model coordinates (640x640).
            transform_metadata: Metadata dict returned by preprocess().
            clip: Whether to clamp coordinates to original image dimensions.

        Returns:
            Box in original source image pixel coordinates.
        """
        ...


