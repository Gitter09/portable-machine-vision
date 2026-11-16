"""Per-cavity and per-cap detection on one rig, with no product-specific rules.

One blister card and several bottle caps are two different inspection problems,
but they share a shape: the part arrives in a frame, it is located by comparing
the frame with the stored empty fixture, and then something inside the part has
to be found. Which part that is does not matter here, and neither does its size,
colour or angle, because nothing in this module knows what a part is.

Two decisions shape the module, both from measurements on the operator's own
capture rather than from preference:

* Parts are located by differencing against the stored empty fixture. The
  brightness rule this replaces was fitted to a bright part on a dark deck, and
  on that same capture it returned a window over bare background with every
  blister pack outside it, and rejected a bottle cap for being 0.64 percent of
  the frame. A fixture difference is the object-independent form of the question.
* Every part is cropped on its own before inference. A cap hole measures 4.9 px
  in a whole-frame 640 view and 51.5 px cropped to a single cap, so a scene view
  throws away the resolution the defect is made of. No output head recovers it
  afterwards.

Counting is derived, never predicted. A model that emits "there are 9 pills"
cannot be checked against a picture; one that emits nine boxes can.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..preprocessing.agnostic_prep import Box, ProductAgnosticPreprocessor
from ..preprocessing.state_counts import STATE_CLASSES, summarise

__all__ = ["STATE_CLASSES", "InstanceDetector", "summarise"]


class InstanceDetector:
    """Runs a detector on every located part and maps results back to the frame.

    Detection happens per part crop, so the same model serves one card and a
    dozen caps without being told which it is looking at.
    """

    def __init__(
        self,
        preprocessor: ProductAgnosticPreprocessor,
        model: Any,
        confidence: float = 0.25,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.confidence = float(confidence)

    def detect(self, image_bgr: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        # A bare fixture raises here, which is the correct outcome: there is no
        # part to inspect, and returning an empty result would be
        # indistinguishable from a part with no defects in it.
        instances = self.preprocessor.preprocess_instances(image_bgr)

        detections: list[dict[str, Any]] = []
        for instance in instances:
            result = self.model.predict(
                instance["model_bgr"], conf=self.confidence, verbose=False
            )
            for prediction in result:
                names = getattr(prediction, "names", {}) or {}
                for box in prediction.boxes:
                    source = self.preprocessor.inverse_map(
                        Box(
                            float(box.xyxy[0][0]),
                            float(box.xyxy[0][1]),
                            float(box.xyxy[0][2] - box.xyxy[0][0]),
                            float(box.xyxy[0][3] - box.xyxy[0][1]),
                        ),
                        instance["metadata"],
                    )
                    class_id = int(box.cls[0])
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": str(names.get(class_id, class_id)),
                            "confidence": float(box.conf[0]),
                            "box_source": source.as_dict(),
                            "part_box_source": dict(instance["box"]),
                            "part_index": instance["index"],
                        }
                    )

        payload = summarise(detections)
        payload["detections"] = detections
        payload["part_count"] = len(instances)
        payload["possibly_merged"] = instances[0]["possibly_merged"] if instances else 0
        payload["latency_ms"] = (time.perf_counter() - started) * 1000
        return payload
