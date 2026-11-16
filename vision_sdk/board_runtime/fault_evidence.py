"""Return rejected alignment frames to Admin without saving images on the board."""

from datetime import datetime, timezone
import hashlib

import cv2

from preprocessing.fixed_ref_prep import AlignmentError

MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def inspection_error(error, frame):
    body = {"code": "INSPECTION_FAILED", "message": str(error)}
    if not isinstance(error, AlignmentError):
        return body, b""
    body.update(
        code="ALIGNMENT_FAILED", retryable=error.retryable,
        alignment=error.diagnostics, captured_at=datetime.now(timezone.utc).isoformat(),
    )
    if frame is None:
        body.update(evidence_error="no captured frame", retryable=False)
        return body, b""
    body.update(source_width=int(frame.shape[1]), source_height=int(frame.shape[0]))
    try:
        # Preserve the original pixels for replay whenever the protocol limit
        # permits it. A high quality full-resolution JPEG is the size fallback.
        ok, encoded = cv2.imencode(".png", frame)
        image_format = "png"
        if not ok or encoded.nbytes > MAX_EVIDENCE_BYTES:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_format = "jpg"
        if not ok or encoded.nbytes > MAX_EVIDENCE_BYTES:
            raise RuntimeError("fault frame could not be encoded within the protocol limit")
        payload = encoded.tobytes()
        body.update(image_format=image_format, image_sha256=hashlib.sha256(payload).hexdigest())
        return body, payload
    except Exception as encoding_error:
        body.update(evidence_error=str(encoding_error), retryable=False)
        return body, b""
