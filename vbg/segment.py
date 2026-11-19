"""Person segmentation using the MediaPipe Tasks API (ImageSegmenter).

The legacy ``mediapipe.solutions`` API was removed in MediaPipe >= 1.0, so this
module uses the Tasks API and lazily downloads the Selfie Segmenter TFLite model
on first use. The public contract is unchanged:

    PersonSegmenter(model_selection=0).segment(frame_bgr) -> float32 mask (H, W) in [0, 1]
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Selfie Segmenter (square). ImageSegmenter resizes arbitrary inputs internally,
# so the same model serves both "general" and "landscape" requests.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "selfie_segmenter.tflite"


def _ensure_model() -> str:
    """Return a local path to the segmenter model, downloading it if necessary."""
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 0:
        return str(_MODEL_PATH)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _MODEL_PATH.with_suffix(".tflite.part")
    print(f"[vbg] Downloading segmentation model to {_MODEL_PATH} ...")
    try:
        req = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "virtual-bg"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp_path, _MODEL_PATH)
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Could not download the segmentation model from {_MODEL_URL}: {e}. "
            "Check your internet connection, or place 'selfie_segmenter.tflite' in "
            f"'{_MODEL_DIR}' manually."
        ) from e
    return str(_MODEL_PATH)


_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class PersonSegmenter:
    """Segment a person from a BGR frame using MediaPipe ImageSegmenter.

    Attributes:
        model_selection: 0 = general, 1 = landscape. Both currently use the same
            Selfie Segmenter model; retained for API compatibility.
    """

    def __init__(self, model_selection: int = 0, smooth: float = 0.6) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to import mediapipe Tasks API: {e}. "
                "Install a recent mediapipe: 'pip install -U mediapipe'."
            ) from e

        self._mp = mp
        self._model_selection = model_selection
        self._smooth = max(0.0, min(1.0, float(smooth)))
        self._prev: np.ndarray | None = None
        self._seg = None
        self._closed = False

        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._seg = vision.ImageSegmenter.create_from_options(options)

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 person mask in [0.0, 1.0] with shape (H, W).

        Args:
            frame_bgr: Input frame in BGR color space, shape (H, W, 3).

        Returns:
            2D float32 mask with same H, W as input, values in [0.0, 1.0].
        """
        if self._closed or self._seg is None:
            raise RuntimeError("Segmenter is closed")

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._seg.segment(mp_image)

        mask = None
        if result.confidence_masks:
            # confidence_masks[0] is the foreground (person) probability.
            mask = result.confidence_masks[0].numpy_view()
        elif result.category_mask is not None:
            mask = result.category_mask.numpy_view()

        if mask is None:
            h, w = frame_bgr.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(axis=2)
        mask = np.array(mask, dtype=np.float32, copy=True)
        np.clip(mask, 0.0, 1.0, out=mask)
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(
                mask,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Fill small holes / drop tiny specks so the background cannot poke through.
        mask = self._close_holes(mask)

        # Temporal smoothing (EMA) to stop frame-to-frame flicker.
        if self._smooth > 0.0 and self._prev is not None and self._prev.shape == mask.shape:
            a = self._smooth
            mask = (a * mask + (1.0 - a) * self._prev).astype(np.float32)
        self._prev = mask.copy()
        return mask

    @staticmethod
    def _close_holes(mask: np.ndarray) -> np.ndarray:
        """Morphological close: fill pinholes and remove isolated specks."""
        m8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        return m8.astype(np.float32) / 255.0

    def close(self) -> None:
        """Release MediaPipe resources. Idempotent."""
        if not self._closed and self._seg is not None:
            close_fn = getattr(self._seg, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._closed = True
            self._seg = None

    def __enter__(self) -> "PersonSegmenter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()