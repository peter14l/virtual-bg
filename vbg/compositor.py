from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from vbg.config import Config


class Compositor:
    """Composite foreground frames with background using alpha mask."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._bg_image: Optional[np.ndarray] = None
        self._bg_image_path: Optional[str] = None
        self._bg_image_size: Optional[Tuple[int, int]] = None
        self._color_bgr: Tuple[int, int, int] = (30, 30, 30)
        self.kind: str = "blur"
        self.set_background(config.background, config.background_image, config.background_color)

    def set_background(
        self,
        kind: str,
        image_path: Optional[str] = None,
        color: Tuple[int, int, int] = (30, 30, 30)
    ) -> None:
        """Set background mode."""
        if kind not in ("blur", "image", "color"):
            raise ValueError(f"kind must be 'blur', 'image', or 'color', got {kind}")

        if kind == "image":
            if not image_path:
                raise ValueError("image_path required for 'image' background")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Failed to load background image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Failed to load background image: {image_path}")
            self._bg_image = img
            self._bg_image_path = image_path
            self._bg_image_size = None

        elif kind == "color":
            if not (isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
                raise ValueError("color must be a 3-tuple of ints in [0, 255]")
            self._color_bgr = (color[2], color[1], color[0])
            self._bg_image = None
            self._bg_image_path = None

        else:  # blur
            self._bg_image = None
            self._bg_image_path = None

        self.kind = kind

    def compose(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Composite frame with background using mask."""
        if self.config.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
            mask = cv2.flip(mask, 1)

        h, w = frame_bgr.shape[:2]

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(2)
        mask = mask.astype(np.float32, copy=False)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0)
        np.clip(mask, 0.0, 1.0, out=mask)

        if self.kind == "blur":
            # Resolution-independent blur: sigma scales with frame size so the
            # effect looks the same at 480p, 720p or 1080p. blur_strength is a
            # relative multiplier (25 = 1.0x default).
            amount = max(int(self.config.blur_strength), 1)
            sigma = 0.03 * min(h, w) * (amount / 25.0)
            if sigma < 0.1:
                sigma = 0.1
            bg = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)

        elif self.kind == "color":
            bg = np.full_like(frame_bgr, self._color_bgr, dtype=np.uint8)

        else:  # image
            if self._bg_image is None and self._bg_image_path:
                self._bg_image = cv2.imread(self._bg_image_path)
                if self._bg_image is None:
                    raise FileNotFoundError(f"Failed to load background image: {self._bg_image_path}")

            if self._bg_image_size != (w, h):
                self._bg_image = cv2.resize(self._bg_image, (w, h), interpolation=cv2.INTER_LINEAR)
                self._bg_image_size = (w, h)
            bg = self._bg_image

        m = mask[..., None]
        out = frame_bgr * m + bg * (1.0 - m)
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8, copy=False)
