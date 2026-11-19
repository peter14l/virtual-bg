"""Synthetic test-pattern source so the pipeline runs without a physical webcam.

Useful for demos, CI smoke tests, and verifying the segmentation/compositing
stages on machines with no camera or no virtual-camera driver installed.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class TestPatternCapture:
    """A fake capture that emits a moving pattern, matching WebcamCapture's API."""

    __test__ = False  # Prevent pytest from treating this as a test suite

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._i = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._i = 0

    def read(self) -> np.ndarray:
        with self._lock:
            img = np.full((self.height, self.width, 3), 60, dtype=np.uint8)
            cx, cy = self.width // 2, self.height // 2
            radius = max(8, min(self.width, self.height) // 3)
            cv2.circle(img, (cx, cy), radius, (80, 200, 120), -1)
            x = int((self._i * 8) % max(self.width, 1))
            cv2.rectangle(
                img,
                (x - 30, self.height // 4),
                (x + 30, (3 * self.height) // 4),
                (200, 180, 60),
                -1,
            )
            self._i += 1
            return img

    def release(self) -> None:
        pass

    def __enter__(self) -> "TestPatternCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()