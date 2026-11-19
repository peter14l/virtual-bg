from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class WebcamCapture:
    """Thread-safe wrapper around cv2.VideoCapture."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Open the camera, configure it, and warm up the sensor."""
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera index {self.index}. "
                "Check that the device is connected, camera privacy permissions are enabled, "
                "and that no other application (e.g. OBS Studio, Zoom, Teams, or browser) "
                "is currently holding the physical webcam."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        for _ in range(5):
            cap.read()
        self._cap = cap

    def read(self) -> Optional[np.ndarray]:
        """Read one BGR frame, or None on failure/EOF."""
        with self._lock:
            cap = self._cap
            if cap is None:
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            return frame

    def release(self) -> None:
        """Release the camera. Idempotent and safe to call repeatedly."""
        with self._lock:
            cap = self._cap
            if cap is not None:
                cap.release()
                self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        return None
