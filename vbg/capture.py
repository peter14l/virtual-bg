from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class WebcamCapture:
    """Thread-safe, decoupled capture wrapper around cv2.VideoCapture.

    A background reader thread continuously acquires frames so that `read()`
    always returns the latest frame instantly without camera buffer latency
    or pipeline starvation.
    """

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        """Open the camera, warm up the sensor, and launch background acquisition."""
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

        initial_frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None:
                initial_frame = f

        self._cap = cap
        if initial_frame is not None:
            self._latest_frame = self._format_frame(initial_frame)

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="vbg-webcam-reader")
        self._thread.start()

    def _format_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame

    def _worker(self) -> None:
        while self._running:
            with self._lock:
                cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            formatted = self._format_frame(frame)
            with self._lock:
                self._latest_frame = formatted

    def read(self) -> Optional[np.ndarray]:
        """Read the most recent BGR frame, or None if unavailable/stopped."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            # Fallback if background worker hasn't populated yet but cap is active
            if self._cap is not None:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self._latest_frame = self._format_frame(frame)
                    return self._latest_frame.copy()
            return None

    def release(self) -> None:
        """Stop background acquisition and release camera resources. Idempotent."""
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
            self._thread = None

        with self._lock:
            cap = self._cap
            if cap is not None:
                cap.release()
                self._cap = None
            self._latest_frame = None

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        return None
