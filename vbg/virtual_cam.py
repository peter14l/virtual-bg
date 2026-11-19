"""Virtual camera interface using pyvirtualcam.

On Windows, pyvirtualcam publishes frames through a DirectShow capture device.
It bundles no driver itself, so one of these must be registered on the system:

* ``obs`` backend -> "OBS Virtual Camera" (ships with OBS Studio). RECOMMENDED:
  works with browsers (Chrome/Meet/Zoom/Teams) because it negotiates RGB24/I420
  correctly.
* ``unitycapture`` backend -> "Unity Video Capture" (the bundled Unity Capture
  filter, MIT licensed). Install it with ``python run.py --install``. Works with
  ffmpeg-based apps but NOT browsers (its filter ignores ``SetFormat(RGB24)``).

If no backend is forced, pyvirtualcam tries every available backend and uses the
first one that works (OBS is tried before Unity Capture). The device name shown
inside Zoom/Teams/browsers comes from the driver ("OBS Virtual Camera" or "Unity
Video Capture"), not from ``name``.
"""

from __future__ import annotations

import inspect
from typing import Optional

import cv2
import numpy as np


class VirtualCamera:
    """Virtual camera that exposes a BGR frame stream as a selectable webcam."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        name: str,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the virtual camera.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames per second.
            name: Human-readable label used in logs/errors (not the device name).
            backend: Force a pyvirtualcam backend ("obs" | "unitycapture"), or
                None to try all available backends automatically.
            device: Force a specific device name, or None to use any available.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.backend = backend
        self.device = device
        self._cam = None
        self._backend_used: Optional[str] = None
        self.device_name = name

    def start(self) -> None:
        """
        Start the virtual camera.

        Raises:
            RuntimeError: If pyvirtualcam fails to initialize. See the module
                docstring for the Windows driver requirement.
        """
        try:
            import pyvirtualcam
        except Exception as e:
            raise RuntimeError(
                f"pyvirtualcam is not available: {e}. Install it with 'pip install pyvirtualcam'."
            ) from e

        kwargs = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

        # fmt: newer pyvirtualcam wants a PixelFormat enum, older wants a string.
        pixel_format = getattr(pyvirtualcam, "PixelFormat", None)
        if pixel_format is not None and hasattr(pixel_format, "BGR"):
            kwargs["fmt"] = pixel_format.BGR
        else:
            kwargs["fmt"] = "BGR"

        # Only pass backend/device when explicitly requested; otherwise let
        # pyvirtualcam auto-select whichever driver is installed.
        if self.backend:
            kwargs["backend"] = self.backend
        if self.device:
            kwargs["device"] = self.device

        try:
            self._cam = pyvirtualcam.Camera(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to start virtual camera ({self.name}): {e}.\n"
                "No virtual-camera driver was found. Fix it with one of:\n"
                "  * install OBS Studio        (provides 'OBS Virtual Camera', works with browsers)\n"
                "  * python run.py --install   (installs Unity Capture; ffmpeg-based apps only)\n"
                "On Linux install v4l2loopback. "
                "Use --preview-only to run without a virtual camera."
            ) from e

        # Remember which backend pyvirtualcam picked, for logs.
        self._backend_used = getattr(self._cam, "_backend_name", None)

        # The device name apps will see (e.g. "Unity Video Capture").
        # NB: pyvirtualcam exposes `device` as a string attribute (not a method).
        try:
            self.device_name = self._cam.device
        except Exception:
            self.device_name = self.name

    def send(self, frame_bgr: np.ndarray) -> None:
        """
        Send a BGR frame to the virtual camera.

        Args:
            frame_bgr: BGR uint8 frame with shape (height, width, 3).
                If shape or dtype differs, it will be resized/converted.

        Raises:
            RuntimeError: If camera not started or driver error occurs.
        """
        if self._cam is None:
            raise RuntimeError("VirtualCamera not started")

        frame = frame_bgr
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise RuntimeError(f"Unsupported channel count: {frame.shape[2]}")

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        try:
            self._cam.send(frame)
        except Exception as e:
            raise RuntimeError(f"Failed to send frame to virtual camera: {e}") from e

        try:
            self._cam.sleep_until_next_frame()
        except NotImplementedError:
            pass

    def stop(self) -> None:
        """Stop the virtual camera. Idempotent."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            finally:
                self._cam = None

    def __enter__(self) -> "VirtualCamera":
        """Context manager entry: starts the camera."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: stops the camera."""
        self.stop()