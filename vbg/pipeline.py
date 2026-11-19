"""End-to-end pipeline: capture -> segment -> composite -> virtual camera."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from vbg.capture import WebcamCapture
from vbg.compositor import Compositor
from vbg.config import Config
from vbg.segment import PersonSegmenter
from vbg.virtual_cam import VirtualCamera


class Pipeline:
    """Wire the capture, segmentation, compositing and virtual-camera stages together.

    ``run`` blocks until ``stop`` is called, the frame budget is reached, or the
    user interrupts (Ctrl+C / ESC in preview mode).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._stop = threading.Event()
        self._capture = None
        self._segmenter: Optional[PersonSegmenter] = None
        self._compositor: Optional[Compositor] = None
        self._vcam: Optional[VirtualCamera] = None

    def _make_capture(self):
        if self.config.source == "test":
            from vbg.testpattern import TestPatternCapture

            return TestPatternCapture(
                self.config.camera_index,
                self.config.width,
                self.config.height,
                self.config.fps,
            )
        return WebcamCapture(
            self.config.camera_index,
            self.config.width,
            self.config.height,
            self.config.fps,
        )

    def _setup(self) -> None:
        self._capture = self._make_capture()
        self._capture.start()

        self._segmenter = PersonSegmenter(
            self.config.model_selection,
            smooth=self.config.smooth,
            refine_edges=self.config.guided_filter,
            motion_adaptive=self.config.motion_adaptive,
        )
        self._compositor = Compositor(self.config)

        if not self.config.preview_only:
            self._vcam = VirtualCamera(
                self.config.width,
                self.config.height,
                self.config.fps,
                self.config.vcam_name,
                backend=self.config.vcam_backend,
                device=self.config.vcam_device,
            )
            self._vcam.start()

    def _teardown(self) -> None:
        if self._vcam is not None:
            self._vcam.stop()
            self._vcam = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None
        if self.config.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def run(self) -> None:
        """Run the blocking main loop. Always cleans up on exit."""
        self._setup()

        if self._vcam is not None:
            dev = getattr(self._vcam, "device_name", None) or self.config.vcam_name
            print(f"[vbg] Virtual camera ready: {dev}")
            print(f'[vbg] Select "{dev}" as the camera in Zoom/Teams/your browser.')
            print(
                "[vbg] The physical camera is now in use by vbg; other apps must use "
                "the virtual camera above."
            )

        frames = 0
        try:
            while not self._stop.is_set():
                frame = self._capture.read()  # type: ignore[union-attr]
                if frame is None:
                    time.sleep(0.01)
                    continue

                mask = self._segmenter.segment(frame)  # type: ignore[union-attr]
                out = self._compositor.compose(frame, mask)  # type: ignore[union-attr]

                if self._vcam is not None:
                    self._vcam.send(out)

                if self.config.show_preview:
                    try:
                        cv2.imshow("virtual-bg", out)
                        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                            break
                    except Exception:
                        self.config.show_preview = False

                frames += 1
                if self.config.max_frames and frames >= self.config.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._teardown()

    def stop(self) -> None:
        """Signal the main loop to stop. Safe to call from another thread."""
        self._stop.set()