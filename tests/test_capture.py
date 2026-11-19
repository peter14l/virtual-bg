import numpy as np
import pytest

from vbg.capture import WebcamCapture


def test_format_frame_dimensions_and_channels():
    cap = WebcamCapture(index=0, width=64, height=48, fps=30)

    # 1. Grayscale to BGR
    gray = np.zeros((48, 64), dtype=np.uint8)
    formatted = cap._format_frame(gray)
    assert formatted.shape == (48, 64, 3)

    # 2. Resize from arbitrary dimensions
    other = np.zeros((100, 200, 3), dtype=np.uint8)
    formatted = cap._format_frame(other)
    assert formatted.shape == (48, 64, 3)

    # 3. Cast non-uint8 dtype
    float_frame = np.zeros((48, 64, 3), dtype=np.float32)
    formatted = cap._format_frame(float_frame)
    assert formatted.dtype == np.uint8


def test_release_when_not_started_is_safe():
    cap = WebcamCapture(index=999, width=64, height=48, fps=30)
    cap.release()  # safe and idempotent
    assert cap.read() is None
