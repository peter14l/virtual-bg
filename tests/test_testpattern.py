import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vbg.testpattern import TestPatternCapture  # noqa: E402


def test_pattern_shape_and_dtype():
    cap = TestPatternCapture(width=160, height=120, fps=30)
    cap.start()
    frame = cap.read()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_pattern_advances():
    cap = TestPatternCapture(width=160, height=120)
    cap.start()
    a = cap.read()
    b = cap.read()
    assert not np.array_equal(a, b)


def test_context_manager():
    with TestPatternCapture(width=64, height=64) as cap:
        assert cap.read().shape == (64, 64, 3)