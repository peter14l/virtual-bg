import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from vbg.segment import PersonSegmenter  # noqa: E402


@pytest.fixture(scope="module")
def segmenter():
    seg = PersonSegmenter(model_selection=0)
    yield seg
    seg.close()


def test_segment_shape_dtype_range(segmenter):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = segmenter.segment(frame)
    assert mask.shape == (120, 160)
    assert mask.dtype == np.float32
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_segment_non_square(segmenter):
    frame = np.zeros((90, 200, 3), dtype=np.uint8)
    mask = segmenter.segment(frame)
    assert mask.shape == (90, 200)


def test_close_is_idempotent(segmenter):
    # Calling close twice must not raise.
    segmenter.close()
    segmenter.close()


def test_close_holes_fills_pinholes():
    mask = np.ones((40, 40), dtype=np.float32)
    mask[18:22, 18:22] = 0.0  # a small hole that would let the background poke through
    out = PersonSegmenter._close_holes(mask)
    assert out[20, 20] > 0.9


def test_same_frame_is_stable():
    seg = PersonSegmenter(0, smooth=0.6)
    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        m1 = seg.segment(frame)
        m2 = seg.segment(frame)
        assert np.allclose(m1, m2, atol=1e-4)
    finally:
        seg.close()


def test_smooth_param_is_clamped():
    seg = PersonSegmenter(0, smooth=5.0)
    try:
        assert seg._smooth == 1.0
    finally:
        seg.close()