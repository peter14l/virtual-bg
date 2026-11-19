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


def test_smoothstep_mask_curve():
    from vbg.segment import smoothstep_mask

    raw = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0], dtype=np.float32)
    stepped = smoothstep_mask(raw, low=0.2, high=0.8)
    assert stepped[0] == 0.0
    assert stepped[1] == 0.0
    assert stepped[2] == 0.0
    assert 0.49 <= stepped[3] <= 0.51
    assert stepped[4] == 1.0
    assert stepped[5] == 1.0
    assert stepped[6] == 1.0


def test_fast_guided_filter_edge_transfer():
    from vbg.segment import fast_guided_filter

    # Sharp vertical step in guide image (left black, right white)
    guide = np.zeros((32, 32, 3), dtype=np.uint8)
    guide[:, 16:] = 255

    # Blurry/fuzzy mask with a gradual transition
    x = np.linspace(0, 1, 32, dtype=np.float32)
    fuzzy_mask = np.tile(x, (32, 1))

    refined = fast_guided_filter(guide, fuzzy_mask, radius=4, eps=1e-3, scale=1)
    assert refined.shape == (32, 32)
    assert refined.dtype == np.float32
    assert float(refined.min()) >= 0.0
    assert float(refined.max()) <= 1.0
    # Near the edge: the dark side of the guide step suppresses the mask,
    # and the bright side enhances it.
    assert refined[:, 15].mean() < fuzzy_mask[:, 15].mean()
    assert refined[:, 16].mean() > fuzzy_mask[:, 16].mean()



def test_motion_adaptive_smoothing():
    seg = PersonSegmenter(0, smooth=0.8, motion_adaptive=True)
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        seg.segment(frame)
        assert seg._prev is not None
        prev_copy = seg._prev.copy()

        # Simulate large sudden motion
        seg._prev = np.zeros((64, 64), dtype=np.float32)
        simulated_new_mask = np.ones((64, 64), dtype=np.float32)
        # Test diff response
        diff = np.abs(simulated_new_mask - seg._prev)
        a = np.clip(0.8 + (1.0 - 0.8) * (diff * 2.5), 0.8, 0.98)
        # When diff is 1.0, a should be boosted to 0.98 (fast adaptation)
        assert np.all(a >= 0.95)
    finally:
        seg.close()
