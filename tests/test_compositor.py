import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vbg.config import Config  # noqa: E402
from vbg.compositor import Compositor  # noqa: E402


def _frame(h=48, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_blur_compose_shape_and_dtype():
    comp = Compositor(Config(width=64, height=48, background="blur"))
    out = comp.compose(_frame(), np.zeros((48, 64), dtype=np.float32))
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_color_background_is_applied():
    # RGB (255,0,0) must appear as BGR (0,0,255)
    comp = Compositor(Config(width=32, height=32, background="color", background_color=(255, 0, 0)))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.float32)  # fully background
    out = comp.compose(frame, mask)
    assert tuple(int(v) for v in out[0, 0]) == (0, 0, 255)


def test_full_foreground_keeps_original_frame():
    comp = Compositor(Config(width=16, height=16, background="color", background_color=(0, 0, 0)))
    frame = np.full((16, 16, 3), 200, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.float32)  # fully foreground
    out = comp.compose(frame, mask)
    # Blur/feather may soften edges slightly, but the interior should stay ~200
    assert abs(int(out[8, 8].mean()) - 200) <= 2


def test_image_mode_missing_file_raises(tmp_path):
    cfg = Config(width=16, height=16, background="image", background_image=str(tmp_path / "nope.png"))
    with pytest.raises(FileNotFoundError):
        Compositor(cfg)


def test_invalid_kind_raises():
    cfg = Config(width=8, height=8, background="blur")
    comp = Compositor(cfg)
    with pytest.raises(ValueError):
        comp.set_background("bogus")


def test_blur_strength_affects_output():
    # A noisy frame + all-background mask: stronger blur must smooth it more.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.float32)

    low = Compositor(Config(width=160, height=120, background="blur", blur_strength=5)).compose(frame, mask)
    high = Compositor(Config(width=160, height=120, background="blur", blur_strength=60)).compose(frame, mask)

    assert not np.array_equal(low, high)
    low_var = cv2.Laplacian(cv2.cvtColor(low, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    high_var = cv2.Laplacian(cv2.cvtColor(high, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert high_var < low_var