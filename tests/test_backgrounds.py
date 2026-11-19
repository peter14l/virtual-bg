import cv2
import numpy as np
import pytest

from vbg import backgrounds


def test_filename_from_url():
    assert backgrounds._filename_from_url("https://x.com/a/b/photo.jpg") == "photo.jpg"
    assert backgrounds._filename_from_url("https://x.com/a/b/") == "background.img"


def test_download_from_file_url(tmp_path):
    # Build a tiny source image and "download" it via a file:// URL (no network).
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), np.full((12, 12, 3), 128, np.uint8))

    dest_dir = tmp_path / "dl"
    out = backgrounds.download_image(src.as_uri(), dest_dir=dest_dir)

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.parent == dest_dir


def test_download_reuses_existing(tmp_path):
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), np.zeros((8, 8, 3), np.uint8))

    dest_dir = tmp_path / "dl"
    first = backgrounds.download_image(src.as_uri(), dest_dir=dest_dir)
    mtime = first.stat().st_mtime
    second = backgrounds.download_image(src.as_uri(), dest_dir=dest_dir)
    assert second == first
    assert second.stat().st_mtime == mtime