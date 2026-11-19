import pytest

from vbg import setup


def test_filter_urls_present():
    assert "UnityCaptureFilter32.dll" in setup.FILTER_DLL_URLS
    assert "UnityCaptureFilter64.dll" in setup.FILTER_DLL_URLS
    assert all(u.startswith("https://") for u in setup.FILTER_DLL_URLS.values())


def test_device_name():
    assert setup.DEVICE_NAME == "Unity Video Capture"


def test_is_admin_returns_bool():
    assert isinstance(setup._is_admin(), bool)


def test_third_party_dir_points_at_project():
    assert setup.THIRD_PARTY_DIR.name == "UnityCapture"