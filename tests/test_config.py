import pytest

from vbg.config import Config


def test_defaults():
    c = Config()
    assert c.background == "blur"
    assert c.blur_strength % 2 == 1
    assert c.width == 1280 and c.height == 720


def test_even_blur_strength_becomes_odd():
    c = Config(blur_strength=24)
    assert c.blur_strength % 2 == 1


def test_invalid_background_rejected():
    with pytest.raises(ValueError):
        Config(background="nope")


def test_invalid_blur_rejected():
    with pytest.raises(ValueError):
        Config(blur_strength=0)


def test_roundtrip(tmp_path):
    c = Config(camera_index=2, background="color", background_color=(10, 20, 30))
    p = tmp_path / "cfg.json"
    c.save(str(p))

    loaded = Config.load(str(p))
    assert loaded.camera_index == 2
    assert loaded.background == "color"
    # JSON serializes tuples as lists
    assert tuple(loaded.background_color) == (10, 20, 30)


def test_load_missing_file_returns_defaults():
    c = Config.load("this_file_should_not_exist.json")
    assert c.camera_index == 0


def test_from_dict_ignores_unknown_keys():
    c = Config.from_dict({"camera_index": 5, "bogus_key": 1})
    assert c.camera_index == 5