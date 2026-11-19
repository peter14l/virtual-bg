import numpy as np
import pytest

pytest.importorskip("cv2")

from vbg.config import Config  # noqa: E402
from vbg.pipeline import Pipeline  # noqa: E402


def test_pipeline_constructs_and_stops_safely():
    p = Pipeline(Config())
    p.stop()  # must not raise even before run()


def test_cli_build_parser_defaults():
    from vbg.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.background == "blur"
    assert args.width == 1280
    assert args.camera == 0


def test_cli_color_parsing():
    from vbg.cli import build_parser

    args = build_parser().parse_args(["--background", "color", "--color", "10", "20", "30"])
    assert args.color == [10, 20, 30]


def test_cli_backend_and_device():
    from vbg.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "unitycapture", "--device", "Unity Video Capture"]
    )
    assert args.backend == "unitycapture"
    assert args.device == "Unity Video Capture"


def test_cli_setup_flags():
    from vbg.cli import build_parser

    assert build_parser().parse_args(["--install"]).install is True
    assert build_parser().parse_args(["--uninstall"]).uninstall is True
    assert build_parser().parse_args(["--check"]).check is True