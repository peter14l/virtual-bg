"""Command-line interface for virtual-bg."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from vbg.config import Config
from vbg.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="virtual-bg",
        description=(
            "Apply ML background removal and virtual backgrounds to any video-call "
            "app via a virtual camera (no green screen required)."
        ),
    )
    p.add_argument("--camera", type=int, default=0, help="Physical webcam index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    p.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    p.add_argument(
        "--background",
        choices=["blur", "image", "color"],
        default="blur",
        help="Background mode (default: blur)",
    )
    p.add_argument("--image", default=None, help="Background image path (for --background image)")
    p.add_argument(
        "--background-url",
        default=None,
        help="Download an image from a URL and use it as the background",
    )
    p.add_argument(
        "--download-background",
        default=None,
        help="Download a background image to assets/backgrounds and exit",
    )
    p.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=None,
        help="Background color as R G B (for --background color)",
    )
    p.add_argument("--blur", type=int, default=25, help="Blur strength for blur mode (default: 25)")
    p.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help="Segmentation model: 0 = general, 1 = landscape (default: 0)",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="Temporal mask smoothing 0..1 (higher = less flicker, more lag; default: 0.6)",
    )
    p.add_argument("--name", default="Virtual Background Cam", help="Virtual camera display name")
    p.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirroring")
    p.add_argument("--preview", action="store_true", help="Show a local preview window")
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Process frames but do NOT publish to a virtual camera (no driver needed)",
    )
    p.add_argument(
        "--source",
        choices=["camera", "test"],
        default="camera",
        help="Frame source: 'camera' (physical webcam) or 'test' (synthetic pattern)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until quit); useful for testing",
    )
    p.add_argument("--config", default="config.json", help="Config file path (default: config.json)")
    p.add_argument(
        "--save-config",
        action="store_true",
        help="Save the current settings to the config file and exit",
    )
    p.add_argument(
        "--backend",
        choices=["obs", "unitycapture"],
        default=None,
        help="Force a virtual-camera backend (default: auto-detect)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force a specific virtual-camera device name (default: any available)",
    )
    p.add_argument("--install", action="store_true", help="Install the bundled virtual-camera driver and exit")
    p.add_argument("--uninstall", action="store_true", help="Uninstall the virtual-camera driver and exit")
    p.add_argument("--check", action="store_true", help="Check which virtual-camera backends are available and exit")
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup / diagnostic commands short-circuit before any capture work.
    if args.install:
        from vbg import setup

        return setup.install()
    if args.uninstall:
        from vbg import setup

        return setup.uninstall()
    if args.check:
        from vbg import setup

        return setup.check()

    # Download-only mode: fetch a background image and exit.
    if args.download_background:
        from vbg.backgrounds import download_image

        try:
            path = download_image(args.download_background)
        except Exception as e:
            print(f"error: could not download background: {e}", file=sys.stderr)
            return 1
        print(f"Saved background to {path}")
        return 0

    cfg = Config.load(args.config)
    cfg.camera_index = args.camera
    cfg.width = args.width
    cfg.height = args.height
    cfg.fps = args.fps
    cfg.background = args.background
    cfg.background_image = args.image
    cfg.background_url = args.background_url
    cfg.blur_strength = args.blur
    cfg.model_selection = args.model
    cfg.smooth = args.smooth
    cfg.vcam_name = args.name
    cfg.mirror = not args.no_mirror
    cfg.show_preview = args.preview
    cfg.preview_only = args.preview_only
    cfg.source = args.source
    cfg.max_frames = args.max_frames
    cfg.vcam_backend = args.backend
    cfg.vcam_device = args.device
    cfg.config_path = args.config
    if args.color is not None:
        cfg.background_color = (args.color[0], args.color[1], args.color[2])

    # Resolve an internet background: --background-url implies image mode.
    if cfg.background_url:
        from vbg.backgrounds import download_image

        cfg.background = "image"
        if not cfg.background_image:
            try:
                cfg.background_image = str(download_image(cfg.background_url))
            except Exception as e:
                print(f"error: could not download background: {e}", file=sys.stderr)
                return 1

    if cfg.background == "image" and not cfg.background_image:
        print(
            "error: --background image requires --image <path> or --background-url <url>",
            file=sys.stderr,
        )
        return 1

    if args.save_config:
        cfg.save(args.config)
        print(f"Saved configuration to {args.config}")
        return 0

    try:
        Pipeline(cfg).run()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0