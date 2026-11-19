# vbg — Virtual Background

Applies ML background removal and virtual backgrounds to ANY video-calling app
(Zoom, Teams, WhatsApp Desktop, Instagram Web, browsers, …) via a virtual
camera — **no green screen required**. Person segmentation runs locally with
MediaPipe; the composited result is published to a virtual webcam that other
apps can select.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On first run the Selfie Segmenter model (~250 KB) is downloaded automatically
into `models/`.

## Set up the virtual camera (Windows, one time)

`pyvirtualcam` publishes frames through a DirectShow capture device. There are two
backends; pick based on what you need.

### Recommended: OBS Virtual Camera (works with browsers)

Chrome, Edge, Meet, Zoom and Teams use MediaFoundation, which needs a camera that
negotiates RGB24/I420 properly. **OBS Virtual Camera** does this; the bundled Unity
Capture filter does not (browsers see black with it).

1. Install OBS Studio: `winget install OBSProject.OBSStudio` (or from
   https://obsproject.com).
2. Run the app — it auto-detects "OBS Virtual Camera" (OBS need not be running):

```bash
python run.py --check       # should show [OK] obs
```

### Lightweight fallback: Unity Capture (ffmpeg-based apps only)

If you only need ffmpeg-based consumers (feeding OBS itself, custom pipelines) and
want to avoid the OBS download, use the bundled MIT-licensed Unity Capture filter:

```bash
python run.py --install
```

This downloads the filter into `third_party/UnityCapture/` and registers it (UAC
prompt — approve it). The device **"Unity Video Capture"** appears as a selectable
camera, but **browsers will show black** with it.

Other commands:

```bash
python run.py --check       # which virtual-camera backends are usable right now
python run.py --uninstall   # unregister the Unity Capture driver
```

On **Linux**, install `v4l2loopback`. On **macOS**, install OBS Studio.

## Usage

Select **"OBS Virtual Camera"** inside Zoom/Teams/your browser, then run:

```bash
python run.py --camera 0 --background blur
```

Modes:

```bash
python run.py --background blur                       # blur the room behind you
python run.py --background image --image bg.png        # use a local image
python run.py --background color --color 20 30 40      # solid color (R G B)
```

Backgrounds straight from the internet:

```bash
# Download an image and use it immediately as the background
python run.py --background image --background-url https://example.com/beach.jpg

# Or just download it into assets/backgrounds for later reuse
python run.py --download-background https://example.com/beach.jpg
```

Blur strength (resolution-independent; higher = blurrier, default 25):

```bash
python run.py --background blur --blur 60
```

Reduce background "spots" / edge flicker (temporal mask smoothing, default 0.6; 0 = off):

```bash
python run.py --smooth 0.7
```

Force a specific backend/device:

```bash
python run.py --backend obs --device "OBS Virtual Camera"   # browsers / Zoom / Teams
python run.py --backend unitycapture                        # ffmpeg-based apps only
```

Preview and testing without a virtual-camera driver:

```bash
python run.py --preview                               # local preview window
python run.py --preview-only --source test            # no camera/driver needed
python run.py --source test --preview-only --max-frames 30   # CI smoke test
```

Persist settings:

```bash
python run.py --background color --color 0 32 96 --save-config
```

Run `python run.py --help` for all options.

## Troubleshooting

### The camera "doesn't turn on" in my browser or another app
While vbg is running it holds the **physical** webcam, and most Windows webcams
can only be opened by one app at a time. Other apps must use the **virtual**
camera instead:

1. Confirm the driver is installed: `python run.py --check`
2. In Zoom / Teams / your browser, select **"OBS Virtual Camera"** (recommended) as
   the camera — **not** your physical webcam. With only Unity Capture installed,
   browsers show black; use OBS instead.
3. Keep vbg running: it feeds the virtual camera. When vbg exits, the virtual
   camera stops and the physical camera is released.
4. If the virtual camera doesn't appear in the list, fully restart the
   browser/app after installing the driver.

## How it works

```
WebcamCapture → PersonSegmenter (MediaPipe) → Compositor → VirtualCamera
   (BGR)           (float32 mask)              (BGR)         (DirectShow/v4l2)
```

- `capture` grabs BGR frames from the webcam (or a synthetic test pattern).
- `segment` produces a per-pixel person mask in `[0, 1]`, with hole-filling and
  temporal (EMA) smoothing to prevent the background poking through as spots.
- `compositor` blends the frame over a blurred / image / solid-color background.
  The blur is resolution-independent (sigma scales with frame size), so it looks
  the same at 480p, 720p or 1080p.
- `virtual_cam` publishes the result via pyvirtualcam to "OBS Virtual Camera"
  (recommended; works with browsers) or "Unity Video Capture" (ffmpeg-based apps).

## Project structure

```
.
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── run.py
├── models/               # auto-downloaded TFLite model (gitignored)
├── third_party/          # bundled Unity Capture driver (gitignored, created by --install)
├── assets/backgrounds/
├── tests/
└── vbg/
    ├── __init__.py
    ├── __main__.py       # python -m vbg
    ├── config.py         # Config dataclass + JSON load/save
    ├── capture.py        # WebcamCapture
    ├── testpattern.py    # synthetic source for tests/demos
    ├── segment.py        # PersonSegmenter (MediaPipe Tasks API)
    ├── compositor.py     # Compositor
    ├── virtual_cam.py    # VirtualCamera (pyvirtualcam)
    ├── setup.py          # install/uninstall/check the virtual-camera driver
    ├── backgrounds.py    # download background images from URLs
    ├── pipeline.py       # Pipeline (capture→segment→composite→vcam)
    └── cli.py            # CLI entry point
```

## Tests

```bash
python -m pytest -q
```

## Status

Working reference implementation: capture, segmentation, compositing, CLI,
virtual-camera publishing, and a self-contained Windows driver installer are
implemented. Next steps: a signed installer/executable, native low-memory core,
and macOS/Linux packaging. See `ARCHITECTURE.md`.

## Support & Donations

If you find `vbg` useful and want to support its maintenance and ongoing development, donations are gratefully accepted:

- **Ko-fi**: [ko-fi.com/peter20](https://ko-fi.com/peter20)
- **UPI (India)**: `9831060419@fam`
