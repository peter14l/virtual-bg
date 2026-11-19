# Architecture — vbg (Virtual Background)

This document defines the **frozen interface contract** for modules in this project. All implementations must conform to these signatures and data conventions.

## Data conventions

- **Frames** crossing module boundaries are **BGR uint8** `numpy.ndarray` of shape `(H, W, 3)`.
- **Masks** are **float32** `numpy.ndarray` of shape `(H, W)` with values in `[0, 1]`.
- **`Config.background_color`** is stored as **RGB** `(R, G, B)`. Any module that needs BGR (e.g. OpenCV drawing/compositing) **must convert** from RGB to BGR.

## Module contracts

### `vbg/capture.py`

```python
class WebcamCapture:
    def __init__(self, index: int, width: int, height: int, fps: int): ...
    def start(self) -> None: ...
    def read(self) -> Optional[np.ndarray]: ...
        # Returns BGR uint8 HxWx3 frame, or None if unavailable
    def release(self) -> None: ...
```

### `vbg/segment.py`

```python
class PersonSegmenter:
    def __init__(self, model_selection: int = 0, smooth: float = 0.6): ...
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray: ...
        # Returns float32 mask HxW in range [0, 1]; hole-filled and temporally
        # smoothed (EMA) to suppress flicker
    def close(self) -> None: ...
```

### `vbg/compositor.py`

```python
class Compositor:
    def __init__(self, config: Config): ...
    def set_background(
        self,
        kind: str,
        image_path: Optional[str] = None,
        color: Tuple[int, int, int] = (30, 30, 30),
    ) -> None: ...
    def compose(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray: ...
        # Returns BGR uint8 HxWx3 composited frame
```

### `vbg/virtual_cam.py`

```python
class VirtualCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        name: str,
        backend: Optional[str] = None,   # None = auto, or "obs" | "unitycapture"
        device: Optional[str] = None,    # None = any available device
    ): ...
    def start(self) -> None: ...
    def send(self, frame_bgr: np.ndarray) -> None: ...
    def stop(self) -> None: ...
```

`VirtualCamera` is a thin wrapper over `pyvirtualcam`. On Windows it needs a
DirectShow backend; `vbg/setup.py` provisions the bundled Unity Capture filter so
no external OBS install is required.

### `vbg/setup.py`

```python
def download_filters(dest: Path = THIRD_PARTY_DIR) -> list[Path]: ...
def install() -> int: ...      # download + register (elevated); process exit code
def uninstall() -> int: ...    # unregister; process exit code
def check() -> int: ...        # probe available backends; process exit code
```

### `vbg/backgrounds.py`

```python
def download_image(url: str, dest_dir: Path = ASSETS_DIR, filename: str | None = None) -> Path: ...
    # Download an image to dest_dir (reuses an existing file) and return its path.
def list_backgrounds() -> list[Path]: ...
    # Locally available background images under assets/backgrounds.
```

### `vbg/testpattern.py`

```python
class TestPatternCapture:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30): ...
    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def release(self) -> None: ...
```

Synthetic frame source used for demos and CI smoke tests; mirrors the
`WebcamCapture` interface.

### `vbg/pipeline.py`

```python
class Pipeline:
    def __init__(self, config: Config): ...
    def run(self) -> None: ...
        # Blocking main loop
    def stop(self) -> None: ...
```

### `vbg/cli.py`

```python
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Optional[list] = None) -> int: ...
```

### `run.py` (repository root)

```python
# Calls vbg.cli.main()
```

## Notes

- Modules must not change the above public signatures.
- `Config` is defined in `vbg/config.py` and is the shared configuration object passed across modules where applicable.
- Implementations should be defensive about input shapes/dtypes but must honor the stated contracts at module boundaries.