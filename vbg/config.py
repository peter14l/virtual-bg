from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class Config:
    # Capture
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Background: "blur" | "image" | "color"
    background: str = "blur"
    background_image: Optional[str] = None
    background_url: Optional[str] = None   # download + use as image background
    background_color: Tuple[int, int, int] = (30, 30, 30)  # RGB
    blur_strength: int = 25          # odd int >= 1
    # Segmentation
    model_selection: int = 0         # 0 = general, 1 = landscape
    smooth: float = 0.6              # temporal mask smoothing (0 = off, 1 = max)
    guided_filter: bool = True       # edge refinement using fast guided filter
    motion_adaptive: bool = True     # adapt temporal smoothing to motion to avoid ghosting
    # Virtual camera
    vcam_name: str = "Virtual Background Cam"
    vcam_backend: Optional[str] = None   # None = auto, or "obs" | "unitycapture"
    vcam_device: Optional[str] = None    # None = any available device
    # Misc
    mirror: bool = True
    show_preview: bool = False
    preview_only: bool = False      # skip the virtual camera; process frames only
    source: str = "camera"          # "camera" | "test"
    max_frames: int = 0             # 0 = unlimited; >0 stops after N frames (testing)
    config_path: str = "config.json"

    def __post_init__(self) -> None:
        if self.background not in ("blur", "image", "color"):
            raise ValueError(f"background must be blur|image|color, got {self.background}")
        if self.blur_strength < 1:
            raise ValueError("blur_strength must be >= 1")
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1  # must be odd for GaussianBlur
        if not (0 <= len(self.background_color) == 3):
            raise ValueError("background_color must be an (R,G,B) tuple")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        allowed = {f for f in cls.__dataclass_fields__}  # noqa
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)