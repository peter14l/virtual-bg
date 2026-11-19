"""Download background images from the internet and manage the local library."""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "backgrounds"


def _filename_from_url(url: str) -> str:
    """Derive a safe local filename from a URL path."""
    path = urllib.parse.urlparse(url).path
    name = "background" if path.endswith("/") else Path(path).name
    if not name:
        name = "background"
    if "." not in name:
        name += ".img"
    return name


def download_image(url: str, dest_dir: Path = ASSETS_DIR, filename: str | None = None) -> Path:
    """Download an image to ``dest_dir`` and return the local path.

    If a file with the same name already exists and is non-empty, it is reused
    (no re-download).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (filename or _filename_from_url(url))

    if target.exists() and target.stat().st_size > 0:
        return target

    req = urllib.request.Request(url, headers={"User-Agent": "virtual-bg"})
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return target


def list_backgrounds() -> list[Path]:
    """Return the locally available background images."""
    if not ASSETS_DIR.exists():
        return []
    return sorted(
        p for p in ASSETS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )