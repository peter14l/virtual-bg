"""Provision the Windows virtual-camera driver (Unity Capture DirectShow filter).

pyvirtualcam needs a DirectShow capture device on Windows. This module downloads
the MIT-licensed Unity Capture filter (https://github.com/schellingb/UnityCapture),
installs it into ``third_party/UnityCapture`` and registers it with ``regsvr32``
(elevated). After registration the device shows up as "Unity Video Capture".

NOTE: Unity Capture works with ffmpeg-based apps but NOT browsers -- its filter
ignores ``SetFormat(RGB24)``, so Chrome/Meet/Zoom see a black frame. For browsers
install OBS Studio, which registers "OBS Virtual Camera" (pyvirtualcam's preferred
backend, tried first by auto-detect). Run ``python run.py --backend obs`` to force it.

Usage:
    python run.py --install     # download + register (asks for admin)
    python run.py --uninstall   # unregister
    python run.py --check       # report which backends are available
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent / "third_party" / "UnityCapture"

_BASE_URL = "https://raw.githubusercontent.com/schellingb/UnityCapture/master/Install"
FILTER_DLL_URLS = {
    "UnityCaptureFilter32.dll": f"{_BASE_URL}/UnityCaptureFilter32.dll",
    "UnityCaptureFilter64.dll": f"{_BASE_URL}/UnityCaptureFilter64.dll",
}

DEVICE_NAME = "Unity Video Capture"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _elevate_and_rerun(flag: str) -> int:
    """Re-launch this program with the given flag under UAC elevation."""
    exe = sys.executable
    script = f'"{Path(sys.argv[0]).resolve()}"'
    params = f"{script} {flag}"
    try:
        ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, params, None, 1)
    except Exception as e:  # pragma: no cover
        print(f"Could not request elevation: {e}")
        return 1
    return 0 if int(ret) > 32 else 1


def download_filters(dest: Path = THIRD_PARTY_DIR) -> list[Path]:
    """Download the Unity Capture filter DLLs into ``dest`` if missing."""
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in FILTER_DLL_URLS.items():
        target = dest / name
        if not (target.exists() and target.stat().st_size > 0):
            print(f"Downloading {name} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "virtual-bg"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        paths.append(target)
    return paths


def _regsvr32(dll: Path, unregister: bool = False) -> bool:
    args = ["regsvr32", "/s"]
    if unregister:
        args.append("/u")
    args.append(str(dll))
    try:
        subprocess.run(args, check=True)
        return True
    except Exception as e:
        print(f"regsvr32 failed for {dll.name}: {e}")
        return False


def install() -> int:
    """Download and register the Unity Capture filter. Returns a process exit code."""
    if not _is_windows():
        print("Automatic install is only supported on Windows.")
        print("On Linux: install v4l2loopback. On macOS: install OBS Studio.")
        return 1

    if not _is_admin():
        print("Administrator rights are required to register the virtual-camera driver.")
        print("A UAC prompt will appear; approve it to continue.")
        return _elevate_and_rerun("--install")

    try:
        dlls = download_filters()
    except Exception as e:
        print(f"Download failed: {e}")
        print("Check your internet connection and try again.")
        return 1

    ok = True
    for dll in dlls:
        ok = _regsvr32(dll) and ok

    if ok:
        print(f'Installed. Select "{DEVICE_NAME}" as the camera in Zoom/Teams/browsers.')
        return 0
    print("Registration failed. Try running this command as administrator.")
    return 1


def uninstall() -> int:
    """Unregister the Unity Capture filter. Returns a process exit code."""
    if not _is_windows():
        print("Automatic uninstall is only supported on Windows.")
        return 1

    if not _is_admin():
        print("Administrator rights are required to unregister the driver.")
        return _elevate_and_rerun("--uninstall")

    ok = True
    for dll in THIRD_PARTY_DIR.glob("UnityCaptureFilter*.dll"):
        ok = _regsvr32(dll, unregister=True) and ok
    if ok:
        print("Uninstalled.")
        return 0
    return 1


def check() -> int:
    """Report which pyvirtualcam backends are usable right now."""
    try:
        import pyvirtualcam
    except Exception as e:
        print(f"pyvirtualcam is not installed: {e}")
        return 1

    print("Probing available virtual-camera backends ...")
    any_ok = False
    for name in ("obs", "unitycapture"):
        try:
            cam = pyvirtualcam.Camera(2, 2, 5, backend=name)
            cam.close()
            print(f"  [OK]   {name}")
            any_ok = True
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    if any_ok:
        print("A virtual camera is ready.")
        print('  * Browsers / Zoom / Teams: use "OBS Virtual Camera" (recommended).')
        print('  * ffmpeg-based apps: "Unity Video Capture" also works.')
        return 0
    print("No backend available. Run 'python run.py --install' (Windows) or install OBS Studio.")
    return 1