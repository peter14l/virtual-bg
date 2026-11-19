"""Allow `python -m vbg`."""

from vbg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())