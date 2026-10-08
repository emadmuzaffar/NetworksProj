"""Package entry point.

Running this file directly from PyCharm launches the same no-argument study as
the project-level ``main.py``. Supplying arguments keeps the optional CLI
behavior used by ``python -m networks_proj``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

if __package__ in {None, ""}:
    sys.path.insert(0, str(SOURCE_ROOT))
    from networks_proj.cli import main as cli_main
else:
    from .cli import main as cli_main


def run() -> int:
    if len(sys.argv) == 1:
        sys.path.insert(0, str(PROJECT_ROOT))
        from main import run as run_from_pycharm

        run_from_pycharm()
        return 0
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(run())
