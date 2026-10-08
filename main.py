"""Run this file with PyCharm's green Run button.

Choose the study file by changing CONFIG_FILE below. No command-line arguments
are required. Progress and the final data location appear in the Run console.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from networks_proj.config import ExperimentConfig  # noqa: E402
from networks_proj.experiment import ExperimentRunner  # noqa: E402


# Change this one line to select a different experiment configuration.
CONFIG_FILE = PROJECT_ROOT / "examples" / "study.json"


def show_progress(message: str) -> None:
    print(message, flush=True)


def run() -> Path:
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    output_directory = Path(raw_config.get("output_directory", "runs"))
    if not output_directory.is_absolute():
        raw_config["output_directory"] = str(PROJECT_ROOT / output_directory)
    config = ExperimentConfig.from_dict(raw_config)
    runner = ExperimentRunner(config, progress=show_progress)
    show_progress(f"Saving this run to: {runner.storage.run_directory}")
    result = runner.run()
    show_progress(f"Study data saved successfully in: {result}")
    return result


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        print("\nThe study stopped with an error:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
