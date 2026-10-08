from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig
from .experiment import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="networks-study",
        description="Run a configurable multi-agent social-relations study.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config", required=True, type=Path, help="Path to a JSON study config"
    )
    run_parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override output_directory from the config",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a config and print the expected call count"
    )
    validate_parser.add_argument("--config", required=True, type=Path)
    return parser


def load_config(path: Path, output_directory: Path | None = None) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if output_directory is not None:
        raw["output_directory"] = str(output_directory)
    return ExperimentConfig.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            args.config,
            getattr(args, "output_directory", None),
        )
        if args.action == "validate":
            print("Configuration is valid.")
            print(f"Estimated provider calls: {config.estimated_provider_calls()}")
            return 0

        runner = ExperimentRunner(config)
        print(f"Starting {config.study_name!r}.")
        print(f"Estimated provider calls: {config.estimated_provider_calls()}")
        print(f"Run directory: {runner.storage.run_directory}")
        completed_directory = runner.run()
        print(f"Experiment completed: {completed_directory}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
