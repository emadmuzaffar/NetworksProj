from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import SafetyConfig


class SafetyStop(RuntimeError):
    """Raised when a runtime safety condition stops the experiment."""


@dataclass
class SafetyMonitor:
    config: SafetyConfig
    run_directory: Path
    estimated_calls: int
    provider_calls: int = 0

    def initial_check(self) -> None:
        if self.estimated_calls > self.config.max_provider_calls:
            raise SafetyStop(
                f"Experiment estimates {self.estimated_calls} provider calls, above the "
                f"configured maximum of {self.config.max_provider_calls}"
            )
        if not self.run_directory.is_dir():
            raise SafetyStop("Run directory does not exist")
        self.checkpoint("initialized")

    def checkpoint(self, stage: str) -> None:
        if (self.run_directory / self.config.stop_file_name).exists():
            raise SafetyStop(
                f"Manual stop file {self.config.stop_file_name!r} was detected"
            )
        if self.provider_calls > self.config.max_provider_calls:
            raise SafetyStop("Provider-call limit was exceeded")
        if self.config.checker_command:
            self._run_external_checker(stage)

    def before_provider_call(self) -> None:
        self.checkpoint("before_provider_call")
        if self.provider_calls + 1 > self.config.max_provider_calls:
            raise SafetyStop("The next call would exceed the provider-call limit")
        self.provider_calls += 1

    def _run_external_checker(self, stage: str) -> None:
        request = {
            "stage": stage,
            "run_directory": str(self.run_directory),
            "provider_calls": self.provider_calls,
            "estimated_provider_calls": self.estimated_calls,
        }
        try:
            completed = subprocess.run(
                self.config.checker_command,
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=self.config.checker_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SafetyStop(f"External safety checker failed: {exc}") from exc
        if completed.returncode != 0:
            raise SafetyStop(
                f"External safety checker exited with {completed.returncode}: "
                f"{completed.stderr.strip()[:1000]}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SafetyStop("External safety checker did not return JSON") from exc
        if response.get("allowed") is not True:
            raise SafetyStop(str(response.get("reason", "External safety check denied run")))
