from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Protocol

from .config import ProviderConfig
from .models import GenerationRequest


class ProviderError(RuntimeError):
    """Raised when an AI provider cannot return a usable response."""


class AIProvider(Protocol):
    def generate(self, request: GenerationRequest) -> str:
        """Generate one response for one logical agent."""


class CommandProvider:
    """Adapter for any executable that accepts one JSON request on stdin.

    The executable may print plain text, or a JSON object containing a `text`,
    `response`, or `content` string. A new process is used for each call so the
    study owns all persistent agent state rather than relying on CLI history.
    """

    def __init__(self, command: tuple[str, ...], timeout_seconds: float) -> None:
        if not command:
            raise ValueError("A command provider requires a non-empty command")
        self.command = command
        self.timeout_seconds = timeout_seconds

    def generate(self, request: GenerationRequest) -> str:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request.to_dict()),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"Provider command failed to run: {exc}") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise ProviderError(
                f"Provider command exited with {completed.returncode}: {stderr[:1000]}"
            )
        output = completed.stdout.strip()
        if not output:
            raise ProviderError("Provider command returned an empty response")
        try:
            envelope = json.loads(output)
        except json.JSONDecodeError:
            return output
        if isinstance(envelope, dict):
            for key in ("text", "response", "content"):
                if isinstance(envelope.get(key), str):
                    return envelope[key]
        return output


class DeterministicProvider:
    """Offline provider for validating a protocol without using model calls."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.call_count = 0

    def _score(self, request: GenerationRequest, label: str, low: int, high: int) -> int:
        material = (
            f"{self.seed}|{self.call_count}|{request.agent_id}|{label}|"
            f"{request.metadata.get('trial')}|{request.metadata.get('partner_id')}"
        )
        value = int(hashlib.sha256(material.encode()).hexdigest()[:12], 16)
        return low + value % (high - low + 1)

    def generate(self, request: GenerationRequest) -> str:
        self.call_count += 1
        task = request.metadata.get("task")
        low = int(request.metadata.get("scale_min", 1))
        high = int(request.metadata.get("scale_max", 5))

        if task == "questionnaire":
            answers = {
                item: self._score(request, item, low, high)
                for item in request.metadata["items"]
            }
            return json.dumps(
                {"answers": answers, "reflection": "Offline deterministic response."}
            )
        if task == "rating":
            return json.dumps(
                {
                    "warmth": self._score(request, "warmth", low, high),
                    "competence": self._score(request, "competence", low, high),
                    "rationale": "Deterministic test rating based on the interaction.",
                }
            )
        if task == "memory":
            partner = request.metadata["partner_id"]
            return json.dumps(
                {
                    "memory": (
                        f"I interacted with {partner}; they contributed ideas and I should "
                        "build on our previous exchange next time."
                    )
                }
            )
        if task == "interaction":
            partner = request.metadata["partner_id"]
            turn = request.metadata["turn"]
            return (
                f"From {request.agent_id} to {partner} on turn {turn}: "
                "I would compare the options, explain my priorities, and invite your view."
            )
        raise ProviderError(f"Unknown deterministic task: {task!r}")


def build_provider(config: ProviderConfig, seed: int) -> AIProvider:
    if config.type == "deterministic":
        return DeterministicProvider(seed)
    if config.type == "command":
        return CommandProvider(config.command, config.timeout_seconds)
    raise ValueError(f"Unsupported provider type: {config.type}")
