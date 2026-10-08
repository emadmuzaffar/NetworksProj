from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONNAIRE = [
    "I see myself as someone who is talkative.",
    "I see myself as someone who tends to find fault with others.",
    "I see myself as someone who does a thorough job.",
    "I see myself as someone who is relaxed and handles stress well.",
    "I see myself as someone who has an active imagination.",
]


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    prompt: str
    expected_questionnaire: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaConfig":
        return cls(
            name=str(data["name"]),
            prompt=str(data.get("prompt", "")),
            expected_questionnaire={
                str(key): int(value)
                for key, value in data.get("expected_questionnaire", {}).items()
            },
        )


@dataclass(frozen=True)
class ProviderConfig:
    type: str = "deterministic"
    command: tuple[str, ...] = ()
    timeout_seconds: float = 120.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        return cls(
            type=str(data.get("type", "deterministic")),
            command=tuple(str(part) for part in data.get("command", [])),
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
        )


@dataclass(frozen=True)
class SafetyConfig:
    max_provider_calls: int = 10_000
    stop_file_name: str = "STOP"
    checker_command: tuple[str, ...] = ()
    checker_timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafetyConfig":
        return cls(
            max_provider_calls=int(data.get("max_provider_calls", 10_000)),
            stop_file_name=str(data.get("stop_file_name", "STOP")),
            checker_command=tuple(str(part) for part in data.get("checker_command", [])),
            checker_timeout_seconds=float(data.get("checker_timeout_seconds", 30.0)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    study_name: str = "persona-differentiation"
    num_agents: int = 4
    trials: int = 1
    round_robin_repetitions: int = 5
    turns_per_dyad: int = 10
    seed: int = 252
    output_directory: str = "runs"
    interaction_topic: str = (
        "Discuss how a group should allocate limited resources fairly."
    )
    base_system_prompt: str = (
        "You are a participant in a social-relations study. Stay in character, "
        "respond naturally, and do not claim to be another participant."
    )
    rating_scale_min: int = 1
    rating_scale_max: int = 5
    max_transcript_messages: int = 20
    max_memories_per_partner: int = 25
    questionnaire: tuple[str, ...] = tuple(DEFAULT_QUESTIONNAIRE)
    personas: tuple[PersonaConfig, ...] = ()
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        known = {
            "study_name",
            "num_agents",
            "trials",
            "round_robin_repetitions",
            "turns_per_dyad",
            "seed",
            "output_directory",
            "interaction_topic",
            "base_system_prompt",
            "rating_scale_min",
            "rating_scale_max",
            "max_transcript_messages",
            "max_memories_per_partner",
        }
        kwargs = {key: data[key] for key in known if key in data}
        if "questionnaire" in data:
            kwargs["questionnaire"] = tuple(str(item) for item in data["questionnaire"])
        kwargs["personas"] = tuple(
            PersonaConfig.from_dict(item) for item in data.get("personas", [])
        )
        kwargs["provider"] = ProviderConfig.from_dict(data.get("provider", {}))
        kwargs["safety"] = SafetyConfig.from_dict(data.get("safety", {}))
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provider"]["command"] = list(self.provider.command)
        data["safety"]["checker_command"] = list(self.safety.checker_command)
        return data

    def validate(self) -> None:
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        for name in ("trials", "round_robin_repetitions", "turns_per_dyad"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not self.questionnaire:
            raise ValueError("questionnaire must contain at least one item")
        if self.rating_scale_min >= self.rating_scale_max:
            raise ValueError("rating_scale_min must be less than rating_scale_max")
        if self.max_transcript_messages < 1 or self.max_memories_per_partner < 1:
            raise ValueError("transcript and memory limits must be at least 1")
        if self.provider.type not in {"deterministic", "command"}:
            raise ValueError("provider.type must be 'deterministic' or 'command'")
        if self.provider.type == "command" and not self.provider.command:
            raise ValueError("provider.command is required for a command provider")
        for persona in self.personas:
            for item, score in persona.expected_questionnaire.items():
                if item not in self.questionnaire:
                    raise ValueError(
                        f"Persona {persona.name!r} has an unknown questionnaire item: {item!r}"
                    )
                if not self.rating_scale_min <= score <= self.rating_scale_max:
                    raise ValueError(
                        f"Expected questionnaire score for {persona.name!r} is out of range"
                    )

    def estimated_provider_calls(self) -> int:
        pairs = self.num_agents * (self.num_agents - 1) // 2
        questionnaire_calls = self.num_agents * self.trials
        per_encounter = self.turns_per_dyad + 4  # two ratings and two memories
        interaction_calls = (
            pairs * self.round_robin_repetitions * per_encounter * self.trials
        )
        return questionnaire_calls + interaction_calls
