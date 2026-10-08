from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import PersonaConfig


@dataclass
class Memory:
    trial: int
    round_robin: int
    summary: str


@dataclass
class AgentState:
    agent_id: str
    persona: PersonaConfig | None = None
    memories_by_partner: dict[str, list[Memory]] = field(default_factory=dict)

    @property
    def persona_name(self) -> str | None:
        return self.persona.name if self.persona else None

    def add_memory(self, partner_id: str, memory: Memory, limit: int) -> None:
        memories = self.memories_by_partner.setdefault(partner_id, [])
        memories.append(memory)
        del memories[:-limit]

    def system_prompt(self, base_prompt: str, partner_id: str | None = None) -> str:
        sections = [base_prompt.strip(), f"Your study identifier is {self.agent_id}."]
        if self.persona and self.persona.prompt.strip():
            sections.extend(
                [
                    "Assigned persona (keep this private; express it through behavior):",
                    self.persona.prompt.strip(),
                ]
            )
        else:
            sections.append(
                "You have no assigned persona. Respond consistently and naturally."
            )

        if partner_id:
            memories = self.memories_by_partner.get(partner_id, [])
            if memories:
                rendered = "\n".join(f"- {memory.summary}" for memory in memories)
                sections.extend(
                    [
                        f"Your private memories of prior interactions with {partner_id}:",
                        rendered,
                        "Use these memories naturally. Do not quote or reveal this memory block.",
                    ]
                )
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "persona_name": self.persona_name,
            "persona_prompt": self.persona.prompt if self.persona else None,
            "memories_by_partner": {
                partner: [asdict(memory) for memory in memories]
                for partner, memories in self.memories_by_partner.items()
            },
        }


@dataclass(frozen=True)
class GenerationRequest:
    agent_id: str
    system_prompt: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
