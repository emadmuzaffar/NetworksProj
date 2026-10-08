from __future__ import annotations

import itertools
import random
import statistics
from pathlib import Path
from typing import Any, Callable

from .config import ExperimentConfig, PersonaConfig
from .models import AgentState, GenerationRequest, Memory
from .parsing import StructuredResponseError, bounded_integer, parse_json_object
from .prompts import (
    interaction_prompt,
    memory_prompt,
    questionnaire_prompt,
    rating_prompt,
)
from .provider import AIProvider, ProviderError, build_provider
from .safety import SafetyMonitor, SafetyStop
from .storage import RunStorage


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        provider: AIProvider | None = None,
        storage: RunStorage | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.random = random.Random(config.seed)
        self.provider = provider or build_provider(config.provider, config.seed)
        self.storage = storage or RunStorage.create(config)
        self.progress = progress or (lambda _message: None)
        self.agents = self._instantiate_agents()
        self.safety = SafetyMonitor(
            config.safety,
            self.storage.run_directory,
            config.estimated_provider_calls(),
        )
        self.previous_questionnaires: dict[str, dict[str, int]] = {}
        self.questionnaire_records: list[dict[str, Any]] = []
        self.rating_records: list[dict[str, Any]] = []
        self.completed_trials = 0
        self.completed_encounters = 0

    def _instantiate_agents(self) -> list[AgentState]:
        assignments: list[PersonaConfig | None]
        if self.config.personas:
            full_cycles, remainder = divmod(
                self.config.num_agents, len(self.config.personas)
            )
            assignments = list(self.config.personas) * full_cycles
            remainder_pool = list(self.config.personas)
            self.random.shuffle(remainder_pool)
            assignments.extend(remainder_pool[:remainder])
            self.random.shuffle(assignments)
        else:
            assignments = [None] * self.config.num_agents
        return [
            AgentState(agent_id=f"agent-{index + 1:03d}", persona=assignments[index])
            for index in range(self.config.num_agents)
        ]

    def run(self) -> Path:
        self.safety.initial_check()
        self.progress(
            f"Starting {self.config.study_name}: {self.config.num_agents} agents, "
            f"{self.config.trials} trial(s), "
            f"{self.config.estimated_provider_calls()} expected provider calls."
        )
        self.storage.append_event(
            "run_started",
            {
                "study_name": self.config.study_name,
                "seed": self.config.seed,
                "estimated_provider_calls": self.config.estimated_provider_calls(),
                "agents": [
                    {"agent_id": agent.agent_id, "persona_name": agent.persona_name}
                    for agent in self.agents
                ],
            },
        )
        self._save_state(status="running")

        try:
            for trial in range(1, self.config.trials + 1):
                self._run_trial(trial)
                self.completed_trials = trial
                self._save_state(status="running")
            self.storage.append_event(
                "run_completed",
                {
                    "provider_calls": self.safety.provider_calls,
                    "completed_trials": self.completed_trials,
                    "completed_encounters": self.completed_encounters,
                },
            )
            self.storage.write_json("summary.json", self._build_summary())
            self._save_state(status="completed")
            self.progress(
                f"Completed all {self.completed_trials} trial(s) and "
                f"{self.completed_encounters} dyadic encounter(s)."
            )
        except (ProviderError, StructuredResponseError, SafetyStop, KeyboardInterrupt) as exc:
            self.storage.append_event(
                "run_stopped",
                {
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "provider_calls": self.safety.provider_calls,
                },
            )
            self._save_state(status="stopped", stop_reason=str(exc))
            raise

        return self.storage.run_directory

    def _run_trial(self, trial: int) -> None:
        self.progress(
            f"Trial {trial}/{self.config.trials}: running personality questionnaires."
        )
        self.storage.append_event("trial_started", {"trial": trial})
        trial_questionnaires: dict[str, dict[str, int]] = {}
        for agent in self.agents:
            answers, reflection = self._questionnaire(agent, trial)
            trial_questionnaires[agent.agent_id] = answers
            metrics = self._individual_questionnaire_metrics(agent, answers)
            self.questionnaire_records.append(
                {
                    "trial": trial,
                    "agent_id": agent.agent_id,
                    "persona_name": agent.persona_name,
                    "answers": answers,
                    "metrics": metrics,
                }
            )
            self.storage.append_event(
                "questionnaire",
                {
                    "trial": trial,
                    "agent_id": agent.agent_id,
                    "persona_name": agent.persona_name,
                    "answers": answers,
                    "reflection": reflection,
                    "metrics": metrics,
                },
            )
            self.previous_questionnaires[agent.agent_id] = answers

        self.storage.append_event(
            "questionnaire_trial_summary",
            {
                "trial": trial,
                "metrics": self._trial_questionnaire_metrics(trial_questionnaires),
            },
        )
        self.safety.checkpoint(f"trial_{trial}_post_questionnaire")
        self._save_state(status="running")
        self.progress(
            f"Trial {trial}/{self.config.trials}: questionnaire checkpoint passed."
        )

        pairs = list(itertools.combinations(self.agents, 2))
        for repetition in range(1, self.config.round_robin_repetitions + 1):
            scheduled_pairs = list(pairs)
            self.random.shuffle(scheduled_pairs)
            for first, second in scheduled_pairs:
                self._run_encounter(trial, repetition, first, second)
        self.storage.append_event("trial_completed", {"trial": trial})

    def _questionnaire(
        self, agent: AgentState, trial: int
    ) -> tuple[dict[str, int], str]:
        request = GenerationRequest(
            agent_id=agent.agent_id,
            system_prompt=agent.system_prompt(self.config.base_system_prompt),
            prompt=questionnaire_prompt(
                self.config.questionnaire,
                self.config.rating_scale_min,
                self.config.rating_scale_max,
            ),
            metadata={
                "task": "questionnaire",
                "trial": trial,
                "items": list(self.config.questionnaire),
                "scale_min": self.config.rating_scale_min,
                "scale_max": self.config.rating_scale_max,
            },
        )
        parsed = parse_json_object(self._generate(request))
        raw_answers = parsed.get("answers")
        if not isinstance(raw_answers, dict):
            raise StructuredResponseError("Questionnaire response is missing 'answers'")
        if set(raw_answers) != set(self.config.questionnaire):
            raise StructuredResponseError(
                "Questionnaire answers must contain every configured item exactly once"
            )
        answers = {
            item: bounded_integer(
                raw_answers[item],
                self.config.rating_scale_min,
                self.config.rating_scale_max,
                f"questionnaire answer for {item!r}",
            )
            for item in self.config.questionnaire
        }
        return answers, str(parsed.get("reflection", ""))

    def _run_encounter(
        self,
        trial: int,
        repetition: int,
        first: AgentState,
        second: AgentState,
    ) -> None:
        dyad_id = "__".join(sorted((first.agent_id, second.agent_id)))
        self.safety.checkpoint(
            f"trial_{trial}_round_{repetition}_before_{dyad_id}"
        )
        transcript: list[dict[str, Any]] = []
        pair = (first, second)
        for turn in range(self.config.turns_per_dyad):
            speaker_index = (turn + repetition - 1) % 2
            speaker = pair[speaker_index]
            listener = pair[1 - speaker_index]
            request = GenerationRequest(
                agent_id=speaker.agent_id,
                system_prompt=speaker.system_prompt(
                    self.config.base_system_prompt, listener.agent_id
                ),
                prompt=interaction_prompt(
                    self.config.interaction_topic,
                    listener.agent_id,
                    turn,
                    self.config.turns_per_dyad,
                    transcript,
                    self.config.max_transcript_messages,
                ),
                metadata={
                    "task": "interaction",
                    "trial": trial,
                    "round_robin": repetition,
                    "dyad_id": dyad_id,
                    "turn": turn + 1,
                    "partner_id": listener.agent_id,
                },
            )
            text = self._generate(request).strip()
            if not text:
                raise StructuredResponseError("Interaction response was empty")
            message = {"speaker": speaker.agent_id, "text": text}
            transcript.append(message)
            self.storage.append_event(
                "interaction_message",
                {
                    "trial": trial,
                    "round_robin": repetition,
                    "dyad_id": dyad_id,
                    "turn": turn + 1,
                    "speaker_id": speaker.agent_id,
                    "listener_id": listener.agent_id,
                    "text": text,
                },
            )

        ratings: dict[str, dict[str, Any]] = {}
        for rater, target in ((first, second), (second, first)):
            rating = self._rate(rater, target, transcript, trial, repetition, dyad_id)
            ratings[rater.agent_id] = rating
            rating_record = {
                "trial": trial,
                "round_robin": repetition,
                "dyad_id": dyad_id,
                "rater_id": rater.agent_id,
                "target_id": target.agent_id,
                **rating,
            }
            self.rating_records.append(rating_record)
            self.storage.append_event(
                "partner_rating",
                rating_record,
            )

        for rememberer, partner in ((first, second), (second, first)):
            summary = self._create_memory(
                rememberer,
                partner,
                transcript,
                ratings[rememberer.agent_id],
                trial,
                repetition,
                dyad_id,
            )
            rememberer.add_memory(
                partner.agent_id,
                Memory(trial=trial, round_robin=repetition, summary=summary),
                self.config.max_memories_per_partner,
            )
            self.storage.append_event(
                "memory_update",
                {
                    "trial": trial,
                    "round_robin": repetition,
                    "dyad_id": dyad_id,
                    "agent_id": rememberer.agent_id,
                    "partner_id": partner.agent_id,
                    "memory": summary,
                },
            )

        self.completed_encounters += 1
        pairs_per_trial = (
            self.config.num_agents
            * (self.config.num_agents - 1)
            // 2
            * self.config.round_robin_repetitions
        )
        total_encounters = pairs_per_trial * self.config.trials
        self.progress(
            f"Encounter {self.completed_encounters}/{total_encounters} complete "
            f"({dyad_id}, trial {trial}, round {repetition})."
        )
        self._save_state(status="running")

    def _rate(
        self,
        rater: AgentState,
        target: AgentState,
        transcript: list[dict[str, Any]],
        trial: int,
        repetition: int,
        dyad_id: str,
    ) -> dict[str, Any]:
        request = GenerationRequest(
            agent_id=rater.agent_id,
            system_prompt=rater.system_prompt(
                self.config.base_system_prompt, target.agent_id
            ),
            prompt=rating_prompt(
                target.agent_id,
                transcript,
                self.config.max_transcript_messages,
                self.config.rating_scale_min,
                self.config.rating_scale_max,
            ),
            metadata={
                "task": "rating",
                "trial": trial,
                "round_robin": repetition,
                "dyad_id": dyad_id,
                "partner_id": target.agent_id,
                "scale_min": self.config.rating_scale_min,
                "scale_max": self.config.rating_scale_max,
            },
        )
        parsed = parse_json_object(self._generate(request))
        return {
            "warmth": bounded_integer(
                parsed.get("warmth"),
                self.config.rating_scale_min,
                self.config.rating_scale_max,
                "warmth",
            ),
            "competence": bounded_integer(
                parsed.get("competence"),
                self.config.rating_scale_min,
                self.config.rating_scale_max,
                "competence",
            ),
            "rationale": str(parsed.get("rationale", "")),
        }

    def _create_memory(
        self,
        rememberer: AgentState,
        partner: AgentState,
        transcript: list[dict[str, Any]],
        rating: dict[str, Any],
        trial: int,
        repetition: int,
        dyad_id: str,
    ) -> str:
        request = GenerationRequest(
            agent_id=rememberer.agent_id,
            system_prompt=rememberer.system_prompt(
                self.config.base_system_prompt, partner.agent_id
            ),
            prompt=memory_prompt(
                partner.agent_id,
                transcript,
                self.config.max_transcript_messages,
                rating,
            ),
            metadata={
                "task": "memory",
                "trial": trial,
                "round_robin": repetition,
                "dyad_id": dyad_id,
                "partner_id": partner.agent_id,
            },
        )
        parsed = parse_json_object(self._generate(request))
        memory = parsed.get("memory")
        if not isinstance(memory, str) or not memory.strip():
            raise StructuredResponseError("Memory response is missing non-empty 'memory'")
        return memory.strip()

    def _generate(self, request: GenerationRequest) -> str:
        self.safety.before_provider_call()
        try:
            return self.provider.generate(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Provider raised {type(exc).__name__}: {exc}") from exc

    def _individual_questionnaire_metrics(
        self, agent: AgentState, answers: dict[str, int]
    ) -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {
            "persona_items_scored": 0,
            "persona_mean_absolute_error": None,
            "persona_accuracy": None,
            "within_agent_consistency": None,
        }
        scale_width = self.config.rating_scale_max - self.config.rating_scale_min
        if agent.persona and agent.persona.expected_questionnaire:
            differences = [
                abs(answers[item] - expected)
                for item, expected in agent.persona.expected_questionnaire.items()
            ]
            mean_error = statistics.fmean(differences)
            metrics.update(
                {
                    "persona_items_scored": len(differences),
                    "persona_mean_absolute_error": mean_error,
                    "persona_accuracy": 1.0 - mean_error / scale_width,
                }
            )
        previous = self.previous_questionnaires.get(agent.agent_id)
        if previous:
            mean_difference = statistics.fmean(
                abs(answers[item] - previous[item]) for item in self.config.questionnaire
            )
            metrics["within_agent_consistency"] = 1.0 - mean_difference / scale_width
        return metrics

    def _trial_questionnaire_metrics(
        self, questionnaires: dict[str, dict[str, int]]
    ) -> dict[str, Any]:
        per_item: dict[str, dict[str, float]] = {}
        for item in self.config.questionnaire:
            scores = [answers[item] for answers in questionnaires.values()]
            per_item[item] = {
                "mean": statistics.fmean(scores),
                "population_standard_deviation": statistics.pstdev(scores),
            }
        agent_pairs = list(itertools.combinations(questionnaires.values(), 2))
        pairwise_differences = [
            statistics.fmean(
                abs(first[item] - second[item]) for item in self.config.questionnaire
            )
            for first, second in agent_pairs
        ]
        scale_width = self.config.rating_scale_max - self.config.rating_scale_min
        mean_pairwise = statistics.fmean(pairwise_differences)
        return {
            "per_item": per_item,
            "mean_pairwise_absolute_difference": mean_pairwise,
            "across_agent_consistency": 1.0 - mean_pairwise / scale_width,
        }

    def _save_state(self, status: str, stop_reason: str | None = None) -> None:
        self.storage.checkpoint(
            {
                "status": status,
                "stop_reason": stop_reason,
                "provider_calls": self.safety.provider_calls,
                "completed_trials": self.completed_trials,
                "completed_encounters": self.completed_encounters,
                "agents": [agent.to_dict() for agent in self.agents],
            }
        )

    def _build_summary(self) -> dict[str, Any]:
        per_agent: dict[str, dict[str, Any]] = {}
        received_warmth_means: list[float] = []
        received_competence_means: list[float] = []
        for agent in self.agents:
            received = [
                record
                for record in self.rating_records
                if record["target_id"] == agent.agent_id
            ]
            given = [
                record
                for record in self.rating_records
                if record["rater_id"] == agent.agent_id
            ]
            received_warmth = statistics.fmean(
                record["warmth"] for record in received
            )
            received_competence = statistics.fmean(
                record["competence"] for record in received
            )
            received_warmth_means.append(received_warmth)
            received_competence_means.append(received_competence)
            per_agent[agent.agent_id] = {
                "persona_name": agent.persona_name,
                "mean_received_warmth": received_warmth,
                "mean_received_competence": received_competence,
                "mean_given_warmth": statistics.fmean(
                    record["warmth"] for record in given
                ),
                "mean_given_competence": statistics.fmean(
                    record["competence"] for record in given
                ),
                "stored_memories": sum(
                    len(memories) for memories in agent.memories_by_partner.values()
                ),
            }

        trajectories: dict[str, list[dict[str, Any]]] = {}
        for record in self.rating_records:
            directed_pair = f"{record['rater_id']}->{record['target_id']}"
            trajectories.setdefault(directed_pair, []).append(
                {
                    "trial": record["trial"],
                    "round_robin": record["round_robin"],
                    "warmth": record["warmth"],
                    "competence": record["competence"],
                }
            )

        final_questionnaires = {
            record["agent_id"]: record
            for record in self.questionnaire_records
            if record["trial"] == self.config.trials
        }
        return {
            "study_name": self.config.study_name,
            "seed": self.config.seed,
            "provider_calls": self.safety.provider_calls,
            "completed_trials": self.completed_trials,
            "completed_encounters": self.completed_encounters,
            "differentiation": {
                "standard_deviation_of_mean_received_warmth": statistics.pstdev(
                    received_warmth_means
                ),
                "standard_deviation_of_mean_received_competence": statistics.pstdev(
                    received_competence_means
                ),
                "final_questionnaire": self._trial_questionnaire_metrics(
                    {
                        agent_id: record["answers"]
                        for agent_id, record in final_questionnaires.items()
                    }
                ),
            },
            "per_agent": per_agent,
            "directed_relationship_trajectories": trajectories,
        }
