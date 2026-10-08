from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from networks_proj.config import ExperimentConfig
from networks_proj.experiment import ExperimentRunner


class ExperimentRunnerTests(unittest.TestCase):
    def test_small_protocol_saves_every_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig.from_dict(
                {
                    "study_name": "test-study",
                    "num_agents": 2,
                    "trials": 1,
                    "round_robin_repetitions": 2,
                    "turns_per_dyad": 2,
                    "seed": 7,
                    "output_directory": temporary,
                    "personas": [],
                    "provider": {"type": "deterministic"},
                    "safety": {"max_provider_calls": 20},
                }
            )
            runner = ExperimentRunner(config)
            run_directory = runner.run()

            events = [
                json.loads(line)
                for line in (run_directory / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            counts = {
                event_type: sum(
                    event["event_type"] == event_type for event in events
                )
                for event_type in {
                    "questionnaire",
                    "interaction_message",
                    "partner_rating",
                    "memory_update",
                }
            }
            self.assertEqual(counts["questionnaire"], 2)
            self.assertEqual(counts["interaction_message"], 4)
            self.assertEqual(counts["partner_rating"], 4)
            self.assertEqual(counts["memory_update"], 4)

            state = json.loads((run_directory / "state.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["provider_calls"], 14)
            self.assertEqual(state["completed_encounters"], 2)
            for agent in state["agents"]:
                memories = list(agent["memories_by_partner"].values())
                self.assertEqual(len(memories), 1)
                self.assertEqual(len(memories[0]), 2)

            summary = json.loads((run_directory / "summary.json").read_text())
            self.assertIn("differentiation", summary)
            self.assertEqual(len(summary["directed_relationship_trajectories"]), 2)

    def test_persona_assignment_is_seeded_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = {
                "num_agents": 5,
                "output_directory": temporary,
                "personas": [
                    {"name": "a", "prompt": "Persona A"},
                    {"name": "b", "prompt": "Persona B"},
                ],
            }
            first = ExperimentRunner(ExperimentConfig.from_dict(raw))
            second = ExperimentRunner(ExperimentConfig.from_dict(raw))
            first_names = [agent.persona_name for agent in first.agents]
            second_names = [agent.persona_name for agent in second.agents]
            self.assertEqual(first_names, second_names)
            self.assertEqual(
                sorted((first_names.count("a"), first_names.count("b"))), [2, 3]
            )


if __name__ == "__main__":
    unittest.main()
