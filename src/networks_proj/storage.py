from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStorage:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.events_path = run_directory / "events.jsonl"
        self.state_path = run_directory / "state.json"

    @classmethod
    def create(cls, config: ExperimentConfig) -> "RunStorage":
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.study_name).strip("-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{safe_name or 'study'}-{timestamp}-{uuid.uuid4().hex[:8]}"
        run_directory = Path(config.output_directory).expanduser().resolve() / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        storage = cls(run_directory)
        storage.write_json("config.json", config.to_dict())
        return storage

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "event_type": event_type, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def checkpoint(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.state_path)

    def write_json(self, name: str, value: Any) -> None:
        path = self.run_directory / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
