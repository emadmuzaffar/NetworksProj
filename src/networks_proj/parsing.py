from __future__ import annotations

import json
from typing import Any


class StructuredResponseError(ValueError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a fenced block or brief leading prose."""

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = None
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break

    if not isinstance(value, dict):
        raise StructuredResponseError("Expected a JSON object from the provider")
    return value


def bounded_integer(value: Any, low: int, high: int, field_name: str) -> int:
    if isinstance(value, bool):
        raise StructuredResponseError(f"{field_name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise StructuredResponseError(f"{field_name} must be an integer") from exc
    if not low <= integer <= high:
        raise StructuredResponseError(
            f"{field_name} must be between {low} and {high}; received {integer}"
        )
    return integer
