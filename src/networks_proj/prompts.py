from __future__ import annotations

import json
from typing import Any


def questionnaire_prompt(items: tuple[str, ...], low: int, high: int) -> str:
    numbered = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))
    return f"""Complete this personality questionnaire as your current character.

Use a {low}-{high} Likert scale, where {low} means strongly disagree and {high} means strongly agree.

Items:
{numbered}

Return only valid JSON in this exact shape:
{{
  "answers": {{"exact item text": integer}},
  "reflection": "one brief optional explanation"
}}

Include every item exactly once. Do not add scores outside {low}-{high}."""


def transcript_text(messages: list[dict[str, Any]], limit: int) -> str:
    selected = messages[-limit:]
    if not selected:
        return "(No messages yet.)"
    return "\n".join(f"{item['speaker']}: {item['text']}" for item in selected)


def interaction_prompt(
    topic: str,
    partner_id: str,
    turn: int,
    turns_per_dyad: int,
    transcript: list[dict[str, Any]],
    transcript_limit: int,
) -> str:
    return f"""You are in a private dyadic interaction with {partner_id}.

Shared discussion task:
{topic}

Conversation so far:
{transcript_text(transcript, transcript_limit)}

This is turn {turn + 1} of {turns_per_dyad}.
Reply directly to {partner_id} in one concise message. Do not output JSON."""


def rating_prompt(
    partner_id: str,
    transcript: list[dict[str, Any]],
    transcript_limit: int,
    low: int,
    high: int,
) -> str:
    return f"""Privately rate {partner_id} using only the interaction below.

Warmth means warm, kind, and well-intentioned.
Competence means competent, capable, and effective.

Interaction:
{transcript_text(transcript, transcript_limit)}

Return only valid JSON:
{{
  "warmth": integer from {low} to {high},
  "competence": integer from {low} to {high},
  "rationale": "brief explanation"
}}"""


def memory_prompt(
    partner_id: str,
    transcript: list[dict[str, Any]],
    transcript_limit: int,
    rating: dict[str, Any],
) -> str:
    rating_json = json.dumps(rating, ensure_ascii=False)
    return f"""Create your private memory of this interaction with {partner_id}.

Interaction:
{transcript_text(transcript, transcript_limit)}

Your private rating:
{rating_json}

Write a concise first-person memory covering:
- what {partner_id} seemed like,
- the relationship or interaction pattern,
- what you may expect or do next time.

Do not invent facts outside the interaction. Return only valid JSON:
{{"memory": "first-person memory, at most 120 words"}}"""
