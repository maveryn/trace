#!/usr/bin/env python3
"""Strict, reusable answer parsing for TRACE benchmark evaluation."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def last_boxed(text: Any) -> str | None:
    """Return the content of the final balanced ``\\boxed{...}`` expression."""

    raw = _clean_cell(text)
    values: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", raw):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(raw) and depth:
            if raw[pos] == "{":
                depth += 1
            elif raw[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            values.append(raw[start : pos - 1])
    return values[-1] if values else None


def clean_final_answer(text: Any) -> str:
    value = _clean_cell(text)
    value = re.sub(r"</?answer>", "", value, flags=re.I).strip()
    value = re.sub(r"^\s*[:：,\-]+\s*", "", value).strip()
    value = value.strip("` \n\t\r").strip("\"'")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) > 1 and len(lines[0]) <= 160:
        value = lines[0]
    return value.strip().strip("\"'")


def extract_final_answer(value: Any) -> tuple[str, str]:
    """Extract a final answer while preserving the parsing method for audits.

    The deterministic contracts accepted here are the response wrappers used by
    TRACE evaluations. Unwrapped responses are returned unchanged rather than
    guessed from arbitrary reasoning text.
    """

    text = _clean_cell(value)
    if not text:
        return "", "empty"

    tagged = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S))
    if tagged:
        return clean_final_answer(tagged[-1].group(1)), "answer_tag"

    boxed = last_boxed(text)
    if boxed is not None:
        return clean_final_answer(boxed), "boxed"

    decoder = json.JSONDecoder()
    json_answers: list[Any] = []
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "answer" in payload:
            json_answers.append(payload["answer"])
    if json_answers:
        return clean_final_answer(json_answers[-1]), "json_answer"

    final_markers = list(
        re.finditer(
            r"(?is)(?:^|\n|\b)(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:|：)\s*(.+)",
            text,
        )
    )
    if final_markers:
        return clean_final_answer(final_markers[-1].group(1)), "answer_marker"

    return text, "raw"


def parse_binary_score(value: Any) -> float | None:
    """Parse an explicit binary judge decision; return ``None`` if malformed."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        if float(value) in {0.0, 1.0}:
            return float(value)
        return None

    text = _clean_cell(value).lower()
    if text in {"1", "true", "yes", "correct"}:
        return 1.0
    if text in {"0", "false", "no", "incorrect"}:
        return 0.0

    match = re.search(
        r"[\"']?(?:score|judg(?:e)?ment|judge\s+output|correct)[\"']?\s*[:=]\s*"
        r"(?:\*{1,3}|_{1,3})?\s*([01])\s*(?:\*{1,3}|_{1,3})?"
        r"(?=\s|[.!,:;)}\]]|$)",
        text,
    )
    if match:
        return float(match.group(1))
    return None
