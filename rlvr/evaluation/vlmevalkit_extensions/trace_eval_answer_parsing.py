"""Narrow final-answer wrapper handling for pinned VLMEvalKit scorers."""

from __future__ import annotations

import math
import re
from typing import Any


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.I | re.S)
_OFFICIAL_OPTION_RE = re.compile(r"\banswer\s*:\s*([A-D])\b", flags=re.I)
_ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.*?)\s*$", flags=re.I | re.M)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _last_boxed(text: str) -> str | None:
    values: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            values.append(text[start : pos - 1])
    return values[-1] if values else None


def _atomic_abcd(value: Any) -> str | None:
    text = _text(value).strip()
    answer = re.fullmatch(r"answer\s*:\s*(.*)", text, flags=re.I | re.S)
    if answer:
        text = answer.group(1).strip()

    while text:
        previous = text
        wrappers = (
            r"\\\(\s*(.*?)\s*\\\)",
            r"\\\[\s*(.*?)\s*\\\]",
            r"\$\s*(.*?)\s*\$",
            r"\\(?:text|mathrm|mathbf)\s*\{\s*(.*?)\s*\}",
            r"[\(\[\{]\s*(.*?)\s*[\)\]\}]",
        )
        for pattern in wrappers:
            wrapped = re.fullmatch(pattern, text, flags=re.I | re.S)
            if wrapped:
                text = wrapped.group(1).strip()
                break
        if text == previous:
            break

    return text.upper() if re.fullmatch(r"[A-D]", text, flags=re.I) else None


def unwrap_single_answer_block(value: Any) -> str:
    """Unwrap one nonempty ``<answer>`` block; otherwise preserve the input."""

    text = _text(value)
    matches = list(_ANSWER_TAG_RE.finditer(text))
    if len(matches) != 1:
        return text
    answer = matches[0].group(1).strip()
    return answer if answer else text


def extract_unambiguous_abcd(value: Any) -> str:
    """Extend VLMEvalKit's ``Answer: X`` parser with explicit final wrappers.

    Only explicit ``Answer: X`` markers, a single answer block, and the final
    balanced box are considered. Conflicting explicit candidates remain
    unresolved, matching the scorer's existing ``Z`` sentinel.
    """

    text = _text(value)
    candidates = [match.group(1).upper() for match in _OFFICIAL_OPTION_RE.finditer(text)]

    tags = list(_ANSWER_TAG_RE.finditer(text))
    if len(tags) == 1:
        tagged = tags[0].group(1).strip()
        tagged_box = _last_boxed(tagged)
        tagged_candidate = _atomic_abcd(tagged_box) if tagged_box is not None else _atomic_abcd(tagged)
        if tagged_candidate is None:
            answer_lines = list(_ANSWER_LINE_RE.finditer(tagged))
            if answer_lines:
                tagged_candidate = _atomic_abcd(answer_lines[-1].group(1))
        if tagged_candidate is not None:
            candidates.append(tagged_candidate)

    boxed = _last_boxed(text)
    boxed_candidate = _atomic_abcd(boxed) if boxed is not None else None
    if boxed_candidate is not None:
        candidates.append(boxed_candidate)

    distinct = list(dict.fromkeys(candidates))
    return distinct[0] if len(distinct) == 1 else "Z"
