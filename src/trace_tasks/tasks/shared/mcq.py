"""Shared helpers for deterministic integer multiple-choice options."""

from __future__ import annotations

from typing import Dict, List


def option_label_for_index(index: int) -> str:
    """Return canonical option label for one 0-based MCQ index."""
    idx = int(index)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return labels[idx] if 0 <= idx < len(labels) else f"Option{idx + 1}"


def build_integer_mcq_options(
    rng,
    *,
    correct_value: int,
    min_value: int,
    max_value: int,
    option_count: int = 5,
    min_delta: int = 5,
    max_delta: int = 30,
) -> Dict[str, object]:
    """Build integer MCQ choices around one correct value.

    Distractors are sampled uniformly without replacement from values that satisfy:
    - `min_value <= value <= max_value`,
    - `min_delta <= abs(value - correct_value) <= max_delta`.
    """
    total_options = max(2, int(option_count))
    answer_value = int(correct_value)
    low = int(min_value)
    high = int(max_value)
    delta_low = max(1, int(min_delta))
    delta_high = max(delta_low, int(max_delta))
    candidates = [
        int(value)
        for value in range(low, high + 1)
        if int(value) != int(answer_value) and delta_low <= abs(int(value) - int(answer_value)) <= delta_high
    ]
    if len(candidates) < (total_options - 1):
        raise ValueError("insufficient MCQ distractor candidates")

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    distractors = [int(value) for value in shuffled[: total_options - 1]]
    options = [int(answer_value), *distractors]
    rng.shuffle(options)
    correct_index = int(options.index(int(answer_value)))
    return {
        "options": [int(value) for value in options],
        "correct_index": int(correct_index),
    }


def format_lettered_options(options: List[int]) -> str:
    """Format one integer option list into lettered MCQ lines."""
    lines = []
    for idx, value in enumerate(options):
        label = option_label_for_index(idx)
        lines.append(f"{label}. {int(value)}")
    return "\n".join(lines)
