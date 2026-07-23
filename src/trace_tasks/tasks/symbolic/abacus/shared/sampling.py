"""Option sampling helpers for symbolic abacus readout tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default

from .rules import DEFAULT_OPTION_LABELS, digits_for_abacus_value, value_from_digits


def resolve_six_option_labels(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve the fixed six visible option labels for abacus MCQ tasks."""

    raw_labels = params.get("option_label_support", group_default(gen_defaults, "option_label_support", DEFAULT_OPTION_LABELS))
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    option_count = int(params.get("option_count", group_default(gen_defaults, "option_count", len(DEFAULT_OPTION_LABELS))))
    if int(option_count) != 6:
        raise ValueError("abacus readout option tasks require exactly 6 visual options")
    if len(labels) != int(option_count):
        raise ValueError("abacus option label support must contain exactly 6 labels")
    if len(set(labels)) != len(labels):
        raise ValueError("abacus option labels must be unique")
    return tuple(str(label) for label in labels)


def candidate_value_distractors(target_value: int, *, min_value: int = 0, max_value: int = 999) -> list[int]:
    """Return plausible nearby and digit-edit distractors for a three-digit abacus value."""

    target_digits = digits_for_abacus_value(int(target_value))
    candidates: list[int] = []
    seen: set[int] = {int(target_value)}

    def add(value: int) -> None:
        if int(min_value) <= int(value) <= int(max_value) and int(value) not in seen:
            seen.add(int(value))
            candidates.append(int(value))

    for digit_index in range(3):
        for delta in (-1, 1, -2, 2, -5, 5):
            edited = list(target_digits)
            edited[digit_index] = int(edited[digit_index]) + int(delta)
            if 0 <= int(edited[digit_index]) <= 9:
                add(value_from_digits((int(edited[0]), int(edited[1]), int(edited[2]))))
    for first, second in ((0, 1), (1, 2), (0, 2)):
        edited = list(target_digits)
        edited[first], edited[second] = edited[second], edited[first]
        add(value_from_digits((int(edited[0]), int(edited[1]), int(edited[2]))))
    for offset in (-100, 100, -50, 50, -20, 20, -10, 10, -5, 5, -2, 2, -1, 1):
        add(int(target_value) + int(offset))
    return [int(value) for value in candidates]


def choose_value_options(
    *,
    instance_seed: int,
    seed_namespace: str,
    target_value: int,
    option_labels: Sequence[str],
    correct_label: str,
    min_value: int = 0,
    max_value: int = 999,
) -> dict[str, int]:
    """Bind one target integer and unique distractors to visible option labels."""

    labels = tuple(str(label) for label in option_labels)
    rng = spawn_rng(int(instance_seed), f"{seed_namespace}.value_options")
    distractor_pool = candidate_value_distractors(int(target_value), min_value=int(min_value), max_value=int(max_value))
    rng.shuffle(distractor_pool)
    distractors: list[int] = []
    seen: set[int] = {int(target_value)}
    for value in distractor_pool:
        if int(value) not in seen:
            seen.add(int(value))
            distractors.append(int(value))
        if len(distractors) >= len(labels) - 1:
            break
    while len(distractors) < len(labels) - 1:
        value = int(rng.randint(int(min_value), int(max_value)))
        if int(value) not in seen:
            seen.add(int(value))
            distractors.append(int(value))
    out: dict[str, int] = {}
    distractor_index = 0
    for label in labels:
        if str(label) == str(correct_label):
            out[str(label)] = int(target_value)
        else:
            out[str(label)] = int(distractors[distractor_index])
            distractor_index += 1
    return dict(out)


def choose_digit_options(
    *,
    instance_seed: int,
    seed_namespace: str,
    target_digit: int,
    option_labels: Sequence[str],
    correct_label: str,
) -> dict[str, int]:
    """Bind one target digit and five unique digit distractors to option labels."""

    labels = tuple(str(label) for label in option_labels)
    if not 0 <= int(target_digit) <= 9:
        raise ValueError("abacus digit option target must be in 0..9")
    rng = spawn_rng(int(instance_seed), f"{seed_namespace}.digit_options")
    pool = [digit for digit in range(10) if int(digit) != int(target_digit)]
    rng.shuffle(pool)
    distractors = [int(digit) for digit in pool[: len(labels) - 1]]
    out: dict[str, int] = {}
    distractor_index = 0
    for label in labels:
        if str(label) == str(correct_label):
            out[str(label)] = int(target_digit)
        else:
            out[str(label)] = int(distractors[distractor_index])
            distractor_index += 1
    return dict(out)


__all__ = [
    "candidate_value_distractors",
    "choose_digit_options",
    "choose_value_options",
    "resolve_six_option_labels",
]
