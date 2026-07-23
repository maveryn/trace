"""Sampling primitives for symbolic clock-display scenes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice_with_probabilities
from ....shared.config_defaults import group_default
from ....shared.time_format import clock_hand_angle_gap_deg, clock_total_minutes

DEFAULT_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


def resolve_clock_time_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    fallback_hour_min: int,
    fallback_hour_max: int,
    fallback_minute_min: int,
    fallback_minute_max: int,
    fallback_minute_step: int,
    context: str,
) -> Tuple[Tuple[int, int], Tuple[int, int, int], Tuple[int, ...]]:
    """Resolve hour/minute supports and canonical 12-hour clock times."""

    hour_min = int(params.get("hour_min", group_default(gen_defaults, "hour_min", fallback_hour_min)))
    hour_max = int(params.get("hour_max", group_default(gen_defaults, "hour_max", fallback_hour_max)))
    minute_min = int(params.get("minute_min", group_default(gen_defaults, "minute_min", fallback_minute_min)))
    minute_max = int(params.get("minute_max", group_default(gen_defaults, "minute_max", fallback_minute_max)))
    minute_step = int(params.get("minute_step", group_default(gen_defaults, "minute_step", fallback_minute_step)))
    if not (1 <= hour_min <= hour_max <= 12):
        raise ValueError(f"{context} hours must satisfy 1 <= min <= max <= 12")
    if not (0 <= minute_min <= minute_max <= 59):
        raise ValueError(f"{context} minutes must satisfy 0 <= min <= max <= 59")
    if minute_step <= 0:
        raise ValueError(f"{context} minute_step must be positive")
    values = tuple(
        clock_total_minutes(hour, minute)
        for hour in range(hour_min, hour_max + 1)
        for minute in range(minute_min, minute_max + 1, minute_step)
    )
    return (int(hour_min), int(hour_max)), (int(minute_min), int(minute_max), int(minute_step)), values


def feasible_clock_times(times: Tuple[int, ...], *, min_hand_angle_gap_deg: float) -> Tuple[int, ...]:
    """Filter time values to analog displays with separated hour/minute hands."""

    return tuple(
        int(total)
        for total in times
        if float(clock_hand_angle_gap_deg(int(total))) >= float(min_hand_angle_gap_deg)
    )


def resolve_text_option_labels(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    option_count: int = 6,
    fallback: Sequence[str] = DEFAULT_OPTION_LABELS,
) -> Tuple[str, ...]:
    """Resolve fixed visible option labels for clock MCQ tasks."""

    raw_labels = params.get(
        "option_label_support",
        group_default(gen_defaults, "option_label_support", tuple(fallback)),
    )
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    resolved_count = int(
        params.get("option_count", group_default(gen_defaults, "option_count", int(option_count)))
    )
    if int(resolved_count) != int(option_count):
        raise ValueError(f"clock option tasks require exactly {int(option_count)} options")
    if len(labels) < int(option_count):
        raise ValueError(f"clock option tasks require at least {int(option_count)} option labels")
    labels = labels[: int(option_count)]
    if len(set(labels)) != len(labels):
        raise ValueError("clock option labels must be unique")
    return tuple(str(label) for label in labels)


def sample_correct_option_label(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    seed_namespace: str,
    labels: Sequence[str],
) -> Tuple[str, Mapping[str, float]]:
    """Sample or validate the correct option label for a clock MCQ task."""

    option_labels = tuple(str(label) for label in labels)
    explicit = params.get("answer_label", params.get("correct_label"))
    if explicit is not None:
        label = str(explicit).strip()
        if label not in option_labels:
            raise ValueError("explicit answer_label/correct_label is outside option labels")
        return label, {str(candidate): (1.0 if str(candidate) == label else 0.0) for candidate in option_labels}
    rng = spawn_rng(int(instance_seed), f"{str(seed_namespace)}.correct_option_label")
    label, probabilities = uniform_choice_with_probabilities(
        rng,
        option_labels,
        sort_keys=True,
    )
    return str(label), {str(key): float(value) for key, value in probabilities.items()}


def option_value_map(
    *,
    labels: Sequence[str],
    correct_label: str,
    correct_value: Any,
    distractors: Iterable[Any],
) -> dict[str, Any]:
    """Bind one correct value and ordered distractors to visible option labels."""

    option_labels = tuple(str(label) for label in labels)
    if str(correct_label) not in option_labels:
        raise ValueError("correct_label must be in labels")
    distinct_distractors: list[Any] = []
    seen = {correct_value}
    for value in distractors:
        if value in seen:
            continue
        distinct_distractors.append(value)
        seen.add(value)
        if len(distinct_distractors) >= len(option_labels) - 1:
            break
    if len(distinct_distractors) < len(option_labels) - 1:
        raise ValueError("not enough distinct option distractors")
    mapping: dict[str, Any] = {}
    distractor_index = 0
    for label in option_labels:
        if str(label) == str(correct_label):
            mapping[str(label)] = correct_value
        else:
            mapping[str(label)] = distinct_distractors[distractor_index]
            distractor_index += 1
    return mapping


def nearby_integer_distractors(
    *,
    correct_value: int,
    support_values: Iterable[int],
    preferred_offsets: Sequence[int],
    min_value: int | None = None,
    max_value: int | None = None,
) -> Tuple[int, ...]:
    """Return ordered integer distractors near a correct value."""

    correct = int(correct_value)
    values: list[int] = []
    seen = {correct}
    for offset in preferred_offsets:
        for candidate in (correct + int(offset), correct - int(offset)):
            if min_value is not None and candidate < int(min_value):
                continue
            if max_value is not None and candidate > int(max_value):
                continue
            if candidate in seen:
                continue
            values.append(int(candidate))
            seen.add(int(candidate))
    for candidate in support_values:
        value = int(candidate)
        if min_value is not None and value < int(min_value):
            continue
        if max_value is not None and value > int(max_value):
            continue
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(int(value) for value in values)


__all__ = [
    "DEFAULT_OPTION_LABELS",
    "feasible_clock_times",
    "nearby_integer_distractors",
    "option_value_map",
    "resolve_clock_time_support",
    "resolve_text_option_labels",
    "sample_correct_option_label",
]
