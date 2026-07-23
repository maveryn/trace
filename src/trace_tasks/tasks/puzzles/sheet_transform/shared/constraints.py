"""Identity-free validation helpers for sheet-transform options."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import integer_range_choice, support_probability_map


def single_correct_option(
    option_specs: Sequence[Mapping[str, Any]],
    *,
    expected_label: str,
    expected_choice_id: str,
    context: str,
) -> Mapping[str, Any]:
    """Return the one correct option after checking answer label and id binding."""

    correct = [option for option in option_specs if bool(option.get("is_correct", False))]
    if len(correct) != 1:
        raise ValueError(f"{context} must have exactly one correct option")
    option = correct[0]
    if str(option["option_label"]) != str(expected_label):
        raise ValueError(f"{context} answer drifted from option specs")
    if str(option["option_choice_id"]) != str(expected_choice_id):
        raise ValueError(f"{context} correct option id drifted from option specs")
    return option


def resolve_correct_option_index(
    params: Mapping[str, Any],
    *,
    option_count: int,
    rng,
) -> tuple[int, dict[str, float], str]:
    """Resolve the answer option slot with traceable support stratification."""

    support = tuple(range(int(option_count)))
    explicit = params.get("correct_option_index")
    if explicit is not None:
        index = int(explicit)
        if index not in set(support):
            raise ValueError("correct_option_index must fall inside option_count")
        return (
            int(index),
            support_probability_map(support, selected=int(index), sort_keys=True),
            "explicit_param",
        )
    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        index = int(sample_cursor) % int(option_count)
        return (
            int(index),
            support_probability_map(support, sort_keys=True),
            "sample_cursor_stratified",
        )
    index, probabilities = integer_range_choice(rng, 0, int(option_count) - 1)
    return int(index), dict(probabilities), "rng_uniform"


__all__ = ["resolve_correct_option_index", "single_correct_option"]
