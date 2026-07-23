"""Sampling helpers for isometric harbor tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile
from trace_tasks.tasks.shared.config_defaults import group_default


@dataclass(frozen=True)
class CountTaskSampleSpec:
    """Resolved sampling state for one harbor count task instance."""

    selected_key: str
    prompt_query_key: str
    query_probabilities: dict[str, float]
    target_count: int
    target_count_probabilities: dict[str, float]
    answer_count_support: tuple[int, ...]
    answer_count_probabilities: dict[str, float]
    canvas_width: int
    canvas_height: int
    canvas_profile: str
    canvas_profile_probabilities: dict[str, float]


def support_ints(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    support_key: str,
    fallback: Sequence[int],
) -> tuple[int, ...]:
    """Resolve a numeric support list from params/defaults."""

    raw = params.get(str(support_key), group_default(defaults, str(support_key), tuple(fallback)))
    values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in values))
    if not support:
        raise ValueError(f"{support_key} must include at least one value")
    return support


def select_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback: Sequence[int],
    namespace: str,
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    """Select a count from finite support, using an internal cursor when present."""

    support = support_ints(params, defaults, support_key=str(support_key), fallback=fallback)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be one of {support}")
        return value, support_probability_map(support, selected=value, sort_keys=True), support
    sample_namespace = str(namespace)
    if params.get("_sample_cursor") is not None:
        sample_namespace = f"{sample_namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), sample_namespace)
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(value), dict(probabilities), support


__all__ = ["CountTaskSampleSpec", "select_count", "support_ints"]
