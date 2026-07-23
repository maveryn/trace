"""Shared sampling helpers for isometric quarry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


@dataclass(frozen=True)
class LabelTaskSampleSpec:
    """Common resolved sampling state for quarry letter-label tasks."""

    selected_key: str
    prompt_key: str
    selection_probabilities: dict[str, float]
    candidate_count: int
    candidate_count_probabilities: dict[str, float]
    canvas_width: int
    canvas_height: int
    canvas_profile: str
    canvas_profile_probabilities: dict[str, float]


def support_values(
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
) -> tuple[int, dict[str, float]]:
    """Select one count from a support list with deterministic task-local sampling."""

    support = support_values(params, defaults, support_key=str(support_key), fallback=fallback)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be one of {support}")
        return value, support_probability_map(support, selected=value, sort_keys=True)
    rng = spawn_rng(int(instance_seed), str(namespace))
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(value), dict(probabilities)


def sample_label_task_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    identity_label: str,
    query_options: Sequence[str],
    default_query_id: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    max_candidate_count: int,
    fallback_width: int = 1200,
    fallback_height: int = 800,
) -> LabelTaskSampleSpec:
    """Resolve query, candidate count, and canvas profile for one label task."""

    selected_key, selection_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in query_options),
        default_query_id=str(default_query_id),
        task_id=str(identity_label),
        namespace=f"{identity_label}:query",
    )
    candidate_count, candidate_count_probabilities = select_count(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=generation_defaults,
        support_key="candidate_count_support",
        explicit_key="candidate_count",
        fallback=(4,),
        namespace=f"{identity_label}:candidate_count",
    )
    if int(candidate_count) > int(max_candidate_count):
        raise ValueError(f"candidate_count must be at most {max_candidate_count}")
    profile = resolve_canvas_profile(
        params=task_params,
        defaults=rendering_defaults,
        fallback_width=int(fallback_width),
        fallback_height=int(fallback_height),
        instance_seed=int(instance_seed),
        namespace=f"{identity_label}:canvas_profile",
    )
    return LabelTaskSampleSpec(
        selected_key=str(selected_key),
        prompt_key=str(selected_key),
        selection_probabilities=dict(selection_probabilities),
        candidate_count=int(candidate_count),
        candidate_count_probabilities=dict(candidate_count_probabilities),
        canvas_width=int(profile.width),
        canvas_height=int(profile.height),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
    )


__all__ = [
    "LabelTaskSampleSpec",
    "sample_label_task_spec",
    "select_count",
    "support_values",
]
