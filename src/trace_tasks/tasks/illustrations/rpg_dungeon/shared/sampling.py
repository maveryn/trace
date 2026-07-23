"""Sampling helpers for RPG dungeon public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.shared.config_defaults import group_default


@dataclass(frozen=True)
class HazardRenderCounts:
    """Neutral chamber counts needed to render a reachable hazardous subset."""

    reachable_total: int
    hazard_total: int
    reachable_hazard_total: int


def choose_hazard_render_counts(
    *,
    instance_seed: int,
    namespace: str,
    total_chambers: int,
    target_safe_total: int,
    max_hazard_total: int,
) -> HazardRenderCounts:
    """Choose reachable and hazardous chamber counts for a requested safe subset."""

    candidates: list[HazardRenderCounts] = []
    for reachable_total in range(max(1, int(target_safe_total)), int(total_chambers)):
        reachable_hazards = int(reachable_total) - int(target_safe_total)
        if not 0 <= int(reachable_hazards) <= int(max_hazard_total):
            continue
        unreachable_slots = int(total_chambers) - int(reachable_total)
        if int(reachable_hazards) == 0:
            candidates.append(HazardRenderCounts(int(reachable_total), 1, 0))
            continue
        max_extra_unreachable = min(
            int(max_hazard_total) - int(reachable_hazards),
            int(unreachable_slots),
        )
        for extra_unreachable in range(max_extra_unreachable + 1):
            candidates.append(
                HazardRenderCounts(
                    reachable_total=int(reachable_total),
                    hazard_total=int(reachable_hazards) + int(extra_unreachable),
                    reachable_hazard_total=int(reachable_hazards),
                )
            )
    if not candidates:
        raise ValueError(
            "no RPG dungeon hazard-count plan for "
            f"total={total_chambers}, target={target_safe_total}"
        )
    rng = spawn_rng(
        int(instance_seed),
        f"{namespace}:{int(total_chambers)}:{int(target_safe_total)}",
    )
    return rng.choice(candidates)


def select_count_from_support(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    max_value: int | None = None,
) -> tuple[int, dict[str, float]]:
    """Select an integer count from task params, defaults, or fallback support."""

    raw_support = params.get(str(support_key), group_default(gen_defaults, str(support_key), tuple(fallback_support)))
    if isinstance(raw_support, int):
        support = (int(raw_support),)
    else:
        support = tuple(dict.fromkeys(int(value) for value in raw_support))
    if max_value is not None:
        support = tuple(value for value in support if int(value) <= int(max_value))
    if not support:
        raise ValueError(f"{support_key} must contain at least one value")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be one of {support}")
        return int(value), support_probability_map(support, selected=int(value), sort_keys=True)
    sample_namespace = str(namespace)
    if params.get("_sample_cursor") is not None:
        sample_namespace = f"{sample_namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), sample_namespace)
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(value), dict(probabilities)


__all__ = [
    "HazardRenderCounts",
    "choose_hazard_render_counts",
    "select_count_from_support",
]
