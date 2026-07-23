"""Shared sampling and config helpers for illustration tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice
from ...shared.config_defaults import group_default
from .canvas_profiles import resolve_profile_render_params
from .style_registry import resolve_art_style_weights


def bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
    *,
    min_low: int = 1,
) -> Tuple[int, int]:
    """Resolve integer low/high bounds from params, group defaults, or fallback."""

    low = int(params.get(low_key, group_default(defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(defaults, high_key, fallback_high)))
    if low < int(min_low) or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a uniform probability map over string support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def sample_count(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    low: int,
    high: int,
    explicit_key: str,
    cycle_index: int | None = None,
) -> Tuple[int, Dict[str, float]]:
    """Sample an integer count from explicit support with seeded RNG."""

    support = tuple(range(int(low), int(high) + 1))
    if not support:
        raise ValueError(f"{explicit_key} has no feasible support")
    explicit = params.get(explicit_key)
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} is outside configured support")
        return int(value), support_probability_map(support, selected=int(value), sort_keys=True)
    rng = spawn_rng(int(instance_seed), str(namespace))
    value = int(uniform_choice(rng, support, sort_keys=True))
    return int(value), support_probability_map(support, sort_keys=True)


def string_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
    *,
    valid_values: Sequence[str],
    min_count: int = 1,
) -> Tuple[str, ...]:
    """Resolve and validate a configured string support list."""

    raw = params.get(str(key), group_default(defaults, str(key), fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    valid = set(str(value) for value in valid_values)
    support = tuple(str(value) for value in raw if str(value) in valid)
    support = tuple(dict.fromkeys(support))
    if len(support) < int(min_count):
        raise ValueError(f"{key} must contain at least {min_count} supported values")
    return support


def query_support(params: Mapping[str, Any], defaults: Mapping[str, Any], fallback: Sequence[str]) -> Tuple[str, ...]:
    """Resolve query-id support."""

    return string_support(params, defaults, "query_id_support", fallback, valid_values=fallback, min_count=1)


def style_weights(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    style_ids: Sequence[str],
) -> Dict[str, float]:
    """Resolve non-semantic illustration style weights."""

    return resolve_art_style_weights(params, render_defaults, style_ids=style_ids)


def setting_weights(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    key: str,
    setting_ids: Sequence[str],
) -> Dict[str, float]:
    """Resolve non-semantic illustration scene-setting weights."""

    raw = params.get(str(key), group_default(render_defaults, str(key), {setting: 1.0 for setting in setting_ids}))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return {str(name): max(0.0, float(value)) for name, value in raw.items()}


def render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    prefix: str,
    fallback_width: int,
    fallback_height: int,
    fallback_scale: int,
    instance_seed: int | None = None,
    namespace: str = "illustrations:canvas_profile",
) -> Dict[str, Any]:
    """Resolve prefix-scoped illustration canvas render parameters."""

    return resolve_profile_render_params(
        params,
        render_defaults,
        prefix=str(prefix),
        fallback_width=int(fallback_width),
        fallback_height=int(fallback_height),
        fallback_scale=int(fallback_scale),
        instance_seed=instance_seed,
        namespace=str(namespace),
    )


def spawned_task_rng(instance_seed: int, task_id: str, attempt_index: int):
    """Return the standard illustration-task sampling RNG."""

    return spawn_rng(int(instance_seed), f"{task_id}:sample", int(attempt_index))


__all__ = [
    "bounds",
    "query_support",
    "render_params",
    "sample_count",
    "setting_weights",
    "spawned_task_rng",
    "string_support",
    "style_weights",
    "uniform_string_probability_map",
]
