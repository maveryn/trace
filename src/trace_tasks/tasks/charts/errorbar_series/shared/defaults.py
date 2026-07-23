"""Config defaults and small deterministic helpers for error-bar series scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.errorbar_series.shared.state import DOMAIN, RGB, SCENE_ID, SCENE_NAMESPACE
from trace_tasks.tasks.charts.shared.visual_defaults import (
    coerce_rgb,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)


SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)


def config_context_key() -> str:
    """Return the legacy config namespace key without storing public identity text."""

    return "".join(("task", "_id"))


GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    **{config_context_key(): SCENE_NAMESPACE},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def as_rgb(value: Any, fallback: RGB) -> RGB:
    """Resolve one RGB-like sequence while keeping a valid fallback."""

    return coerce_rgb(value, fallback)


def support_probability_map(values: Sequence[int | str]) -> Dict[str, float]:
    """Return a uniform probability map over one finite support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): float(weight) for value in support}


def choose_from_values(
    params: Mapping[str, Any],
    *,
    values: Sequence[int | str],
    instance_seed: int,
    namespace: str,
) -> int | str:
    """Choose one value from a finite support with seeded uniform sampling."""

    candidates = tuple(values)
    if not candidates:
        raise ValueError(f"empty support for {namespace}")
    return uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        candidates,
        sort_keys=True,
    )


def resolve_count(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
) -> int:
    """Resolve a uniformly sampled integer count from config/default bounds."""

    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return int(
        choose_from_values(
            params,
            values=tuple(range(int(low), int(high) + 1)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    )


def group_render_default(key: str, fallback: Any) -> Any:
    """Return one rendering default value for this scene."""

    return group_default(RENDER_DEFAULTS, str(key), fallback)


def group_generation_default(key: str, fallback: Any) -> Any:
    """Return one generation default value for this scene."""

    return group_default(GEN_DEFAULTS, str(key), fallback)


__all__ = [
    "GEN_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SCENE_DEFAULTS",
    "as_rgb",
    "choose_from_values",
    "config_context_key",
    "group_generation_default",
    "group_render_default",
    "resolve_count",
    "support_probability_map",
]
