"""Scene defaults for dumbbell chart primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.core.sampling import support_probability_map, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    render_style_seed,
    sample_chart_font_family as sample_shared_chart_font_family,
    resolve_chart_render_rgb,
)
from trace_tasks.tasks.charts.dumbbell.shared.state import RGB


DOMAIN = "charts"
SCENE_ID = "dumbbell"
SCENE_NAMESPACE = "charts_dumbbell"
SUPPORTED_SCENE_VARIANTS = ("horizontal_dumbbell",)
SUPPORTED_RANK_N = (2, 3, 4)

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

SCENE_LOAD_BY_VARIANT = {"horizontal_dumbbell": 0.58}


def render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer rendering parameter."""

    return int(params.get(str(key), RENDER_DEFAULTS.get(str(key), int(fallback))))


def render_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    """Resolve one RGB rendering parameter, including configured option lists."""

    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer generation parameter."""

    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def balanced_int(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    low: int,
    high: int,
) -> tuple[int, dict[str, float]]:
    """Select one integer from an inclusive support using deterministic balancing."""

    values = [int(value) for value in range(int(low), int(high) + 1)]
    if not values:
        raise ValueError(f"empty integer support for {namespace}")
    selected = uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        tuple(values),
        sort_keys=True,
    )
    return int(selected), support_probability_map(tuple(values))


def balanced_choice(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    values: Sequence[int],
) -> tuple[int, dict[str, float]]:
    """Select one integer from an explicit support using deterministic balancing."""

    support = tuple(int(value) for value in values)
    if not support:
        raise ValueError(f"empty integer support for {namespace}")
    selected = uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        support,
        sort_keys=True,
    )
    return int(selected), support_probability_map(support)


def sample_chart_font_family(params: Mapping[str, Any], *, instance_seed: int) -> str:
    """Sample the chart font family for all text in one dumbbell chart."""

    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


__all__ = [
    "DOMAIN",
    "GEN_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SCENE_ID",
    "SCENE_LOAD_BY_VARIANT",
    "SCENE_NAMESPACE",
    "SUPPORTED_RANK_N",
    "SUPPORTED_SCENE_VARIANTS",
    "balanced_choice",
    "balanced_int",
    "generation_int",
    "render_int",
    "render_rgb",
    "render_style_seed",
    "sample_chart_font_family",
]
