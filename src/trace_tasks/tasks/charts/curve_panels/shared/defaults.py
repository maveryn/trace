"""Defaults and neutral config helpers for curve-panel charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .....core.sampling import uniform_choice
from .....core.scene_config import get_scene_defaults
from .....core.seed import spawn_rng
from ....shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.visual_defaults import (
    coerce_rgb,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    render_style_seed,
    resolve_chart_render_int,
    resolve_chart_render_rgb,
)
from .state import RGB

SCENE_NAMESPACE = "charts.curve_panels"
SCENE_ID = "curve_panels"
SCENE_VARIANT = "multipanel_line_grid"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (SCENE_VARIANT,)
PANEL_LABELS: Tuple[str, ...] = tuple("ABCDEFGHI")

_CONFIG_CONTEXT_KEY = "_".join(("task", "id"))
TASK_GROUP_DEFAULTS = get_scene_defaults("charts", "curve_panels")
GENERATION_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(
        TASK_GROUP_DEFAULTS if isinstance(TASK_GROUP_DEFAULTS, Mapping) else {},
        **{_CONFIG_CONTEXT_KEY: SCENE_NAMESPACE},
    )
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(
    scene_id=SCENE_ID, apply_prob=0.0
)


def as_rgb(value: Any, fallback: RGB) -> RGB:
    """Coerce a config value to an RGB tuple."""

    return coerce_rgb(value, fallback)


def resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer rendering parameter."""

    return resolve_chart_render_int(params, RENDER_DEFAULTS, str(key), int(fallback), namespace=SCENE_NAMESPACE)


def resolve_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    """Resolve an RGB rendering parameter."""

    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer generation parameter."""

    return int(
        params.get(
            str(key), group_default(GENERATION_DEFAULTS, str(key), int(fallback))
        )
    )


def without_sample_cursor(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return params without the balanced sampling cursor."""

    copied = dict(params)
    copied.pop("_sample_cursor", None)
    return copied


def balanced_choice(
    values: Sequence[Any],
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Any:
    """Choose one value from a non-empty support."""

    support = list(values)
    if not support:
        raise ValueError(f"empty support for {namespace}")
    return uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        support,
        sort_keys=True,
    )


def palette(params: Mapping[str, Any]) -> Tuple[RGB, ...]:
    """Resolve the curve color palette."""

    raw = params.get(
        "method_palette_rgb", RENDER_DEFAULTS.get("method_palette_rgb", ())
    )
    colors: list[RGB] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) >= 3
            ):
                colors.append(as_rgb(item, (0, 0, 0)))
    return (
        tuple(colors)
        if colors
        else (
            (39, 105, 176),
            (203, 79, 72),
            (52, 145, 98),
            (138, 91, 184),
            (211, 139, 44),
            (48, 147, 168),
        )
    )


def panel_count(
    params: Mapping[str, Any], *, instance_seed: int, min_required: int = 4
) -> int:
    """Sample a visible panel count."""

    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="panel_count_min",
        max_key="panel_count_max",
        fallback_min=6,
        fallback_max=8,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    low = max(4, int(min_required), int(low))
    high = min(len(PANEL_LABELS), int(high))
    if int(low) > int(high):
        raise ValueError("panel_count support is empty")
    return int(
        balanced_choice(
            list(range(int(low), int(high) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.panel_count",
        )
    )


def panel_answer_index_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return panel-index support for tasks that balance answer panels."""

    _low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="panel_count_min",
        max_key="panel_count_max",
        fallback_min=6,
        fallback_max=8,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    high = min(len(PANEL_LABELS), max(4, int(high)))
    return tuple(range(int(high)))


def method_count(
    params: Mapping[str, Any], *, instance_seed: int, min_required: int = 3
) -> int:
    """Sample a visible method/curve count."""

    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="method_count_min",
        max_key="method_count_max",
        fallback_min=5,
        fallback_max=5,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    low = max(3, int(min_required), int(low))
    high = int(high)
    if int(low) > int(high):
        raise ValueError("method_count support is empty")
    return int(
        balanced_choice(
            list(range(int(low), int(high) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.method_count",
        )
    )


def method_count_max(params: Mapping[str, Any]) -> int:
    """Return the maximum configured method count."""

    _low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="method_count_min",
        max_key="method_count_max",
        fallback_min=5,
        fallback_max=5,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return max(3, int(high))


__all__ = [
    "GENERATION_DEFAULTS",
    "PANEL_LABELS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "SUPPORTED_SCENE_VARIANTS",
    "balanced_choice",
    "generation_int",
    "method_count",
    "method_count_max",
    "palette",
    "panel_answer_index_support",
    "panel_count",
    "resolve_int",
    "resolve_rgb",
    "without_sample_cursor",
]
