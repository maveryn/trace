"""Shared sampling helpers for construction-site illustration tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from .state import (
    CONSTRUCTION_COLOR_NAMES,
    CONSTRUCTION_EQUIPMENT_TYPES,
    CONSTRUCTION_MATERIAL_TYPES,
    CONSTRUCTION_SETTING_IDS,
    CONSTRUCTION_TOOL_TYPES,
    CONSTRUCTION_ZONE_TYPES,
)
from ...shared.object_library import STYLE_IDS
from ...shared.task_support import (
    bounds as _shared_bounds,
    query_support,
    render_params as _shared_render_params,
    sample_count,
    spawned_task_rng,
    string_support,
    style_weights as _shared_style_weights,
    uniform_string_probability_map,
)


def bounds(params: Mapping[str, Any], defaults: Mapping[str, Any], low_key: str, high_key: str, fallback_low: int, fallback_high: int) -> Tuple[int, int]:
    """Resolve integer low/high bounds from params, group defaults, or fallback."""

    return _shared_bounds(params, defaults, low_key, high_key, fallback_low, fallback_high, min_low=0)


def color_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve construction color support."""

    return string_support(params, defaults, "color_name_support", CONSTRUCTION_COLOR_NAMES, valid_values=CONSTRUCTION_COLOR_NAMES, min_count=2)


def material_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve construction material support."""

    return string_support(params, defaults, "material_type_support", CONSTRUCTION_MATERIAL_TYPES, valid_values=CONSTRUCTION_MATERIAL_TYPES, min_count=2)


def equipment_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve construction equipment support."""

    return string_support(params, defaults, "equipment_type_support", CONSTRUCTION_EQUIPMENT_TYPES, valid_values=CONSTRUCTION_EQUIPMENT_TYPES, min_count=2)


def zone_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve construction zone support."""

    return string_support(params, defaults, "zone_support", CONSTRUCTION_ZONE_TYPES, valid_values=CONSTRUCTION_ZONE_TYPES, min_count=2)


def tool_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve construction worker tool support."""

    return string_support(params, defaults, "tool_type_support", CONSTRUCTION_TOOL_TYPES, valid_values=CONSTRUCTION_TOOL_TYPES, min_count=1)


def render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    fallback_width: int,
    fallback_height: int,
    fallback_scale: int,
    instance_seed: int | None = None,
    namespace: str = "construction_site:canvas_profile",
) -> Dict[str, Any]:
    """Resolve construction canvas render parameters."""

    return _shared_render_params(
        params,
        render_defaults,
        prefix="construction",
        fallback_width=fallback_width,
        fallback_height=fallback_height,
        fallback_scale=fallback_scale,
        instance_seed=instance_seed,
        namespace=namespace,
    )


def style_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    """Resolve illustration style weights."""

    return _shared_style_weights(params, render_defaults, style_ids=STYLE_IDS)


def setting_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    """Resolve construction setting weights."""

    raw = params.get("construction_setting_weights", group_default(render_defaults, "construction_setting_weights", {setting: 1.0 for setting in CONSTRUCTION_SETTING_IDS}))
    if not isinstance(raw, Mapping):
        raise ValueError("construction_setting_weights must be a mapping")
    return {str(key): max(0.0, float(value)) for key, value in raw.items()}


__all__ = [
    "bounds",
    "color_support",
    "equipment_support",
    "material_support",
    "query_support",
    "render_params",
    "sample_count",
    "setting_weights",
    "spawned_task_rng",
    "style_weights",
    "tool_support",
    "uniform_string_probability_map",
    "zone_support",
]
