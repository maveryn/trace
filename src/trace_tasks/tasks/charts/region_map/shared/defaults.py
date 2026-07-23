"""Region-map scene defaults and semantic-axis helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from ...shared.visual_defaults import load_chart_scene_background_defaults, load_chart_scene_noise_defaults
from .assets import SUPPORTED_GEOGRAPHIC_MAP_VARIANTS as _SUPPORTED_GEOGRAPHIC_MAP_VARIANTS
from .assets import normalize_geographic_map_variant


DOMAIN = "charts"
SCENE_ID = "region_map"
SCENE_NAMESPACE = "charts_region_map"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("synthetic_region_map", "geographic_region_map")
SUPPORTED_THRESHOLD_DIRECTIONS: Tuple[str, ...] = ("greater_than", "less_than")
SUPPORTED_MARKER_RENDER_VARIANTS: Tuple[str, ...] = ("proportional_bubble",)

_TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)

POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

_SCENE_VARIANT_LOADS: Dict[str, float] = {
    "synthetic_region_map": 0.58,
    "geographic_region_map": 0.72,
}


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Select the visible map grammar for tasks that support both map variants."""

    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


def fixed_scene_variant_probabilities(scene_variant: str) -> Dict[str, float]:
    """Return one-hot scene-variant metadata for objectives with a fixed map grammar."""

    normalized = str(scene_variant)
    if normalized not in SUPPORTED_SCENE_VARIANTS:
        raise ValueError(f"unsupported region-map scene variant: {scene_variant}")
    return {variant: 1.0 if variant == normalized else 0.0 for variant in SUPPORTED_SCENE_VARIANTS}


def resolve_geographic_map_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Select the bundled geographic map asset without exposing public task identity."""

    alias_params = dict(params)
    if "map_asset_variant" in alias_params and "geographic_map_variant" not in alias_params:
        alias_params["geographic_map_variant"] = alias_params.get("map_asset_variant")
    if "map_asset_id" in alias_params and "geographic_map_variant" not in alias_params:
        alias_params["geographic_map_variant"] = normalize_geographic_map_variant(alias_params.get("map_asset_id"))
    return resolve_chart_axis_variant_for_namespace(
        params=alias_params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=_SUPPORTED_GEOGRAPHIC_MAP_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.geographic_map_variant",
        explicit_key="geographic_map_variant",
        weights_key="geographic_map_variant_weights",
        balance_flag_key="balanced_geographic_map_variant_sampling",
    )


def resolve_marker_render_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Select the neutral marker-layer rendering mode for region-map marker tasks."""

    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_MARKER_RENDER_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.marker_render_variant",
        explicit_key="marker_render_variant",
        weights_key="marker_render_variant_weights",
        balance_flag_key="balanced_marker_render_variant_sampling",
    )


__all__ = [
    "DOMAIN",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_MARKER_RENDER_VARIANTS",
    "SUPPORTED_THRESHOLD_DIRECTIONS",
    "_GEN_DEFAULTS",
    "_PROMPT_DEFAULTS",
    "_RENDER_DEFAULTS",
    "fixed_scene_variant_probabilities",
    "resolve_geographic_map_variant",
    "resolve_marker_render_variant",
    "resolve_scene_variant",
]
