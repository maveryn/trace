"""Scene-local defaults and variant resolution for spinner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Tuple

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.common import resolve_symbolic_axis_variant
from ...shared.visual_defaults import load_symbolic_noise_defaults

from .rendering import SUPPORTED_SPINNER_SCENE_VARIANTS, SpinnerRenderParams


SCENE_ID = "spinner"
_SCENE_DEFAULTS = get_scene_defaults("symbolic", SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)


def load_spinner_defaults(public_task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load scene defaults for one public spinner probability task."""

    return load_scene_generation_rendering_prompt_defaults(
        "symbolic",
        SCENE_ID,
        task_id=str(public_task_id),
    )


def resolve_spinner_scene_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the non-semantic spinner scene variant."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SPINNER_SCENE_VARIANTS,
        task_id=str(public_task_id),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_spinner_render_params(render_defaults: Mapping[str, Any]) -> SpinnerRenderParams:
    """Resolve pixel geometry for spinner rendering."""

    return SpinnerRenderParams(
        canvas_width=int(render_defaults.get("canvas_width", 1100)),
        canvas_height=int(render_defaults.get("canvas_height", 780)),
        single_center_x_px=int(render_defaults.get("single_center_x_px", 550)),
        single_center_y_px=int(render_defaults.get("single_center_y_px", 360)),
        single_radius_px=int(render_defaults.get("single_radius_px", 235)),
        pair_left_center_x_px=int(render_defaults.get("pair_left_center_x_px", 350)),
        pair_right_center_x_px=int(render_defaults.get("pair_right_center_x_px", 750)),
        pair_center_y_px=int(render_defaults.get("pair_center_y_px", 360)),
        pair_radius_px=int(render_defaults.get("pair_radius_px", 180)),
        panel_padding_px=int(render_defaults.get("panel_padding_px", 34)),
        panel_corner_radius_px=int(render_defaults.get("panel_corner_radius_px", 22)),
        sector_outline_width_px=int(render_defaults.get("sector_outline_width_px", 3)),
        pointer_width_px=int(render_defaults.get("pointer_width_px", 5)),
        hub_radius_px=int(render_defaults.get("hub_radius_px", 18)),
        badge_width_px=int(render_defaults.get("badge_width_px", 54)),
        badge_height_px=int(render_defaults.get("badge_height_px", 46)),
        number_font_size_px=int(render_defaults.get("number_font_size_px", 21)),
        title_font_size_px=int(render_defaults.get("title_font_size_px", 28)),
        subtitle_font_size_px=int(render_defaults.get("subtitle_font_size_px", 17)),
    )


def with_spinner_style_overrides(render_params: SpinnerRenderParams, scene_style: Any) -> SpinnerRenderParams:
    """Bind the sampled symbolic scene style into spinner renderer controls."""

    return replace(
        render_params,
        style_overrides={
            "panel_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
            "panel_outline": tuple(int(value) for value in scene_style.panel_border_rgb),
            "sector_outline": tuple(int(value) for value in scene_style.grid_rgb),
            "text": tuple(int(value) for value in scene_style.text_rgb),
            "muted_text": tuple(int(value) for value in scene_style.text_rgb),
            "text_stroke": tuple(int(value) for value in scene_style.text_stroke_rgb),
            "badge_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
            "badge_outline": tuple(int(value) for value in scene_style.panel_border_rgb),
            "shape_fill": tuple(int(value) for value in scene_style.text_rgb),
            "shape_outline": tuple(int(value) for value in scene_style.text_stroke_rgb),
            "hub_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
            "pointer": tuple(int(value) for value in scene_style.mark_rgb),
        },
    )


__all__ = [
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "load_spinner_defaults",
    "resolve_spinner_render_params",
    "resolve_spinner_scene_variant",
    "with_spinner_style_overrides",
]
