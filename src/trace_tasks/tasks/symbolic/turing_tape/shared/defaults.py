"""Config/default resolution for the symbolic Turing tape scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .....core.seed import hash64
from ....shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ....shared.font_assets import font_asset_version, sample_font_family
from ...shared.common import resolve_symbolic_axis_variant
from ...shared.unit_size_jitter import resolve_symbolic_unit_size_scale, scale_symbolic_px
from ...shared.visual_defaults import load_symbolic_noise_defaults

from .state import SCENE_ID, SCENE_VARIANTS, TuringRenderParams


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def load_turing_defaults(task_identifier: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one Turing task."""

    return load_scene_generation_rendering_prompt_defaults(
        "symbolic",
        SCENE_ID,
        task_id=str(task_identifier),
    )


def resolve_scene_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    task_identifier: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the non-semantic Turing tape scene variant."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(task_identifier),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> TuringRenderParams:
    """Resolve pixel geometry, unit jitter, and font metadata."""

    unit_scale, unit_meta = resolve_symbolic_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="turing_tape.unit_size",
    )
    font_params = {**dict(render_defaults), **dict(params)}
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="turing_tape.font",
        params=font_params,
    )
    return TuringRenderParams(
        canvas_width=int(group_default(render_defaults, "canvas_width", 1040)),
        canvas_height=int(group_default(render_defaults, "canvas_height", 880)),
        cell_size_px=scale_symbolic_px(group_default(render_defaults, "cell_size_px", 54), unit_scale, min_px=28),
        grid_gap_px=scale_symbolic_px(group_default(render_defaults, "grid_gap_px", 2), unit_scale, min_px=1),
        panel_padding_px=scale_symbolic_px(group_default(render_defaults, "panel_padding_px", 28), unit_scale, min_px=12),
        panel_corner_radius_px=scale_symbolic_px(group_default(render_defaults, "panel_corner_radius_px", 22), unit_scale, min_px=8),
        panel_border_width_px=scale_symbolic_px(group_default(render_defaults, "panel_border_width_px", 3), unit_scale, min_px=1),
        grid_line_width_px=scale_symbolic_px(group_default(render_defaults, "grid_line_width_px", 2), unit_scale, min_px=1),
        option_card_width_px=scale_symbolic_px(group_default(render_defaults, "option_card_width_px", 150), unit_scale, min_px=80),
        option_card_height_px=scale_symbolic_px(group_default(render_defaults, "option_card_height_px", 116), unit_scale, min_px=70),
        option_gap_px=scale_symbolic_px(group_default(render_defaults, "option_gap_px", 18), unit_scale, min_px=8),
        option_grid_cell_px=scale_symbolic_px(group_default(render_defaults, "option_grid_cell_px", 24), unit_scale, min_px=9),
        label_font_size_px=scale_symbolic_px(group_default(render_defaults, "label_font_size_px", 22), unit_scale, min_px=12),
        small_font_size_px=scale_symbolic_px(group_default(render_defaults, "small_font_size_px", 16), unit_scale, min_px=10),
        arrow_width_px=scale_symbolic_px(group_default(render_defaults, "arrow_width_px", 6), unit_scale, min_px=2),
        unit_size_jitter=dict(unit_meta),
        layout_seed=int(hash64(int(instance_seed), "turing_tape.layout", 0)),
        font_family=str(font_family),
    )


def style_meta_with_font(style_meta: Mapping[str, Any], render_params: TuringRenderParams) -> Dict[str, Any]:
    """Attach sampled readout font metadata used by rendered scene text."""

    return {
        **dict(style_meta),
        "font_family": str(render_params.font_family),
        "font": {
            "source": "global_font_pool",
            "font_family": str(render_params.font_family),
            "font_asset_version": font_asset_version(),
            "scope": "single_turing_tape_panel",
        },
    }
