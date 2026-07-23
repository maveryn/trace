"""Identity-free sampling helpers for balance-scale puzzle scenes."""

from __future__ import annotations

from itertools import cycle
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import integer_range_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.common import get_int_param, get_int_range
from trace_tasks.tasks.puzzles.shared.symbol_rendering import (
    PUZZLE_OBJECT_COLOR_BY_TYPE,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.puzzles.shared.common import resolve_puzzle_axis_variant

from .state import (
    BalanceScaleRenderParams,
    BalanceSceneAxes,
    SCENE_VARIANTS,
    TARGET_CUE_MODES,
)

BALANCE_OBJECT_TYPES: tuple[str, ...] = (
    "circle",
    "triangle",
    "diamond",
    "hexagon",
    "star",
    "pentagon",
)


def resolve_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    include_target_cue: bool,
) -> BalanceSceneAxes:
    """Resolve the scene style axis and optional target-cue axis."""

    scene_variant, scene_probs = resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )
    if bool(include_target_cue):
        target_cue_mode, target_probs = resolve_puzzle_axis_variant(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            supported_variants=TARGET_CUE_MODES,
            task_id=str(namespace),
            explicit_key="target_cue_mode",
            weights_key="target_cue_mode_weights",
            balance_flag_key="balanced_target_cue_mode_sampling",
            axis_namespace="target_cue_mode",
        )
    else:
        target_cue_mode = "query_row_only"
        target_probs = {"query_row_only": 1.0}
    return BalanceSceneAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={
            str(key): float(value) for key, value in scene_probs.items()
        },
        target_cue_mode=str(target_cue_mode),
        target_cue_mode_probabilities={
            str(key): float(value) for key, value in target_probs.items()
        },
    )


def resolve_integer_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, list[int]]:
    """Select one integer target answer from the configured support."""

    low, high = get_int_range(
        params,
        gen_defaults,
        min_key="answer_min",
        max_key="answer_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    support = [int(value) for value in range(int(low), int(high) + 1)]
    if not support:
        raise ValueError("balance-scale answer support cannot be empty")
    rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
    selected, _probabilities = integer_range_choice(
        rng,
        int(support[0]),
        int(support[-1]),
    )
    return int(selected), support


def resolve_panel_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> int:
    """Select the number of visible scale panels for equation tasks."""

    low, high = get_int_range(
        params,
        gen_defaults,
        min_key="scale_panel_count_min",
        max_key="scale_panel_count_max",
        fallback_min=2,
        fallback_max=3,
    )
    support = [int(value) for value in range(int(low), int(high) + 1)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.scale_panel_count")
    selected, _probabilities = integer_range_choice(
        rng,
        int(support[0]),
        int(support[-1]),
    )
    return int(selected)


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> BalanceScaleRenderParams:
    """Resolve pixel sizes and unit-size jitter for the balance renderer."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.unit_size",
    )
    return BalanceScaleRenderParams(
        canvas_width=get_int_param(params, render_defaults, "canvas_width", 1120),
        canvas_height=get_int_param(params, render_defaults, "canvas_height", 840),
        scene_margin_left_px=get_int_param(
            params,
            render_defaults,
            "scene_margin_left_px",
            54,
        ),
        scene_margin_right_px=get_int_param(
            params,
            render_defaults,
            "scene_margin_right_px",
            54,
        ),
        scene_margin_top_px=get_int_param(
            params,
            render_defaults,
            "scene_margin_top_px",
            48,
        ),
        scene_margin_bottom_px=get_int_param(
            params,
            render_defaults,
            "scene_margin_bottom_px",
            48,
        ),
        panel_padding_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "panel_padding_px", 26),
            unit_scale,
            min_px=16,
        ),
        panel_corner_radius_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "panel_corner_radius_px", 20),
            unit_scale,
            min_px=8,
        ),
        panel_border_width_px=get_int_param(
            params,
            render_defaults,
            "panel_border_width_px",
            3,
        ),
        scale_panel_gap_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "scale_panel_gap_px", 20),
            unit_scale,
            min_px=12,
        ),
        query_row_height_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "query_row_height_px", 116),
            unit_scale,
            min_px=82,
        ),
        beam_width_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "beam_width_px", 620),
            unit_scale,
            min_px=420,
        ),
        pan_width_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "pan_width_px", 230),
            unit_scale,
            min_px=170,
        ),
        pan_height_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "pan_height_px", 82),
            unit_scale,
            min_px=58,
        ),
        token_size_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "token_size_px", 46),
            unit_scale,
            min_px=34,
        ),
        token_gap_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "token_gap_px", 8),
            unit_scale,
            min_px=5,
        ),
        line_width_px=get_int_param(params, render_defaults, "line_width_px", 3),
        value_font_size_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "value_font_size_px", 24),
            unit_scale,
            min_px=18,
        ),
        label_font_size_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "label_font_size_px", 22),
            unit_scale,
            min_px=16,
        ),
        query_font_size_px=scale_puzzle_px(
            get_int_param(params, render_defaults, "query_font_size_px", 30),
            unit_scale,
            min_px=22,
        ),
        unit_size_jitter=dict(unit_meta),
    )


def object_specs_for_labels(
    labels: Sequence[str], *, offset: int
) -> Dict[str, Dict[str, Any]]:
    """Assign deterministic puzzle icon shapes and colors to object labels."""

    specs: Dict[str, Dict[str, Any]] = {}
    object_types = tuple(str(item) for item in BALANCE_OBJECT_TYPES)
    offset_index = int(offset)
    if offset_index < 0:
        raise ValueError("balance object type offset must be non-negative")
    while offset_index >= len(object_types):
        offset_index -= len(object_types)
    rotated_object_types = object_types[offset_index:] + object_types[:offset_index]
    for label, object_type in zip(labels, cycle(rotated_object_types)):
        specs[str(label)] = {
            "object_label": str(label),
            "object_type": str(object_type),
            "fill_rgb": list(PUZZLE_OBJECT_COLOR_BY_TYPE[object_type]),
        }
    return specs


__all__ = [
    "object_specs_for_labels",
    "BALANCE_OBJECT_TYPES",
    "resolve_integer_answer",
    "resolve_panel_count",
    "resolve_render_params",
    "resolve_scene_axes",
]
