"""Config resolution for paper-fold result puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import support_probability_map, weighted_support_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.paper_fold_common import (
    PuzzleFoldResultRenderParams,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb

from .state import (
    FOLD_SCENE_VARIANTS,
    OVERLAY_MARK_SHAPES,
    OVERLAY_SCENE_VARIANTS,
    SUPPORTED_CUT_HOLE_SHAPES,
    SUPPORTED_FOLD_AXES,
    SUPPORTED_FOLD_COUNTS,
)


@dataclass(frozen=True)
class PaperFoldGenerationDefaults:
    """Resolved generation bounds for one paper-fold task."""

    option_count_min: int
    option_count_max: int
    grid_size: int
    mark_count_min: int
    mark_count_max: int


def resolve_generation_defaults(
    generation_defaults: Mapping[str, Any],
) -> PaperFoldGenerationDefaults:
    """Resolve stable generation bounds from scene config."""

    return PaperFoldGenerationDefaults(
        option_count_min=int(group_default(generation_defaults, "option_count_min", 4)),
        option_count_max=int(group_default(generation_defaults, "option_count_max", 4)),
        grid_size=int(group_default(generation_defaults, "grid_size", 6)),
        mark_count_min=int(group_default(generation_defaults, "mark_count_min", 3)),
        mark_count_max=int(group_default(generation_defaults, "mark_count_max", 5)),
    )


def _support_from_config(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    options_key: str,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    """Return one explicit string support from params, config, or fallback."""

    raw = params.get(str(options_key), group_default(defaults, str(options_key), None))
    if raw is None:
        return tuple(str(item) for item in fallback)
    values = tuple(str(item).strip() for item in raw if str(item).strip())
    if not values:
        raise ValueError(f"{options_key} must contain at least one value")
    return values


def _select_supported_value(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    explicit_key: str,
    weights_key: str,
    options_key: str,
    fallback: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select one configured support value using explicit or weighted sampling."""

    support = _support_from_config(
        params,
        defaults,
        options_key=str(options_key),
        fallback=tuple(fallback),
    )
    explicit = params.get(str(explicit_key), group_default(defaults, str(explicit_key), None))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(support):
            raise ValueError(f"{explicit_key} must be one of {support}")
        return selected, support_probability_map(support, selected=selected)

    raw_weights = params.get(str(weights_key), group_default(defaults, str(weights_key), None))
    weights = (
        {str(key): float(value) for key, value in raw_weights.items()}
        if isinstance(raw_weights, Mapping)
        else None
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = weighted_support_choice(
        rng,
        support,
        weights=weights,
        sort_keys=False,
    )
    return str(selected), dict(probabilities)


def resolve_fold_axis(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve vertical versus horizontal fold as a semantic parameter axis."""

    return _select_supported_value(
        params,
        generation_defaults,
        explicit_key="fold_axis",
        weights_key="fold_axis_weights",
        options_key="fold_axis_options",
        fallback=SUPPORTED_FOLD_AXES,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.fold_axis",
    )


def resolve_scene_variant(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one paper-fold visual scene treatment."""

    return _select_supported_value(
        params,
        generation_defaults,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        options_key="scene_variant_options",
        fallback=FOLD_SCENE_VARIANTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scene_variant",
    )


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> PuzzleFoldResultRenderParams:
    """Resolve rendering parameters for one paper-fold scene."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(fallback),
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.fold_projection_render",
        )

    def _rgb(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            str(key),
            fallback,
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.fold_projection_render",
        )

    unit_size_scale, unit_size_jitter = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.sheet_transform.fold_projection.unit_size",
    )
    return PuzzleFoldResultRenderParams(
        canvas_width=int(_int("canvas_width", 1200)),
        canvas_height=int(_int("canvas_height", 940)),
        scene_margin_left_px=int(_int("scene_margin_left_px", 64)),
        scene_margin_right_px=int(_int("scene_margin_right_px", 64)),
        scene_margin_top_px=int(_int("scene_margin_top_px", 56)),
        scene_margin_bottom_px=int(_int("scene_margin_bottom_px", 56)),
        reference_panel_height_px=scale_puzzle_px(
            _int("reference_panel_height_px", 330),
            unit_size_scale,
            min_px=200,
        ),
        reference_panel_padding_px=scale_puzzle_px(
            _int("reference_panel_padding_px", 28),
            unit_size_scale,
            min_px=12,
        ),
        reference_to_options_gap_px=scale_puzzle_px(
            _int("reference_to_options_gap_px", 34),
            unit_size_scale,
            min_px=18,
        ),
        option_gap_px=scale_puzzle_px(
            _int("option_gap_px", 36),
            unit_size_scale,
            min_px=10,
        ),
        option_row_gap_px=scale_puzzle_px(
            _int("option_row_gap_px", 20),
            unit_size_scale,
            min_px=10,
        ),
        option_label_gap_px=scale_puzzle_px(
            _int("option_label_gap_px", 10),
            unit_size_scale,
            min_px=8,
        ),
        paper_corner_radius_px=scale_puzzle_px(
            _int("paper_corner_radius_px", 18),
            unit_size_scale,
            min_px=8,
        ),
        panel_corner_radius_px=int(_int("panel_corner_radius_px", 28)),
        border_width_px=scale_puzzle_px(
            _int("border_width_px", 3),
            unit_size_scale,
            min_px=2,
        ),
        option_label_font_size_px=scale_puzzle_px(
            _int("option_label_font_size_px", 28),
            unit_size_scale,
            min_px=20,
        ),
        panel_fill_rgb=_rgb("panel_fill_rgb", (248, 249, 252)),
        paper_fill_rgb=_rgb("paper_fill_rgb", (255, 252, 245)),
        paper_shadow_rgb=_rgb("paper_shadow_rgb", (236, 230, 215)),
        border_color_rgb=_rgb("border_color_rgb", (86, 94, 108)),
        text_color_rgb=_rgb("text_color_rgb", (30, 34, 40)),
        text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
        fold_line_rgb=_rgb("fold_line_rgb", (100, 116, 145)),
        grid_line_rgb=_rgb("grid_line_rgb", (209, 214, 223)),
        arrow_rgb=_rgb("arrow_rgb", (54, 102, 180)),
        instruction_fill_rgb=_rgb("instruction_fill_rgb", (238, 243, 250)),
        cut_hole_fill_rgb=_rgb("cut_hole_fill_rgb", (38, 45, 56)),
        cut_hole_outline_rgb=_rgb("cut_hole_outline_rgb", (255, 255, 255)),
        cut_hole_shape="circle",
        unit_size_scale=float(unit_size_scale),
        unit_size_jitter=dict(unit_size_jitter),
    )


__all__ = [
    "PaperFoldGenerationDefaults",
    "resolve_fold_axis",
    "resolve_generation_defaults",
    "resolve_render_params",
    "resolve_scene_variant",
]


# Fold-cut defaults.
@dataclass(frozen=True)
class PaperFoldCutGenerationDefaults:
    """Resolved generation bounds for one fold-cut task."""

    option_count_min: int
    option_count_max: int
    grid_size: int
    cut_count_min: int
    cut_count_max: int


def resolve_fold_cut_generation_defaults(
    generation_defaults: Mapping[str, Any],
) -> PaperFoldCutGenerationDefaults:
    """Resolve stable generation bounds from scene config."""

    return PaperFoldCutGenerationDefaults(
        option_count_min=int(group_default(generation_defaults, "option_count_min", 4)),
        option_count_max=int(group_default(generation_defaults, "option_count_max", 4)),
        grid_size=int(group_default(generation_defaults, "grid_size", 6)),
        cut_count_min=int(group_default(generation_defaults, "cut_count_min", 1)),
        cut_count_max=int(group_default(generation_defaults, "cut_count_max", 2)),
    )


def _fold_cut_support_from_config(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    options_key: str,
    fallback: Sequence[Any],
) -> tuple[Any, ...]:
    """Return one explicit support from params, config, or fallback."""

    raw = params.get(str(options_key), group_default(defaults, str(options_key), None))
    if raw is None:
        return tuple(fallback)
    values = tuple(item for item in raw if str(item).strip())
    if not values:
        raise ValueError(f"{options_key} must contain at least one value")
    return values


def _select_fold_cut_supported_value(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    explicit_key: str,
    weights_key: str,
    options_key: str,
    fallback: Sequence[Any],
    instance_seed: int,
    namespace: str,
) -> tuple[Any, dict[str, float]]:
    """Select one configured support value using explicit or weighted sampling."""

    support = _fold_cut_support_from_config(
        params,
        defaults,
        options_key=str(options_key),
        fallback=tuple(fallback),
    )
    support_keys = {str(item) for item in support}
    explicit = params.get(str(explicit_key), group_default(defaults, str(explicit_key), None))
    if explicit is not None:
        selected_key = str(explicit)
        if selected_key not in support_keys:
            raise ValueError(f"{explicit_key} must be one of {tuple(support_keys)}")
        selected = next(item for item in support if str(item) == selected_key)
        return selected, support_probability_map(support, selected=selected)

    raw_weights = params.get(str(weights_key), group_default(defaults, str(weights_key), None))
    weights = (
        {str(key): float(value) for key, value in raw_weights.items()}
        if isinstance(raw_weights, Mapping)
        else None
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = weighted_support_choice(
        rng,
        support,
        weights=weights,
        sort_keys=False,
    )
    return selected, dict(probabilities)


def resolve_fold_cut_fold_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Resolve one-fold versus two-fold puzzles as a generation axis."""

    selected, probabilities = _select_fold_cut_supported_value(
        params,
        generation_defaults,
        explicit_key="fold_count",
        weights_key="fold_count_weights",
        options_key="fold_count_options",
        fallback=SUPPORTED_FOLD_COUNTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.fold_count",
    )
    fold_count = int(selected)
    if fold_count not in set(SUPPORTED_FOLD_COUNTS):
        raise ValueError("fold_count must be 1 or 2")
    return int(fold_count), dict(probabilities)


def resolve_fold_cut_axis(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve vertical versus horizontal as a fold-orientation axis."""

    selected, probabilities = _select_fold_cut_supported_value(
        params,
        generation_defaults,
        explicit_key="fold_axis",
        weights_key="fold_axis_weights",
        options_key="fold_axis_options",
        fallback=SUPPORTED_FOLD_AXES,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.fold_axis",
    )
    return str(selected), dict(probabilities)


def resolve_fold_cut_scene_variant(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one fold-cut visual scene treatment."""

    selected, probabilities = _select_fold_cut_supported_value(
        params,
        generation_defaults,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        options_key="scene_variant_options",
        fallback=FOLD_SCENE_VARIANTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scene_variant",
    )
    return str(selected), dict(probabilities)


def resolve_cut_hole_shape(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the visual shape used for folded cuts and unfolded holes."""

    selected, probabilities = _select_fold_cut_supported_value(
        params,
        render_defaults,
        explicit_key="cut_hole_shape",
        weights_key="cut_hole_shape_weights",
        options_key="cut_hole_shape_options",
        fallback=SUPPORTED_CUT_HOLE_SHAPES,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.cut_hole_shape",
    )
    return str(selected), dict(probabilities)


def resolve_fold_cut_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[PuzzleFoldResultRenderParams, dict[str, float]]:
    """Resolve rendering parameters for one fold-cut scene."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(fallback),
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.fold_cut_render",
        )

    def _rgb(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            str(key),
            fallback,
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.fold_cut_render",
        )

    unit_size_scale, unit_size_jitter = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.sheet_transform.fold_cut.unit_size",
    )
    cut_hole_shape, cut_hole_shape_probabilities = resolve_cut_hole_shape(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return (
        PuzzleFoldResultRenderParams(
            canvas_width=int(_int("canvas_width", 1200)),
            canvas_height=int(_int("canvas_height", 940)),
            scene_margin_left_px=int(_int("scene_margin_left_px", 64)),
            scene_margin_right_px=int(_int("scene_margin_right_px", 64)),
            scene_margin_top_px=int(_int("scene_margin_top_px", 56)),
            scene_margin_bottom_px=int(_int("scene_margin_bottom_px", 56)),
            reference_panel_height_px=scale_puzzle_px(
                _int("reference_panel_height_px", 330),
                unit_size_scale,
                min_px=200,
            ),
            reference_panel_padding_px=scale_puzzle_px(
                _int("reference_panel_padding_px", 28),
                unit_size_scale,
                min_px=12,
            ),
            reference_to_options_gap_px=scale_puzzle_px(
                _int("reference_to_options_gap_px", 34),
                unit_size_scale,
                min_px=18,
            ),
            option_gap_px=scale_puzzle_px(
                _int("option_gap_px", 36),
                unit_size_scale,
                min_px=10,
            ),
            option_row_gap_px=scale_puzzle_px(
                _int("option_row_gap_px", 20),
                unit_size_scale,
                min_px=10,
            ),
            option_label_gap_px=scale_puzzle_px(
                _int("option_label_gap_px", 10),
                unit_size_scale,
                min_px=8,
            ),
            paper_corner_radius_px=scale_puzzle_px(
                _int("paper_corner_radius_px", 18),
                unit_size_scale,
                min_px=8,
            ),
            panel_corner_radius_px=int(_int("panel_corner_radius_px", 28)),
            border_width_px=scale_puzzle_px(
                _int("border_width_px", 3),
                unit_size_scale,
                min_px=2,
            ),
            option_label_font_size_px=scale_puzzle_px(
                _int("option_label_font_size_px", 28),
                unit_size_scale,
                min_px=20,
            ),
            panel_fill_rgb=_rgb("panel_fill_rgb", (248, 249, 252)),
            paper_fill_rgb=_rgb("paper_fill_rgb", (255, 252, 245)),
            paper_shadow_rgb=_rgb("paper_shadow_rgb", (236, 230, 215)),
            border_color_rgb=_rgb("border_color_rgb", (86, 94, 108)),
            text_color_rgb=_rgb("text_color_rgb", (30, 34, 40)),
            text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
            fold_line_rgb=_rgb("fold_line_rgb", (100, 116, 145)),
            grid_line_rgb=_rgb("grid_line_rgb", (209, 214, 223)),
            arrow_rgb=_rgb("arrow_rgb", (54, 102, 180)),
            instruction_fill_rgb=_rgb("instruction_fill_rgb", (238, 243, 250)),
            cut_hole_fill_rgb=_rgb("cut_hole_fill_rgb", (38, 45, 56)),
            cut_hole_outline_rgb=_rgb("cut_hole_outline_rgb", (255, 255, 255)),
            cut_hole_shape=str(cut_hole_shape),
            unit_size_scale=float(unit_size_scale),
            unit_size_jitter=dict(unit_size_jitter),
        ),
        dict(cut_hole_shape_probabilities),
    )



# Overlay defaults.
@dataclass(frozen=True)
class OverlayRenderParams:
    """Resolved rendering knobs for transparent-sheet overlay scenes."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    reference_panel_height_px: int
    reference_panel_padding_px: int
    source_paper_size_px: int
    source_gap_px: int
    reference_to_options_gap_px: int
    option_paper_size_px: int
    option_gap_px: int
    option_row_gap_px: int
    option_label_gap_px: int
    paper_corner_radius_px: int
    panel_corner_radius_px: int
    border_width_px: int
    option_label_font_size_px: int
    combine_symbol_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    paper_fill_rgb: Tuple[int, int, int]
    paper_shadow_rgb: Tuple[int, int, int]
    border_color_rgb: Tuple[int, int, int]
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    mark_fill_rgb: Tuple[int, int, int]
    mark_outline_rgb: Tuple[int, int, int]
    mark_shape: str
    mark_shape_probabilities: Dict[str, float]
    instruction_fill_rgb: Tuple[int, int, int]
    unit_size_jitter: Dict[str, Any]


def _overlay_support_from_config(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    options_key: str,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    """Return one explicit string support from params, config, or fallback."""

    raw = params.get(str(options_key), group_default(defaults, str(options_key), None))
    if raw is None:
        return tuple(str(item) for item in fallback)
    values = tuple(str(item).strip() for item in raw if str(item).strip())
    if not values:
        raise ValueError(f"{options_key} must contain at least one value")
    return values


def _select_overlay_supported_value(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    explicit_key: str,
    weights_key: str,
    options_key: str,
    fallback: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select one configured support value using explicit or weighted sampling."""

    support = _overlay_support_from_config(
        params,
        defaults,
        options_key=str(options_key),
        fallback=tuple(fallback),
    )
    explicit = params.get(str(explicit_key), group_default(defaults, str(explicit_key), None))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(support):
            raise ValueError(f"{explicit_key} must be one of {support}")
        return selected, support_probability_map(support, selected=selected)

    raw_weights = params.get(str(weights_key), group_default(defaults, str(weights_key), None))
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
    } if isinstance(raw_weights, Mapping) else None
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = weighted_support_choice(
        rng,
        support,
        weights=weights,
        sort_keys=False,
    )
    return str(selected), dict(probabilities)


def resolve_overlay_scene_variant(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one overlay visual scene treatment."""

    return _select_overlay_supported_value(
        params,
        generation_defaults,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        options_key="scene_variant_options",
        fallback=OVERLAY_SCENE_VARIANTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scene_variant",
    )


def resolve_overlay_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> OverlayRenderParams:
    """Resolve rendering parameters and visual mark-style metadata."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(fallback),
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.overlay_render",
        )

    def _triple(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            str(key),
            fallback,
            instance_seed=int(instance_seed),
            namespace="puzzles.sheet_transform.overlay_render",
        )

    mark_shape, mark_shape_probabilities = _select_overlay_supported_value(
        params,
        render_defaults,
        explicit_key="mark_shape",
        weights_key="mark_shape_weights",
        options_key="mark_shape_options",
        fallback=OVERLAY_MARK_SHAPES,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.mark_shape",
    )
    unit_size_scale, unit_size_jitter = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.sheet_transform.overlay.unit_size",
    )

    return OverlayRenderParams(
        canvas_width=int(_int("canvas_width", 1200)),
        canvas_height=int(_int("canvas_height", 860)),
        scene_margin_left_px=int(_int("scene_margin_left_px", 64)),
        scene_margin_right_px=int(_int("scene_margin_right_px", 64)),
        scene_margin_top_px=int(_int("scene_margin_top_px", 56)),
        scene_margin_bottom_px=int(_int("scene_margin_bottom_px", 56)),
        reference_panel_height_px=scale_puzzle_px(
            _int("reference_panel_height_px", 294),
            unit_size_scale,
            min_px=200,
        ),
        reference_panel_padding_px=scale_puzzle_px(
            _int("reference_panel_padding_px", 24),
            unit_size_scale,
            min_px=12,
        ),
        source_paper_size_px=scale_puzzle_px(
            _int("source_paper_size_px", 158),
            unit_size_scale,
            min_px=150,
        ),
        source_gap_px=scale_puzzle_px(
            _int("source_gap_px", 120),
            unit_size_scale,
            min_px=48,
        ),
        reference_to_options_gap_px=scale_puzzle_px(
            _int("reference_to_options_gap_px", 34),
            unit_size_scale,
            min_px=18,
        ),
        option_paper_size_px=scale_puzzle_px(
            _int("option_paper_size_px", 158),
            unit_size_scale,
            min_px=150,
        ),
        option_gap_px=scale_puzzle_px(
            _int("option_gap_px", 22),
            unit_size_scale,
            min_px=12,
        ),
        option_row_gap_px=scale_puzzle_px(
            _int("option_row_gap_px", 22),
            unit_size_scale,
            min_px=12,
        ),
        option_label_gap_px=scale_puzzle_px(
            _int("option_label_gap_px", 12),
            unit_size_scale,
            min_px=8,
        ),
        paper_corner_radius_px=scale_puzzle_px(
            _int("paper_corner_radius_px", 18),
            unit_size_scale,
            min_px=8,
        ),
        panel_corner_radius_px=int(_int("panel_corner_radius_px", 28)),
        border_width_px=scale_puzzle_px(
            _int("border_width_px", 3),
            unit_size_scale,
            min_px=2,
        ),
        option_label_font_size_px=scale_puzzle_px(
            _int("option_label_font_size_px", 28),
            unit_size_scale,
            min_px=20,
        ),
        combine_symbol_font_size_px=scale_puzzle_px(
            _int("combine_symbol_font_size_px", 46),
            unit_size_scale,
            min_px=24,
        ),
        panel_fill_rgb=_triple("panel_fill_rgb", (248, 249, 252)),
        paper_fill_rgb=_triple("paper_fill_rgb", (255, 252, 245)),
        paper_shadow_rgb=_triple("paper_shadow_rgb", (236, 230, 215)),
        border_color_rgb=_triple("border_color_rgb", (86, 94, 108)),
        text_color_rgb=_triple("text_color_rgb", (30, 34, 40)),
        text_stroke_rgb=_triple("text_stroke_rgb", (255, 255, 255)),
        mark_fill_rgb=_triple("mark_fill_rgb", (53, 96, 164)),
        mark_outline_rgb=_triple("mark_outline_rgb", (36, 48, 66)),
        mark_shape=str(mark_shape),
        mark_shape_probabilities=dict(mark_shape_probabilities),
        instruction_fill_rgb=_triple("instruction_fill_rgb", (238, 243, 250)),
        unit_size_jitter=dict(unit_size_jitter),
    )



__all__ = [
    "OverlayRenderParams",
    "PaperFoldCutGenerationDefaults",
    "PaperFoldGenerationDefaults",
    "resolve_cut_hole_shape",
    "resolve_fold_axis",
    "resolve_fold_cut_axis",
    "resolve_fold_cut_fold_count",
    "resolve_fold_cut_generation_defaults",
    "resolve_fold_cut_render_params",
    "resolve_fold_cut_scene_variant",
    "resolve_generation_defaults",
    "resolve_overlay_render_params",
    "resolve_overlay_scene_variant",
    "resolve_render_params",
    "resolve_scene_variant",
]
