"""Config and visual-style defaults for Tents puzzle scenes."""

from __future__ import annotations

from dataclasses import replace
from math import floor
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.sampling import weighted_support_choice
from trace_tasks.tasks.puzzles.shared.common import get_int_range
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.color_distance import coerce_rgb as _rgb
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import (
    SUPPORTED_PALETTE_VARIANTS,
    SUPPORTED_SCENE_VARIANTS,
    TentsRenderParams,
)

STYLE_COLORS = {
    "tents_classic": {
        "panel_fill": (249, 248, 241),
        "grid_fill": (255, 253, 245),
        "cell_a": (253, 252, 245),
        "cell_b": (247, 250, 240),
        "grid_line": (158, 147, 120),
        "heavy_line": (92, 86, 72),
        "clue_fill": (238, 234, 217),
        "candidate_fill": (255, 240, 168),
        "candidate_outline": (140, 112, 38),
        "tree_fill": (55, 132, 83),
        "tree_outline": (33, 84, 54),
        "tent_fill": (202, 91, 75),
        "tent_shadow": (130, 62, 54),
        "accent": (50, 86, 150),
    },
    "tents_card": {
        "panel_fill": (244, 249, 251),
        "grid_fill": (250, 253, 255),
        "cell_a": (251, 253, 255),
        "cell_b": (242, 248, 252),
        "grid_line": (155, 178, 192),
        "heavy_line": (69, 91, 108),
        "clue_fill": (226, 238, 246),
        "candidate_fill": (255, 236, 164),
        "candidate_outline": (127, 99, 34),
        "tree_fill": (48, 139, 112),
        "tree_outline": (31, 86, 74),
        "tent_fill": (197, 93, 111),
        "tent_shadow": (122, 58, 72),
        "accent": (53, 103, 177),
    },
    "tents_blueprint": {
        "panel_fill": (239, 246, 252),
        "grid_fill": (247, 252, 255),
        "cell_a": (250, 253, 255),
        "cell_b": (240, 248, 253),
        "grid_line": (139, 171, 196),
        "heavy_line": (43, 82, 118),
        "clue_fill": (222, 237, 248),
        "candidate_fill": (255, 239, 150),
        "candidate_outline": (124, 96, 30),
        "tree_fill": (55, 128, 112),
        "tree_outline": (30, 78, 68),
        "tent_fill": (189, 87, 95),
        "tent_shadow": (115, 52, 60),
        "accent": (37, 90, 150),
    },
}

PALETTE_COLORS = {
    "garden": {
        "clue_fill": (238, 234, 217),
        "candidate_fill": (255, 240, 168),
        "candidate_outline": (140, 112, 38),
        "candidate_label_fill": (255, 250, 215),
        "tree_fill": (55, 132, 83),
        "tree_outline": (33, 84, 54),
        "tent_fill": (202, 91, 75),
        "tent_shadow": (130, 62, 54),
        "tent_flap_fill": (245, 178, 142),
    },
    "autumn": {
        "clue_fill": (244, 228, 205),
        "candidate_fill": (255, 225, 153),
        "candidate_outline": (145, 91, 32),
        "candidate_label_fill": (255, 246, 212),
        "tree_fill": (136, 115, 48),
        "tree_outline": (83, 72, 34),
        "tent_fill": (183, 86, 47),
        "tent_shadow": (111, 55, 33),
        "tent_flap_fill": (244, 177, 116),
    },
    "lake": {
        "clue_fill": (222, 239, 242),
        "candidate_fill": (187, 231, 232),
        "candidate_outline": (39, 108, 125),
        "candidate_label_fill": (231, 249, 248),
        "tree_fill": (46, 130, 124),
        "tree_outline": (24, 78, 78),
        "tent_fill": (64, 107, 182),
        "tent_shadow": (39, 68, 119),
        "tent_flap_fill": (169, 207, 244),
    },
    "violet": {
        "clue_fill": (235, 229, 247),
        "candidate_fill": (226, 211, 250),
        "candidate_outline": (99, 76, 154),
        "candidate_label_fill": (246, 240, 255),
        "tree_fill": (64, 128, 101),
        "tree_outline": (34, 76, 62),
        "tent_fill": (131, 92, 183),
        "tent_shadow": (82, 57, 118),
        "tent_flap_fill": (212, 190, 240),
    },
    "slate": {
        "clue_fill": (229, 234, 238),
        "candidate_fill": (239, 219, 155),
        "candidate_outline": (93, 80, 51),
        "candidate_label_fill": (251, 246, 225),
        "tree_fill": (73, 118, 88),
        "tree_outline": (42, 70, 54),
        "tent_fill": (100, 115, 135),
        "tent_shadow": (57, 68, 83),
        "tent_flap_fill": (192, 202, 214),
    },
}


def resolve_grid_size(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
    """Sample or pin one Tents grid size from configured row/column ranges."""

    row_range = get_int_range(
        params,
        generation_defaults,
        min_key="grid_rows_min",
        max_key="grid_rows_max",
        fallback_min=6,
        fallback_max=8,
    )
    col_range = get_int_range(
        params,
        generation_defaults,
        min_key="grid_cols_min",
        max_key="grid_cols_max",
        fallback_min=6,
        fallback_max=8,
    )
    rows = int(
        params.get("grid_rows", rng.randint(int(row_range[0]), int(row_range[1])))
    )
    cols = int(
        params.get("grid_cols", rng.randint(int(col_range[0]), int(col_range[1])))
    )
    if not int(row_range[0]) <= int(rows) <= int(row_range[1]):
        raise ValueError("grid_rows falls outside configured range")
    if not int(col_range[0]) <= int(cols) <= int(col_range[1]):
        raise ValueError("grid_cols falls outside configured range")
    return int(rows), int(cols), tuple(row_range), tuple(col_range)


def resolve_scene_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[str, Dict[str, float]]:
    """Sample a visual Tents scene variant from explicit support."""

    explicit = params.get("scene_variant")
    if explicit is not None:
        value = str(explicit)
        if value not in SUPPORTED_SCENE_VARIANTS:
            raise ValueError(f"unsupported tents scene_variant: {value}")
        return value, {
            item: (1.0 if item == value else 0.0) for item in SUPPORTED_SCENE_VARIANTS
        }
    return weighted_support_choice(
        rng,
        SUPPORTED_SCENE_VARIANTS,
        weights=generation_defaults.get("scene_variant_weights"),
    )


def resolve_palette_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[str, Dict[str, float]]:
    """Sample a Tents color palette from explicit support."""

    explicit = params.get("palette_variant")
    if explicit is not None:
        value = str(explicit)
        if value not in SUPPORTED_PALETTE_VARIANTS:
            raise ValueError(f"unsupported tents palette_variant: {value}")
        return value, {
            item: (1.0 if item == value else 0.0) for item in SUPPORTED_PALETTE_VARIANTS
        }
    return weighted_support_choice(
        rng,
        SUPPORTED_PALETTE_VARIANTS,
        weights=generation_defaults.get("palette_variant_weights"),
    )


def resolve_option_count(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback: int,
) -> int:
    """Resolve the number of candidate labels for option-style Tents tasks."""

    count = int(
        params.get(
            "option_count",
            group_default(generation_defaults, "option_count", int(fallback)),
        )
    )
    return max(4, min(6, int(count)))


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> TentsRenderParams:
    """Resolve render dimensions, font sizes, and unit-size jitter."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.tents.unit_size",
    )
    render_params = TentsRenderParams(
        canvas_width=int(render_defaults.get("canvas_width", 1100)),
        canvas_height=int(render_defaults.get("canvas_height", 860)),
        cell_size_px=scale_puzzle_px(
            render_defaults.get("cell_size_px", 62), unit_scale, min_px=22
        ),
        left_clue_width_px=scale_puzzle_px(
            render_defaults.get("left_clue_width_px", 78), unit_scale, min_px=38
        ),
        top_clue_height_px=scale_puzzle_px(
            render_defaults.get("top_clue_height_px", 78), unit_scale, min_px=38
        ),
        grid_line_width_px=scale_puzzle_px(
            render_defaults.get("grid_line_width_px", 2), unit_scale, min_px=1
        ),
        heavy_line_width_px=scale_puzzle_px(
            render_defaults.get("heavy_line_width_px", 4), unit_scale, min_px=2
        ),
        panel_padding_px=scale_puzzle_px(
            render_defaults.get("panel_padding_px", 26), unit_scale, min_px=12
        ),
        panel_corner_radius_px=scale_puzzle_px(
            render_defaults.get("panel_corner_radius_px", 18), unit_scale, min_px=7
        ),
        clue_font_size_px=scale_puzzle_px(
            render_defaults.get("clue_font_size_px", 28), unit_scale, min_px=14
        ),
        candidate_font_size_px=scale_puzzle_px(
            render_defaults.get("candidate_font_size_px", 30), unit_scale, min_px=14
        ),
        text_color_rgb=_rgb(render_defaults.get("text_color_rgb"), (28, 32, 38)),
        text_stroke_rgb=_rgb(render_defaults.get("text_stroke_rgb"), (255, 255, 255)),
        style_overrides={},
        unit_size_jitter=dict(unit_meta),
    )
    return _fit_render_params_to_canvas(
        render_params,
        rows=grid_rows,
        cols=grid_cols,
    )


def _fit_render_params_to_canvas(
    render_params: TentsRenderParams,
    *,
    rows: int | None,
    cols: int | None,
) -> TentsRenderParams:
    """Shrink repeated Tents units when a sampled grid would exceed the canvas."""

    if rows is None or cols is None or int(rows) <= 0 or int(cols) <= 0:
        return render_params

    rows = int(rows)
    cols = int(cols)
    total_width = (
        int(render_params.left_clue_width_px)
        + (cols * int(render_params.cell_size_px))
        + (2 * int(render_params.panel_padding_px))
    )
    total_height = (
        int(render_params.top_clue_height_px)
        + (rows * int(render_params.cell_size_px))
        + (2 * int(render_params.panel_padding_px))
    )
    fit_scale = min(
        1.0,
        float(render_params.canvas_width) / max(1.0, float(total_width)),
        float(render_params.canvas_height) / max(1.0, float(total_height)),
    )
    if fit_scale >= 1.0:
        return render_params

    def scaled(value: int, *, min_px: int) -> int:
        return max(int(min_px), int(floor(float(value) * float(fit_scale))))

    fitted = replace(
        render_params,
        cell_size_px=scaled(render_params.cell_size_px, min_px=22),
        left_clue_width_px=scaled(render_params.left_clue_width_px, min_px=32),
        top_clue_height_px=scaled(render_params.top_clue_height_px, min_px=32),
        grid_line_width_px=scaled(render_params.grid_line_width_px, min_px=1),
        heavy_line_width_px=scaled(render_params.heavy_line_width_px, min_px=2),
        panel_padding_px=scaled(render_params.panel_padding_px, min_px=8),
        panel_corner_radius_px=scaled(render_params.panel_corner_radius_px, min_px=6),
        clue_font_size_px=scaled(render_params.clue_font_size_px, min_px=12),
        candidate_font_size_px=scaled(render_params.candidate_font_size_px, min_px=12),
    )

    max_cell_width = int(
        floor(
            (
                float(fitted.canvas_width)
                - float(fitted.left_clue_width_px)
                - (2.0 * float(fitted.panel_padding_px))
            )
            / float(cols)
        )
    )
    max_cell_height = int(
        floor(
            (
                float(fitted.canvas_height)
                - float(fitted.top_clue_height_px)
                - (2.0 * float(fitted.panel_padding_px))
            )
            / float(rows)
        )
    )
    max_fit_cell_size = min(int(max_cell_width), int(max_cell_height))
    if max_fit_cell_size < 22:
        raise ValueError(
            "Tents grid cannot fit within the configured canvas while preserving "
            "the minimum cell size"
        )
    fitted_cell_size = min(int(fitted.cell_size_px), int(max_fit_cell_size))
    meta = dict(fitted.unit_size_jitter)
    meta["fit_to_canvas"] = {
        "enabled": True,
        "scale": float(fit_scale),
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "content_size_before_fit_px": [int(total_width), int(total_height)],
        "content_size_after_fit_px": [
            int(fitted.left_clue_width_px)
            + (int(cols) * int(fitted_cell_size))
            + (2 * int(fitted.panel_padding_px)),
            int(fitted.top_clue_height_px)
            + (int(rows) * int(fitted_cell_size))
            + (2 * int(fitted.panel_padding_px)),
        ],
    }
    return replace(fitted, cell_size_px=int(fitted_cell_size), unit_size_jitter=meta)


__all__ = [
    "PALETTE_COLORS",
    "STYLE_COLORS",
    "resolve_grid_size",
    "resolve_option_count",
    "resolve_palette_variant",
    "resolve_render_params",
    "resolve_scene_variant",
]
