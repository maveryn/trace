"""Config and visual-axis defaults for toggle-grid puzzle scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.sampling import integer_range_choice, weighted_support_choice
from trace_tasks.tasks.puzzles.shared.common import get_int_range
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import SUPPORTED_SCENE_VARIANTS, ToggleRenderParams


def resolve_grid_size(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
    """Sample a row/column size from configured inclusive ranges."""

    row_range = get_int_range(
        params,
        generation_defaults,
        min_key="grid_rows_min",
        max_key="grid_rows_max",
        fallback_min=4,
        fallback_max=5,
    )
    col_range = get_int_range(
        params,
        generation_defaults,
        min_key="grid_cols_min",
        max_key="grid_cols_max",
        fallback_min=4,
        fallback_max=5,
    )
    rows, _row_probs = integer_range_choice(rng, int(row_range[0]), int(row_range[1]))
    cols, _col_probs = integer_range_choice(rng, int(col_range[0]), int(col_range[1]))
    return int(rows), int(cols), tuple(row_range), tuple(col_range)


def resolve_press_count(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[int, Tuple[int, int]]:
    """Sample the number of sequential presses shown in the result task."""

    press_range = get_int_range(
        params,
        generation_defaults,
        min_key="result_press_count_min",
        max_key="result_press_count_max",
        fallback_min=2,
        fallback_max=4,
    )
    press_count, _probabilities = integer_range_choice(
        rng,
        int(press_range[0]),
        int(press_range[1]),
    )
    return int(press_count), tuple(press_range)


def resolve_option_count(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback: int = 5,
) -> int:
    """Resolve the number of labeled options rendered for one task."""

    option_count = int(
        params.get(
            "option_count",
            group_default(generation_defaults, "option_count", int(fallback)),
        )
    )
    if not 2 <= int(option_count) <= 8:
        raise ValueError("toggle option_count must be in 2..8")
    return int(option_count)


def resolve_scene_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> Tuple[str, Dict[str, float]]:
    """Sample a visual scene variant from explicit support."""

    explicit = params.get("scene_variant")
    if explicit is not None:
        value = str(explicit)
        if value not in SUPPORTED_SCENE_VARIANTS:
            raise ValueError(f"unsupported toggle scene_variant: {value}")
        return value, {
            item: (1.0 if item == value else 0.0)
            for item in SUPPORTED_SCENE_VARIANTS
        }
    return weighted_support_choice(
        rng,
        SUPPORTED_SCENE_VARIANTS,
        weights=generation_defaults.get("scene_variant_weights"),
    )


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> ToggleRenderParams:
    """Resolve concrete render dimensions and font sizes."""

    return ToggleRenderParams(
        canvas_width=int(
            params.get(
                "canvas_width",
                group_default(rendering_defaults, "canvas_width", 1120),
            )
        ),
        canvas_height=int(
            params.get(
                "canvas_height",
                group_default(rendering_defaults, "canvas_height", 860),
            )
        ),
        main_cell_size_px=int(
            params.get(
                "main_cell_size_px",
                group_default(rendering_defaults, "main_cell_size_px", 72),
            )
        ),
        mini_cell_size_px=int(
            params.get(
                "mini_cell_size_px",
                group_default(rendering_defaults, "mini_cell_size_px", 28),
            )
        ),
        panel_title_font_size_px=int(
            params.get(
                "panel_title_font_size_px",
                group_default(rendering_defaults, "panel_title_font_size_px", 22),
            )
        ),
        option_font_size_px=int(
            params.get(
                "option_font_size_px",
                group_default(rendering_defaults, "option_font_size_px", 22),
            )
        ),
    )
