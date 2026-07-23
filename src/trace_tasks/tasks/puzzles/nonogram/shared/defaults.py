"""Config and render defaults for nonogram puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.sampling import (
    support_probability_map,
    uniform_choice_with_probabilities,
)
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)

from .state import OPTION_COUNTS, SCENE_VARIANTS


@dataclass(frozen=True)
class NonogramRenderParams:
    """Resolved render controls for one nonogram scene."""

    canvas_width: int
    canvas_height: int
    margin_top_px: int
    left_clue_width_px: int
    top_clue_height_px: int
    cell_size_px: int
    grid_line_width_px: int
    heavy_line_width_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    option_y_px: int
    panel_corner_radius_px: int
    clue_font_size_px: int
    option_label_font_size_px: int
    question_font_size_px: int
    unit_size_jitter: dict[str, Any]


def _int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _int_range(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    low = _int_param(params, defaults, min_key, fallback_min)
    high = _int_param(params, defaults, max_key, fallback_max)
    if int(low) > int(high):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(low), int(high)


def resolve_grid_size(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    rng,
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    """Resolve nonogram row/column counts from params/defaults."""

    row_range = _int_range(
        params,
        generation_defaults,
        min_key="grid_rows_min",
        max_key="grid_rows_max",
        fallback_min=6,
        fallback_max=9,
    )
    col_range = _int_range(
        params,
        generation_defaults,
        min_key="grid_cols_min",
        max_key="grid_cols_max",
        fallback_min=6,
        fallback_max=9,
    )
    rows = int(params.get("grid_rows", rng.randint(row_range[0], row_range[1])))
    cols = int(params.get("grid_cols", rng.randint(col_range[0], col_range[1])))
    if not row_range[0] <= int(rows) <= row_range[1]:
        raise ValueError("grid_rows falls outside configured range")
    if not col_range[0] <= int(cols) <= col_range[1]:
        raise ValueError("grid_cols falls outside configured range")
    return int(rows), int(cols), tuple(row_range), tuple(col_range)


def resolve_option_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    rng,
) -> tuple[int, dict[str, float]]:
    """Sample a supported option count uniformly unless explicitly pinned."""

    support = tuple(
        int(value)
        for value in generation_defaults.get("option_count_choices", OPTION_COUNTS)
    )
    if not support:
        raise ValueError("nonogram option_count_choices cannot be empty")
    explicit = params.get("option_count")
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported nonogram option_count: {explicit}")
        return (
            int(selected),
            support_probability_map(support, selected=selected, sort_keys=True),
        )
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        support,
        sort_keys=True,
    )
    return int(selected), dict(probabilities)


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Sample one nonogram visual variant uniformly unless explicitly pinned."""

    explicit = params.get("scene_variant")
    if explicit is not None:
        selected = str(explicit).strip()
        if selected not in set(SCENE_VARIANTS):
            raise ValueError(f"unsupported scene_variant: {explicit}")
        return (
            str(selected),
            support_probability_map(SCENE_VARIANTS, selected=selected, sort_keys=True),
        )
    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        SCENE_VARIANTS,
        sort_keys=True,
    )
    return str(selected), dict(probabilities)


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> NonogramRenderParams:
    """Resolve render parameters with puzzle unit-size jitter metadata."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.nonogram.unit_size",
    )
    return NonogramRenderParams(
        canvas_width=_int_param(params, rendering_defaults, "canvas_width", 1100),
        canvas_height=_int_param(params, rendering_defaults, "canvas_height", 820),
        margin_top_px=_int_param(params, rendering_defaults, "margin_top_px", 50),
        left_clue_width_px=_int_param(params, rendering_defaults, "left_clue_width_px", 156),
        top_clue_height_px=_int_param(params, rendering_defaults, "top_clue_height_px", 108),
        cell_size_px=scale_puzzle_px(
            rendering_defaults.get("cell_size_px", 46),
            unit_scale,
            min_px=28,
        ),
        grid_line_width_px=scale_puzzle_px(
            rendering_defaults.get("grid_line_width_px", 2),
            unit_scale,
            min_px=1,
        ),
        heavy_line_width_px=scale_puzzle_px(
            rendering_defaults.get("heavy_line_width_px", 4),
            unit_scale,
            min_px=2,
        ),
        option_panel_width_px=scale_puzzle_px(
            rendering_defaults.get("option_panel_width_px", 146),
            unit_scale,
            min_px=112,
        ),
        option_panel_height_px=scale_puzzle_px(
            rendering_defaults.get("option_panel_height_px", 126),
            unit_scale,
            min_px=92,
        ),
        option_gap_px=scale_puzzle_px(
            rendering_defaults.get("option_gap_px", 14),
            unit_scale,
            min_px=8,
        ),
        option_y_px=_int_param(params, rendering_defaults, "option_y_px", 650),
        panel_corner_radius_px=scale_puzzle_px(
            rendering_defaults.get("panel_corner_radius_px", 10),
            unit_scale,
            min_px=5,
        ),
        clue_font_size_px=scale_puzzle_px(
            rendering_defaults.get("clue_font_size_px", 21),
            unit_scale,
            min_px=14,
        ),
        option_label_font_size_px=scale_puzzle_px(
            rendering_defaults.get("option_label_font_size_px", 23),
            unit_scale,
            min_px=14,
        ),
        question_font_size_px=_int_param(
            params,
            rendering_defaults,
            "question_font_size_px",
            28,
        ),
        unit_size_jitter=dict(unit_meta),
    )


def sample_nonogram_font(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> str:
    """Sample one global font family for clue text and option labels."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.label_font",
        params={**dict(rendering_defaults), **dict(params)},
    )


def font_trace_record(font_family: str) -> dict[str, Any]:
    """Build trace metadata for the sampled nonogram font."""

    return {
        **get_font_family_record(str(font_family)).to_trace(),
        "source": "global_font_pool",
        "font_asset_version": font_asset_version(),
        "scope": "nonogram_clues_and_option_labels",
    }


__all__ = [
    "NonogramRenderParams",
    "font_trace_record",
    "resolve_grid_size",
    "resolve_option_count",
    "resolve_render_params",
    "resolve_scene_variant",
    "sample_nonogram_font",
]
