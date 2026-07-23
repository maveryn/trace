"""Configuration resolution for pipe-flow repair puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.sampling import (
    integer_range_choice,
    support_probability_map,
    uniform_choice_with_probabilities,
    weighted_support_choice,
)
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import (
    GAP_SIZE_VARIANTS,
    GRID_SIZE_VARIANTS,
    LABEL_POOL,
    SCENE_VARIANTS,
    Color,
    RenderParams,
)


@dataclass(frozen=True)
class PipeFlowDefaults:
    """Fallback defaults for the pipe-flow scene."""

    grid_size_min: int = 5
    grid_size_max: int = 7
    path_length_min: int = 8
    path_length_max: int = 16
    candidate_count_min: int = 4
    candidate_count_max: int = 4
    branch_count_min: int = 0
    branch_count_max: int = 0
    branch_length_min: int = 0
    branch_length_max: int = 0
    canvas_width: int = 760
    canvas_height: int = 720
    scene_margin_px: int = 48
    panel_padding_px: int = 18
    panel_corner_radius_px: int = 22
    panel_border_width_px: int = 3
    cell_gap_px: int = 3
    cell_size_min_px: int = 24
    cell_size_max_px: int = 34
    cell_border_width_px: int = 2
    pipe_width_px: int = 10
    source_dest_font_size_px: int = 15
    tile_label_font_size_px: int = 16
    panel_fill_rgb: Color = (248, 250, 252)
    cell_fill_rgb: Color = (255, 255, 255)
    grid_line_rgb: Color = (196, 203, 215)
    pipe_rgb: Color = (57, 143, 205)
    pipe_shadow_rgb: Color = (224, 232, 242)
    source_fill_rgb: Color = (203, 238, 224)
    source_outline_rgb: Color = (41, 120, 83)
    dest_fill_rgb: Color = (251, 229, 177)
    dest_outline_rgb: Color = (164, 107, 28)
    label_fill_rgb: Color = (35, 42, 54)
    label_text_rgb: Color = (255, 255, 255)
    text_stroke_rgb: Color = (255, 255, 255)


DEFAULTS = PipeFlowDefaults()


def to_int(value: Any, fallback: int) -> int:
    """Coerce a config value to int, falling back on invalid input."""

    try:
        return int(value)
    except Exception:
        return int(fallback)


def rgb(value: Any, fallback: Color) -> Color:
    """Resolve one RGB color config value."""

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, to_int(value[0], fallback[0]))),
            max(0, min(255, to_int(value[1], fallback[1]))),
            max(0, min(255, to_int(value[2], fallback[2]))),
        )
    return tuple(int(channel) for channel in fallback)


def rgb_option(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Color,
    *,
    seed: int,
) -> Color:
    """Resolve a color, sampling from the configured option list if present."""

    options = params.get(f"{key}_options", group_default(defaults, f"{key}_options", None))
    if isinstance(options, list) and options:
        rng = spawn_rng(int(seed), f"puzzles.pipe_flow.render.{key}")
        selected, _probabilities = weighted_support_choice(
            rng,
            tuple(range(len(options))),
            sort_keys=True,
        )
        return rgb(options[int(selected)], fallback)
    return rgb(params.get(str(key), group_default(defaults, str(key), fallback)), fallback)


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> RenderParams:
    """Resolve pipe-flow render dimensions, colors, and unit-size jitter."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.pipe_flow.unit_size",
    )

    def scaled_int(key: str, fallback: int, *, min_px: int) -> int:
        raw = to_int(params.get(key, group_default(render_defaults, key, fallback)), fallback)
        return scale_puzzle_px(raw, unit_scale, min_px=int(min_px))

    cell_size_min_px = max(
        1,
        to_int(
            params.get(
                "cell_size_min_px",
                group_default(render_defaults, "cell_size_min_px", DEFAULTS.cell_size_min_px),
            ),
            DEFAULTS.cell_size_min_px,
        ),
    )
    cell_size_max_px = max(
        cell_size_min_px,
        to_int(
            params.get(
                "cell_size_max_px",
                group_default(render_defaults, "cell_size_max_px", DEFAULTS.cell_size_max_px),
            ),
            DEFAULTS.cell_size_max_px,
        ),
    )

    return RenderParams(
        canvas_width=max(
            760,
            to_int(
                params.get(
                    "canvas_width",
                    group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width),
                ),
                DEFAULTS.canvas_width,
            ),
        ),
        canvas_height=max(
            640,
            to_int(
                params.get(
                    "canvas_height",
                    group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height),
                ),
                DEFAULTS.canvas_height,
            ),
        ),
        scene_margin_px=max(48, scaled_int("scene_margin_px", DEFAULTS.scene_margin_px, min_px=36)),
        panel_padding_px=max(12, scaled_int("panel_padding_px", DEFAULTS.panel_padding_px, min_px=12)),
        panel_corner_radius_px=max(
            0,
            scaled_int("panel_corner_radius_px", DEFAULTS.panel_corner_radius_px, min_px=8),
        ),
        panel_border_width_px=max(
            1,
            scaled_int("panel_border_width_px", DEFAULTS.panel_border_width_px, min_px=1),
        ),
        cell_gap_px=max(0, scaled_int("cell_gap_px", DEFAULTS.cell_gap_px, min_px=2)),
        cell_size_min_px=int(cell_size_min_px),
        cell_size_max_px=int(cell_size_max_px),
        cell_border_width_px=max(
            1,
            scaled_int("cell_border_width_px", DEFAULTS.cell_border_width_px, min_px=1),
        ),
        pipe_width_px=max(8, scaled_int("pipe_width_px", DEFAULTS.pipe_width_px, min_px=8)),
        source_dest_font_size_px=max(
            12,
            scaled_int(
                "source_dest_font_size_px",
                DEFAULTS.source_dest_font_size_px,
                min_px=12,
            ),
        ),
        tile_label_font_size_px=max(
            12,
            scaled_int(
                "tile_label_font_size_px",
                DEFAULTS.tile_label_font_size_px,
                min_px=12,
            ),
        ),
        panel_fill_rgb=rgb_option(
            params,
            render_defaults,
            "panel_fill_rgb",
            DEFAULTS.panel_fill_rgb,
            seed=int(instance_seed),
        ),
        cell_fill_rgb=rgb_option(
            params,
            render_defaults,
            "cell_fill_rgb",
            DEFAULTS.cell_fill_rgb,
            seed=int(instance_seed),
        ),
        grid_line_rgb=rgb_option(
            params,
            render_defaults,
            "grid_line_rgb",
            DEFAULTS.grid_line_rgb,
            seed=int(instance_seed),
        ),
        pipe_rgb=rgb_option(
            params,
            render_defaults,
            "pipe_rgb",
            DEFAULTS.pipe_rgb,
            seed=int(instance_seed),
        ),
        pipe_shadow_rgb=rgb_option(
            params,
            render_defaults,
            "pipe_shadow_rgb",
            DEFAULTS.pipe_shadow_rgb,
            seed=int(instance_seed),
        ),
        source_fill_rgb=rgb_option(
            params,
            render_defaults,
            "source_fill_rgb",
            DEFAULTS.source_fill_rgb,
            seed=int(instance_seed),
        ),
        source_outline_rgb=rgb_option(
            params,
            render_defaults,
            "source_outline_rgb",
            DEFAULTS.source_outline_rgb,
            seed=int(instance_seed),
        ),
        dest_fill_rgb=rgb_option(
            params,
            render_defaults,
            "dest_fill_rgb",
            DEFAULTS.dest_fill_rgb,
            seed=int(instance_seed),
        ),
        dest_outline_rgb=rgb_option(
            params,
            render_defaults,
            "dest_outline_rgb",
            DEFAULTS.dest_outline_rgb,
            seed=int(instance_seed),
        ),
        label_fill_rgb=rgb_option(
            params,
            render_defaults,
            "label_fill_rgb",
            DEFAULTS.label_fill_rgb,
            seed=int(instance_seed),
        ),
        label_text_rgb=rgb_option(
            params,
            render_defaults,
            "label_text_rgb",
            DEFAULTS.label_text_rgb,
            seed=int(instance_seed),
        ),
        text_stroke_rgb=rgb_option(
            params,
            render_defaults,
            "text_stroke_rgb",
            DEFAULTS.text_stroke_rgb,
            seed=int(instance_seed),
        ),
        unit_size_jitter=dict(unit_meta),
    )


def resolve_axis_choice(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    supported_values: tuple[str, ...],
    explicit_key: str,
    weights_key: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one explicit-or-weighted generation axis."""

    supported = tuple(str(value) for value in supported_values)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = str(explicit).strip()
        if selected not in set(supported):
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return selected, support_probability_map(supported, selected=selected)

    raw_weights = params.get(
        str(weights_key),
        group_default(gen_defaults, str(weights_key), {value: 1.0 for value in supported}),
    )
    if not isinstance(raw_weights, Mapping):
        raw_weights = {value: 1.0 for value in supported}
    selected, probabilities = weighted_support_choice(
        rng,
        supported,
        weights={str(key): float(value) for key, value in raw_weights.items()},
        sort_keys=False,
    )
    return str(selected), dict(probabilities)


def resolve_grid_size_variant(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the grid-size variant for one pipe-flow sample."""

    return resolve_axis_choice(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_values=GRID_SIZE_VARIANTS,
        explicit_key="grid_size_variant",
        weights_key="grid_size_variant_weights",
    )


def resolve_gap_size_variant(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the missing-region size variant for one pipe-flow sample."""

    return resolve_axis_choice(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_values=GAP_SIZE_VARIANTS,
        explicit_key="gap_size_variant",
        weights_key="gap_size_variant_weights",
    )


def resolve_scene_variant(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the visual scene variant for one pipe-flow sample."""

    return resolve_axis_choice(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_values=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )


def resolve_candidate_count(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[int, dict[str, float]]:
    """Resolve the number of displayed answer options."""

    low = max(
        4,
        min(
            len(LABEL_POOL),
            to_int(
                params.get(
                    "candidate_count_min",
                    group_default(gen_defaults, "candidate_count_min", DEFAULTS.candidate_count_min),
                ),
                DEFAULTS.candidate_count_min,
            ),
        ),
    )
    high = max(
        low,
        min(
            len(LABEL_POOL),
            to_int(
                params.get(
                    "candidate_count_max",
                    group_default(gen_defaults, "candidate_count_max", DEFAULTS.candidate_count_max),
                ),
                DEFAULTS.candidate_count_max,
            ),
        ),
    )
    return integer_range_choice(rng, int(low), int(high))


def resolve_answer_label(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    candidate_count: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the correct candidate label as a task-owned semantic operand."""

    labels = tuple(LABEL_POOL[index] for index in range(int(candidate_count)))
    explicit = params.get("answer_label")
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in set(labels):
            raise ValueError(f"answer_label {selected!r} outside candidate labels")
        return selected, support_probability_map(labels, selected=selected)
    rng = spawn_rng(int(instance_seed), f"{namespace}.answer_label")
    selected, probabilities = uniform_choice_with_probabilities(rng, labels)
    return str(selected), dict(probabilities)
