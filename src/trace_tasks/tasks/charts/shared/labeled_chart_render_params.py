"""Render-parameter resolution for labeled chart task families."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.color_distance import (
    sample_color_palette_with_distance_constraints,
    sample_color_with_distance_constraints,
)
from ...shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.named_colors import darken_color
from ...shared.render_variation import apply_layout_jitter_to_margins
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from .chart_scene_primitives import resolve_chart_render_params
from .chart_scene_types import (
    ChartMarkSpec,
    ChartRenderParams,
    RenderedChartScene,
    SUPPORTED_CHART_SCENE_VARIANTS,
)
from .label_assets import resolve_chart_compact_axis_labels

from .labeled_chart_defaults import LabeledChartDefaults


def resolve_chart_render_params_for_task(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    instance_seed: int | None = None,
) -> ChartRenderParams:
    """Resolve one chart render-parameter block."""

    def _rng_for_key(key: str):
        seed = 0 if instance_seed is None else int(instance_seed)
        return spawn_rng(int(seed), f"chart_render:{str(key)}")

    def _resolve_int(key: str, fallback: int, *, minimum: int = 1) -> int:
        if params.get(str(key)) is not None:
            return max(int(minimum), int(params[str(key)]))
        low_raw = params.get(f"{str(key)}_min", group_default(render_defaults, f"{str(key)}_min", None))
        high_raw = params.get(f"{str(key)}_max", group_default(render_defaults, f"{str(key)}_max", None))
        if low_raw is not None or high_raw is not None:
            default_value = int(group_default(render_defaults, str(key), int(fallback)))
            low = int(default_value if low_raw is None else low_raw)
            high = int(default_value if high_raw is None else high_raw)
            if int(low) > int(high):
                raise ValueError(f"{str(key)}_min must be <= {str(key)}_max")
            return max(int(minimum), int(_rng_for_key(str(key)).randint(int(low), int(high))))
        return max(int(minimum), int(group_default(render_defaults, str(key), int(fallback))))

    def _resolve_float(key: str, fallback: float, *, steps: int = 15) -> float:
        if params.get(str(key)) is not None:
            return float(params[str(key)])
        low_raw = params.get(f"{str(key)}_min", group_default(render_defaults, f"{str(key)}_min", None))
        high_raw = params.get(f"{str(key)}_max", group_default(render_defaults, f"{str(key)}_max", None))
        if low_raw is not None or high_raw is not None:
            default_value = float(group_default(render_defaults, str(key), float(fallback)))
            low = float(default_value if low_raw is None else low_raw)
            high = float(default_value if high_raw is None else high_raw)
            if float(low) > float(high):
                raise ValueError(f"{str(key)}_min must be <= {str(key)}_max")
            step_count = max(1, int(steps))
            offset = _rng_for_key(str(key)).randint(0, int(step_count))
            return float(low + ((high - low) * (float(offset) / float(step_count))))
        return float(group_default(render_defaults, str(key), float(fallback)))

    def _resolve_rgb(key: str, fallback: Sequence[int]) -> Sequence[int]:
        if params.get(str(key)) is not None:
            return params[str(key)]
        options = params.get(f"{str(key)}_options", group_default(render_defaults, f"{str(key)}_options", None))
        if isinstance(options, Sequence) and options and not isinstance(options, (str, bytes)):
            selected = _rng_for_key(str(key)).choice(list(options))
            return selected
        return group_default(render_defaults, str(key), list(fallback))

    def _resolve_bool(key: str, fallback: bool) -> bool:
        value = params.get(str(key), group_default(render_defaults, str(key), bool(fallback)))
        if isinstance(value, str):
            return str(value).strip().lower() in {"1", "true", "yes", "on", "always"}
        return bool(value)

    def _resolve_str(key: str, fallback: str) -> str:
        return str(params.get(str(key), group_default(render_defaults, str(key), str(fallback))))

    def _resolve_choice(key: str, fallback: str) -> str:
        explicit = params.get(str(key), group_default(render_defaults, str(key), None))
        if explicit is not None:
            return str(explicit)
        options = params.get(f"{str(key)}_options", group_default(render_defaults, f"{str(key)}_options", None))
        if isinstance(options, Sequence) and options and not isinstance(options, (str, bytes)):
            return str(_rng_for_key(str(key)).choice(list(options)))
        return str(fallback)

    def _resolve_float_value(key: str, fallback: float) -> float:
        return float(params.get(str(key), group_default(render_defaults, str(key), float(fallback))))

    def _resolve_style() -> str:
        explicit = params.get("guide_line_style", group_default(render_defaults, "guide_line_style", None))
        if explicit is not None:
            return str(explicit)
        styles = params.get("guide_line_styles", group_default(render_defaults, "guide_line_styles", ("dashed", "dotted")))
        if isinstance(styles, Sequence) and styles and not isinstance(styles, (str, bytes)):
            return str(_rng_for_key("guide_line_style").choice(list(styles)))
        return "dashed"

    def _resolve_guide_mode() -> str:
        mode = _resolve_str("guide_line_mode", "off").strip().lower()
        if mode != "variant":
            return str(mode)
        probability = max(0.0, min(1.0, _resolve_float_value("guide_line_prob", 0.5)))
        draw_value = _rng_for_key("guide_line_enabled").random()
        return "always" if float(draw_value) < float(probability) else "off"

    margin_left = int(params.get("plot_margin_left_px", group_default(render_defaults, "plot_margin_left_px", defaults.plot_margin_left_px)))
    margin_right = int(params.get("plot_margin_right_px", group_default(render_defaults, "plot_margin_right_px", defaults.plot_margin_right_px)))
    margin_top = int(params.get("plot_margin_top_px", group_default(render_defaults, "plot_margin_top_px", defaults.plot_margin_top_px)))
    margin_bottom = int(params.get("plot_margin_bottom_px", group_default(render_defaults, "plot_margin_bottom_px", defaults.plot_margin_bottom_px)))
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=render_defaults,
        instance_seed=instance_seed,
        namespace="charts.labeled.layout",
    )

    resolved = {
        "canvas_width": int(params.get("canvas_width", group_default(render_defaults, "canvas_width", defaults.canvas_width))),
        "canvas_height": int(params.get("canvas_height", group_default(render_defaults, "canvas_height", defaults.canvas_height))),
        "plot_margin_left_px": int(margin_left),
        "plot_margin_right_px": int(margin_right),
        "plot_margin_top_px": int(margin_top),
        "plot_margin_bottom_px": int(margin_bottom),
        "axis_line_width_px": _resolve_int("axis_line_width_px", defaults.axis_line_width_px),
        "grid_line_width_px": _resolve_int("grid_line_width_px", defaults.grid_line_width_px),
        "tick_length_px": int(params.get("tick_length_px", group_default(render_defaults, "tick_length_px", defaults.tick_length_px))),
        "label_font_size_px": int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", defaults.label_font_size_px))),
        "tick_font_size_px": int(params.get("tick_font_size_px", group_default(render_defaults, "tick_font_size_px", defaults.tick_font_size_px))),
        "label_stroke_width_px": _resolve_int("label_stroke_width_px", defaults.label_stroke_width_px, minimum=0),
        "label_bold": _resolve_bool("label_bold", True),
        "mark_outline_width_px": _resolve_int("mark_outline_width_px", defaults.mark_outline_width_px),
        "line_width_px": _resolve_int("line_width_px", defaults.line_width_px),
        "point_radius_px": _resolve_int("point_radius_px", defaults.point_radius_px),
        "bar_width_fraction": _resolve_float("bar_width_fraction", defaults.bar_width_fraction),
        "axis_color_rgb": _resolve_rgb("axis_color_rgb", [74, 78, 86]),
        "grid_color_rgb": _resolve_rgb("grid_color_rgb", [224, 227, 232]),
        "mark_fill_rgb": params.get("mark_fill_rgb", group_default(render_defaults, "mark_fill_rgb", [86, 138, 214])),
        "mark_outline_rgb": params.get("mark_outline_rgb", group_default(render_defaults, "mark_outline_rgb", [50, 76, 116])),
        "text_color_rgb": _resolve_rgb("text_color_rgb", [38, 41, 48]),
        "text_stroke_rgb": _resolve_rgb("text_stroke_rgb", [255, 255, 255]),
        "plot_fill_rgb": _resolve_rgb("plot_fill_rgb", [255, 255, 255]),
        "value_axis_window_enabled": _resolve_bool("value_axis_window_enabled", False),
        "value_axis_span_min": int(params.get("value_axis_span_min", group_default(render_defaults, "value_axis_span_min", 10))),
        "value_axis_span_max": int(params.get("value_axis_span_max", group_default(render_defaults, "value_axis_span_max", 25))),
        "value_axis_hard_max": int(params.get("value_axis_hard_max", group_default(render_defaults, "value_axis_hard_max", 99))),
        "value_axis_major_tick_step": int(params.get("value_axis_major_tick_step", group_default(render_defaults, "value_axis_major_tick_step", 5))),
        "value_axis_minor_tick_step": int(params.get("value_axis_minor_tick_step", group_default(render_defaults, "value_axis_minor_tick_step", 1))),
        "value_axis_allow_nonzero_min": _resolve_bool("value_axis_allow_nonzero_min", True),
        "guide_line_mode": _resolve_guide_mode(),
        "guide_line_prob": _resolve_float_value("guide_line_prob", 0.0),
        "guide_line_style": _resolve_style(),
        "guide_line_width_px": int(params.get("guide_line_width_px", group_default(render_defaults, "guide_line_width_px", 1))),
        "guide_line_color_rgb": _resolve_rgb("guide_line_color_rgb", [150, 156, 166]),
        "layout_jitter_dx_px": int(layout_jitter_meta.get("dx_px", 0)),
        "layout_jitter_dy_px": int(layout_jitter_meta.get("dy_px", 0)),
        "layout_jitter_meta": dict(layout_jitter_meta),
        "violin_mode_line_style": _resolve_choice("violin_mode_line_style", "full"),
        "violin_fill_style": _resolve_choice("violin_fill_style", "solid"),
        "violin_width_scale": _resolve_float("violin_width_scale", 1.0),
        "violin_smoothing_scale": _resolve_float("violin_smoothing_scale", 1.0),
        "violin_palette_mode": _resolve_choice("violin_palette_mode", "single"),
        "violin_palette_offset": _resolve_int("violin_palette_offset", 0, minimum=0),
    }
    return resolve_chart_render_params(resolved)

__all__ = [
    'resolve_chart_render_params_for_task',
]
