from __future__ import annotations

from PIL import Image

from trace_tasks.tasks.charts.shared import chart_scene_primitives
from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene


def _base_render_params() -> dict[str, object]:
    return {
        "canvas_width": 640,
        "canvas_height": 420,
        "plot_margin_left_px": 64,
        "plot_margin_right_px": 48,
        "plot_margin_top_px": 40,
        "plot_margin_bottom_px": 72,
        "axis_line_width_px": 3,
        "grid_line_width_px": 1,
        "tick_length_px": 8,
        "label_font_size_px": 20,
        "tick_font_size_px": 16,
        "label_stroke_width_px": 2,
        "mark_outline_width_px": 2,
        "line_width_px": 4,
        "point_radius_px": 7,
        "bar_width_fraction": 0.62,
    }


def test_chart_scene_primitives_public_surface() -> None:
    assert set(chart_scene_primitives.__all__) == {
        "resolve_chart_render_params",
        "value_axis_render_metadata",
    }


def test_resolve_chart_render_params_normalizes_style_values() -> None:
    params = _base_render_params()
    params.update(
        {
            "axis_color_rgb": [-10, 512, 32],
            "value_axis_window_enabled": "yes",
            "guide_line_styles": ["solid", "dotted"],
            "_guide_style_seed": 3,
            "layout_jitter_meta": {"mode": "test"},
        }
    )

    resolved = chart_scene_primitives.resolve_chart_render_params(params)

    assert resolved.axis_color_rgb == (0, 255, 32)
    assert resolved.value_axis_window_enabled is True
    assert resolved.guide_line_style in {"solid", "dotted"}
    assert chart_scene_primitives.resolve_chart_render_params(params).guide_line_style == resolved.guide_line_style
    assert resolved.layout_jitter_meta == {"mode": "test"}


def test_value_axis_render_metadata_keeps_trace_shape() -> None:
    rendered = RenderedChartScene(
        image=Image.new("RGB", (10, 10)),
        mark_traces=(),
        entities=(),
        plot_bbox_px=(1, 2, 8, 9),
        y_axis_max=12,
        y_ticks=(0, 4, 8, 12),
        scene_variant="bar",
        value_axis_min=2,
        value_axis_max=12,
        value_axis_span=10,
        value_axis_major_ticks=(2, 7, 12),
        value_axis_minor_ticks=(2, 3, 4, 5),
        value_axis_window_enabled=True,
        guide_line_style="dashed",
        guide_lines=({"label": "A", "points_px": [[1, 2], [3, 4]]},),
    )

    metadata = chart_scene_primitives.value_axis_render_metadata(rendered)

    assert metadata == {
        "value_axis_min": 2,
        "value_axis_max": 12,
        "value_axis_span": 10,
        "value_axis_major_ticks": [2, 7, 12],
        "value_axis_minor_ticks": [2, 3, 4, 5],
        "value_axis_window_enabled": True,
        "guide_line_style": "dashed",
        "guide_lines": [{"label": "A", "points_px": [[1, 2], [3, 4]]}],
    }
