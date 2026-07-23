"""Tests for bounded geometry graph-paper panel rendering."""

from __future__ import annotations

from trace_tasks.tasks.geometry.graph_paper.line_slope_value import (
    GeometryGraphPaperLineSlopeValueTask,
)
from trace_tasks.tasks.geometry.shared.graph_panel_layout import resolve_graph_panel_layout


def test_graph_panel_layout_bounds_origin_and_metadata() -> None:
    layout = resolve_graph_panel_layout(
        canvas_width=640,
        canvas_height=640,
        graph_cells=20,
        params={},
        render_defaults={},
        origin_fraction_x=0.5,
        origin_fraction_y=0.5,
    )

    panel = layout.panel_bbox_px
    content = layout.content_bbox_px
    origin = layout.graph_origin_px
    assert 0 <= panel[0] < content[0] < content[2] < panel[2] <= 640
    assert 0 <= panel[1] < content[1] < content[3] < panel[3] <= 640
    assert content[0] <= origin[0] <= content[2]
    assert content[1] <= origin[1] <= content[3]
    assert layout.graph_spacing >= 4

    meta = layout.to_metadata()
    assert meta["layout_placement"]["mode"] == "fractional_free_area"
    assert meta["graph_panel_bbox_px"] == list(panel)
    assert meta["graph_origin_px"] == list(origin)
    assert meta["scene_bbox_px"] == list(panel)


def test_graph_paper_task_records_bounded_panel_metadata() -> None:
    output = GeometryGraphPaperLineSlopeValueTask().generate(
        4242,
        params={},
        max_attempts=200,
    )
    render_spec = output.trace_payload["render_spec"]

    assert render_spec["background_style"]["selected_style"] == "graph_paper_panel"
    assert render_spec["layout_placement"]["mode"] == "fractional_free_area"
    assert render_spec["graph_panel_bbox_px"] == render_spec["scene_bbox_px"]
    assert render_spec["graph_origin_px"] == render_spec["graph_coordinate_frame"]["origin_pixel"]

    panel = render_spec["graph_panel_bbox_px"]
    content = render_spec["graph_content_bbox_px"]
    assert panel[0] < content[0] < content[2] < panel[2]
    assert panel[1] < content[1] < content[3] < panel[3]
