"""Tests for shared semantic marker legibility utilities."""

from __future__ import annotations

from PIL import Image, ImageDraw

from trace_tasks.core import error_codes
from trace_tasks.core.validation import _validate_marker_legibility_contract
from trace_tasks.tasks.shared.marker_legibility import (
    SEMANTIC_MARKER_MIN_CONTRAST_RATIO,
    SEMANTIC_MARKER_MIN_LAB_DISTANCE,
    collect_semantic_marker_records,
    draw_semantic_bbox_marker,
    draw_semantic_line_marker,
    resolve_semantic_marker_style,
    semantic_marker_records_summary,
)


def test_resolve_semantic_marker_style_avoids_close_surface_color() -> None:
    surface = (170, 85, 78)
    close_preferred = (196, 82, 72)
    style = resolve_semantic_marker_style(
        instance_seed=9911,
        namespace="test.marker.close_surface",
        role="goal_cell_outline",
        surface_rgbs=(surface,),
        preferred_rgbs=(close_preferred,),
    )

    assert style.passes is True
    assert style.min_effective_contrast_ratio >= SEMANTIC_MARKER_MIN_CONTRAST_RATIO
    assert style.min_effective_lab_distance >= SEMANTIC_MARKER_MIN_LAB_DISTANCE


def test_semantic_marker_records_are_collected_and_summarized() -> None:
    image = Image.new("RGB", (120, 120), (170, 85, 78))
    draw = ImageDraw.Draw(image)
    style = resolve_semantic_marker_style(
        instance_seed=9912,
        namespace="test.marker.collection",
        role="marked_cell",
        surface_rgbs=((170, 85, 78),),
    )

    with collect_semantic_marker_records() as records:
        draw_semantic_bbox_marker(
            draw,
            (20, 20, 80, 80),
            style=style,
            radius=8,
            width=5,
        )

    summary = semantic_marker_records_summary(records)
    assert summary["policy_version"] == "marker_legibility_v1"
    assert summary["drawn_marker_record_count"] == 1
    assert summary["required_marker_count"] == 1
    assert summary["failure_count"] == 0
    assert records[0]["bbox_px"] == [20.0, 20.0, 80.0, 80.0]


def test_semantic_line_marker_records_dashed_pattern_and_alpha() -> None:
    image = Image.new("RGBA", (120, 120), (250, 248, 240, 255))
    draw = ImageDraw.Draw(image)
    style = resolve_semantic_marker_style(
        instance_seed=9914,
        namespace="test.marker.dashed_line",
        role="marked_cell",
        surface_rgbs=((250, 248, 240), (40, 36, 30)),
    )

    with collect_semantic_marker_records() as records:
        draw_semantic_line_marker(
            draw,
            ((20, 90), (100, 20)),
            style=style,
            width=3,
            pattern="dashed",
            dash_px=9,
            gap_px=6,
            alpha=138,
            marker_kind="dashed_slash",
        )

    assert len(records) == 1
    assert records[0]["marker_kind"] == "dashed_slash"
    assert records[0]["marker_pattern"] == "dashed"
    assert records[0]["marker_alpha"] == 138
    assert records[0]["passes"] is True


def test_core_validation_rejects_failing_marker_legibility_record() -> None:
    trace_record = {
        "render_spec": {
            "marker_legibility": {
                "enabled": True,
                "failure_count": 1,
                "records": [
                    {
                        "role": "marked_cell",
                        "required": True,
                        "passes": False,
                        "min_effective_contrast_ratio": 1.1,
                        "min_contrast_required": 3.0,
                        "min_effective_lab_distance": 10.0,
                        "min_lab_distance_required": 40.0,
                        "bbox_px": [10, 10, 30, 30],
                    }
                ],
            }
        }
    }

    errors = _validate_marker_legibility_contract(trace_record, instance_id="iid")
    assert any(error.error_code == error_codes.MARKER_LEGIBILITY_CONTRAST_FAILED for error in errors)
