"""Contract tests for the geometry coordinate-relation task."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from trace_tasks.tasks import create_task
from trace_tasks.tasks.geometry.coordinate_plane.collinear_point_count import TASK_ID as COLLINEAR_TASK_ID
from trace_tasks.tasks.geometry.coordinate_plane.point_in_polygon_count import TASK_ID as POINT_IN_POLYGON_TASK_ID
from trace_tasks.tasks.geometry.coordinate_plane.same_quadrant_point_count import TASK_ID as SAME_QUADRANT_TASK_ID
from trace_tasks.tasks.geometry.coordinate_plane.segment_relation_count import TASK_ID as SEGMENT_RELATION_TASK_ID
from trace_tasks.tasks.geometry.coordinate_plane.segment_relation_count import _segments_intersect
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_label_center


@pytest.mark.parametrize(
    ("task_id", "params", "expected_answer_type", "expected_annotation_type", "expected_annotation_count"),
    (
        (SEGMENT_RELATION_TASK_ID, {"query_id": "parallel_count", "target_count": 2}, "integer", "segment_set", 2),
        (SEGMENT_RELATION_TASK_ID, {"query_id": "perpendicular_count", "target_count": 1}, "integer", "segment_set", 1),
        (COLLINEAR_TASK_ID, {"query_id": "single", "target_count": 3}, "integer", "point_set", 3),
        (SAME_QUADRANT_TASK_ID, {"query_id": "single", "target_count": 3}, "integer", "point_set", 3),
        (POINT_IN_POLYGON_TASK_ID, {"query_id": "single", "target_count": 2}, "integer", "point_set", 2),
        (POINT_IN_POLYGON_TASK_ID, {"query_id": "single", "target_count": 6}, "integer", "point_set", 6),
    ),
)
def test_geometry_coordinate_relation_emits_expected_contract(
    task_id: str,
    params: dict[str, int | str],
    expected_answer_type: str,
    expected_annotation_type: str,
    expected_annotation_count: int,
) -> None:
    out = create_task(task_id).generate(23301, params=params, max_attempts=25)
    assert out.answer_gt.type == expected_answer_type
    assert out.annotation_gt.type == expected_annotation_type
    assert len(out.annotation_gt.value) == int(expected_annotation_count)
    assert out.trace_payload["projected_annotation"][expected_annotation_type] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["params"]["query_id"] == out.query_id


def test_geometry_coordinate_relation_rejects_unsupported_public_query() -> None:
    with pytest.raises(ValueError):
        create_task(SEGMENT_RELATION_TASK_ID).generate(
            23311,
            params={"query_id": "same_quadrant_count"},
            max_attempts=20,
        )


def test_geometry_coordinate_single_query_task_rejects_legacy_internal_query() -> None:
    with pytest.raises(ValueError):
        create_task(SAME_QUADRANT_TASK_ID).generate(
            23312,
            params={"query_id": "same_quadrant_count"},
            max_attempts=20,
        )


def test_geometry_coordinate_point_in_polygon_rejects_retired_count_support() -> None:
    with pytest.raises(ValueError, match="unsupported target_count"):
        create_task(POINT_IN_POLYGON_TASK_ID).generate(
            23310,
            params={"query_id": "single", "target_count": 8},
            max_attempts=20,
        )


def test_geometry_coordinate_relation_uses_centered_segment_window_and_quadrant_point_annotation() -> None:
    segment_out = create_task(SEGMENT_RELATION_TASK_ID).generate(
        23313,
        params={"query_id": "parallel_count", "target_count": 2},
        max_attempts=20,
    )
    segment_frame = segment_out.trace_payload["render_spec"]["graph_coordinate_frame"]
    assert float(segment_frame["origin_fraction_x"]) == pytest.approx(0.5)
    assert float(segment_frame["origin_fraction_y"]) == pytest.approx(0.5)
    assert segment_out.trace_payload["execution_trace"]["matching_segment_ids"]
    segment_render_map = segment_out.trace_payload["render_map"]
    reference_segment = tuple(tuple(int(coord) for coord in point) for point in segment_render_map["reference_segment_graph"])
    candidate_segments = [
        tuple(tuple(int(coord) for coord in point) for point in segment)
        for segment in segment_render_map["candidate_segments_graph"].values()
    ]
    assert max(abs(int(coord)) for point in reference_segment for coord in point) <= 8
    assert all(max(abs(int(coord)) for point in segment for coord in point) <= 8 for segment in candidate_segments)
    all_segments = [reference_segment, *candidate_segments]
    for index, segment in enumerate(all_segments):
        for other in all_segments[index + 1 :]:
            assert not _segments_intersect(segment, other)

    quadrant_out = create_task(SAME_QUADRANT_TASK_ID).generate(
        23314,
        params={"query_id": "single", "target_count": 2},
        max_attempts=20,
    )
    quadrant_frame = quadrant_out.trace_payload["render_spec"]["graph_coordinate_frame"]
    assert float(quadrant_frame["origin_fraction_x"]) == pytest.approx(0.5)
    assert float(quadrant_frame["origin_fraction_y"]) == pytest.approx(0.5)
    assert quadrant_out.annotation_gt.type == "point_set"
    assert all(isinstance(point, list) and len(point) == 2 for point in quadrant_out.annotation_gt.value)


def test_geometry_coordinate_relation_collinear_scene_keeps_reference_and_candidates_inside_centered_board() -> None:
    out = create_task(COLLINEAR_TASK_ID).generate(
        23315,
        params={"query_id": "single", "target_count": 4},
        max_attempts=20,
    )
    frame = out.trace_payload["render_spec"]["graph_coordinate_frame"]
    assert float(frame["origin_fraction_x"]) == pytest.approx(0.5)
    assert float(frame["origin_fraction_y"]) == pytest.approx(0.5)
    render_map = out.trace_payload["render_map"]
    all_points = [
        *render_map["reference_points_graph"],
        *render_map["candidate_points_graph"],
    ]
    assert all(max(abs(int(coord)) for coord in point) <= 8 for point in all_points)
    assert len(render_map["matching_points_graph"]) == 4
    point_a, point_b = render_map["reference_points_graph"]
    dx = int(point_b[0]) - int(point_a[0])
    dy = int(point_b[1]) - int(point_a[1])
    for point in render_map["matching_points_graph"]:
        cross = ((int(point[0]) - int(point_a[0])) * dy) - ((int(point[1]) - int(point_a[1])) * dx)
        assert cross == 0


def test_resolve_text_label_center_avoids_blocked_points_when_placing_labels() -> None:
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(22, bold=True)
    center, bbox = resolve_text_label_center(
        draw,
        text="A",
        anchor=(120.0, 120.0),
        base_direction=(1.0, -1.0),
        offset_px=14.0,
        font=font,
        blocked_points=[(120.0, 120.0), (138.0, 104.0)],
        point_clearance_px=10.0,
        canvas_size=240,
    )
    point_boxes = [
        (110.0, 110.0, 130.0, 130.0),
        (128.0, 94.0, 148.0, 114.0),
    ]
    assert all(
        bbox[2] <= blocked[0] or bbox[0] >= blocked[2] or bbox[3] <= blocked[1] or bbox[1] >= blocked[3]
        for blocked in point_boxes
    )
    assert center != (120.0, 120.0)
