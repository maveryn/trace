"""Contract tests for the geometry transformation matching task."""

from __future__ import annotations

from itertools import combinations

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.geometry.shape_reference.reflection_match import GeometryShapeReferenceReflectionMatchTask
from trace_tasks.tasks.geometry.shape_reference.rotation_match import GeometryShapeReferenceRotationMatchTask
from trace_tasks.tasks.geometry.shape_reference.translation_match import GeometryShapeReferenceTranslationMatchTask


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_point_count", "expected_rule"),
    (
        (GeometryShapeReferenceTranslationMatchTask, {"scene_variant": "triangle"}, 3, "translation"),
        (GeometryShapeReferenceReflectionMatchTask, {"scene_variant": "quadrilateral"}, 4, "reflection"),
        (GeometryShapeReferenceRotationMatchTask, {"scene_variant": "triangle"}, 3, "rotation"),
    ),
)
def test_geometry_transformation_match_emits_expected_contract(
    task_cls,
    params: dict[str, str],
    expected_point_count: int,
    expected_rule: str,
) -> None:
    out = task_cls().generate(23101, params=params, max_attempts=20)
    assert out.answer_gt.type == "option_letter"
    assert isinstance(out.answer_gt.value, str)
    assert len(str(out.answer_gt.value)) == 1
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == expected_point_count
    assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["params"]["query_id"] == out.query_id
    assert out.query_id == "single"
    assert out.trace_payload["execution_trace"]["transform_rule"] == expected_rule
    assert out.trace_payload["execution_trace"]["required_annotation_labels"] == [
        f"vertex_{index}" for index in range(1, expected_point_count + 1)
    ]


def test_geometry_transformation_match_rejects_unsupported_scene_variant() -> None:
    with pytest.raises(ValueError):
        GeometryShapeReferenceTranslationMatchTask().generate(
            23111,
            params={"scene_variant": "circle"},
            max_attempts=20,
        )


def test_geometry_transformation_match_rejects_unsupported_query_id() -> None:
    with pytest.raises(ValueError):
        GeometryShapeReferenceTranslationMatchTask().generate(
            23112,
            params={"scene_variant": "triangle", "query_id": "largest_area"},
            max_attempts=20,
        )


def test_geometry_transformation_match_keeps_candidate_polygons_separated() -> None:
    task = GeometryShapeReferenceReflectionMatchTask()

    for index in range(30):
        out = task.generate(
            int(hash64(0, "geometry_transformation_match_base", index)),
            params={},
            max_attempts=100,
        )
        candidates = out.trace_payload["render_map"]["candidate_vertices_graph_by_label"]
        boxes = {
            label: (
                min(float(point[0]) for point in vertices),
                min(float(point[1]) for point in vertices),
                max(float(point[0]) for point in vertices),
                max(float(point[1]) for point in vertices),
            )
            for label, vertices in candidates.items()
        }
        for left_label, right_label in combinations(sorted(boxes), 2):
            left = boxes[left_label]
            right = boxes[right_label]
            assert (
                float(left[2]) + 0.75 <= float(right[0])
                or float(right[2]) + 0.75 <= float(left[0])
                or float(left[3]) + 0.75 <= float(right[1])
                or float(right[3]) + 0.75 <= float(left[1])
            ), (index, left_label, right_label, left, right)


def test_geometry_transformation_match_translation_cue_is_above_reference_and_left_of_y_axis() -> None:
    task = GeometryShapeReferenceTranslationMatchTask()

    for index in range(30):
        out = task.generate(
            int(hash64(0, "geometry_transformation_match_base.translation", index)),
            params={},
            max_attempts=100,
        )
        cue = out.trace_payload["render_map"]["cue"]
        reference_vertices = out.trace_payload["render_map"]["reference_vertices_graph"]
        reference_top_y = max(int(point[1]) for point in reference_vertices)
        cue_bottom_y = min(int(cue["start_graph"][1]), int(cue["end_graph"][1]))
        assert cue["type"] == "translation_vector"
        assert max(int(cue["start_graph"][0]), int(cue["end_graph"][0])) < 0, (index, cue)
        assert cue_bottom_y - reference_top_y == 4, (index, cue, reference_vertices)
