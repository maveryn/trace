"""Contract tests for the geometry shape-reference relation matching tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.shape_reference.congruent_match import GeometryShapeReferenceCongruentMatchTask
from trace_tasks.tasks.geometry.shape_reference.similar_match import GeometryShapeReferenceSimilarMatchTask


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_relation"),
    (
        (GeometryShapeReferenceCongruentMatchTask, {"scene_variant": "triangle", "winner_label": "A"}, "congruent"),
        (GeometryShapeReferenceSimilarMatchTask, {"scene_variant": "quadrilateral", "winner_label": "B"}, "similar"),
        (GeometryShapeReferenceSimilarMatchTask, {"scene_variant": "triangle", "candidate_count": 4}, "similar"),
        (GeometryShapeReferenceCongruentMatchTask, {"scene_variant": "quadrilateral", "candidate_count": 6}, "congruent"),
    ),
)
def test_geometry_similarity_match_emits_expected_contract(
    task_cls,
    params: dict[str, int | str],
    expected_relation: str,
) -> None:
    out = task_cls().generate(23201, params=params, max_attempts=30)
    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) in set(out.trace_payload["query_spec"]["params"]["candidate_label_pool"])
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) in {3, 4}
    assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["params"]["query_id"] == out.query_id
    assert out.query_id == "single"
    assert out.trace_payload["execution_trace"]["relation_rule"] == expected_relation
    assert out.trace_payload["execution_trace"]["winner_label"] == out.answer_gt.value


def test_geometry_similarity_match_rejects_unsupported_scene_variant() -> None:
    with pytest.raises(ValueError):
        GeometryShapeReferenceSimilarMatchTask().generate(
            23211,
            params={"scene_variant": "circle"},
            max_attempts=20,
        )


def test_geometry_similarity_match_rejects_unsupported_query_id() -> None:
    with pytest.raises(ValueError):
        GeometryShapeReferenceCongruentMatchTask().generate(
            23212,
            params={"scene_variant": "triangle", "query_id": "largest_area"},
            max_attempts=20,
        )
