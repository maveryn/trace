"""Contracts for concentric-circle tangent-chord geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.concentric_chord.chord_length_from_radii import (
    GeometryConcentricChordLengthFromRadiiTask,
)
from trace_tasks.tasks.geometry.concentric_chord.inner_radius_from_chord import GeometryConcentricInnerRadiusFromChordTask
from trace_tasks.tasks.geometry.concentric_chord.shared.defaults import SCENE_ID
from trace_tasks.tasks.geometry.concentric_chord.shared.sampling import (
    MAX_INNER_RADIUS_RATIO,
    MIN_INNER_RADIUS_RATIO,
    PYTHAGOREAN_CASES,
)

TASK_CLASSES = (
    GeometryConcentricChordLengthFromRadiiTask,
    GeometryConcentricInnerRadiusFromChordTask,
)

QUERY_IDS_BY_TASK = {
    GeometryConcentricChordLengthFromRadiiTask: ("single",),
    GeometryConcentricInnerRadiusFromChordTask: ("single",),
}

INTERNAL_QUERY_ID_BY_TASK = {
    GeometryConcentricChordLengthFromRadiiTask: "chord_length_from_radii",
    GeometryConcentricInnerRadiusFromChordTask: "inner_radius_from_chord",
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_concentric_circle_chord_task_emits_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(56001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id
    assert out.answer_gt.type == "integer"
    assert isinstance(out.answer_gt.value, int)
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"O", "A", "B", "T"}
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["query_spec"]["params"]["internal_query_id"] == INTERNAL_QUERY_ID_BY_TASK[task_cls]
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["execution_trace"]["internal_query_id"] == INTERNAL_QUERY_ID_BY_TASK[task_cls]
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["outer_radius"] ** 2 == (
        trace["execution_trace"]["inner_radius"] ** 2
        + trace["execution_trace"]["half_chord"] ** 2
    )


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_concentric_circle_chord_task_is_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(56011, params=params, max_attempts=20)
    out_b = task.generate(56011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_concentric_circle_chord_task_supports_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            56021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.trace_payload["query_spec"]["params"]["internal_query_id"] == INTERNAL_QUERY_ID_BY_TASK[task_cls]
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_concentric_circle_chord_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            56041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_concentric_circle_chord_annotation_uses_construction_points_not_labels(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            56061 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.annotation_gt.type == "point_map"
        assert set(out.annotation_gt.value) == {"O", "A", "B", "T"}
        assert set(out.trace_payload["execution_trace"]["annotation_roles"]) == {
            "O",
            "A",
            "B",
            "T",
        }
        assert all(
            "label" not in str(role)
            for role in out.trace_payload["execution_trace"]["annotation_roles"]
        )
        assert "label_bboxes" in out.trace_payload["render_map"]


def test_concentric_circle_chord_tasks_reject_unknown_query_id() -> None:
    task = GeometryConcentricChordLengthFromRadiiTask()
    with pytest.raises(ValueError):
        task.generate(56031, params={"query_id": "not_a_query"}, max_attempts=20)


def test_concentric_circle_chord_case_pool_has_visible_separation_and_answer_diversity() -> None:
    assert len({case.chord_length for case in PYTHAGOREAN_CASES}) >= 50
    assert len({case.inner_radius for case in PYTHAGOREAN_CASES}) >= 50
    for case in PYTHAGOREAN_CASES:
        ratio = float(case.inner_radius) / float(case.outer_radius)
        assert MIN_INNER_RADIUS_RATIO <= ratio <= MAX_INNER_RADIUS_RATIO
        assert case.outer_radius**2 == case.inner_radius**2 + case.half_chord**2


@pytest.mark.parametrize(
    "task_cls",
    (
        GeometryConcentricChordLengthFromRadiiTask,
        GeometryConcentricInnerRadiusFromChordTask,
    ),
)
def test_concentric_circle_chord_review_sample_spreads_answers(task_cls) -> None:
    task = task_cls()
    counts: dict[int, int] = {}
    for seed in range(56200, 56300):
        out = task.generate(seed, params={}, max_attempts=20)
        answer = int(out.answer_gt.value)
        counts[answer] = counts.get(answer, 0) + 1

    assert len(counts) >= 50
    assert max(counts.values()) <= 4
