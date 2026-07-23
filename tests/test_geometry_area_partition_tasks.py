"""Contracts for area-partition theorem geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.area_partition.total_area_value import (
    AREA_PARTITION_SCENE_ID,
    GeometryAreaPartitionTotalAreaValueTask,
)

TASK_CLASSES = (
    GeometryAreaPartitionTotalAreaValueTask,
)

QUERY_IDS_BY_TASK = {
    GeometryAreaPartitionTotalAreaValueTask: (
        "single",
    ),
}

SCENE_ID_BY_TASK = {
    GeometryAreaPartitionTotalAreaValueTask: AREA_PARTITION_SCENE_ID,
}

VARIANTS_BY_TASK = {
    GeometryAreaPartitionTotalAreaValueTask: {
        "parallelogram_diagonals_quarter",
        "parallelogram_diagonals_midpoint_eighth",
        "triangle_median_half",
        "triangle_midsegment_quarter",
        "triangle_medians_sixth",
    },
}

DENOMINATORS_BY_TASK = {
    GeometryAreaPartitionTotalAreaValueTask: {2, 4, 6, 8},
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(62001, params={}, max_attempts=20)
    scene_id = SCENE_ID_BY_TASK[task_cls]

    assert out.scene_id == scene_id
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_map"
    assert len(out.annotation_gt.value) == 2
    assert set(out.annotation_gt.value) == {"outer_shape", "shaded_region"}
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    shape_type = str(trace["execution_trace"]["shape_type"])
    assert f"in a {shape_type}" in out.prompt
    assert "triangle or parallelogram" not in out.prompt
    assert "equal-area regions" not in out.prompt
    assert trace["query_spec"]["scene_id"] == scene_id
    assert trace["scene_ir"]["scene_id"] == scene_id
    assert trace["witness_symbolic"]["scene_id"] == scene_id
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert set(trace["projected_annotation"]["bbox_map"]) == {
        "outer_shape",
        "shaded_region",
    }
    assert trace["execution_trace"]["annotation_roles"] == [
        "outer_shape",
        "shaded_region",
    ]
    assert "label_bboxes" in trace["render_map"]
    assert "partition_bbox" in trace["render_map"]
    assert "given_area" in trace["render_map"]["label_bboxes"]
    assert "target" in trace["render_map"]["label_bboxes"]
    slot_values = trace["query_spec"]["prompt_variant"]["slot_values"]
    assert slot_values["shape_type"] == shape_type

    shaded_area = int(trace["execution_trace"]["shaded_area"])
    denominator = int(trace["execution_trace"]["shaded_fraction_denominator"])
    assert denominator in DENOMINATORS_BY_TASK[task_cls]
    assert trace["execution_trace"]["shaded_fraction_numerator"] == 1
    assert out.answer_gt.value == int(shaded_area * denominator)
    assert trace["execution_trace"]["answer_value"] == int(shaded_area * denominator)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(62011, params=params, max_attempts=20)
    out_b = task.generate(62011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"]
        == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            62021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_tasks_sample_all_scene_variants(task_cls) -> None:
    task = task_cls()
    variants = set()
    denominators = set()
    for index in range(18):
        out = task.generate(
            62041 + index,
            params={"_sampling_index": index},
            max_attempts=20,
        )
        trace = out.trace_payload["execution_trace"]
        variants.add(trace["scene_variant"])
        denominators.add(trace["shaded_fraction_denominator"])

    assert variants == VARIANTS_BY_TASK[task_cls]
    assert denominators == DENOMINATORS_BY_TASK[task_cls]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_tasks_sample_broad_numeric_answers(task_cls) -> None:
    task = task_cls()
    answers = {
        task.generate(62101 + index, params={}, max_attempts=20).answer_gt.value
        for index in range(60)
    }

    assert len(answers) >= 30


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            62061 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        assert out.annotation_gt.type == "bbox_map"
        for x0, y0, x1, y1 in out.annotation_gt.value.values():
            assert 0.0 <= x0 < x1 <= float(width)
            assert 0.0 <= y0 < y1 <= float(height)
            assert (x1 - x0) > 8.0
            assert (y1 - y0) > 8.0


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_area_partition_readout_labels_stay_visible(task_cls) -> None:
    task = task_cls()
    seeds = (364306975400184, 62001, 62061, 62062, 62063)
    for seed in seeds:
        out = task.generate(seed, params={}, max_attempts=20)
        width, height = out.image.size
        label_bboxes = out.trace_payload["render_map"]["label_bboxes"]
        assert set(label_bboxes) == {"given_area", "target"}
        for x0, y0, x1, y1 in label_bboxes.values():
            assert 0.0 <= x0 < x1 <= float(width)
            assert 0.0 <= y0 < y1 <= float(height)
            assert (x1 - x0) > 8.0
            assert (y1 - y0) > 8.0


def test_area_partition_tasks_reject_unknown_query_id() -> None:
    for task_cls in TASK_CLASSES:
        task = task_cls()
        with pytest.raises(ValueError):
            task.generate(62031, params={"query_id": "not_a_query"}, max_attempts=20)
