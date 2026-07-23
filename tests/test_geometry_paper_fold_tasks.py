from __future__ import annotations

from collections import Counter

import pytest

from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.geometry.paper_fold.shared.construction import fold_segment_geometry

TASK_ID = "task_geometry__paper_fold__paper_fold_angle_value"
SEGMENT_TASK_ID = "task_geometry__paper_fold__folded_segment_length_value"
ANNOTATION_KEYS = {"target_angle_cue", "given_angle_label"}


def test_paper_fold_angle_value_uses_single_query_number_answer_and_bbox_map() -> None:
    out = create_task(TASK_ID).generate(
        instance_seed=2026062301,
        params={},
        max_attempts=50,
    )

    assert out.scene_id == "paper_fold"
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert isinstance(out.answer_gt.value, float)
    assert round(float(out.answer_gt.value), 1) == float(out.answer_gt.value)

    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == ANNOTATION_KEYS
    for bbox in out.annotation_gt.value.values():
        assert len(bbox) == 4
        assert bbox[0] <= bbox[2]
        assert bbox[1] <= bbox[3]

    projected = out.trace_payload["projected_annotation"]
    assert projected["type"] == "bbox_map"
    assert projected["bbox_map"] == out.annotation_gt.value
    assert projected["pixel_bbox_map"] == out.annotation_gt.value

    query_spec = out.trace_payload["query_spec"]
    assert query_spec["query_id"] == "single"
    assert query_spec["params"]["query_id"] == "single"
    assert query_spec["prompt_variant"]["prompt_schema_version"] == "v1"
    assert query_spec["template_id"] == "geometry_paper_fold_measurement_v1"


def test_paper_fold_rejects_retired_query_id() -> None:
    task = create_task(TASK_ID)
    assert task.generate(instance_seed=17, params={"query_id": "single"}, max_attempts=50).query_id == "single"

    with pytest.raises(ValueError, match="unsupported query_id"):
        task.generate(
            instance_seed=17,
            params={"query_id": "fold_angle_from_total_label"},
            max_attempts=50,
        )


def test_paper_fold_explicit_geometry_overrides_still_bind_same_trace() -> None:
    out = create_task(TASK_ID).generate(
        instance_seed=2026062302,
        params={"height_units": 16, "folded_offset_units": 10},
        max_attempts=50,
    )

    witness = out.trace_payload["witness_symbolic"]
    assert witness["height_units"] == 16.0
    assert witness["folded_offset_units"] == 10.0
    assert witness["answer_value"] == out.answer_gt.value
    assert witness["formula_family"] == "fold_bisector_with_straight_angle"
    assert out.trace_payload["execution_trace"]["reasoning_steps"] == 2
    assert witness["given_angle_degrees"] == 64.0
    assert witness["total_angle_degrees"] == 116.0
    assert round((180.0 - witness["given_angle_degrees"]) / 2.0, 1) == out.answer_gt.value


def test_paper_fold_samples_broad_answer_support() -> None:
    task = create_task(TASK_ID)
    answers = []
    support_sizes = set()
    for seed in range(100):
        out = task.generate(instance_seed=seed, params={}, max_attempts=50)
        answers.append(out.answer_gt.value)
        support = out.trace_payload["query_spec"]["params"]["answer_support"]
        support_sizes.add(len(support))

    assert support_sizes == {190}
    assert len(set(answers)) >= 60
    assert max(Counter(answers).values()) <= 4


def test_paper_folded_segment_length_uses_integer_answer_and_scalar_segment() -> None:
    out = create_task(SEGMENT_TASK_ID).generate(
        instance_seed=2026062801,
        params={},
        max_attempts=50,
    )

    assert out.scene_id == "paper_fold"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert isinstance(out.answer_gt.value, int)

    assert out.annotation_gt.type == "segment"
    assert len(out.annotation_gt.value) == 2
    for point in out.annotation_gt.value:
        assert len(point) == 2

    projected = out.trace_payload["projected_annotation"]
    assert projected["type"] == "segment"
    assert projected["segment"] == out.annotation_gt.value
    assert projected["pixel_segment"] == out.annotation_gt.value

    query_spec = out.trace_payload["query_spec"]
    assert query_spec["query_id"] == "single"
    assert query_spec["params"]["formula_family"] == "pythagorean_leg_then_fold_correspondence"
    assert query_spec["prompt_variant"]["prompt_schema_version"] == "v1"
    assert query_spec["template_id"] == "geometry_paper_fold_measurement_v1"


def test_paper_folded_segment_explicit_triple_overrides_bind_same_trace() -> None:
    fp = create_task(SEGMENT_TASK_ID).generate(
        instance_seed=2026062802,
        params={"leg_ae": 6, "leg_af": 8, "target_segment": "FP"},
        max_attempts=50,
    )
    ep = create_task(SEGMENT_TASK_ID).generate(
        instance_seed=2026062802,
        params={"leg_ae": 6, "leg_af": 8, "target_segment": "EP"},
        max_attempts=50,
    )

    fp_witness = fp.trace_payload["witness_symbolic"]
    assert fp.answer_gt.value == 8
    assert fp_witness["leg_ae"] == 6
    assert fp_witness["leg_af"] == 8
    assert fp_witness["crease_ef"] == 10
    assert fp_witness["known_leg_segment"] == "AE"
    assert fp_witness["target_segment"] == "FP"
    assert fp_witness["pythagorean_unknown_original_segment"] == "AF"
    assert fp.trace_payload["execution_trace"]["reasoning_steps"] == 2

    ep_witness = ep.trace_payload["witness_symbolic"]
    assert ep.answer_gt.value == 6
    assert ep_witness["known_leg_segment"] == "AF"
    assert ep_witness["target_segment"] == "EP"
    assert ep_witness["pythagorean_unknown_original_segment"] == "AE"


def test_paper_folded_segment_reflected_point_lands_on_bottom_edge() -> None:
    geometry = fold_segment_geometry(6, 8)

    assert geometry.folded_point_units[1] == geometry.height_units
    assert geometry.folded_point_units[1] > geometry.leg_ae


def test_paper_folded_segment_samples_large_answer_support() -> None:
    task = create_task(SEGMENT_TASK_ID)
    answers = []
    support_sizes = set()
    target_segments = Counter()
    for seed in range(100):
        out = task.generate(instance_seed=seed, params={}, max_attempts=50)
        answers.append(out.answer_gt.value)
        params = out.trace_payload["query_spec"]["params"]
        support_sizes.add(len(params["answer_support"]))
        target_segments[str(params["target_segment"])] += 1

    assert min(support_sizes) >= 100
    assert len(set(answers)) >= 65
    assert max(Counter(answers).values()) <= 4
    assert set(target_segments) == {"EP", "FP"}
