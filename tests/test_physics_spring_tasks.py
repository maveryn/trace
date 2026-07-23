"""Contract tests for the split physics spring tasks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.spring.shared.annotations import SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX
from trace_tasks.tasks.physics.spring.spring_extension_difference import (
    PhysicsSpringExtensionDifferenceTask,
)
from trace_tasks.tasks.physics.spring.spring_missing_value import PhysicsSpringMissingValueTask
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query_id", "expected_answer", "expected_annotation_count"),
    (
        (
            PhysicsSpringMissingValueTask,
            {"scene_variant": "paired_springs", "query_id": "missing_weight_for_extension", "target_answer": 4},
            "missing_weight_for_extension",
            4,
            4,
        ),
        (
            PhysicsSpringMissingValueTask,
            {"scene_variant": "staggered_springs", "query_id": "missing_extension_for_weight", "target_answer": 6},
            "missing_extension_for_weight",
            6,
            4,
        ),
        (
            PhysicsSpringExtensionDifferenceTask,
            {"scene_variant": "textured_spring", "query_id": "single", "target_answer": 4},
            "single",
            4,
            2,
        ),
    ),
)
def test_physics_spring_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_query_id: str,
    expected_answer: int,
    expected_annotation_count: int,
) -> None:
    out = task_cls().generate(28001, params=params, max_attempts=40)
    trace = out.trace_payload
    execution = trace["execution_trace"]


    assert out.answer_gt.type == "integer"

    assert int(out.answer_gt.value) == int(expected_answer)

    if expected_query_id in {"missing_weight_for_extension", "missing_extension_for_weight"}:
        assert out.annotation_gt.type == "bbox_map"
        assert set(out.annotation_gt.value.keys()) == {
            "reference_weight",
            "reference_extension",
            "query_weight",
            "query_extension",
        }
        assert len(out.annotation_gt.value) == int(expected_annotation_count)
        for key in ("reference_extension", "query_extension"):
            bbox = out.annotation_gt.value[key]
            assert float(bbox[3]) - float(bbox[1]) >= SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX
    else:
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(expected_annotation_count)
        for bbox in out.annotation_gt.value:
            assert float(bbox[3]) - float(bbox[1]) >= SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX

    assert out.query_id == expected_query_id

    assert trace["query_spec"]["query_id"] == expected_query_id

    assert trace["query_spec"]["params"]["query_id"] == expected_query_id
    expected_internal_query = (
        "extension_difference" if expected_query_id == "single" else expected_query_id
    )
    assert trace["query_spec"]["params"]["internal_query_id"] == expected_internal_query

    assert execution["query_id"] == expected_query_id
    assert execution["internal_query_id"] == expected_internal_query

    assert int(execution["target_answer"]) == int(expected_answer)

    assert trace["projected_annotation"]["type"] == out.annotation_gt.type
    if expected_query_id in {"missing_weight_for_extension", "missing_extension_for_weight"}:
        assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
        assert trace["render_map"]["annotation_bbox_map_px"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert trace["render_map"]["annotation_bboxes_px"] == out.annotation_gt.value
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_spring_diagram_offset"

    assert int(execution["scale_factor"]) in {1, 2, 3}
    if expected_query_id == "missing_weight_for_extension":

        assert execution["internal_query_id"] == "missing_weight_for_extension"

        assert execution["right_measurement"]["shown_weight_value"] is None

        assert execution["right_measurement"]["shown_extension_value"] == execution["right_measurement"]["true_extension_value"]
    elif expected_query_id == "missing_extension_for_weight":

        assert execution["internal_query_id"] == "missing_extension_for_weight"

        assert execution["right_measurement"]["shown_weight_value"] == execution["right_measurement"]["true_weight_value"]

        assert execution["right_measurement"]["shown_extension_value"] is None
    else:

        assert expected_query_id == "single"
        left_extension = int(execution["left_measurement"]["shown_extension_value"])
        right_extension = int(execution["right_measurement"]["shown_extension_value"])

        assert abs(left_extension - right_extension) == int(expected_answer)


def test_physics_spring_missing_value_is_deterministic() -> None:
    params = {
        "scene_variant": "staggered_springs",
        "query_id": "missing_weight_for_extension",
        "target_answer": 5,
        "accent_color_name": "purple",
    }
    task = PhysicsSpringMissingValueTask()
    out_a = task.generate(28021, params=params, max_attempts=40)
    out_b = task.generate(28021, params=params, max_attempts=40)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_spring_extension_difference_accepts_explicit_accent_color() -> None:
    out = PhysicsSpringExtensionDifferenceTask().generate(
        28031,
        params={
            "scene_variant": "paired_springs",
            "target_answer": 4,
            "accent_color_name": "orange",
        },
        max_attempts=40,
    )

    assert str(out.trace_payload["execution_trace"]["accent_color_name"]) == "orange"

    assert str(out.trace_payload["render_map"]["accent_color_name"]) == "orange"


def test_physics_spring_tasks_reject_unknown_scene_variant() -> None:
    with pytest.raises(ValueError):
        PhysicsSpringMissingValueTask().generate(
            28041,
            params={"scene_variant": "coiled_trio", "query_id": "missing_weight_for_extension"},
            max_attempts=20,
        )


def test_physics_spring_tasksseeded_sampler_decouples_answer_support() -> None:
    missing_task = PhysicsSpringMissingValueTask()
    answers_by_query: dict[tuple[str, str | None], set[int]] = {
        ("missing_weight_for_extension", "weight"): set(),
        ("missing_extension_for_weight", "extension"): set(),
        ("single", None): set(),
    }
    scenes_by_query: dict[tuple[str, str | None], Counter[str]] = defaultdict(Counter)
    combos: Counter[tuple[str, str | None, str]] = Counter()
    for query_id, solve_for in (
        ("missing_weight_for_extension", "weight"),
        ("missing_extension_for_weight", "extension"),
    ):
        for sampling_index in range(72):
            out = missing_task.generate(
                28100 + sampling_index,
                params={"query_id": query_id},
                max_attempts=60,
            )
            scene_variant = str(out.trace_payload["query_spec"]["params"]["scene_variant"])
            query_key = (str(out.query_id), str(solve_for))
            answers_by_query[query_key].add(int(out.answer_gt.value))
            scenes_by_query[query_key][scene_variant] += 1
            combos[(*query_key, scene_variant)] += 1

    difference_task = PhysicsSpringExtensionDifferenceTask()
    for sampling_index in range(36):
        out = difference_task.generate(
            28250 + sampling_index,
            params={"query_id": "single"},
            max_attempts=60,
        )
        query_id = str(out.query_id)
        scene_variant = str(out.trace_payload["query_spec"]["params"]["scene_variant"])
        query_key = (query_id, None)
        answers_by_query[query_key].add(int(out.answer_gt.value))
        scenes_by_query[query_key][scene_variant] += 1
        combos[(*query_key, scene_variant)] += 1


    assert answers_by_query[("missing_weight_for_extension", "weight")] == set(range(1, 9))

    assert answers_by_query[("missing_extension_for_weight", "extension")] == set(range(1, 13))

    assert answers_by_query[("single", None)] == {2, 4, 8, 10, 12}
    for query_key, counts in scenes_by_query.items():

        assert set(counts.keys()) == {"paired_springs", "staggered_springs", "textured_spring"}, query_key

    assert len(combos) == 9


def test_physics_spring_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/spring/physics_spring_v1.json").read_text(encoding="utf-8"))

    assert len(bundle["templates"]["query"]["missing_weight_for_extension"]) == 5

    assert len(bundle["templates"]["query"]["missing_extension_for_weight"]) == 5

    assert len(bundle["templates"]["query"]["spring_extension_difference"]) == 5


def test_physics_spring_tasks_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_spring"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_spring",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__spring__spring_missing_value",
                count=2,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_physics__spring__spring_extension_difference",
                count=2,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=40,
        sampling_seed=81,
    )
    final_path = build_dataset(config, code_hash="physics-mechanics-spring-extension-smoke")

    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 4

    assert all(record["domain"] == "physics" for record in train_records)

    assert all(record["scene_id"] == "spring" for record in train_records)

    assert {record["task"] for record in train_records} == {
        "task_physics__spring__spring_missing_value",
        "task_physics__spring__spring_extension_difference",
    }

    assert {record["query_id"] for record in train_records} <= {"missing_weight_for_extension", "missing_extension_for_weight", "single"}

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))

    assert int(build_report["accepted_counts_by_task"]["task_physics__spring__spring_missing_value"]) == 2

    assert int(build_report["accepted_counts_by_task"]["task_physics__spring__spring_extension_difference"]) == 2

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))

    assert validation["total_errors"] == 0
