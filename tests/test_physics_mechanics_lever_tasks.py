"""Contract tests for the split physics mechanics lever tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.lever.missing_weight_balance_value import (
    PhysicsLeverMissingWeightBalanceValueTask,
)
from trace_tasks.tasks.physics.lever.side_torque_value import PhysicsLeverSideTorqueValueTask
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_internal_query_id", "expected_answer"),
    (
        (
            PhysicsLeverSideTorqueValueTask,
            {"scene_variant": "center_fulcrum", "torque_side": "left", "target_answer": 8},
            "left_torque",
            8,
        ),
        (
            PhysicsLeverSideTorqueValueTask,
            {"scene_variant": "offset_fulcrum", "torque_side": "right", "target_answer": 12},
            "right_torque",
            12,
        ),
        (
            PhysicsLeverMissingWeightBalanceValueTask,
            {"scene_variant": "textured_beam", "target_answer": 5},
            "missing_weight_to_balance",
            5,
        ),
    ),
)
def test_physics_mechanics_lever_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_internal_query_id: str,
    expected_answer: int,
) -> None:
    out = task_cls().generate(25001, params=params, max_attempts=40)
    trace = out.trace_payload
    execution = trace["execution_trace"]


    assert out.answer_gt.type == "integer"

    assert int(out.answer_gt.value) == int(expected_answer)

    expected_annotation_type = "bbox_set_map" if expected_internal_query_id == "missing_weight_to_balance" else "bbox_set"
    assert out.annotation_gt.type == expected_annotation_type

    assert out.query_id == "single"

    assert trace["query_spec"]["query_id"] == "single"

    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["internal_query_id"] == expected_internal_query_id

    assert execution["query_id"] == "single"
    assert execution["internal_query_id"] == expected_internal_query_id

    assert str(trace["query_spec"]["params"]["accent_color_name"]) == str(trace["execution_trace"]["accent_color_name"])

    assert str(trace["render_map"]["accent_color_name"]) == str(trace["execution_trace"]["accent_color_name"])

    assert int(execution["target_answer"]) == int(expected_answer)

    if expected_internal_query_id == "missing_weight_to_balance":
        assert trace["projected_annotation"]["bbox_set_map"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_lever_diagram_offset"
    if expected_internal_query_id == "missing_weight_to_balance":

        assert trace["render_map"]["missing_weight_marker_bbox_px"] in out.annotation_gt.value["target_weight"]

        assert set(execution["annotation_entity_ids"]) == {"known_weights", "target_weight"}
        assert "missing_weight_marker" in execution["witness_entity_ids"]

        assert execution["placeholder_side"] in {"left", "right"}
        flattened_annotation = [box for boxes in out.annotation_gt.value.values() for box in boxes]
        assert len(flattened_annotation) == len(execution["relevant_weight_ids"])
        assert len(flattened_annotation) >= 2
    else:

        assert str(execution["internal_query_id"]) in {"left_torque", "right_torque"}

        assert len(out.annotation_gt.value) == len(execution["relevant_weight_ids"])

        assert len(out.annotation_gt.value) >= 1
        queried_side = str(execution["torque_side"])
        for spec in execution["weight_specs"]:
            if bool(spec["relevant_to_query"]):

                assert str(spec["side"]) == queried_side


def test_physics_mechanics_missing_weight_balance_value_is_deterministic() -> None:
    params = {
        "scene_variant": "textured_beam",
        "target_answer": 6,
    }
    task = PhysicsLeverMissingWeightBalanceValueTask()
    out_a = task.generate(25021, params=params, max_attempts=40)
    out_b = task.generate(25021, params=params, max_attempts=40)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_mechanics_side_torque_value_accepts_explicit_accent_color() -> None:
    out = PhysicsLeverSideTorqueValueTask().generate(
        25029,
        params={
            "scene_variant": "center_fulcrum",
            "torque_side": "left",
            "target_answer": 8,
            "accent_color_name": "purple",
        },
        max_attempts=40,
    )

    assert str(out.trace_payload["execution_trace"]["accent_color_name"]) == "purple"

    assert str(out.trace_payload["render_map"]["accent_color_name"]) == "purple"


def test_physics_mechanics_lever_tasks_reject_unknown_scene_variant() -> None:
    with pytest.raises(ValueError):
        PhysicsLeverSideTorqueValueTask().generate(
            25031,
            params={"scene_variant": "swinging_beam", "torque_side": "left"},
            max_attempts=20,
        )


def test_physics_mechanics_lever_tasksseeded_sampler_decouples_answer_support() -> None:
    side_task = PhysicsLeverSideTorqueValueTask()
    answers_by_query: dict[tuple[str, str | None], set[int]] = {
        ("side_torque", "left"): set(),
        ("side_torque", "right"): set(),
        ("missing_weight_to_balance", None): set(),
    }
    for sampling_index in range(120):
        out = side_task.generate(
            25100 + sampling_index,
            params={},
            max_attempts=60,
        )
        execution = out.trace_payload["execution_trace"]
        answers_by_query[("side_torque", str(execution["torque_side"]))].add(int(out.answer_gt.value))

    missing_task = PhysicsLeverMissingWeightBalanceValueTask()
    for sampling_index in range(46):
        out = missing_task.generate(
            25200 + sampling_index,
            params={},
            max_attempts=60,
        )
        answers_by_query[("missing_weight_to_balance", None)].add(int(out.answer_gt.value))


    assert answers_by_query[("side_torque", "left")].issubset(set(range(2, 25)))
    assert len(answers_by_query[("side_torque", "left")]) >= 18

    assert answers_by_query[("side_torque", "right")].issubset(set(range(2, 25)))
    assert len(answers_by_query[("side_torque", "right")]) >= 18

    assert answers_by_query[("missing_weight_to_balance", None)] == set(range(1, 7))


def test_physics_mechanics_lever_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/lever/physics_lever_v1.json").read_text(encoding="utf-8"))

    assert len(bundle["templates"]["task"]["side_torque_value_query"]) == 5

    assert len(bundle["templates"]["task"]["missing_weight_balance_value_query"]) == 5


def test_physics_mechanics_lever_tasks_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_mechanics_lever"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_mechanics_lever",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__lever__side_torque_value",
                count=2,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_physics__lever__missing_weight_balance_value",
                count=2,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=40,
        sampling_seed=52,
    )
    final_path = build_dataset(config, code_hash="physics-mechanics-lever-balance-smoke")

    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 4

    assert all(record["domain"] == "physics" for record in train_records)

    assert all(record["scene_id"] == "lever" for record in train_records)

    assert {record["task"] for record in train_records} == {
        "task_physics__lever__side_torque_value",
        "task_physics__lever__missing_weight_balance_value",
    }

    assert {record["query_id"] for record in train_records} == {"single"}

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))

    assert int(build_report["accepted_counts_by_task"]["task_physics__lever__side_torque_value"]) == 2

    assert int(build_report["accepted_counts_by_task"]["task_physics__lever__missing_weight_balance_value"]) == 2

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))

    assert validation["total_errors"] == 0
