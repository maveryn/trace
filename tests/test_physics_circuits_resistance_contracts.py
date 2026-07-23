"""Contract tests for physics equivalent-circuit tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.circuit_equivalent.total_capacitance_value import (
    PhysicsCircuitEquivalentTotalCapacitanceValueTask,
)
from trace_tasks.tasks.physics.circuit_equivalent.total_resistance_value import (
    PhysicsCircuitEquivalentTotalResistanceValueTask,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_prefix", "expected_kind", "expected_answer"),
    (
        (
            PhysicsCircuitEquivalentTotalResistanceValueTask,
            {"scene_variant": "series_parallel", "target_answer": 8},
            "R",
            "resistor",
            8,
        ),
        (
            PhysicsCircuitEquivalentTotalCapacitanceValueTask,
            {"scene_variant": "series_parallel", "target_answer": 4, "parallel_block_count_options": [2]},
            "C",
            "capacitor",
            4,
        ),
        (
            PhysicsCircuitEquivalentTotalCapacitanceValueTask,
            {"scene_variant": "series_parallel", "target_answer": 6},
            "C",
            "capacitor",
            6,
        ),
    ),
)
def test_physics_circuit_equivalent_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_prefix: str,
    expected_kind: str,
    expected_answer: int,
) -> None:
    out = task_cls().generate(26001, params=params, max_attempts=40)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "circuit_equivalent"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.annotation_gt.value == trace["render_map"]["annotation_bbox_px"]
    x0, y0, x1, y1 = [float(value) for value in out.annotation_gt.value]
    assert 0 <= x0 < x1 <= out.image.size[0]
    assert 0 <= y0 < y1 <= out.image.size[1]
    component_bboxes = trace["render_map"]["component_bboxes_px"]
    assert component_bboxes
    assert all(str(key).startswith(expected_prefix) for key in component_bboxes)
    assert all(len(value) == 4 for value in component_bboxes.values())

    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert execution["component_kind"] == expected_kind
    assert int(execution["target_answer"]) == int(expected_answer)
    assert int(execution["equivalent_value"]) == int(expected_answer)
    assert trace["render_spec"]["component_kind"] == expected_kind
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_equivalent_circuit_diagram_offset"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["type"] == "bbox"
    assert trace["witness_symbolic"]["id"] == "equivalent_circuit_network"
    assert trace["witness_symbolic"]["component_entity_ids"] == list(trace["render_map"]["component_entity_ids"].values())
    assert len(execution["component_specs"]) == len(component_bboxes)
    assert [spec["label"] for spec in execution["component_specs"]] == list(component_bboxes.keys())

    assert str(params["scene_variant"]) == "series_parallel"
    assert len(execution["parallel_blocks"]) in {1, 2}
    assert len(execution["outer_series_values"]) == 2
    assert len(execution["inter_block_series_values"]) == max(0, len(execution["parallel_blocks"]) - 1)
    assert (
        sum(int(value) > 0 for value in execution["outer_series_values"])
        + sum(int(value) > 0 for value in execution["inter_block_series_values"])
    ) >= 1
    if params.get("parallel_block_count_options") == [2]:
        assert len(execution["parallel_blocks"]) == 2


def test_physics_circuits_equivalent_tasks_are_deterministic() -> None:
    params = {
        "scene_variant": "series_parallel",
        "target_answer": 3,
        "accent_color_name": "cyan",
    }
    task = PhysicsCircuitEquivalentTotalCapacitanceValueTask()
    out_a = task.generate(26021, params=params, max_attempts=40)
    out_b = task.generate(26021, params=params, max_attempts=40)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize(
    ("task_cls", "params"),
    (
        (PhysicsCircuitEquivalentTotalResistanceValueTask, {"scene_variant": "series_parallel", "target_answer": 1}),
        (PhysicsCircuitEquivalentTotalCapacitanceValueTask, {"scene_variant": "series_parallel", "target_answer": 999}),
    ),
)
def test_physics_circuits_equivalent_tasks_reject_infeasible_target_answer(
    task_cls: type,
    params: dict[str, int | str],
) -> None:
    with pytest.raises(ValueError, match="unsupported target_answer"):
        task_cls().generate(26029, params=params, max_attempts=40)


def test_physics_circuits_equivalent_tasks_reject_unknown_scene_variant() -> None:
    with pytest.raises(ValueError):
        PhysicsCircuitEquivalentTotalResistanceValueTask().generate(
            26031,
            params={"scene_variant": "bridge_network"},
            max_attempts=20,
        )


def test_physics_circuits_equivalent_rejects_pure_series_or_parallel_scene_variants() -> None:
    for task_cls in (PhysicsCircuitEquivalentTotalResistanceValueTask, PhysicsCircuitEquivalentTotalCapacitanceValueTask):
        for scene_variant in ("series", "parallel"):
            with pytest.raises(ValueError, match="unsupported scene_variant"):
                task_cls().generate(
                    26032,
                    params={"scene_variant": scene_variant},
                    max_attempts=20,
                )


def test_physics_circuits_equivalent_tasks_reject_retired_query_id_param() -> None:
    with pytest.raises(ValueError, match="unsupported query_id"):
        PhysicsCircuitEquivalentTotalResistanceValueTask().generate(
            26033,
            params={"query_id": "total_capacitance"},
            max_attempts=20,
        )


def test_physics_circuits_equivalent_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/physics/circuit_equivalent/physics_circuit_equivalent_v1.json").read_text(encoding="utf-8")
    )

    assert bundle["schema_version"] == "v1"
    assert len(bundle["templates"]["query"]["single"]) == 5
    assert len(bundle["templates"]["task"]["total_resistance_value_query"]) == 5
    assert len(bundle["templates"]["task"]["total_capacitance_value_query"]) == 5
    assert len(set(bundle["templates"]["output"]["answer_and_annotation"])) == 5


def test_physics_circuits_equivalent_tasks_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_circuits_equivalent"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_circuits_equivalent",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__circuit_equivalent__total_resistance_value",
                count=2,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_physics__circuit_equivalent__total_capacitance_value",
                count=2,
                params={},
            ),
        ],
        strict_repro=False,
        max_attempts_per_instance=40,
        sampling_seed=61,
    )
    final_path = build_dataset(config, code_hash="physics-circuits-equivalent-smoke")

    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "physics" for record in train_records)
    assert {record["task"] for record in train_records} == {
        "task_physics__circuit_equivalent__total_resistance_value",
        "task_physics__circuit_equivalent__total_capacitance_value",
    }
    assert {record["scene_id"] for record in train_records} == {"circuit_equivalent"}
    assert {record["query_id"] for record in train_records} == {"single"}

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_physics__circuit_equivalent__total_resistance_value"]) == 2
    assert int(build_report["accepted_counts_by_task"]["task_physics__circuit_equivalent__total_capacitance_value"]) == 2

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
