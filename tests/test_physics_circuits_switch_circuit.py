"""Contract tests for physics switch-circuit lit-bulb counting."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.switch_circuit.lit_bulb_count import (
    PhysicsSwitchCircuitLitBulbCountTask,
)
from trace_tasks.tasks.physics.switch_circuit.shared.circuitry import (
    lit_bulbs_from_edges,
    make_edges,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from tests.helpers import read_jsonl


def _assert_bbox_set_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox_set"
    for bbox in out.annotation_gt.value:
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height


def test_physics_switch_circuit_target_answer_support() -> None:
    task = PhysicsSwitchCircuitLitBulbCountTask()
    for target_answer in range(6):
        out = task.generate(
            28800 + target_answer,
            params={"target_answer": target_answer},
            max_attempts=20,
        )
        render_map = out.trace_payload["render_map"]

        assert out.scene_id == "switch_circuit"
        assert out.query_id == "single"
        assert out.trace_payload["execution_trace"]["internal_query_id"] == "lit_bulb_count"
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == target_answer
        assert len(out.annotation_gt.value) == target_answer
        assert render_map["lit_bulbs"] == out.trace_payload["execution_trace"]["lit_bulbs"]
        assert render_map["annotation_bbox_set"] == out.annotation_gt.value
        assert len(render_map["bulb_bboxes"]) == 5
        assert len(render_map["switch_bboxes"]) == 5
        _assert_bbox_set_in_bounds(out)


def test_physics_switch_circuit_explicit_switch_state_logic() -> None:
    states = {
        "S1": "closed",
        "S2": "closed",
        "S3": "open",
        "S4": "closed",
        "S5": "open",
    }
    out = PhysicsSwitchCircuitLitBulbCountTask().generate(
        28831,
        params={"target_answer": 3, "switch_states": states},
        max_attempts=20,
    )

    assert out.answer_gt.value == 3
    assert out.trace_payload["execution_trace"]["lit_bulbs"] == ["B1", "B2", "B4"]
    assert len(out.annotation_gt.value) == 3
    assert out.annotation_gt.value == [
        out.trace_payload["render_map"]["bulb_bboxes"]["B1"],
        out.trace_payload["render_map"]["bulb_bboxes"]["B2"],
        out.trace_payload["render_map"]["bulb_bboxes"]["B4"],
    ]


def test_physics_switch_circuit_empty_annotation_for_zero_count() -> None:
    out = PhysicsSwitchCircuitLitBulbCountTask().generate(
        28841,
        params={
            "target_answer": 0,
            "switch_states": {
                "S1": "open",
                "S2": "open",
                "S3": "closed",
                "S4": "closed",
                "S5": "open",
            },
        },
        max_attempts=20,
    )

    assert out.answer_gt.value == 0
    assert out.annotation_gt.type == "bbox_set"
    assert out.annotation_gt.value == []
    assert out.trace_payload["projected_annotation"]["bbox_set"] == []
    assert out.trace_payload["execution_trace"]["lit_bulbs"] == []


def test_physics_switch_circuit_graph_edge_lighting_logic() -> None:
    states = {
        "S1": True,
        "S2": True,
        "S3": False,
        "S4": True,
        "S5": False,
    }

    assert lit_bulbs_from_edges(make_edges(states)) == ("B1", "B2", "B4")
    assert lit_bulbs_from_edges(make_edges({label: False for label in states})) == ()
    assert lit_bulbs_from_edges(make_edges({label: True for label in states})) == ("B1", "B2", "B3", "B4", "B5")


def test_physics_switch_circuit_open_s2_middle_branch_is_not_lit() -> None:
    states = {
        "S1": True,
        "S2": False,
        "S3": True,
        "S4": True,
        "S5": True,
    }

    assert lit_bulbs_from_edges(make_edges(states)) == ("B1", "B2", "B5")


def test_physics_switch_circuit_is_deterministic() -> None:
    params = {
        "target_answer": 4,
        "accent_color_name": "cyan",
    }
    task = PhysicsSwitchCircuitLitBulbCountTask()
    out_a = task.generate(28851, params=params, max_attempts=20)
    out_b = task.generate(28851, params=params, max_attempts=20)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_switch_circuit_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("physics", "switch_circuit")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__switch_circuit__lit_bulb_count",
    )
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/switch_circuit/physics_switch_circuit_v1.json").read_text(encoding="utf-8"))

    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert set(generation["scene_variant_weights"]) == {"mixed_branch"}
    assert list(generation["target_answer_support"]) == [0, 1, 2, 3, 4, 5]
    assert int(rendering["canvas_width"]) == 1280
    assert str(prompt["bundle_id"]) == "physics_switch_circuit_v1"
    assert str(prompt["task_key"]) == "lit_bulb_count_query"
    assert "single" in bundle["templates"]["query"]
    assert len(bundle["templates"]["query"]["single"]) == 5
    assert "scene:switch_circuit_diagram" in bundle["required_slots_by_key"]


def test_physics_switch_circuit_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_switch_circuit"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_switch_circuit",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__switch_circuit__lit_bulb_count",
                count=2,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=89,
    )
    final_path = build_dataset(config, code_hash="physics-switch-circuit-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 2
    assert all(record["domain"] == "physics" for record in train_records)
    assert {record["task"] for record in train_records} == {"task_physics__switch_circuit__lit_bulb_count"}
    assert {record["scene_id"] for record in train_records} == {"switch_circuit"}
    assert {record["query_id"] for record in train_records} == {"single"}

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
