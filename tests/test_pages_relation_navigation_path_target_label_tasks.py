"""Behavior tests for navigation-flow page tasks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.navigation_flow.navigation_path_target_label import (
    PagesNavigationFlowNavigationPathTargetLabelTask,
)
from trace_tasks.tasks.pages.navigation_flow.same_group_target_label import (
    PagesNavigationFlowSameGroupTargetLabelTask,
)
from tests.helpers import extract_prompt_json_example, read_jsonl


TASK_CASES = (
    (
        "menu_path_target_label",
        "menu_path_target_label",
        "menu_path",
    ),
    (
        "sidebar_tree_target_label",
        "sidebar_tree_target_label",
        "sidebar_tree",
    ),
    (
        "ribbon_group_command_label",
        "ribbon_group_command_label",
        "ribbon_group",
    ),
)
TASK_ID = "task_pages__navigation_flow__navigation_path_target_label"
TASK_CLS = PagesNavigationFlowNavigationPathTargetLabelTask
SAME_GROUP_TASK_ID = "task_pages__navigation_flow__same_group_target_label"
SAME_GROUP_TASK_CLS = PagesNavigationFlowSameGroupTargetLabelTask


def test_navigation_flow_public_task_contracts_match_trace() -> None:
    for prompt_key, query_id, surface in TASK_CASES:
        out = TASK_CLS().generate(78200, params={"query_id": query_id}, max_attempts=20)
        assert out.scene_id == "navigation_flow"
        assert out.query_id == query_id
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        execution = out.trace_payload["execution_trace"]
        assert execution["query_id"] == query_id
        assert execution["prompt_query_key"] == prompt_key
        assert execution["source_query_id"] == prompt_key
        assert execution["navigation_surface"] == surface
        assert execution["target_label"] == out.answer_gt.value
        assert out.annotation_gt.value == execution["target_control"]["bbox_px"]
        assert set(execution["query_id_probabilities"].keys()) == {
            "menu_path_target_label",
            "sidebar_tree_target_label",
            "ribbon_group_command_label",
        }
        assert execution["query_id_probabilities"][query_id] == 1.0
        assert out.trace_payload["query_spec"]["scene_id"] == "navigation_flow"
        assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_navigation_flow_same_group_task_contracts_match_trace() -> None:
    for _prompt_key, _query_id, surface in TASK_CASES:
        out = SAME_GROUP_TASK_CLS().generate(79200, params={"navigation_surface": surface}, max_attempts=20)
        assert out.scene_id == "navigation_flow"
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        execution = out.trace_payload["execution_trace"]
        assert execution["query_id"] == "single"
        assert execution["prompt_query_key"] == "same_group_target_label"
        assert execution["source_query_id"] == "same_group_target_label"
        assert execution["navigation_surface"] == surface
        assert execution["target_label"] == out.answer_gt.value
        assert execution["same_group_answer_control_id"] == execution["target_control_id"]
        assert execution["reference_control_id"] != execution["same_group_answer_control_id"]
        assert len(execution["same_group_control_ids"]) == 2
        assert set(execution["same_group_control_ids"]) == {
            execution["reference_control_id"],
            execution["same_group_answer_control_id"],
        }
        assert out.annotation_gt.value == execution["target_control"]["bbox_px"]
        assert "reference_overlay" not in out.trace_payload["render_spec"]
        assert f'"{execution["reference_label"]}"' in out.prompt
        assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_navigation_flow_prompt_examples_match_annotation_contracts() -> None:
    expected_examples = {
        "menu_path_target_label": {
            "annotation": [112, 316, 286, 354],
            "answer": "G",
        },
        "sidebar_tree_target_label": {
            "annotation": [112, 316, 286, 354],
            "answer": "G",
        },
        "ribbon_group_command_label": {
            "annotation": [112, 316, 286, 354],
            "answer": "G",
        },
    }
    for prompt_key, query_id, _surface in TASK_CASES:
        out = TASK_CLS().generate(78200, params={"query_id": query_id}, max_attempts=20)
        assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == expected_examples[prompt_key]
        assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "G"}

    same_group_out = SAME_GROUP_TASK_CLS().generate(78200, params={"navigation_surface": "menu_path"}, max_attempts=20)
    assert extract_prompt_json_example(same_group_out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [112, 316, 286, 354],
        "answer": "G",
    }
    assert extract_prompt_json_example(same_group_out.prompt_variants["answer_only"]) == {"answer": "G"}


def test_navigation_flow_sampling_defaults_cover_scene_axes_and_answers() -> None:
    scene_variants: Counter[str] = Counter()
    information_treatments: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    surfaces: Counter[str] = Counter()
    for index in range(144):
        out = TASK_CLS().generate(
            hash64(78300, "navigation_flow.navigation_path_target_label", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        render_spec = out.trace_payload["render_spec"]
        assert "style_variant" not in execution
        assert render_spec["information_scene_style"]["kind"] == "information_scene_style"
        scene_variants[str(execution["scene_variant"])] += 1
        information_treatments[str(render_spec["information_scene_style"]["treatment"])] += 1
        answers[str(execution["target_label"])] += 1
        surfaces[str(execution["navigation_surface"])] += 1
    assert set(scene_variants.keys()) == {
        "office_document",
        "creative_workspace",
        "developer_ide",
        "cad_workspace",
        "scientific_plotter",
        "os_file_manager",
    }
    assert set(surfaces.keys()) == {"menu_path", "sidebar_tree", "ribbon_group"}
    assert any(value.startswith("dark_") for value in information_treatments)
    assert any(not value.startswith("dark_") for value in information_treatments)
    assert len(answers) >= 12

    same_group_surfaces: Counter[str] = Counter()
    same_group_answers: Counter[str] = Counter()
    for index in range(96):
        out = SAME_GROUP_TASK_CLS().generate(
            hash64(79300, "navigation_flow.same_group_target_label", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        same_group_surfaces[str(execution["navigation_surface"])] += 1
        same_group_answers[str(execution["target_label"])] += 1
    assert set(same_group_surfaces.keys()) == {"menu_path", "sidebar_tree", "ribbon_group"}
    assert len(same_group_answers) >= 12


def test_navigation_flow_deterministic() -> None:
    task = TASK_CLS()
    params = {
        "query_id": "ribbon_group_command_label",
        "scene_variant": "cad_workspace",
        "information_scene_treatments": ["dark_console_panel"],
        "target_label": "M",
    }
    out_a = task.generate(78400, params=params, max_attempts=20)
    out_b = task.generate(78400, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()

    same_group_task = SAME_GROUP_TASK_CLS()
    same_group_params = {
        "navigation_surface": "menu_path",
        "scene_variant": "developer_ide",
        "information_scene_treatments": ["light_blueprint"],
        "target_label": "M",
    }
    same_group_a = same_group_task.generate(79400, params=same_group_params, max_attempts=20)
    same_group_b = same_group_task.generate(79400, params=same_group_params, max_attempts=20)
    assert same_group_a.answer_gt.to_dict() == same_group_b.answer_gt.to_dict()
    assert same_group_a.annotation_gt.to_dict() == same_group_b.annotation_gt.to_dict()
    assert same_group_a.trace_payload["execution_trace"] == same_group_b.trace_payload["execution_trace"]
    assert same_group_a.prompt == same_group_b.prompt
    assert same_group_a.image.tobytes() == same_group_b.image.tobytes()


def test_navigation_flow_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / TASK_ID
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{TASK_ID}",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id=TASK_ID, count=4, params={}),
            BuildTaskConfig(task_id=SAME_GROUP_TASK_ID, count=4, params={}),
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="navigation-flow-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 8
    assert all(record["domain"] == "pages" for record in train_records)
    curriculum_records = read_jsonl(final_path / "curriculum_index.jsonl")
    assert len(curriculum_records) == 8
    assert {record["task"] for record in curriculum_records} == {TASK_ID, SAME_GROUP_TASK_ID}
    assert {record["query_id"] for record in curriculum_records if record["task"] == TASK_ID}.issubset(
        {"menu_path_target_label", "sidebar_tree_target_label", "ribbon_group_command_label"}
    )
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][TASK_ID]) == 4
    assert int(build_report["accepted_counts_by_task"][SAME_GROUP_TASK_ID]) == 4
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
