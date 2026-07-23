"""Behavior tests for migrated workspace tasks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.workspace.context_control_count import (
    OBJECTIVE_KEY as COUNT_OBJECTIVE_KEY,
)
from trace_tasks.tasks.pages.workspace.context_control_count import TASK_ID as COUNT_TASK_ID
from trace_tasks.tasks.pages.workspace.context_control_count import PagesWorkspaceContextControlCountTask
from trace_tasks.tasks.pages.workspace.context_guide_control_label import (
    OBJECTIVE_KEY as CONTEXT_GUIDE_OBJECTIVE_KEY,
)
from trace_tasks.tasks.pages.workspace.context_guide_control_label import TASK_ID as CONTEXT_GUIDE_TASK_ID
from trace_tasks.tasks.pages.workspace.context_guide_control_label import PagesWorkspaceContextGuideControlLabelTask
from trace_tasks.tasks.pages.workspace.control_label import (
    OBJECTIVE_KEY as CONTROL_OBJECTIVE_KEY,
)
from trace_tasks.tasks.pages.workspace.control_label import TASK_ID as CONTROL_TASK_ID
from trace_tasks.tasks.pages.workspace.control_label import PagesWorkspaceControlLabelTask
from trace_tasks.tasks.pages.workspace._lifecycle import SUPPORTED_WORKSPACE_VARIANTS

from tests.helpers import extract_prompt_json_example, read_jsonl


def test_pages_workspace_control_label_contract_matches_trace() -> None:
    task = PagesWorkspaceControlLabelTask()
    for offset, workspace_variant in enumerate(SUPPORTED_WORKSPACE_VARIANTS):
        out = task.generate(89200 + offset, params={"workspace_variant": workspace_variant}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        query_spec = out.trace_payload["query_spec"]
        witness = out.trace_payload["witness_symbolic"]

        assert out.scene_id == "workspace"
        assert out.query_id == SINGLE_QUERY_ID
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert out.annotation_gt.value == execution["target_control"]["bbox_px"]
        assert witness["type"] == "bbox"
        assert witness["value"] == out.annotation_gt.value
        assert execution["query_id"] == SINGLE_QUERY_ID
        assert execution["prompt_query_key"] == CONTROL_OBJECTIVE_KEY
        assert execution["source_query_id"] == CONTROL_OBJECTIVE_KEY
        assert execution["workspace_variant"] == workspace_variant
        assert query_spec["query_id"] == SINGLE_QUERY_ID
        assert query_spec["params"]["prompt_query_key"] == CONTROL_OBJECTIVE_KEY
        assert query_spec["params"]["query_id_probabilities"] == {SINGLE_QUERY_ID: 1.0}
        assert query_spec["prompt_variant"]["prompt_schema_version"] == "v1"


def test_pages_workspace_context_control_count_contract_matches_trace() -> None:
    task = PagesWorkspaceContextControlCountTask()
    params = {
        "workspace_variant": "canvas_tool",
        "target_context_index": 1,
        "target_state_id": "orange_warning",
        "answer_value": 3,
    }
    out = task.generate(89300, params=params, max_attempts=20)
    execution = out.trace_payload["execution_trace"]
    witness = out.trace_payload["witness_symbolic"]
    controls_by_id = {str(record["control_id"]): dict(record) for record in execution["controls"]}
    counted_ids = [str(value) for value in execution["counted_control_ids"]]
    expected_boxes = [controls_by_id[control_id]["bbox_px"] for control_id in counted_ids]

    assert out.scene_id == "workspace"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.to_dict() == {"type": "integer", "value": 3}
    assert out.annotation_gt.type == "bbox_set"
    assert out.annotation_gt.value == expected_boxes
    assert witness["type"] == "bbox_set"
    assert witness["counted_control_ids"] == counted_ids
    assert len(counted_ids) == 3
    for control_id in counted_ids:
        record = controls_by_id[control_id]
        assert int(record["context_index"]) == 1
        assert record["state_id"] == "orange_warning"
    assert execution["prompt_query_key"] == COUNT_OBJECTIVE_KEY
    assert execution["source_query_id"] == COUNT_OBJECTIVE_KEY
    assert execution["target_state_phrase"] == "orange warning"


def test_pages_workspace_context_guide_control_label_contract_matches_trace() -> None:
    task = PagesWorkspaceContextGuideControlLabelTask()
    out = task.generate(89330, params={"workspace_variant": "property_panel"}, max_attempts=20)
    execution = out.trace_payload["execution_trace"]
    query_spec = out.trace_payload["query_spec"]
    witness = out.trace_payload["witness_symbolic"]

    assert out.scene_id == "workspace"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == execution["target_label"]
    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == execution["target_control"]["bbox_px"]
    assert witness["type"] == "bbox"
    assert witness["value"] == out.annotation_gt.value
    assert set(witness["support_bbox_map"].keys()) == {
        "context_cue_guide",
        "context_row",
        "action_code_header",
        "target_control",
    }
    assert execution["prompt_query_key"] == CONTEXT_GUIDE_OBJECTIVE_KEY
    assert execution["source_query_id"] == CONTEXT_GUIDE_OBJECTIVE_KEY
    assert execution["context_cue_label"]
    assert int(execution["context_count"]) == 3
    assert query_spec["params"]["prompt_query_key"] == CONTEXT_GUIDE_OBJECTIVE_KEY
    assert "context_cue_guide" in execution["annotation_role_support_ids"]
    assert "action_cue_guide" not in execution["annotation_role_support_ids"]


def test_pages_workspace_context_control_count_allows_zero_annotation() -> None:
    out = PagesWorkspaceContextControlCountTask().generate(
        89301,
        params={"target_context_index": 0, "target_state_id": "blue_highlighted", "answer_value": 0},
        max_attempts=20,
    )
    assert out.answer_gt.to_dict() == {"type": "integer", "value": 0}
    assert out.annotation_gt.to_dict() == {"type": "bbox_set", "value": []}
    assert out.trace_payload["execution_trace"]["counted_control_ids"] == []


def test_pages_workspace_prompt_examples_match_annotation_contract() -> None:
    control_out = PagesWorkspaceControlLabelTask().generate(89400, params={}, max_attempts=20)
    count_out = PagesWorkspaceContextControlCountTask().generate(89401, params={}, max_attempts=20)
    context_guide_out = PagesWorkspaceContextGuideControlLabelTask().generate(89403, params={}, max_attempts=20)

    assert extract_prompt_json_example(control_out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [520, 360, 690, 444],
        "answer": "G",
    }
    assert extract_prompt_json_example(control_out.prompt_variants["answer_only"]) == {"answer": "G"}
    assert extract_prompt_json_example(count_out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [[410, 378, 560, 418], [580, 378, 730, 418]],
        "answer": 2,
    }
    assert extract_prompt_json_example(count_out.prompt_variants["answer_only"]) == {"answer": 2}
    assert extract_prompt_json_example(context_guide_out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [520, 360, 690, 444],
        "answer": "G",
    }
    assert extract_prompt_json_example(context_guide_out.prompt_variants["answer_only"]) == {"answer": "G"}


def test_pages_workspace_sampling_defaults_cover_axes_and_answers() -> None:
    control_task = PagesWorkspaceControlLabelTask()
    count_task = PagesWorkspaceContextControlCountTask()
    context_guide_task = PagesWorkspaceContextGuideControlLabelTask()
    scene_variants: Counter[str] = Counter()
    workspace_variants: Counter[str] = Counter()
    information_treatments: Counter[str] = Counter()
    label_answers: Counter[str] = Counter()
    count_answers: Counter[int] = Counter()
    context_counts: Counter[int] = Counter()

    for index in range(120):
        control_out = control_task.generate(hash64(89500, CONTROL_TASK_ID, index), params={}, max_attempts=20)
        control_execution = control_out.trace_payload["execution_trace"]
        render_spec = control_out.trace_payload["render_spec"]
        scene_variants[str(control_execution["scene_variant"])] += 1
        workspace_variants[str(control_execution["workspace_variant"])] += 1
        information_treatments[str(render_spec["information_scene_style"]["treatment"])] += 1
        label_answers[str(control_execution["target_label"])] += 1
        context_counts[int(control_execution["context_count"])] += 1

        count_out = count_task.generate(hash64(89501, COUNT_TASK_ID, index), params={}, max_attempts=20)
        count_execution = count_out.trace_payload["execution_trace"]
        count_answers[int(count_execution["answer_value"])] += 1

        context_guide_out = context_guide_task.generate(hash64(89503, CONTEXT_GUIDE_TASK_ID, index), params={}, max_attempts=20)
        context_guide_execution = context_guide_out.trace_payload["execution_trace"]
        label_answers[str(context_guide_execution["target_label"])] += 1

    assert set(scene_variants.keys()) == {
        "office_document",
        "creative_workspace",
        "developer_ide",
        "cad_workspace",
        "scientific_plotter",
        "os_file_manager",
    }
    assert set(workspace_variants.keys()) == set(SUPPORTED_WORKSPACE_VARIANTS)
    assert any(value.startswith("dark_") for value in information_treatments)
    assert any(not value.startswith("dark_") for value in information_treatments)
    assert set(context_counts.keys()) == {3, 4}
    assert len(label_answers) >= 12
    assert set(count_answers.keys()) == {0, 1, 2, 3, 4, 5}


def test_pages_workspace_task_is_deterministic() -> None:
    task = PagesWorkspaceContextControlCountTask()
    params = {
        "query_id": SINGLE_QUERY_ID,
        "scene_variant": "cad_workspace",
        "workspace_variant": "file_dialog",
        "information_scene_treatments": ["dark_console_panel"],
        "target_state_id": "gray_disabled",
        "answer_value": 2,
    }
    out_a = task.generate(89600, params=params, max_attempts=20)
    out_b = task.generate(89600, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_workspace_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / CONTROL_TASK_ID
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{CONTROL_TASK_ID}",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id=CONTROL_TASK_ID, count=2, params={}),
            BuildTaskConfig(task_id=COUNT_TASK_ID, count=2, params={}),
            BuildTaskConfig(task_id=CONTEXT_GUIDE_TASK_ID, count=2, params={}),
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash=f"{CONTROL_TASK_ID}-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 6
    assert all(record["domain"] == "pages" for record in train_records)
    assert {str(record["task"]) for record in train_records} == {
        CONTROL_TASK_ID,
        COUNT_TASK_ID,
        CONTEXT_GUIDE_TASK_ID,
    }
    assert all(record["query_id"] == SINGLE_QUERY_ID for record in train_records)
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][CONTROL_TASK_ID]) == 2
    assert int(build_report["accepted_counts_by_task"][COUNT_TASK_ID]) == 2
    assert int(build_report["accepted_counts_by_task"][CONTEXT_GUIDE_TASK_ID]) == 2
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
