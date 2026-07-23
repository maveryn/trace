"""Behavior tests for GUI control-board and record-table count tasks."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.control_board.control_state_condition_count import (
    PagesControlBoardControlStateConditionCountTask,
)
from trace_tasks.tasks.pages.control_board.state_extremum_group_label import (
    PagesControlBoardStateExtremumGroupLabelTask,
)
from trace_tasks.tasks.pages.record_table.enabled_action_for_type_count import (
    PagesRecordTableEnabledActionForTypeCountTask,
)
from trace_tasks.tasks.pages.record_table.selected_rows_with_status_count import (
    PagesRecordTableSelectedRowsWithStatusCountTask,
)
from trace_tasks.tasks.pages.record_table.value_threshold_in_group_count import (
    PagesRecordTableValueThresholdInGroupCountTask,
)

from tests.helpers import extract_prompt_json_example, read_jsonl


TABLE_PROMPT_KEYS = (
    "enabled_action_for_type_count",
    "selected_rows_with_status_count",
    "value_threshold_in_group_count",
)


def test_control_board_count_contract_matches_trace() -> None:
    task = PagesControlBoardControlStateConditionCountTask()
    for prompt_key in task.supported_query_ids:
        output = task.generate(55200, params={"query_id": prompt_key}, max_attempts=20)
        execution = output.trace_payload["execution_trace"]
        assert output.scene_id == "control_board"
        assert output.query_id == prompt_key
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert execution["query_id"] == prompt_key
        assert execution["prompt_query_key"] == prompt_key
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)
        assert len(execution["matching_control_ids"]) == int(output.answer_gt.value)
        assert output.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"


def test_control_board_state_extremum_group_contract_matches_trace() -> None:
    task = PagesControlBoardStateExtremumGroupLabelTask()
    for prompt_key in task.supported_query_ids:
        output = task.generate(55208, params={"query_id": prompt_key}, max_attempts=20)
        execution = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        answer_group = str(output.answer_gt.value)

        assert output.scene_id == "control_board"
        assert output.query_id == prompt_key
        assert output.answer_gt.type == "string"
        assert output.annotation_gt.type == "bbox"
        assert execution["query_id"] == prompt_key
        assert execution["prompt_query_key"] == prompt_key
        assert answer_group in render_map["group_bboxes_by_name"]
        assert output.annotation_gt.value == render_map["group_bboxes_by_name"][answer_group]
        assert output.trace_payload["projected_annotation"] == {"bbox": output.annotation_gt.value}
        assert output.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"

        counts: Counter[str] = Counter()
        for control in execution["controls"]:
            disabled = not bool(control["enabled"])
            selected_enabled = bool(control["selected"]) and bool(control["enabled"])
            if prompt_key == "disabled_extremum_group_label" and disabled:
                counts[str(control["group_name"])] += 1
            if prompt_key == "selected_enabled_extremum_group_label" and selected_enabled:
                counts[str(control["group_name"])] += 1
        for group in render_map["group_bboxes_by_name"]:
            counts.setdefault(str(group), 0)
        max_count = max(counts.values())
        assert counts[answer_group] == max_count
        assert sum(1 for value in counts.values() if int(value) == int(max_count)) == 1
        assert int(execution["target_state_count"]) == max_count


def test_record_table_count_contract_matches_trace() -> None:
    cases = (
        (PagesRecordTableEnabledActionForTypeCountTask(), "enabled_action_for_type_count"),
        (PagesRecordTableSelectedRowsWithStatusCountTask(), "selected_rows_with_status_count"),
        (PagesRecordTableValueThresholdInGroupCountTask(), "value_threshold_in_group_count"),
    )
    for task, branch in cases:
        output = task.generate(55240, params={"query_id": "single"}, max_attempts=20)
        execution = output.trace_payload["execution_trace"]
        assert output.scene_id == "record_table"
        assert output.query_id == "single"
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert execution["query_id"] == "single"
        assert execution["prompt_query_key"] == branch
        assert output.trace_payload["query_spec"]["prompt_variant"]["query_key"] == branch
        assert output.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)


def test_control_board_prompt_examples_match_integer_contract() -> None:
    task = PagesControlBoardControlStateConditionCountTask()
    out = task.generate(55200, params={"query_id": "disabled_controls_in_group_count"}, max_attempts=20)
    assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [[96, 218, 221, 300], [233, 218, 358, 300], [370, 310, 495, 392]],
        "answer": 3,
    }
    assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": 3}


def test_control_board_state_extremum_prompt_examples_match_string_bbox_contract() -> None:
    task = PagesControlBoardStateExtremumGroupLabelTask()
    out = task.generate(55208, params={"query_id": "disabled_extremum_group_label"}, max_attempts=20)
    assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [[72, 190, 612, 430]],
        "answer": "Review",
    }
    assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "Review"}


def test_control_board_balanced_sampling_defaults_cover_axes_and_answers() -> None:
    task = PagesControlBoardControlStateConditionCountTask()
    scene_variants: Counter[str] = Counter()
    information_treatments: Counter[str] = Counter()
    answers_by_prompt_key: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for index in range(150):
        out = task.generate(hash64(55300, task.task_id, index), params={}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        render_spec = out.trace_payload["render_spec"]
        prompt_key = str(execution["prompt_query_key"])
        assert execution["query_id"] in task.supported_query_ids
        assert execution["prompt_query_key"] == execution["query_id"]
        assert "style_variant" not in execution
        assert render_spec["information_scene_style"]["kind"] == "information_scene_style"
        scene_variants[str(execution["scene_variant"])] += 1
        information_treatments[str(render_spec["information_scene_style"]["treatment"])] += 1
        answers_by_prompt_key[prompt_key][int(execution["answer_value"])] += 1
    assert set(scene_variants.keys()) == {
        "office_document",
        "creative_workspace",
        "developer_ide",
        "cad_workspace",
        "scientific_plotter",
        "os_file_manager",
    }
    assert any(value.startswith("dark_") for value in information_treatments)
    assert any(not value.startswith("dark_") for value in information_treatments)
    assert set(answers_by_prompt_key["disabled_controls_in_group_count"].keys()).issubset({2, 3, 4, 5, 6, 7})
    assert set(answers_by_prompt_key["selected_enabled_controls_in_group_count"].keys()).issubset({2, 3, 4, 5, 6, 7})
    assert all(len(values) >= 5 for values in answers_by_prompt_key.values())


def test_control_board_state_extremum_defaults_cover_branches_groups_and_answers() -> None:
    task = PagesControlBoardStateExtremumGroupLabelTask()
    scene_variants: Counter[str] = Counter()
    answers_by_prompt_key: defaultdict[str, Counter[str]] = defaultdict(Counter)
    target_counts_by_prompt_key: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for index in range(120):
        out = task.generate(hash64(55308, task.task_id, index), params={}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        prompt_key = str(execution["prompt_query_key"])
        assert execution["query_id"] in task.supported_query_ids
        assert execution["prompt_query_key"] == execution["query_id"]
        scene_variants[str(execution["scene_variant"])] += 1
        answers_by_prompt_key[prompt_key][str(execution["answer_value"])] += 1
        target_counts_by_prompt_key[prompt_key][int(execution["target_state_count"])] += 1
    assert set(scene_variants.keys()) == {
        "office_document",
        "creative_workspace",
        "developer_ide",
        "cad_workspace",
        "scientific_plotter",
        "os_file_manager",
    }
    assert set(answers_by_prompt_key) == {
        "disabled_extremum_group_label",
        "selected_enabled_extremum_group_label",
    }
    assert all(set(values.keys()) == {"Layout", "Editing", "Review", "Output"} for values in answers_by_prompt_key.values())
    assert all(set(values.keys()).issubset({3, 4, 5}) for values in target_counts_by_prompt_key.values())
    assert all(len(values) == 3 for values in target_counts_by_prompt_key.values())


def test_record_table_balanced_sampling_defaults_cover_branches_and_answers() -> None:
    tasks = (
        PagesRecordTableEnabledActionForTypeCountTask(),
        PagesRecordTableSelectedRowsWithStatusCountTask(),
        PagesRecordTableValueThresholdInGroupCountTask(),
    )
    prompt_keys: Counter[str] = Counter()
    answers_by_prompt_key: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for task in tasks:
        for index in range(75):
            out = task.generate(hash64(55350, task.task_id, index), params={}, max_attempts=20)
            execution = out.trace_payload["execution_trace"]
            prompt_key = str(out.trace_payload["query_spec"]["prompt_variant"]["query_key"])
            assert execution["query_id"] == "single"
            prompt_keys[prompt_key] += 1
            answers_by_prompt_key[prompt_key][int(execution["answer_value"])] += 1
    assert set(prompt_keys.keys()) == set(TABLE_PROMPT_KEYS)
    assert set(answers_by_prompt_key["enabled_action_for_type_count"].keys()).issubset({2, 3, 4, 5, 6})
    assert set(answers_by_prompt_key["selected_rows_with_status_count"].keys()).issubset({2, 3, 4, 5, 6, 7})
    assert set(answers_by_prompt_key["value_threshold_in_group_count"].keys()).issubset({2, 3, 4, 5, 6, 7})


def test_gui_counting_filter_count_deterministic() -> None:
    task = PagesRecordTableValueThresholdInGroupCountTask()
    params = {"query_id": "single", "scene_variant": "cad_workspace", "style_variant": "contrast", "answer_value": 5}
    out_a = task.generate(55400, params=params, max_attempts=20)
    out_b = task.generate(55400, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_control_board_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__control_board__control_state_condition_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__control_board__control_state_condition_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__control_board__control_state_condition_count",
                count=5,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="gui-counting-filter-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 5
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["scene_id"] == "control_board" for record in train_records)
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__control_board__control_state_condition_count"]) == 5
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
