"""Behavior tests for migrated web-action page tasks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.web_action.action_target_label import (
    SUPPORTED_QUERY_IDS as ACTION_TARGET_QUERY_IDS,
)
from trace_tasks.tasks.pages.web_action.action_target_label import (
    TASK_ID as ACTION_TARGET_TASK_ID,
)
from trace_tasks.tasks.pages.web_action.action_target_label import PagesWebActionActionTargetLabelTask
from trace_tasks.tasks.pages.web_action.guide_code_target_count import (
    SUPPORTED_QUERY_IDS as GUIDE_CODE_COUNT_QUERY_IDS,
)
from trace_tasks.tasks.pages.web_action.guide_code_target_count import (
    TASK_ID as GUIDE_CODE_COUNT_TASK_ID,
)
from trace_tasks.tasks.pages.web_action.guide_code_target_count import PagesWebActionGuideCodeTargetCountTask

from tests.helpers import extract_prompt_json_example, read_jsonl


CONTROL_FAMILY_BY_ACTION_QUERY = {
    "click_target_label": "click_target_label",
    "type_field_label": "type_field_label",
    "select_option_label": "select_option_label",
}
TARGET_ROLE_BY_ACTION_QUERY = {
    "click_target_label": "target_button",
    "type_field_label": "target_input",
    "select_option_label": "target_option",
}
CONTROL_FAMILY_BY_COUNT_QUERY = {
    "click_guide_code_target_count": "click_target_label",
    "type_field_guide_code_target_count": "type_field_label",
    "select_option_guide_code_target_count": "select_option_label",
}


def _bbox_union(left: list[float], right: list[float]) -> list[float]:
    return [
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    ]


def test_pages_web_action_target_contracts_match_trace() -> None:
    task = PagesWebActionActionTargetLabelTask()
    for offset, query_id in enumerate(ACTION_TARGET_QUERY_IDS):
        out = task.generate(99200 + offset, params={"query_id": query_id}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        query_spec = out.trace_payload["query_spec"]
        render_map = out.trace_payload["render_map"]
        target_control_id = str(execution["target_control_id"])
        expected_annotation = _bbox_union(
            render_map["control_bboxes_by_id"][target_control_id],
            render_map["candidate_label_badge_bboxes_by_id"][target_control_id],
        )

        assert out.scene_id == "web_action"
        assert out.query_id == query_id
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert out.annotation_gt.value == expected_annotation
        assert execution["query_id"] == query_id
        assert execution["prompt_query_key"] == query_id
        assert execution["source_query_id"] == query_id
        assert execution["control_family_key"] == CONTROL_FAMILY_BY_ACTION_QUERY[query_id]
        assert execution["target_annotation_role"] == TARGET_ROLE_BY_ACTION_QUERY[query_id]
        assert execution["target_annotation_control_id"] == target_control_id
        assert query_spec["query_id"] == query_id
        assert query_spec["params"]["query_id_probabilities"][query_id] == 1.0
        assert query_spec["prompt_variant"]["prompt_schema_version"] == "v1"
        assert out.trace_payload["projected_annotation"] == {
            "type": "bbox",
            "bbox": expected_annotation,
            "pixel_bbox": expected_annotation,
        }


def test_pages_web_action_guide_code_count_contracts_match_trace() -> None:
    task = PagesWebActionGuideCodeTargetCountTask()
    for offset, query_id in enumerate(GUIDE_CODE_COUNT_QUERY_IDS):
        out = task.generate(99280 + offset, params={"query_id": query_id}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        query_spec = out.trace_payload["query_spec"]
        render_map = out.trace_payload["render_map"]
        guide_code = str(execution["guide_code"])
        matching_control_ids = [
            str(control["control_id"])
            for control in execution["controls"]
            if str(control["action_code_label"]) == guide_code
        ]
        expected_annotations = [
            _bbox_union(
                render_map["control_bboxes_by_id"][control_id],
                render_map["candidate_label_badge_bboxes_by_id"][control_id],
            )
            for control_id in matching_control_ids
        ]

        assert out.scene_id == "web_action"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert out.answer_gt.value == len(matching_control_ids)
        assert out.annotation_gt.value == expected_annotations
        assert execution["query_id"] == query_id
        assert execution["prompt_query_key"] == query_id
        assert execution["source_query_id"] == query_id
        assert execution["control_family_key"] == CONTROL_FAMILY_BY_COUNT_QUERY[query_id]
        assert execution["matching_control_ids"] == matching_control_ids
        assert execution["matching_control_count"] == len(matching_control_ids)
        assert query_spec["query_id"] == query_id
        assert query_spec["params"]["query_id_probabilities"][query_id] == 1.0
        assert out.trace_payload["projected_annotation"] == {
            "type": "bbox_set",
            "bbox_set": expected_annotations,
            "pixel_bbox_set": expected_annotations,
        }
        assert 3 <= int(out.answer_gt.value) <= 6


def test_pages_web_action_prompt_examples_match_annotation_contract() -> None:
    target_task = PagesWebActionActionTargetLabelTask()
    count_task = PagesWebActionGuideCodeTargetCountTask()
    for offset, query_id in enumerate(ACTION_TARGET_QUERY_IDS):
        out = target_task.generate(99300 + offset, params={"query_id": query_id}, max_attempts=20)
        assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == {
            "annotation": [410, 378, 584, 442],
            "answer": "G",
        }
        assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "G"}
    for offset, query_id in enumerate(GUIDE_CODE_COUNT_QUERY_IDS):
        out = count_task.generate(99340 + offset, params={"query_id": query_id}, max_attempts=20)
        assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == {
            "annotation": [[410, 378, 584, 442], [410, 452, 584, 516], [410, 526, 584, 590]],
            "answer": 3,
        }
        assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": 3}


def test_pages_web_action_sampling_defaults_cover_axes_and_answers() -> None:
    task = PagesWebActionActionTargetLabelTask()
    query_ids: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    information_treatments: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    for index in range(180):
        out = task.generate(hash64(99400, ACTION_TARGET_TASK_ID, index), params={}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        render_spec = out.trace_payload["render_spec"]
        assert "style_variant" not in execution
        assert render_spec["information_scene_style"]["kind"] == "information_scene_style"
        query_ids[str(execution["query_id"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        information_treatments[str(render_spec["information_scene_style"]["treatment"])] += 1
        answers[str(execution["target_label"])] += 1

    assert set(query_ids.keys()) == set(ACTION_TARGET_QUERY_IDS)
    assert set(scene_variants.keys()) == {
        "shop_catalog",
        "travel_booking",
        "support_center",
        "learning_portal",
        "finance_portal",
        "content_cms",
    }
    assert any(value.startswith("dark_") for value in information_treatments)
    assert any(not value.startswith("dark_") for value in information_treatments)
    assert len(answers) >= 12

    count_task = PagesWebActionGuideCodeTargetCountTask()
    count_query_ids: Counter[str] = Counter()
    count_answers: Counter[int] = Counter()
    for index in range(90):
        out = count_task.generate(hash64(99480, GUIDE_CODE_COUNT_TASK_ID, index), params={}, max_attempts=20)
        count_query_ids[str(out.query_id)] += 1
        count_answers[int(out.answer_gt.value)] += 1
    assert set(count_query_ids.keys()) == set(GUIDE_CODE_COUNT_QUERY_IDS)
    assert set(count_answers.keys()).issubset({3, 4, 5, 6})


def test_pages_web_action_task_is_deterministic() -> None:
    task = PagesWebActionActionTargetLabelTask()
    params = {
        "query_id": "select_option_label",
        "scene_variant": "finance_portal",
        "information_scene_treatments": ["dark_console_panel"],
        "target_label": "M",
    }
    out_a = task.generate(99500, params=params, max_attempts=20)
    out_b = task.generate(99500, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_web_action_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / ACTION_TARGET_TASK_ID
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{ACTION_TARGET_TASK_ID}",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id=ACTION_TARGET_TASK_ID, count=4, params={}),
            BuildTaskConfig(task_id=GUIDE_CODE_COUNT_TASK_ID, count=4, params={}),
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash=f"{ACTION_TARGET_TASK_ID}-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 8
    assert all(record["domain"] == "pages" for record in train_records)
    assert {record["task"] for record in train_records} == {
        ACTION_TARGET_TASK_ID,
        GUIDE_CODE_COUNT_TASK_ID,
    }
    assert {record["query_id"] for record in train_records}.issubset(
        set(ACTION_TARGET_QUERY_IDS) | set(GUIDE_CODE_COUNT_QUERY_IDS)
    )
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][ACTION_TARGET_TASK_ID]) == 4
    assert int(build_report["accepted_counts_by_task"][GUIDE_CODE_COUNT_TASK_ID]) == 4
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
