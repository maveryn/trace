"""Tests for icon pair-grid reference transform match selection."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.pair_grid.reference_transform_match_label import IconsPairGridReferenceTransformMatchLabelTask
from trace_tasks.tasks.icons.shared.icon_assets import icon_transform_signature
from trace_tasks.tasks.icons.shared.icon_transform import IDENTITY_TRANSFORM_ID
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def test_icons_pair_grid_reference_transform_match_label_contract_matches_scene() -> None:
    task = IconsPairGridReferenceTransformMatchLabelTask()
    out = task.generate(
        14210,
        params={
            "answer_label": "C",
            "transform_ids": ["rot90", "rot180", "rot270", "flip_h", "flip_v"],
        },
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_entities = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "scene"]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert trace["scene_ir"]["scene_kind"] == "icons_reference_pair_transform_match_label"
    assert trace["scene_ir"]["query_id"] == "single"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id_probabilities"] == {"single": 1.0}
    assert execution["question_format"] == "select_scene_cell_matching_reference_transform"
    assert out.query_id == "single"
    assert execution["query_id"] == "single"
    assert int(execution["option_count"]) == 4
    assert len(scene_entities) == 4
    assert len(set(str(entity["label"]) for entity in scene_entities)) == 4

    reference_transform_id = str(execution["reference_transform_id"])
    matching = [entity for entity in scene_entities if str(entity["transform_id"]) == reference_transform_id]
    assert [str(entity["label"]) for entity in matching] == ["C"]
    selected_bbox = list(matching[0]["cell_bbox_xyxy"])
    assert out.annotation_gt.value == selected_bbox
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == selected_bbox
    assert trace["projected_annotation"]["pixel_bbox"] == selected_bbox
    assert trace["witness_symbolic"]["answer_label"] == "C"

    style = trace["render_spec"]["style"]
    assert int(style["text_legibility"]["failure_count"]) == 0
    assert {str(record["role"]) for record in style["text_legibility"]["records"]} >= {
        "icon_panel_header_text",
        "icon_cell_label_text",
    }

    for entity in scene_entities:
        label = str(entity["label"])
        icon_id = str(entity["icon_id"])
        transform_id = str(entity["transform_id"])
        assert bool(entity["is_match"]) == (label == "C")
        identity_sig = icon_transform_signature(icon_id, 72, IDENTITY_TRANSFORM_ID)
        active_sig = icon_transform_signature(icon_id, 72, transform_id)
        reference_sig = icon_transform_signature(icon_id, 72, reference_transform_id)
        assert active_sig != identity_sig
        if label == "C":
            assert transform_id == reference_transform_id
        else:
            assert transform_id != reference_transform_id
            assert active_sig != reference_sig


def test_icons_pair_grid_reference_transform_match_label_prompt_example_matches_contract() -> None:
    task = IconsPairGridReferenceTransformMatchLabelTask()
    out = task.generate(14212, params={"answer_label": "A"}, max_attempts=200)
    assert _extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "A"}
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_and_annotation == {"annotation": [336, 104, 506, 274], "answer": "A"}


def test_icons_pair_grid_reference_transform_match_label_deterministic_and_build_smoke(tmp_path: Path) -> None:
    task = IconsPairGridReferenceTransformMatchLabelTask()
    out_a = task.generate(14200, params={}, max_attempts=200)
    out_b = task.generate(14200, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()

    task_id = "task_icons__pair_grid__reference_transform_match_label"
    output_root = tmp_path / task_id
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{task_id}",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=task_id, count=4, params={"query_id": "single"})],
        strict_repro=False,
        max_attempts_per_instance=200,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="icons-pair-grid-transform-match-label-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "icons" for record in train_records)
    assert all(record["scene_id"] == "pair_grid" for record in train_records)
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][task_id]) == 4
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
