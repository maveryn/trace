"""Tests for icon pair-grid reference color-pair match selection."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.pair_grid.reference_color_pair_match_label import IconsPairGridReferenceColorPairMatchLabelTask
from trace_tasks.tasks.icons.shared.icon_transform import IDENTITY_TRANSFORM_ID
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def test_icons_pair_grid_reference_color_pair_match_label_contract_matches_scene() -> None:
    task = IconsPairGridReferenceColorPairMatchLabelTask()
    out = task.generate(51200, params={"answer_label": "D"}, max_attempts=200)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_entities = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "scene"]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.query_id == "single"
    assert execution["query_id"] == "single"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id_probabilities"] == {"single": 1.0}
    assert trace["scene_ir"]["scene_kind"] == "icons_reference_pair_color_pair_match_label"
    assert execution["question_format"] == "select_scene_cell_matching_reference_color_pair"
    assert int(execution["option_count"]) == 6
    assert len(scene_entities) == 6
    assert len(set(str(entity["label"]) for entity in scene_entities)) == 6

    reference_pair = trace["render_map"]["anchors"]["reference_pair"]
    reference_color_pair = [list(reference_pair["left_tint_rgb"]), list(reference_pair["right_tint_rgb"])]
    assert reference_color_pair == execution["reference_color_pair_rgb"]
    assert reference_pair["left_tint_rgb"] != reference_pair["right_tint_rgb"]
    assert reference_pair["transform_id"] == IDENTITY_TRANSFORM_ID
    assert float(reference_pair["left_size_scale"]) == 1.0
    assert float(reference_pair["right_size_scale"]) == 1.0

    matching = []
    scene_icon_ids = {str(entity["icon_id"]) for entity in scene_entities}
    assert str(reference_pair["icon_id"]) not in scene_icon_ids
    assert len(scene_icon_ids) == 6
    for entity in scene_entities:
        color_pair = [list(entity["left_tint_rgb"]), list(entity["right_tint_rgb"])]
        is_match = color_pair == reference_color_pair
        assert bool(entity["is_match"]) == is_match
        assert str(entity["transform_id"]) == IDENTITY_TRANSFORM_ID
        assert float(entity["left_size_scale"]) == 1.0
        assert float(entity["right_size_scale"]) == 1.0
        if is_match:
            matching.append(entity)
    assert [str(entity["label"]) for entity in matching] == ["D"]

    selected_bbox = list(matching[0]["cell_bbox_xyxy"])
    assert out.annotation_gt.value == selected_bbox
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == selected_bbox
    assert trace["projected_annotation"]["pixel_bbox"] == selected_bbox
    assert trace["witness_symbolic"]["answer_label"] == "D"

    style = trace["render_spec"]["style"]
    assert int(style["text_legibility"]["failure_count"]) == 0
    assert len(style["sampled_palette_rgb"]) >= 4


def test_icons_pair_grid_reference_color_pair_match_label_prompt_example_matches_contract() -> None:
    task = IconsPairGridReferenceColorPairMatchLabelTask()
    out = task.generate(51202, params={"answer_label": "A"}, max_attempts=200)
    assert _extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "A"}
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_and_annotation == {"annotation": [336, 104, 506, 274], "answer": "A"}


def test_icons_pair_grid_reference_color_pair_match_label_deterministic_and_build_smoke(tmp_path: Path) -> None:
    task = IconsPairGridReferenceColorPairMatchLabelTask()
    out_a = task.generate(51201, params={}, max_attempts=200)
    out_b = task.generate(51201, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()

    task_id = "task_icons__pair_grid__reference_color_pair_match_label"
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
    final_path = build_dataset(config, code_hash="icons-pair-grid-color-pair-match-label-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "icons" for record in train_records)
    assert all(record["scene_id"] == "pair_grid" for record in train_records)
    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][task_id]) == 4
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
