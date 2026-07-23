"""Behavior tests for the icon mirror-symmetry match-label task."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.tasks.icons.mirror_grid.mirror_symmetry_match_label import (
    IconsMirrorGridMirrorSymmetryMatchLabelTask,
)


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_icons_relation_mirror_symmetry_match_contract_matches_scene() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    out = task.generate(
        15110,
        params={"mirror_signature": "mirror_diagonal_main", "option_count": 6, "answer_label": "C"},
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_entities = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "scene"]
    reference_entities = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "reference"]
    matching_entities = [entity for entity in scene_entities if bool(entity["is_match"])]

    assert len(reference_entities) == 1
    assert len(scene_entities) == 6
    assert len(matching_entities) == 1
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value) == ["matching_option_cell", "reference_cell"]
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert trace["query_spec"]["prompt_variant_active_key"] == "answer_and_annotation"
    assert trace["scene_ir"]["scene_kind"] == "icons_reference_grid_mirror_symmetry_match_label"
    assert execution["question_format"] == "select_option_cell_matching_reference_mirror_symmetry"
    assert out.query_id == "single"
    assert trace["query_spec"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert execution["internal_query_id"] == "mirror_diagonal_main"
    assert execution["mirror_signature"] == "mirror_diagonal_main"
    assert int(execution["option_count"]) == 6
    assert int(execution["distractor_count"]) == 5
    assert execution["answer_label"] == "C"
    assert execution["matching_cell_label"] == "C"
    assert execution["option_labels"] == ["A", "B", "C", "D", "E", "F"]

    reference_cell = reference_entities[0]
    assert out.annotation_gt.value["reference_cell"] == list(reference_cell["cell_bbox_xyxy"])
    matching_cell = matching_entities[0]
    assert str(matching_cell["label"]) == "C"
    assert out.annotation_gt.value["matching_option_cell"] == list(matching_cell["cell_bbox_xyxy"])
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value

    assert str(reference_cell["symmetry_id"]) == "mirror_diagonal_main"
    assert bool(reference_cell["has_vertical_symmetry"]) is False
    assert bool(reference_cell["has_horizontal_symmetry"]) is False
    assert bool(reference_cell["has_diagonal_main_symmetry"]) is True
    assert bool(reference_cell["has_diagonal_anti_symmetry"]) is False
    assert str(matching_cell["symmetry_id"]) == "mirror_diagonal_main"

    sampled_palette = [
        tuple(int(channel) for channel in color)
        for color in trace["render_spec"]["style"]["sampled_palette_rgb"]
    ]
    assert len(sampled_palette) >= 2
    for entity in scene_entities:
        assert str(entity["label"]) in {"A", "B", "C", "D", "E", "F"}
        assert bool(entity["is_match"]) == (str(entity["label"]) == "C")
        for placement in entity["placements"]:
            assert tuple(int(channel) for channel in placement["tint_rgb"]) in sampled_palette


def test_icons_relation_mirror_symmetry_match_supports_four_options() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    out = task.generate(
        15111,
        params={"mirror_signature": "mirror_horizontal", "option_count": 4, "answer_label": "D"},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    assert out.answer_gt.value == "D"
    assert int(execution["option_count"]) == 4
    assert execution["option_labels"] == ["A", "B", "C", "D"]
    assert execution["matching_cell_label"] == "D"
    assert len([entity for entity in out.trace_payload["scene_ir"]["entities"] if bool(entity.get("is_match"))]) == 1


def test_icons_relation_mirror_symmetry_match_prompt_example_matches_contract() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    out = task.generate(
        15112,
        params={"mirror_signature": "mirror_horizontal", "option_count": 6, "answer_label": "B"},
        max_attempts=200,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "C"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert sorted(answer_and_annotation["annotation"]) == ["matching_option_cell", "reference_cell"]
    assert answer_and_annotation["answer"] == "C"


def test_icons_relation_mirror_symmetry_match_supports_both_axes_reference() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    out = task.generate(
        15114,
        params={"mirror_signature": "mirror_both_axes", "option_count": 6, "answer_label": "E"},
        max_attempts=200,
    )
    reference_cell = next(
        entity for entity in out.trace_payload["scene_ir"]["entities"] if str(entity.get("panel")) == "reference"
    )
    matching_cell = next(
        entity for entity in out.trace_payload["scene_ir"]["entities"] if bool(entity.get("is_match"))
    )
    assert out.answer_gt.value == "E"
    assert str(reference_cell["symmetry_id"]) == "mirror_both_axes"
    assert str(matching_cell["label"]) == "E"
    assert str(matching_cell["symmetry_id"]) == "mirror_both_axes"
    assert bool(reference_cell["has_vertical_symmetry"]) is True
    assert bool(reference_cell["has_horizontal_symmetry"]) is True
    assert bool(reference_cell["has_diagonal_main_symmetry"]) is False
    assert bool(reference_cell["has_diagonal_anti_symmetry"]) is False


def test_icons_relation_mirror_symmetry_match_sampling_defaults() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    option_counts: Counter[int] = Counter()
    signature_counts: Counter[str] = Counter()
    answer_labels: Counter[str] = Counter()
    for index in range(60):
        out = task.generate(151130 + index, params={}, max_attempts=200)
        execution = out.trace_payload["execution_trace"]
        option_count = int(execution["option_count"])
        option_counts[option_count] += 1
        signature_counts[str(execution["mirror_signature"])] += 1
        answer_labels[str(execution["answer_label"])] += 1
        assert out.query_id == "single"
        assert str(execution["query_id"]) == "single"
        assert str(execution["internal_query_id"]) == str(execution["mirror_signature"])
        assert option_count in {4, 6}
        assert str(out.answer_gt.value) in set(execution["option_labels"])
        assert str(out.answer_gt.value) == str(execution["matching_cell_label"])
        assert len([entity for entity in out.trace_payload["scene_ir"]["entities"] if bool(entity.get("is_match"))]) == 1

    assert set(option_counts) == {4, 6}
    assert set(signature_counts) == {
        "mirror_vertical",
        "mirror_horizontal",
        "mirror_diagonal_main",
        "mirror_diagonal_anti",
        "mirror_both_axes",
    }
    assert set(answer_labels) >= {"A", "B", "C", "D"}
