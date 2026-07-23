"""Behavior tests for the icon mirror-grid missing-cell task."""

from __future__ import annotations

import json

from trace_tasks.tasks.icons.mirror_grid.missing_mirror_cell_label import (
    IconsMirrorGridMissingMirrorCellLabelTask,
)


def _prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def test_icons_mirror_grid_missing_cell_contract_matches_scene() -> None:
    task = IconsMirrorGridMissingMirrorCellLabelTask()
    out = task.generate(
        27110,
        params={"mirror_axis": "vertical", "option_count": 4, "answer_label": "C"},
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    grid_cells = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "grid" and "row" in entity]
    option_cells = [entity for entity in trace["scene_ir"]["entities"] if str(entity.get("panel")) == "options"]
    missing_cells = [entity for entity in grid_cells if bool(entity.get("is_missing"))]
    answer_options = [entity for entity in option_cells if bool(entity.get("is_answer"))]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["query_spec"]["query_id"] == "single"
    assert out.query_id == "single"
    assert execution["query_id"] == "single"
    assert execution["question_format"] == "select_option_icon_completing_mirror_symmetry"
    assert trace["scene_ir"]["scene_kind"] == "icons_missing_mirror_cell_label"

    assert len(grid_cells) == 16
    assert len(option_cells) == 4
    assert len(missing_cells) == 1
    assert len(answer_options) == 1
    assert str(answer_options[0]["label"]) == "C"
    assert out.annotation_gt.value == list(answer_options[0]["cell_bbox_xyxy"])

    missing = missing_cells[0]
    assert str(execution["mirror_axis"]) == "vertical"
    assert int(execution["missing_row"]) == int(missing["row"])
    assert int(execution["missing_col"]) == int(missing["col"])
    assert int(execution["counterpart_row"]) == int(missing["row"])
    assert int(execution["counterpart_col"]) == 3 - int(missing["col"])
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_icons_mirror_grid_missing_cell_supports_horizontal_six_options() -> None:
    task = IconsMirrorGridMissingMirrorCellLabelTask()
    out = task.generate(
        27111,
        params={"mirror_axis": "horizontal", "option_count": 6, "answer_label": "F"},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    option_cells = [
        entity for entity in out.trace_payload["scene_ir"]["entities"] if str(entity.get("panel")) == "options"
    ]
    answer_option = next(entity for entity in option_cells if bool(entity.get("is_answer")))
    assert out.answer_gt.value == "F"
    assert int(execution["option_count"]) == 6
    assert execution["option_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert str(execution["mirror_axis"]) == "horizontal"
    assert str(answer_option["label"]) == "F"
    assert out.annotation_gt.value == list(answer_option["cell_bbox_xyxy"])


def test_icons_mirror_grid_missing_cell_deterministic() -> None:
    task = IconsMirrorGridMissingMirrorCellLabelTask()
    out_a = task.generate(27112, params={}, max_attempts=200)
    out_b = task.generate(27112, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_icons_mirror_grid_missing_cell_prompt_example_matches_contract() -> None:
    task = IconsMirrorGridMissingMirrorCellLabelTask()
    out = task.generate(
        27113,
        params={"mirror_axis": "vertical", "option_count": 4, "answer_label": "B"},
        max_attempts=200,
    )
    answer_only = _prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "C"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert len(answer_and_annotation["annotation"]) == 4
    assert answer_and_annotation["answer"] == "C"
