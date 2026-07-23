"""Behavior tests for matchstick puzzle tasks."""

from __future__ import annotations

import json
import re

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.matchstick.equation_repair_stick_label import (
    PuzzlesMatchstickEquationRepairStickLabelTask,
)
from trace_tasks.tasks.puzzles.matchstick.matchstick_number_transform_label import (
    PuzzlesMatchstickNumberTransformLabelTask,
    SUPPORTED_QUERY_IDS as NUMBER_QUERY_IDS,
)
from trace_tasks.tasks.puzzles.matchstick.max_square_count_after_additions_value import (
    PuzzlesMatchstickMaxSquareCountAfterAdditionsValueTask,
)
from trace_tasks.tasks.puzzles.matchstick.shared.rules import (
    equation_is_true,
    optimal_lattice_square_additions,
)
from trace_tasks.tasks.puzzles.matchstick.shared.state import SCENE_VARIANTS


def _extract_answer_and_annotation_example(prompt: str) -> dict[str, object]:
    match = re.search(r'(\{"annotation".*\})\.?$', str(prompt))
    assert match is not None
    return json.loads(match.group(1).rstrip("."))


def test_matchstick_number_transform_uses_keyed_source_and_option_annotation() -> None:
    task = PuzzlesMatchstickNumberTransformLabelTask()

    for index, (query_id, scene_variant) in enumerate(zip(NUMBER_QUERY_IDS * 3, SCENE_VARIANTS)):
        out = task.generate(
            31000 + index,
            params={"query_id": query_id, "scene_variant": scene_variant},
            max_attempts=10,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]
        annotation = {
            str(key): [float(value) for value in bbox]
            for key, bbox in out.annotation_gt.value.items()
        }

        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox_map"
        assert set(annotation) == {"source_number", "selected_option"}
        assert trace["projected_annotation"]["type"] == "bbox_map"
        assert trace["projected_annotation"]["bbox_map"] == annotation
        assert trace["projected_annotation"]["pixel_bbox_map"] == annotation
        assert execution["annotation_role_item_ids"] == {
            "source_number": "source_panel",
            "selected_option": f"option_{out.answer_gt.value}",
        }
        assert execution["supporting_item_ids"] == [
            "source_panel",
            f"option_{out.answer_gt.value}",
        ]
        assert annotation["source_number"] == [
            float(value) for value in render_map["item_bboxes_px"]["source_panel"]
        ]
        assert annotation["selected_option"] == [
            float(value) for value in render_map["item_bboxes_px"][f"option_{out.answer_gt.value}"]
        ]
        assert render_map["annotation_source"] == "keyed_item_bboxes_px"
        assert render["text_style"]["font"]["source"] == "global_font_pool"
        assert render["text_style"]["font"]["font_family"]


def test_matchstick_number_transform_prompt_example_uses_keyed_annotation() -> None:
    out = PuzzlesMatchstickNumberTransformLabelTask().generate(
        31200,
        params={"query_id": "add_one_stick"},
        max_attempts=10,
    )

    answer_and_annotation = _extract_answer_and_annotation_example(
        out.prompt_variants["answer_and_annotation"]
    )
    assert set(answer_and_annotation["annotation"]) == {"source_number", "selected_option"}
    assert answer_and_annotation["answer"] == "B"


def test_matchstick_equation_repair_uses_scalar_segment_annotation() -> None:
    task = PuzzlesMatchstickEquationRepairStickLabelTask()

    for index, scene_variant in enumerate(SCENE_VARIANTS):
        out = task.generate(
            31400 + index,
            params={"query_id": SINGLE_QUERY_ID, "scene_variant": scene_variant},
            max_attempts=20,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render_map = trace["render_map"]
        selected_item_id = str(execution["selected_item_id"])
        annotation = [[float(v) for v in point] for point in out.annotation_gt.value]

        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "segment"
        assert out.query_id == SINGLE_QUERY_ID
        assert trace["projected_annotation"]["type"] == "segment"
        assert trace["projected_annotation"]["segment"] == annotation
        assert trace["projected_annotation"]["pixel_segment"] == annotation
        assert render_map["annotation_source"] == "item_segments_px"
        assert render_map["item_segments_px"][selected_item_id] == annotation
        assert execution["supporting_item_ids"] == [selected_item_id]

        true_removals = [
            row for row in execution["all_removal_outcomes"] if bool(row["is_true"])
        ]
        assert len(true_removals) == 1
        assert true_removals[0]["stick_id"] == execution["repair_stick_id"]
        assert equation_is_true(
            tuple(int(value) for value in execution["repaired_digits"]),
            str(execution["operator"]),
        )
        assert not equation_is_true(
            tuple(int(value) for value in execution["source_digits"]),
            str(execution["operator"]),
        )


def test_matchstick_equation_repair_prompt_example_uses_segment_annotation() -> None:
    out = PuzzlesMatchstickEquationRepairStickLabelTask().generate(
        31500,
        params={"query_id": SINGLE_QUERY_ID},
        max_attempts=20,
    )

    answer_and_annotation = _extract_answer_and_annotation_example(
        out.prompt_variants["answer_and_annotation"]
    )
    assert answer_and_annotation["answer"] == "B"
    assert isinstance(answer_and_annotation["annotation"], list)
    assert len(answer_and_annotation["annotation"]) == 2
    assert all(len(point) == 2 for point in answer_and_annotation["annotation"])


def test_matchstick_square_completion_uses_final_square_bbox_set() -> None:
    task = PuzzlesMatchstickMaxSquareCountAfterAdditionsValueTask()

    for index, scene_variant in enumerate(SCENE_VARIANTS):
        out = task.generate(
            31800 + index,
            params={"query_id": SINGLE_QUERY_ID, "scene_variant": scene_variant},
            max_attempts=40,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render_map = trace["render_map"]
        annotation = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]

        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert out.answer_gt.value == len(annotation)
        assert 1 <= int(execution["add_count"]) <= 2
        assert 1 <= int(out.answer_gt.value) <= 6
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == annotation
        assert trace["projected_annotation"]["pixel_bbox_set"] == annotation
        assert render_map["annotation_source"] == "unit_square_bboxes_px"

        expected_bboxes = [
            [float(value) for value in render_map["item_bboxes_px"][item_id]]
            for item_id in execution["supporting_item_ids"]
        ]
        assert annotation == expected_bboxes
        assert len(execution["completed_square_ids"]) == int(out.answer_gt.value)

        optimal = optimal_lattice_square_additions(
            frozenset(str(edge) for edge in execution["present_edges"]),
            rows=int(execution["rows"]),
            cols=int(execution["cols"]),
            add_count=int(execution["add_count"]),
        )
        assert int(optimal["best_count"]) == int(out.answer_gt.value)
        assert len(tuple(optimal["best_square_sets"])) == 1
        assert tuple(execution["completed_square_ids"]) == tuple(
            tuple(optimal["best_square_sets"])[0]
        )


def test_matchstick_square_completion_prompt_uses_integer_bbox_set_slots() -> None:
    out = PuzzlesMatchstickMaxSquareCountAfterAdditionsValueTask().generate(
        31900,
        params={"query_id": SINGLE_QUERY_ID},
        max_attempts=40,
    )

    assert "option label" not in out.prompt
    assert f"exactly {out.trace_payload['execution_trace']['add_count']}" in out.prompt
    assert "matchstick" in out.prompt
    answer_and_annotation = _extract_answer_and_annotation_example(
        out.prompt_variants["answer_and_annotation"]
    )
    assert isinstance(answer_and_annotation["answer"], int)
    assert isinstance(answer_and_annotation["annotation"], list)
    assert all(len(bbox) == 4 for bbox in answer_and_annotation["annotation"])
