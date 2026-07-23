"""Behavior tests for sheet-transform puzzle tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.sheet_transform.fold_cut_result_label import (
    PuzzlesSheetTransformFoldCutResultLabelTask,
)
from trace_tasks.tasks.puzzles.sheet_transform.fold_projection_result_label import (
    PuzzlesSheetTransformFoldProjectionResultLabelTask,
)
from trace_tasks.tasks.puzzles.sheet_transform.overlay_union_result_label import (
    PuzzlesSheetTransformOverlayUnionResultLabelTask,
)
from tests.helpers import extract_prompt_json_example


def _cell_signature(cells: list[list[int]]) -> tuple[tuple[int, int], ...]:
    """Return a hashable row-major cell signature."""

    return tuple(sorted((int(cell[0]), int(cell[1])) for cell in cells))


def _mark_signature(
    mark_specs: list[dict[str, object]],
) -> tuple[tuple[str, int, int], ...]:
    """Return one hashable signature for a folded-result mark set."""

    return tuple(
        sorted(
            (
                str(mark["object_type"]),
                int(mark["cell"][0]),
                int(mark["cell"][1]),
            )
            for mark in mark_specs
        )
    )


def _assert_common_output(out, *, expected_question_format: str) -> dict:
    """Assert common sheet-transform output fields and return execution trace."""

    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    annotation_bbox = [float(value) for value in out.annotation_gt.value]

    assert str(out.scene_id) == "sheet_transform"
    assert str(out.query_id) == SINGLE_QUERY_ID
    assert str(trace["query_spec"]["query_id"]) == SINGLE_QUERY_ID
    assert str(execution["query_id"]) == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert str(execution["question_format"]) == str(expected_question_format)
    assert int(execution["option_count"]) == 4
    assert out.image.size == (
        int(render["canvas_width"]),
        int(render["canvas_height"]),
    )
    assert str(out.answer_gt.value) == str(execution["answer_option_label"])
    assert trace["projected_annotation"]["bbox"] == annotation_bbox
    expected_bbox = [
        float(value)
        for value in trace["render_map"]["option_choice_bboxes_px"][
            str(execution["correct_option_choice_id"])
        ]
    ]
    assert annotation_bbox == expected_bbox
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    return execution


def test_fold_projection_result_label_contract_matches_option() -> None:
    """The selected option must match the projected single-axis fold result."""

    task = PuzzlesSheetTransformFoldProjectionResultLabelTask()
    out = task.generate(
        26020,
        params={"fold_axis": "vertical", "scene_variant": "fold_card"},
        max_attempts=20,
    )
    execution = _assert_common_output(
        out,
        expected_question_format="fold_result_mcq",
    )
    assert str(execution["fold_axis"]) == "vertical"
    assert int(execution["grid_size"]) == 6
    assert 3 <= int(execution["mark_count"]) <= 5
    option_specs = execution["option_specs"]
    assert [str(option["option_label"]) for option in option_specs] == list("ABCD")
    assert sum(1 for option in option_specs if bool(option["is_correct"])) == 1
    assert len({_mark_signature(option["mark_specs"]) for option in option_specs}) == 4
    winning_option = next(option for option in option_specs if bool(option["is_correct"]))
    assert _mark_signature(winning_option["mark_specs"]) == _mark_signature(
        execution["folded_result_mark_specs"]
    )


def test_fold_cut_result_label_contract_matches_option() -> None:
    """The selected option must match the unfolded fold-and-cut pattern."""

    task = PuzzlesSheetTransformFoldCutResultLabelTask()
    out = task.generate(
        26420,
        params={"fold_axis": "horizontal", "fold_count": 2, "scene_variant": "fold_strip"},
        max_attempts=20,
    )
    execution = _assert_common_output(
        out,
        expected_question_format="fold_cut_unfolded_result_mcq",
    )
    assert int(execution["grid_size"]) == 6
    assert int(execution["fold_count"]) == 2
    assert 1 <= int(execution["cut_count"]) <= 2
    option_specs = execution["option_specs"]
    assert [str(option["option_label"]) for option in option_specs] == list("ABCD")
    assert sum(1 for option in option_specs if bool(option["is_correct"])) == 1
    assert len({_cell_signature(option["cells"]) for option in option_specs}) == 4
    winning_option = next(option for option in option_specs if bool(option["is_correct"]))
    assert _cell_signature(winning_option["cells"]) == _cell_signature(
        execution["unfolded_hole_cells"]
    )


def test_overlay_union_result_label_contract_matches_option() -> None:
    """The selected option must equal the union of both aligned source sheets."""

    task = PuzzlesSheetTransformOverlayUnionResultLabelTask()
    out = task.generate(
        25920,
        params={"scene_variant": "overlay_outline"},
        max_attempts=20,
    )
    execution = _assert_common_output(
        out,
        expected_question_format="overlay_union_mcq",
    )
    assert 4 <= int(execution["grid_size"]) <= 5
    left = set(_cell_signature(execution["left_cells"]))
    right = set(_cell_signature(execution["right_cells"]))
    overlap = set(_cell_signature(execution["overlap_cells"]))
    union = set(_cell_signature(execution["union_cells"]))
    assert overlap <= left
    assert overlap <= right
    assert left | right == union
    assert left & right == overlap
    option_specs = execution["option_specs"]
    assert [str(option["option_label"]) for option in option_specs] == list("ABCD")
    assert sum(1 for option in option_specs if bool(option["is_correct"])) == 1
    assert len({_cell_signature(option["cells"]) for option in option_specs}) == 4
    winning_option = next(option for option in option_specs if bool(option["is_correct"]))
    assert set(_cell_signature(winning_option["cells"])) == union


def test_sheet_transform_prompt_examples_use_scalar_bbox() -> None:
    """Prompt examples should match scalar bbox annotation shape."""

    examples = {
        PuzzlesSheetTransformFoldProjectionResultLabelTask: {
            "annotation": [206, 388, 324, 613],
            "answer": "A",
        },
        PuzzlesSheetTransformFoldCutResultLabelTask: {
            "annotation": [134, 420, 312, 598],
            "answer": "A",
        },
        PuzzlesSheetTransformOverlayUnionResultLabelTask: {
            "annotation": [521, 384, 679, 542],
            "answer": "B",
        },
    }
    for task_cls, expected in examples.items():
        out = task_cls().generate(26090, params={}, max_attempts=30)
        answer_and_annotation = extract_prompt_json_example(
            out.prompt_variants["answer_and_annotation"]
        )
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected
        assert answer_only == {"answer": expected["answer"]}


def test_sheet_transform_tasks_are_deterministic() -> None:
    """Generation should be deterministic for fixed seed and parameters."""

    cases = (
        (
            PuzzlesSheetTransformFoldProjectionResultLabelTask(),
            {"fold_axis": "horizontal", "scene_variant": "fold_card"},
        ),
        (
            PuzzlesSheetTransformFoldCutResultLabelTask(),
            {"fold_count": 2, "scene_variant": "fold_card"},
        ),
        (
            PuzzlesSheetTransformOverlayUnionResultLabelTask(),
            {"scene_variant": "overlay_card"},
        ),
    )
    for task, params in cases:
        out_a = task.generate(26140, params=params, max_attempts=30)
        out_b = task.generate(26140, params=params, max_attempts=30)
        assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
        assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
        assert out_a.trace_payload["execution_trace"] == out_b.trace_payload[
            "execution_trace"
        ]
        assert out_a.prompt == out_b.prompt
        assert out_a.image.tobytes() == out_b.image.tobytes()


def test_sheet_transform_sampling_covers_expected_axes() -> None:
    """Sampling should cover option letters and operation-specific axes."""

    fold_letters: Counter[str] = Counter()
    cut_counts: Counter[int] = Counter()
    overlay_shapes: Counter[str] = Counter()
    for seed in range(26200, 26340):
        fold = PuzzlesSheetTransformFoldProjectionResultLabelTask().generate(
            seed,
            params={},
            max_attempts=30,
        )
        fold_letters[str(fold.answer_gt.value)] += 1
        cut = PuzzlesSheetTransformFoldCutResultLabelTask().generate(
            seed,
            params={},
            max_attempts=30,
        )
        cut_counts[int(cut.trace_payload["execution_trace"]["fold_count"])] += 1
        overlay = PuzzlesSheetTransformOverlayUnionResultLabelTask().generate(
            seed,
            params={},
            max_attempts=30,
        )
        overlay_shapes[str(overlay.trace_payload["execution_trace"]["mark_shape"])] += 1
    assert set(fold_letters) == set("ABCD")
    assert set(cut_counts) == {1, 2}
    assert set(overlay_shapes) == {"circle", "square", "diamond", "rounded_square"}
