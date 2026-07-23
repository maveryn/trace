"""Behavior tests for source-layout nonogram puzzle tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.nonogram.candidate_solution_label import (
    PuzzlesNonogramCandidateSolutionLabelTask,
)
from trace_tasks.tasks.puzzles.nonogram.line_completion_label import (
    PuzzlesNonogramLineCompletionLabelTask,
)
from trace_tasks.tasks.puzzles.nonogram.shared.rules import (
    clue_for_line,
    col_clues_for_grid,
    line_matches_partial,
    row_clues_for_grid,
)
from tests.helpers import extract_prompt_json_example


def test_nonogram_tasks_contract_matches_trace() -> None:
    """Check answer and annotation binding for each nonogram objective."""

    task_cases = (
        (
            PuzzlesNonogramLineCompletionLabelTask(),
            "line_completion_label",
            "line_completion",
        ),
        (
            PuzzlesNonogramCandidateSolutionLabelTask(),
            "candidate_solution_label",
            "candidate_solution",
        ),
    )
    scene_variants = ("nonogram_classic", "nonogram_card", "nonogram_blueprint")

    for task_index, (task, prompt_query_key, mode) in enumerate(task_cases):
        for scene_index, scene_variant in enumerate(scene_variants):
            out = task.generate(
                33100 + (task_index * 20) + scene_index,
                params={"scene_variant": scene_variant},
                max_attempts=10,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]
            render = trace["render_spec"]
            render_map = trace["render_map"]

            assert str(out.query_id) == SINGLE_QUERY_ID
            assert str(out.scene_id) == "nonogram"
            assert out.answer_gt.type == "option_letter"
            assert out.annotation_gt.type == "bbox"
            assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
            assert str(execution["query_id"]) == SINGLE_QUERY_ID
            assert str(execution["internal_query_id"]) == str(prompt_query_key)
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(render["scene_variant"]) == str(scene_variant)
            assert str(execution["dataset"]["mode"]) == str(mode)
            assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
            assert render["text_style"]["font"]["source"] == "global_font_pool"
            assert int(execution["option_count"]) == 4
            assert str(render_map["annotation_source"]) == "option_panel_bboxes_px"

            answer_label = str(out.answer_gt.value)
            correct_option_id = str(execution["correct_option_panel_id"])
            assert correct_option_id == f"option_{answer_label}"
            assert trace["projected_annotation"]["bbox"] == [
                float(value) for value in out.annotation_gt.value
            ]
            assert [float(value) for value in out.annotation_gt.value] == [
                float(value) for value in render_map["option_panel_bboxes_px"][correct_option_id]
            ]

            option_specs = execution["option_specs"]
            assert [str(option["option_label"]) for option in option_specs] == [
                chr(ord("A") + index) for index in range(int(execution["option_count"]))
            ]
            assert [
                str(option["option_label"])
                for option in option_specs
                if bool(option["is_correct"])
            ] == [answer_label]


def test_nonogram_line_completion_options_have_unique_valid_answer() -> None:
    """Only the selected strip should satisfy both clue and visible cells."""

    out = PuzzlesNonogramLineCompletionLabelTask().generate(
        33150,
        params={"option_count": 4},
        max_attempts=10,
    )
    execution = out.trace_payload["execution_trace"]
    clue = [int(value) for value in execution["dataset"]["marked_clue"]]
    partial = execution["dataset"]["partial_line"]
    valid_labels = []
    for option in execution["option_specs"]:
        line = [int(value) for value in option["line"]]
        if clue_for_line(line) == clue and line_matches_partial(line, partial):
            valid_labels.append(str(option["option_label"]))
    assert valid_labels == [str(out.answer_gt.value)]


def test_nonogram_candidate_solution_options_have_unique_valid_answer() -> None:
    """Only the selected grid should satisfy all row and column clues."""

    out = PuzzlesNonogramCandidateSolutionLabelTask().generate(
        33170,
        params={"option_count": 4},
        max_attempts=10,
    )
    execution = out.trace_payload["execution_trace"]
    row_clues = [[int(value) for value in clue] for clue in execution["row_clues"]]
    col_clues = [[int(value) for value in clue] for clue in execution["col_clues"]]
    valid_labels = []
    for option in execution["option_specs"]:
        grid = [[int(value) for value in row] for row in option["grid"]]
        if row_clues_for_grid(grid) == row_clues and col_clues_for_grid(grid) == col_clues:
            valid_labels.append(str(option["option_label"]))
    assert valid_labels == [str(out.answer_gt.value)]


def test_nonogram_prompt_examples_match_selected_variants() -> None:
    """Prompt examples should match scalar bbox annotation shape."""

    expected = {
        "line_completion_label": (
            PuzzlesNonogramLineCompletionLabelTask(),
            {"annotation": [462, 650, 608, 776], "answer": "C"},
            {"answer": "C"},
        ),
        "candidate_solution_label": (
            PuzzlesNonogramCandidateSolutionLabelTask(),
            {"annotation": [622, 650, 768, 776], "answer": "D"},
            {"answer": "D"},
        ),
    }
    for index, (_name, (task, expected_answer_and_annotation, expected_answer_only)) in enumerate(
        expected.items(),
        start=33190,
    ):
        out = task.generate(index, params={}, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only


def test_nonogram_generation_is_deterministic() -> None:
    """Generation should be deterministic for a fixed seed and parameters."""

    task = PuzzlesNonogramCandidateSolutionLabelTask()
    params = {"scene_variant": "nonogram_card", "option_count": 4}
    out_a = task.generate(33220, params=params, max_attempts=10)
    out_b = task.generate(33220, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_nonogram_sampling_covers_visual_and_option_axes() -> None:
    """Scene variants and option counts should be sampled by RNG choices."""

    task = PuzzlesNonogramLineCompletionLabelTask()
    scene_variants = Counter()
    option_counts = Counter()
    answer_labels = Counter()
    for index in range(120):
        out = task.generate(33250 + index, params={}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        scene_variants[str(execution["scene_variant"])] += 1
        option_counts[int(execution["option_count"])] += 1
        answer_labels[str(out.answer_gt.value)] += 1

    assert set(scene_variants) == {"nonogram_blueprint", "nonogram_card", "nonogram_classic"}
    assert set(option_counts) == {4}
    assert set(answer_labels).issubset({"A", "B", "C", "D"})
    assert min(scene_variants.values()) >= 20
    assert min(option_counts.values()) >= 100
