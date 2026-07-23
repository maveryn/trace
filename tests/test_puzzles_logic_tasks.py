"""Behavior tests for Raven-matrix puzzle tasks."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.tasks.puzzles.raven_matrix.raven_analogical_transform_label import (
    PuzzlesRavenMatrixAnalogicalTransformLabelTask,
)
from trace_tasks.tasks.puzzles.raven_matrix.raven_count_progression_label import (
    PuzzlesRavenMatrixCountProgressionLabelTask,
)
from trace_tasks.tasks.puzzles.raven_matrix.raven_feature_binding_label import (
    PuzzlesRavenMatrixFeatureBindingLabelTask,
)
from trace_tasks.tasks.puzzles.raven_matrix.raven_position_progression_label import (
    PuzzlesRavenMatrixPositionProgressionLabelTask,
)
from trace_tasks.tasks.puzzles.raven_matrix.raven_set_operation_label import (
    PuzzlesRavenMatrixSetOperationLabelTask,
)
from trace_tasks.tasks.puzzles.raven_matrix.raven_spatial_transform_label import (
    PuzzlesRavenMatrixSpatialTransformLabelTask,
)
from tests.helpers import extract_prompt_json_example


def _panel_signature(panel_spec: dict) -> str:
    """Return a stable signature for one Raven panel spec."""

    return json.dumps(panel_spec, sort_keys=True, separators=(",", ":"))


def test_puzzle_raven_matrix_label_contract_matches_winning_option_panel() -> None:
    task_cases = (
        (PuzzlesRavenMatrixCountProgressionLabelTask(), "count_progression_matrix"),
        (PuzzlesRavenMatrixSpatialTransformLabelTask(), "spatial_transform_matrix"),
        (PuzzlesRavenMatrixSetOperationLabelTask(), "set_operation_matrix"),
        (PuzzlesRavenMatrixAnalogicalTransformLabelTask(), "analogical_transform_matrix"),
        (PuzzlesRavenMatrixPositionProgressionLabelTask(), "position_progression_matrix"),
        (PuzzlesRavenMatrixFeatureBindingLabelTask(), "feature_binding_matrix"),
    )
    scene_variants = (
        "raven_strip",
        "raven_card",
        "raven_outline",
    )

    for query_id_index, (task, rule_code) in enumerate(task_cases):
        for scene_index, scene_variant in enumerate(scene_variants):
            seed = 25100 + (query_id_index * 20) + scene_index
            out = task.generate(
                seed,
                params={"scene_variant": scene_variant},
                max_attempts=10,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]
            render = trace["render_spec"]
            render_map = trace["render_map"]
            solver = execution["solver_trace"]
            annotation_bbox = [
                float(value) for value in out.annotation_gt.value
            ]

            assert str(out.query_id) == "single"
            assert str(trace["query_spec"]["query_id"]) == "single"
            assert str(execution["query_id"]) == "single"
            assert str(execution["raven_rule_code"]) == str(rule_code)
            assert out.answer_gt.type == "option_letter"
            assert out.annotation_gt.type == "bbox"
            assert sorted(out.prompt_variants.keys()) == [
                "answer_and_annotation",
                "answer_only",
            ]
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(render["scene_variant"]) == str(scene_variant)
            assert not any(
                str(entity["entity_type"]) == "puzzle_raven_panel"
                for entity in trace["scene_ir"]["entities"]
            )
            assert out.image.size == (
                int(render["canvas_width"]),
                int(render["canvas_height"]),
            )
            assert int(execution["matrix_size"]) == 3
            assert int(execution["cell_count"]) == 9
            assert int(execution["visible_matrix_cell_count"]) == 8
            assert int(execution["option_count"]) == 4
            assert str(trace["query_spec"]["prompt_variant"]["query_key"]).startswith(
                "raven_"
            )
            assert str(execution["question_format"]).startswith("raven_")
            assert trace["projected_annotation"]["bbox"] == annotation_bbox
            assert [str(option_id) for option_id in execution["supporting_option_panel_ids"]] == [
                str(execution["correct_option_panel_id"])
            ]
            assert str(out.answer_gt.value) == str(execution["answer_option_label"])
            assert str(out.answer_gt.value) in {"A", "B", "C", "D"}

            expected_bbox = [
                float(value)
                for value in render_map["option_cell_bboxes_px"][
                    str(execution["correct_option_panel_id"])
                ]
            ]
            assert annotation_bbox == expected_bbox
            assert str(render_map["annotation_source"]) == "option_cell_bboxes_px"
            assert all(
                float(bbox[0]) >= 0.0
                for bbox in render_map["matrix_cell_bboxes_px"].values()
            )
            assert all(
                float(bbox[1]) >= 0.0
                for bbox in render_map["matrix_cell_bboxes_px"].values()
            )
            assert all(
                float(bbox[2]) <= float(render["canvas_width"])
                for bbox in render_map["option_panel_bboxes_px"].values()
            )
            assert all(
                float(bbox[3]) <= float(render["canvas_height"])
                for bbox in render_map["option_panel_bboxes_px"].values()
            )
            matrix_cell_bbox = [
                float(value) for value in render_map["matrix_cell_bboxes_px"]["cell_0_0"]
            ]
            matrix_cell_side = float(matrix_cell_bbox[2] - matrix_cell_bbox[0])
            matrix_content_side = float(
                matrix_cell_side - (2.0 * max(8.0, 0.08 * matrix_cell_side))
            )
            option_content_boxes = [
                entity
                for entity in trace["scene_ir"]["entities"]
                if str(entity["entity_type"]) == "puzzle_raven_option_cell"
            ]
            option_panel_boxes = {
                str(entity["entity_id"]): [float(value) for value in entity["bbox_px"]]
                for entity in trace["scene_ir"]["entities"]
                if str(entity["entity_type"]) == "puzzle_raven_option_slot"
            }
            assert len(option_content_boxes) == 4
            assert all(
                float(entity["bbox_px"][2]) - float(entity["bbox_px"][0])
                >= matrix_content_side
                for entity in option_content_boxes
            )
            for entity in option_content_boxes:
                option_id = str(entity["entity_id"]).removesuffix("_cell")
                content_bbox = [float(value) for value in entity["bbox_px"]]
                panel_bbox = option_panel_boxes[option_id]
                assert content_bbox[0] >= panel_bbox[0]
                assert content_bbox[1] >= panel_bbox[1]
                assert content_bbox[2] <= panel_bbox[2]
                assert content_bbox[3] <= panel_bbox[3]

            matrix_rows = execution["matrix_rows"]
            matrix_panel_specs = execution["matrix_panel_specs"]
            unknown_cells = [
                cell
                for row in matrix_rows
                for cell in row
                if bool(cell["is_unknown"])
            ]
            assert len(unknown_cells) == 1
            assert str(unknown_cells[0]["cell_id"]) == "cell_2_2"
            assert int(execution["target_row_index"]) == 2
            assert int(execution["target_col_index"]) == 2
            assert execution["answer_panel_spec"] == matrix_panel_specs[2][2]

            option_specs = execution["option_specs"]
            assert len(option_specs) == 4
            assert [str(option["option_label"]) for option in option_specs] == [
                "A",
                "B",
                "C",
                "D",
            ]
            assert sum(1 for option in option_specs if bool(option["is_correct"])) == 1
            assert len({_panel_signature(dict(option["panel_spec"])) for option in option_specs}) == 4
            winning_option = next(option for option in option_specs if bool(option["is_correct"]))
            assert str(winning_option["option_label"]) == str(out.answer_gt.value)
            assert str(winning_option["option_panel_id"]) == str(
                execution["correct_option_panel_id"]
            )
            assert dict(winning_option["panel_spec"]) == dict(execution["answer_panel_spec"])
            assert str(winning_option["panel_spec"]["panel_kind"]) in {
                "attribute",
                "count",
                "pattern",
            }

            assert str(solver["rule_type"]) == str(rule_code)
            assert str(solver["correct_option_label"]) == str(out.answer_gt.value)
            assert int(solver["correct_option_index"]) == int(
                execution["correct_option_index"]
            )
            if str(rule_code) == "count_progression_matrix":
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "count"
                assert int(execution["answer_panel_spec"]["count"]) == int(
                    solver["count_table"][2][2]
                )
                assert int(execution["answer_panel_spec"]["count"]) == int(
                    solver["answer_count"]
                )
                count_table = [
                    [int(value) for value in row] for row in solver["count_table"]
                ]
                first_delta, second_delta = [
                    int(value) for value in solver["progression_deltas"]
                ]
                assert int(first_delta) == int(second_delta)
                assert int(solver["progression_delta"]) == int(first_delta)
                assert str(solver["progression_axis"]) in {"row", "column"}
                if str(solver["progression_axis"]) == "row":
                    for row in count_table:
                        assert int(row[1] - row[0]) == int(first_delta)
                        assert int(row[2] - row[1]) == int(second_delta)
                else:
                    for col_index in range(3):
                        assert (
                            int(count_table[1][col_index] - count_table[0][col_index])
                            == int(first_delta)
                        )
                        assert (
                            int(count_table[2][col_index] - count_table[1][col_index])
                            == int(second_delta)
                        )
            elif str(rule_code) == "spatial_transform_matrix":
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "pattern"
                assert execution["answer_panel_spec"]["cells"] == solver["answer_cells"]
                assert str(solver["progression_axis"]) in {"row", "column"}
                assert str(solver["rotation_mode"]) in {
                    "row_clockwise_first",
                    "row_counterclockwise_first",
                    "column_clockwise_first",
                    "column_counterclockwise_first",
                }
                assert [str(value) for value in solver["rotation_sequence"]] in [
                    ["identity", "rotate_cw_90", "rotate_ccw_90"],
                    ["identity", "rotate_ccw_90", "rotate_cw_90"],
                ]
                assert "row_transforms" not in solver
                assert "column_transforms" not in solver
                assert "base_cells" not in solver
            elif str(rule_code) == "set_operation_matrix":
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "pattern"
                assert str(solver["operation"]) in {"union", "intersection"}
                assert execution["answer_panel_spec"]["cells"] == solver["answer_cells"]
            elif str(rule_code) == "analogical_transform_matrix":
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "attribute"
                assert str(solver["transform_kind"]) in {
                    "shape_cycle",
                    "color_cycle",
                    "size_cycle",
                }
            elif str(rule_code) == "feature_binding_matrix":
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "attribute"
                assert str(solver["binding_mode"]) in {
                    "row_shape_column_color",
                    "row_color_column_shape",
                    "row_shape_column_size",
                    "row_size_column_shape",
                }
                assert str(solver["row_feature"]) in {"shape", "color", "size"}
                assert str(solver["column_feature"]) in {"shape", "color", "size"}
                assert str(solver["row_feature"]) != str(solver["column_feature"])
                feature_table = solver["feature_table"]
                assert len(feature_table) == 3
                assert all(len(row) == 3 for row in feature_table)
                assert len({str(row[0][solver["row_feature"]]) for row in feature_table}) == 3
                assert len(
                    {
                        str(feature_table[0][col_index][solver["column_feature"]])
                        for col_index in range(3)
                    }
                ) == 3
                assert feature_table[2][2][solver["row_feature"]] == solver[
                    "row_feature_values"
                ][2]
                assert feature_table[2][2][solver["column_feature"]] == solver[
                    "column_feature_values"
                ][2]
            else:
                assert str(execution["answer_panel_spec"]["panel_kind"]) == "pattern"
                assert len(execution["answer_panel_spec"]["cells"]) == 1
                assert execution["answer_panel_spec"]["cells"][0] == solver["answer_position"]
                progression_axis = str(solver["progression_axis"])
                progression_direction = str(solver["progression_direction"])
                progression_mode = str(solver["progression_mode"])
                progression_line_indices = [
                    int(value) for value in solver["progression_line_indices"]
                ]
                assert progression_axis in {"row", "column"}
                assert progression_mode in {
                    "row_left_to_right",
                    "row_right_to_left",
                    "column_top_to_bottom",
                    "column_bottom_to_top",
                }
                assert len(progression_line_indices) == 3
                assert sorted(progression_line_indices) == [0, 1, 2]
                position_table = [
                    [
                        tuple(
                            int(value)
                            for value in matrix_panel_specs[row_index][col_index][
                                "cells"
                            ][0]
                        )
                        for col_index in range(3)
                    ]
                    for row_index in range(3)
                ]
                assert position_table == [
                    [
                        tuple(int(value) for value in coord)
                        for coord in row
                    ]
                    for row in solver["position_table"]
                ]

                if progression_axis == "row":
                    assert progression_direction in {"left_to_right", "right_to_left"}
                    expected_cols = (
                        [0, 1, 2]
                        if progression_direction == "left_to_right"
                        else [2, 1, 0]
                    )
                    for row_index, row in enumerate(position_table):
                        mini_row = int(progression_line_indices[row_index])
                        assert row == [
                            (mini_row, int(mini_col))
                            for mini_col in expected_cols
                        ]
                else:
                    assert progression_direction in {"top_to_bottom", "bottom_to_top"}
                    expected_rows = (
                        [0, 1, 2]
                        if progression_direction == "top_to_bottom"
                        else [2, 1, 0]
                    )
                    for col_index in range(3):
                        mini_col = int(progression_line_indices[col_index])
                        assert [
                            position_table[row_index][col_index]
                            for row_index in range(3)
                        ] == [
                            (int(mini_row), mini_col)
                            for mini_row in expected_rows
                        ]


def test_puzzle_raven_prompt_examples_match_selected_variants() -> None:
    cases = (
        (
            PuzzlesRavenMatrixCountProgressionLabelTask(),
            {},
            {"annotation": [465, 647, 571, 752], "answer": "B"},
            {"answer": "B"},
        ),
        (
            PuzzlesRavenMatrixSpatialTransformLabelTask(),
            {},
            {"annotation": [793, 647, 899, 752], "answer": "D"},
            {"answer": "D"},
        ),
        (
            PuzzlesRavenMatrixSetOperationLabelTask(),
            {},
            {"annotation": [629, 647, 735, 752], "answer": "C"},
            {"answer": "C"},
        ),
        (
            PuzzlesRavenMatrixAnalogicalTransformLabelTask(),
            {},
            {"annotation": [793, 647, 899, 752], "answer": "D"},
            {"answer": "D"},
        ),
        (
            PuzzlesRavenMatrixPositionProgressionLabelTask(),
            {},
            {"annotation": [301, 647, 407, 752], "answer": "A"},
            {"answer": "A"},
        ),
        (
            PuzzlesRavenMatrixFeatureBindingLabelTask(),
            {},
            {"annotation": [629, 647, 735, 752], "answer": "C"},
            {"answer": "C"},
        ),
    )
    for index, (task, params, expected_answer_and_annotation, expected_answer_only) in enumerate(cases, start=25200):
        out = task.generate(index, params=params, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(
            out.prompt_variants["answer_and_annotation"]
        )
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only


def test_puzzle_raven_matrix_label_task_is_deterministic() -> None:
    task = PuzzlesRavenMatrixSpatialTransformLabelTask()
    params = {"scene_variant": "raven_card"}
    out_a = task.generate(25250, params=params, max_attempts=10)
    out_b = task.generate(25250, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_puzzle_raven_count_progression_samples_increasing_and_decreasing_rules() -> None:
    task = PuzzlesRavenMatrixCountProgressionLabelTask()
    observed_signs: set[int] = set()
    observed_axes: set[str] = set()

    for seed in range(1000, 1120):
        out = task.generate(seed, params={}, max_attempts=10)
        solver = out.trace_payload["execution_trace"]["solver_trace"]
        delta = int(solver["progression_delta"])
        observed_signs.add(-1 if delta < 0 else 1)
        observed_axes.add(str(solver["progression_axis"]))
        assert int(delta) != 0
        assert [int(value) for value in solver["progression_deltas"]] == [
            int(delta),
            int(delta),
        ]

    assert observed_signs == {-1, 1}
    assert observed_axes == {"row", "column"}


def test_puzzle_raven_spatial_transform_samples_only_90_degree_rotations() -> None:
    task = PuzzlesRavenMatrixSpatialTransformLabelTask()
    observed_modes: set[str] = set()
    allowed_modes = {
        "row_clockwise_first",
        "row_counterclockwise_first",
        "column_clockwise_first",
        "column_counterclockwise_first",
    }
    allowed_sequences = {
        ("identity", "rotate_cw_90", "rotate_ccw_90"),
        ("identity", "rotate_ccw_90", "rotate_cw_90"),
    }

    for seed in range(1500, 1620):
        out = task.generate(seed, params={}, max_attempts=10)
        solver = out.trace_payload["execution_trace"]["solver_trace"]
        mode = str(solver["rotation_mode"])
        sequence = tuple(str(value) for value in solver["rotation_sequence"])
        observed_modes.add(mode)
        assert mode in allowed_modes
        assert sequence in allowed_sequences
        assert "row_transforms" not in solver
        assert "column_transforms" not in solver
        assert "base_cells" not in solver

    assert observed_modes == allowed_modes


def test_puzzle_raven_position_progression_samples_non_wrapping_modes() -> None:
    task = PuzzlesRavenMatrixPositionProgressionLabelTask()
    observed_modes: set[str] = set()
    allowed_modes = {
        "row_left_to_right",
        "row_right_to_left",
        "column_top_to_bottom",
        "column_bottom_to_top",
    }

    for seed in range(2000, 2120):
        out = task.generate(seed, params={}, max_attempts=10)
        solver = out.trace_payload["execution_trace"]["solver_trace"]
        mode = str(solver["progression_mode"])
        observed_modes.add(mode)
        assert mode in allowed_modes
        assert "progression_step_signed" not in solver
        assert "progression_step_mod3" not in solver
        assert sorted(int(value) for value in solver["progression_line_indices"]) == [
            0,
            1,
            2,
        ]

    assert observed_modes == allowed_modes


def test_puzzle_raven_feature_binding_samples_row_column_modes() -> None:
    task = PuzzlesRavenMatrixFeatureBindingLabelTask()
    observed_modes: set[str] = set()
    allowed_modes = {
        "row_shape_column_color",
        "row_color_column_shape",
        "row_shape_column_size",
        "row_size_column_shape",
    }

    for seed in range(2200, 2320):
        out = task.generate(seed, params={}, max_attempts=10)
        solver = out.trace_payload["execution_trace"]["solver_trace"]
        mode = str(solver["binding_mode"])
        observed_modes.add(mode)
        assert mode in allowed_modes
        assert str(solver["row_feature"]) != str(solver["column_feature"])
        assert len(solver["row_feature_values"]) == 3
        assert len(solver["column_feature_values"]) == 3

    assert observed_modes == allowed_modes


def test_puzzle_raven_sampler_index_covers_answer_letters_per_variant() -> None:
    task_cases = (
        PuzzlesRavenMatrixCountProgressionLabelTask(),
        PuzzlesRavenMatrixSpatialTransformLabelTask(),
        PuzzlesRavenMatrixSetOperationLabelTask(),
        PuzzlesRavenMatrixAnalogicalTransformLabelTask(),
        PuzzlesRavenMatrixPositionProgressionLabelTask(),
        PuzzlesRavenMatrixFeatureBindingLabelTask(),
    )

    for task in task_cases:
        observed_letters = []
        for sampling_index in range(120):
            out = task.generate(
                25290 + sampling_index,
                params={},
                max_attempts=10,
            )
            observed_letters.append(str(out.answer_gt.value))
        letter_counts = Counter(observed_letters)
        assert set(letter_counts) == {"A", "B", "C", "D"}
        assert min(letter_counts.values()) > 0
        assert max(letter_counts.values()) <= 60
