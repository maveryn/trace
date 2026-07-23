"""Behavior tests for cyclic-order puzzle tasks."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.cyclic_order.cyclic_order_equivalent_label import (
    PuzzlesCyclicOrderEquivalentLabelTask,
)
from trace_tasks.tasks.puzzles.cyclic_order.insertion_position_label import (
    PuzzlesCyclicOrderInsertionPositionLabelTask,
)
from trace_tasks.tasks.puzzles.cyclic_order.swap_repair_label import (
    PuzzlesCyclicOrderSwapRepairLabelTask,
)
from trace_tasks.tasks.puzzles.cyclic_order.shared.rules import token_sequences_are_rotation_equivalent
from trace_tasks.tasks.shared.color_distance import color_distance
from tests.helpers import assert_counter_support_within, extract_prompt_json_example


def test_puzzles_cyclic_order_equivalent_label_contract_matches_valid_options() -> None:
    task = PuzzlesCyclicOrderEquivalentLabelTask()
    token_render_styles = (
        "colored_beads",
        "shape_tokens",
        "colored_shape_tokens",
        "outline_shape_tokens",
        "symbol_badges",
    )
    scene_variants = (
        "necklace_board",
        "charm_card_grid",
        "route_loop_diagram",
        "token_ring_outline",
    )
    loop_path_styles = (
        "ellipse",
        "rounded_rect",
        "polygon_loop",
        "wavy_loop",
        "beaded_string",
    )

    for mode_index, token_render_style in enumerate(token_render_styles):
        for scene_index, scene_variant in enumerate(scene_variants):
            loop_path_style = loop_path_styles[(mode_index + scene_index) % len(loop_path_styles)]
            seed = 27320 + (mode_index * 100) + (scene_index * 10)
            out = task.generate(
                seed,
                params={
                    "query_id": SINGLE_QUERY_ID,
                    "token_render_style": token_render_style,
                    "scene_variant": scene_variant,
                    "loop_path_style": loop_path_style,
                },
                max_attempts=10,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]
            render = trace["render_spec"]
            render_map = trace["render_map"]
            solver = execution["solver_trace"]
            annotation_bbox = [float(value) for value in out.annotation_gt.value]

            assert str(out.query_id) == SINGLE_QUERY_ID
            assert str(out.scene_id) == "cyclic_order"
            assert task.supported_query_ids == (SINGLE_QUERY_ID,)
            assert str(execution["token_render_style"]) == str(token_render_style)
            assert str(execution["loop_path_style"]) == str(loop_path_style)
            assert str(execution["query_id"]) == SINGLE_QUERY_ID
            assert str(execution["internal_query_id"]) == SINGLE_QUERY_ID
            assert out.annotation_gt.type == "bbox"
            assert out.answer_gt.type == "option_letter"
            assert str(out.answer_gt.value) == str(execution["answer_option_label"])
            assert int(execution["valid_option_count"]) == 1
            assert str(execution["view_family"]) == "loop_option_label"
            assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(render["scene_variant"]) == str(scene_variant)
            assert str(render["loop_path_style"]) == str(loop_path_style)
            assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
            assert list(execution["option_count_range"]) == [4, 4]
            assert int(execution["option_count"]) == 4
            assert 4 <= int(execution["bead_count"]) <= 5
            assert 4 <= int(execution["bead_count_range"][0]) <= int(execution["bead_count_range"][1]) <= 5
            assert str(execution["question_format"]) == "equivalent_loop_label"
            assert str(execution["equivalence_rule"]) == "same_cyclic_order_up_to_rotation_no_reflection"
            assert trace["projected_annotation"]["bbox"] == annotation_bbox

            option_specs = execution["option_specs"]
            assert len(option_specs) == int(execution["option_count"])
            assert sum(1 for option in option_specs if bool(option["is_valid"])) == 1
            assert [str(option["option_label"]) for option in option_specs] == [
                chr(ord("A") + index) for index in range(int(execution["option_count"]))
            ]

            expected_bbox = [
                float(value)
                for value in render_map["option_choice_bboxes_px"][str(execution["answer_option_choice_id"])]
            ]
            assert annotation_bbox == expected_bbox
            assert [str(value) for value in execution["supporting_option_choice_ids"]] == [
                str(value) for value in execution["valid_option_choice_ids"]
            ]

            reference_sequence = [str(value) for value in execution["reference_token_sequence"]]
            valid_labels = []
            for option in option_specs:
                token_sequence = [str(value) for value in option["token_sequence"]]
                is_equivalent = token_sequences_are_rotation_equivalent(reference_sequence, token_sequence)
                assert bool(is_equivalent) == bool(option["is_valid"])
                if bool(option["is_valid"]):
                    valid_labels.append(str(option["option_label"]))

            assert valid_labels == [str(value) for value in execution["valid_option_labels"]]
            assert valid_labels == [str(value) for value in solver["valid_option_labels"]]
            assert [str(value) for value in execution["valid_option_choice_ids"]] == [
                str(value) for value in solver["valid_option_choice_ids"]
            ]

            if str(token_render_style) in {"colored_beads", "colored_shape_tokens", "symbol_badges"}:
                distinct_colors = sorted(
                    {
                        tuple(int(channel) for channel in token_spec["fill_rgb"])
                        for token_spec in option_specs[0]["bead_specs"]
                    }
                )
                assert len(distinct_colors) == int(execution["bead_count"])
                assert float(execution["min_color_distance"]) == 50.0
                assert str(execution["color_distance_space"]) == "lab"
                for color_a, color_b in combinations(distinct_colors, 2):
                    assert float(color_distance(color_a, color_b, distance_space="lab")) >= 50.0


def test_puzzles_cyclic_order_prompt_examples_match_selected_variants() -> None:
    task = PuzzlesCyclicOrderEquivalentLabelTask()
    out = task.generate(27410, params={}, max_attempts=10)
    assert str(out.query_id) == SINGLE_QUERY_ID
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {"annotation": [574, 463, 746, 617], "answer": "C"}
    assert answer_only == {"answer": "C"}


def test_puzzles_cyclic_order_swap_repair_label_contract_is_unique() -> None:
    task = PuzzlesCyclicOrderSwapRepairLabelTask()
    out = task.generate(
        27440,
        params={
            "query_id": SINGLE_QUERY_ID,
            "token_render_style": "colored_shape_tokens",
            "scene_variant": "necklace_board",
            "loop_path_style": "ellipse",
        },
        max_attempts=10,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation_bbox = [float(value) for value in out.annotation_gt.value]

    assert str(out.query_id) == SINGLE_QUERY_ID
    assert str(out.scene_id) == "cyclic_order"
    assert task.supported_query_ids == (SINGLE_QUERY_ID,)
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert str(execution["query_id"]) == SINGLE_QUERY_ID
    assert str(execution["internal_query_id"]) == "swap_repair"
    assert str(execution["question_format"]) == "swap_repair_option"
    assert str(execution["view_family"]) == "loop_swap_repair"
    assert int(execution["option_count"]) == 4
    assert 4 <= int(execution["bead_count"]) <= 5
    assert int(execution["valid_option_count"]) == 1
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(out.answer_gt.value) == str(execution["answer_option_label"])
    assert trace["projected_annotation"]["bbox"] == annotation_bbox

    expected_bbox = [
        float(value)
        for value in render_map["option_choice_bboxes_px"][str(execution["answer_option_choice_id"])]
    ]
    assert annotation_bbox == expected_bbox

    option_specs = execution["option_specs"]
    assert [str(option["option_label"]) for option in option_specs] == ["A", "B", "C", "D"]
    reference_sequence = [str(value) for value in execution["reference_token_sequence"]]
    valid_labels = []
    for option in option_specs:
        repaired_sequence = [str(value) for value in option["repaired_token_sequence"]]
        is_equivalent = token_sequences_are_rotation_equivalent(
            reference_sequence,
            repaired_sequence,
        )
        assert bool(is_equivalent) == bool(option["is_valid"])
        if bool(option["is_valid"]):
            valid_labels.append(str(option["option_label"]))
            assert str(option["option_choice_id"]) == str(execution["answer_option_choice_id"])

    assert valid_labels == [str(out.answer_gt.value)]
    assert valid_labels == [str(value) for value in execution["valid_option_labels"]]
    assert [str(value) for value in execution["supporting_option_choice_ids"]] == [
        str(value) for value in execution["valid_option_choice_ids"]
    ]


def test_puzzles_cyclic_order_insertion_position_label_contract_is_unique() -> None:
    task = PuzzlesCyclicOrderInsertionPositionLabelTask()
    out = task.generate(
        27460,
        params={
            "query_id": SINGLE_QUERY_ID,
            "token_render_style": "symbol_badges",
            "scene_variant": "route_loop_diagram",
            "loop_path_style": "rounded_rect",
        },
        max_attempts=10,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation_bbox = [float(value) for value in out.annotation_gt.value]

    assert str(out.query_id) == SINGLE_QUERY_ID
    assert str(out.scene_id) == "cyclic_order"
    assert task.supported_query_ids == (SINGLE_QUERY_ID,)
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert str(execution["query_id"]) == SINGLE_QUERY_ID
    assert str(execution["internal_query_id"]) == "insertion_position"
    assert str(execution["question_format"]) == "insertion_position_option"
    assert str(execution["view_family"]) == "loop_insertion_position"
    assert int(execution["option_count"]) == 4
    assert int(execution["bead_count"]) == 5
    assert int(execution["valid_option_count"]) == 1
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(out.answer_gt.value) == str(execution["answer_option_label"])
    assert trace["projected_annotation"]["bbox"] == annotation_bbox

    expected_bbox = [
        float(value)
        for value in render_map["option_choice_bboxes_px"][str(execution["answer_option_choice_id"])]
    ]
    assert annotation_bbox == expected_bbox

    option_specs = execution["option_specs"]
    assert [str(option["option_label"]) for option in option_specs] == ["A", "B", "C", "D"]
    assert [int(option["gap_number"]) for option in option_specs] == [1, 2, 3, 4]
    assert [str(option["gap_label"]) for option in option_specs] == ["A", "B", "C", "D"]
    reference_sequence = [str(value) for value in execution["reference_token_sequence"]]
    partial_sequence = [str(value) for value in execution["partial_token_sequence"]]
    missing_token = str(execution["missing_token"])
    assert len(reference_sequence) == 5
    assert len(partial_sequence) == 4
    assert missing_token not in partial_sequence

    valid_labels = []
    for option in option_specs:
        inserted_sequence = [str(value) for value in option["inserted_token_sequence"]]
        is_equivalent = token_sequences_are_rotation_equivalent(
            reference_sequence,
            inserted_sequence,
        )
        assert bool(is_equivalent) == bool(option["is_valid"])
        if bool(option["is_valid"]):
            valid_labels.append(str(option["option_label"]))
            assert str(option["option_choice_id"]) == str(execution["answer_option_choice_id"])

    assert valid_labels == [str(out.answer_gt.value)]
    assert valid_labels == [str(value) for value in execution["valid_option_labels"]]
    assert [str(value) for value in execution["supporting_option_choice_ids"]] == [
        str(value) for value in execution["valid_option_choice_ids"]
    ]


def test_puzzles_cyclic_order_equivalent_label_task_is_deterministic() -> None:
    task = PuzzlesCyclicOrderEquivalentLabelTask()
    params = {
        "token_render_style": "symbol_badges",
        "scene_variant": "charm_card_grid",
        "loop_path_style": "wavy_loop",
    }
    out_a = task.generate(27480, params=params, max_attempts=10)
    out_b = task.generate(27480, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_puzzles_cyclic_order_sampling_balances_visual_and_answer_axes() -> None:
    token_render_styles = Counter()
    scene_variants = Counter()
    loop_path_styles = Counter()
    answer_labels = Counter()

    task = PuzzlesCyclicOrderEquivalentLabelTask()
    for sampling_index in range(200):
        out = task.generate(
            27520 + sampling_index,
            params={},
            max_attempts=10,
        )
        trace = out.trace_payload["execution_trace"]
        assert str(trace["query_id"]) == SINGLE_QUERY_ID
        assert int(trace["option_count"]) == 4
        token_render_styles[str(trace["token_render_style"])] += 1
        scene_variants[str(trace["scene_variant"])] += 1
        loop_path_styles[str(trace["loop_path_style"])] += 1
        answer_labels[str(trace["answer_option_label"])] += 1

    assert_counter_support_within(
        token_render_styles,
        {"colored_beads", "shape_tokens", "colored_shape_tokens", "outline_shape_tokens", "symbol_badges"},
        expected_per_key=40,
        tolerance=20,
    )
    assert_counter_support_within(
        scene_variants,
        {"necklace_board", "charm_card_grid", "route_loop_diagram", "token_ring_outline"},
        expected_per_key=50,
        tolerance=20,
    )
    assert_counter_support_within(
        loop_path_styles,
        {"ellipse", "rounded_rect", "polygon_loop", "wavy_loop", "beaded_string"},
        expected_per_key=40,
        tolerance=20,
    )
    assert_counter_support_within(
        answer_labels,
        {"A", "B", "C", "D"},
        expected_per_key=50,
        tolerance=20,
    )
