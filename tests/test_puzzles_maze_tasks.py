"""Behavior tests for puzzle maze source-layout tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.maze.exit_reachability_label import (
    PuzzlesMazeExitReachabilityLabelTask,
)
from trace_tasks.tasks.puzzles.maze.nearest_exit_label import (
    PuzzlesMazeNearestExitLabelTask,
)
from tests.helpers import extract_prompt_json_example


def test_puzzle_maze_tasks_contract_matches_maze_trace() -> None:
    """Check answer and annotation binding for each maze objective."""

    task_cases = (
        (
            PuzzlesMazeExitReachabilityLabelTask(),
            "exit_reachability_label",
            "reachable",
            "point",
        ),
        (
            PuzzlesMazeExitReachabilityLabelTask(),
            "exit_reachability_label",
            "unreachable",
            "point",
        ),
        (
            PuzzlesMazeNearestExitLabelTask(),
            "nearest_exit_label",
            None,
            "point",
        ),
    )
    scene_variants = (
        "classic_wall_maze",
        "paper_labyrinth_maze",
        "block_wall_maze",
    )

    for query_index, (task, prompt_query_key, target_reachability, annotation_type) in enumerate(task_cases):
        for scene_index, scene_variant in enumerate(scene_variants):
            seed = 29120 + (query_index * 20) + scene_index
            params = {"scene_variant": scene_variant}
            if target_reachability is not None:
                params["target_reachability"] = target_reachability
            out = task.generate(seed, params=params, max_attempts=10)
            trace = out.trace_payload
            execution = trace["execution_trace"]
            render = trace["render_spec"]
            render_map = trace["render_map"]
            supporting_ids = [str(value) for value in execution["supporting_item_ids"]]
            reachable_labels = [str(value) for value in execution["reachable_exit_labels"]]
            unreachable_labels = [str(value) for value in execution["unreachable_exit_labels"]]

            assert str(out.query_id) == SINGLE_QUERY_ID
            assert str(out.scene_id) == "maze"
            assert out.answer_gt.type == "string"
            assert out.annotation_gt.type == annotation_type
            assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(execution["query_id"]) == SINGLE_QUERY_ID
            assert str(execution["internal_query_id"]) == str(prompt_query_key)
            assert execution["target_reachability"] == target_reachability
            assert str(render["scene_variant"]) == str(scene_variant)
            assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
            assert render["text_style"]["font"]["source"] == "global_font_pool"
            assert render["text_style"]["font"]["font_family"]
            assert render["scene_style"]["font"]["font_family"] == render["text_style"]["font"]["font_family"]
            assert render["scene_style"]["maze"]["wall_color_rgb"]
            assert render["scene_style"]["maze"]["exit_palette_rgb"]
            assert str(execution["view_family"]) == "topology_orthogonal_maze_exit_label"
            assert str(execution["topology_rule"]) == (
                "move_through_open_corridors_from_start_walls_block_motion"
            )
            assert 6 <= int(execution["maze_rows"]) <= 8
            assert 7 <= int(execution["maze_cols"]) <= 10
            if str(prompt_query_key) == "exit_reachability_label":
                assert int(execution["exit_count"]) == 4
            else:
                assert int(execution["exit_count"]) == 4
            assert len(execution["exits"]) == int(execution["exit_count"])
            assert len(reachable_labels) == int(execution["reachable_exit_total"])
            assert len(reachable_labels) + len(unreachable_labels) == int(execution["exit_count"])
            assert str(render_map["annotation_source"]) == str(execution["supporting_annotation_source"])

            if str(prompt_query_key) in {"exit_reachability_label", "nearest_exit_label"}:
                annotation_point = [float(value) for value in out.annotation_gt.value]
                assert trace["projected_annotation"]["point"] == annotation_point
                assert trace["projected_annotation"]["pixel_point"] == annotation_point
                assert annotation_point == [
                    float(value)
                    for value in render_map["item_points_px"][str(supporting_ids[0])]
                ]
                x, y = annotation_point
                assert 0.0 <= x <= float(render["canvas_width"])
                assert 0.0 <= y <= float(render["canvas_height"])

            if str(prompt_query_key) == "exit_reachability_label" and str(target_reachability) == "reachable":
                assert len(reachable_labels) == 1
                assert str(out.answer_gt.value) == str(reachable_labels[0])
                assert len(supporting_ids) == 1
            elif str(prompt_query_key) == "exit_reachability_label" and str(target_reachability) == "unreachable":
                assert len(unreachable_labels) == 1
                assert str(out.answer_gt.value) == str(unreachable_labels[0])
                assert len(supporting_ids) == 1
            else:
                path_lengths = {
                    str(label): int(length)
                    for label, length in execution["exit_path_lengths_by_label"].items()
                }
                nearest_label = str(execution["nearest_exit_label"])
                nearest_length = int(execution["nearest_exit_path_length_edges"])
                assert len(path_lengths) == 4
                assert set(path_lengths) == set(reachable_labels)
                assert len(reachable_labels) == 4
                assert str(out.answer_gt.value) == nearest_label
                assert path_lengths[nearest_label] == nearest_length
                sorted_lengths = sorted(path_lengths.values())
                assert sorted_lengths[0] == nearest_length
                assert sorted_lengths[1] - sorted_lengths[0] >= int(
                    execution["nearest_exit_min_gap_edges"]
                )
                assert int(execution["nearest_exit_margin_edges"]) == (
                    sorted_lengths[1] - sorted_lengths[0]
                )
                assert len(supporting_ids) == 1
                assert supporting_ids[0] == str(execution["nearest_exit_item_id"])
                assert execution["nearest_exit_path_cells"][0] == execution["start_cell"]


def test_puzzle_maze_prompt_examples_match_selected_variants() -> None:
    """Prompt examples should match the public annotation schema for each task."""

    expected = {
        "exit_reachability_label": (
            PuzzlesMazeExitReachabilityLabelTask(),
            {"annotation": [194, 118], "answer": "C"},
            {"answer": "C"},
        ),
        "nearest_exit_label": (
            PuzzlesMazeNearestExitLabelTask(),
            {
                "annotation": [194, 118],
                "answer": "C",
            },
            {"answer": "C"},
        ),
    }
    for index, (_prompt_query_key, (task, expected_answer_and_annotation, expected_answer_only)) in enumerate(
        expected.items(),
        start=29210,
    ):
        out = task.generate(index, params={}, max_attempts=10)
        assert str(out.query_id) == SINGLE_QUERY_ID
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only


def test_puzzle_maze_nearest_exit_label_task_is_deterministic() -> None:
    """Generation should be deterministic for a fixed seed and parameters."""

    task = PuzzlesMazeNearestExitLabelTask()
    params = {"scene_variant": "block_wall_maze"}
    out_a = task.generate(29280, params=params, max_attempts=10)
    out_b = task.generate(29280, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_puzzle_maze_sampling_decouples_visual_variant_and_operands() -> None:
    """Visual variants and task operands should be sampled independently."""

    label_task = PuzzlesMazeExitReachabilityLabelTask()
    nearest_task = PuzzlesMazeNearestExitLabelTask()
    label_scene_combos = Counter()
    nearest_scene_combos = Counter()
    nearest_label_combos = Counter()
    target_combos = Counter()
    exit_counts_by_task: dict[str, set[int]] = {}

    for sampling_index in range(120):
        out = label_task.generate(29320 + sampling_index, params={}, max_attempts=10)
        trace = out.trace_payload["execution_trace"]
        assert str(trace["query_id"]) == SINGLE_QUERY_ID
        assert str(trace["internal_query_id"]) == "exit_reachability_label"
        label_scene_combos[str(trace["scene_variant"])] += 1
        target_combos[(str(trace["target_reachability"]), str(trace["scene_variant"]))] += 1
        exit_counts_by_task.setdefault("exit_reachability_label", set()).add(int(trace["exit_count"]))

    for sampling_index in range(120):
        out = nearest_task.generate(29480 + sampling_index, params={}, max_attempts=10)
        trace = out.trace_payload["execution_trace"]
        assert str(trace["query_id"]) == SINGLE_QUERY_ID
        assert str(trace["internal_query_id"]) == "nearest_exit_label"
        nearest_scene_combos[str(trace["scene_variant"])] += 1
        nearest_label_combos[str(trace["nearest_exit_label"])] += 1
        exit_counts_by_task.setdefault("nearest_exit_label", set()).add(int(trace["exit_count"]))

    expected_scene_support = {"classic_wall_maze", "paper_labyrinth_maze", "block_wall_maze"}
    assert set(label_scene_combos) == expected_scene_support
    assert set(nearest_scene_combos) == expected_scene_support
    assert min(label_scene_combos.values()) >= 10
    assert min(nearest_scene_combos.values()) >= 10
    expected_target_support = {
        ("reachable", "classic_wall_maze"),
        ("reachable", "paper_labyrinth_maze"),
        ("reachable", "block_wall_maze"),
        ("unreachable", "classic_wall_maze"),
        ("unreachable", "paper_labyrinth_maze"),
        ("unreachable", "block_wall_maze"),
    }
    assert set(target_combos) == expected_target_support
    assert min(target_combos.values()) >= 5
    assert len(nearest_label_combos) >= 3
    assert exit_counts_by_task["exit_reachability_label"] == {4}
    assert exit_counts_by_task["nearest_exit_label"] == {4}


def test_puzzle_maze_exit_reachability_label_samples_target_reachability() -> None:
    """Explicit semantic target reachability should not become a public query id."""

    task = PuzzlesMazeExitReachabilityLabelTask()
    cases = {
        "reachable": "reachable",
        "unreachable": "unreachable",
    }
    for index, (target_reachability, expected) in enumerate(cases.items()):
        out = task.generate(
            29440 + index,
            params={"target_reachability": target_reachability},
            max_attempts=10,
        )
        trace = out.trace_payload["execution_trace"]
        assert out.query_id == SINGLE_QUERY_ID
        assert trace["query_id"] == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == "exit_reachability_label"
        assert trace["target_reachability"] == expected
