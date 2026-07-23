"""source-layout contract tests for migrated cell-board puzzle tasks."""

from __future__ import annotations

from collections import deque

import pytest

from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.puzzles.cell_board.shared.topology import (
    coord_distance,
    four_neighbors,
)

_TASK_CASES = [
    (
        "task_puzzles__cell_board__largest_component_size",
        ("single",),
        "bbox_set",
        True,
    ),
    (
        "task_puzzles__cell_board__reachable_region_size",
        ("single",),
        "bbox_set",
        True,
    ),
    (
        "task_puzzles__cell_board__shortest_path_length_value",
        ("single",),
        "segment_set",
        True,
    ),
    (
        "task_puzzles__cell_board__symmetry_violation_count",
        ("single",),
        "segment_set",
        True,
    ),
]


@pytest.mark.parametrize(
    ("task_id", "query_ids", "annotation_type", "cardinality_matches_answer"),
    _TASK_CASES,
)
def test_cell_board_tasks_smoke_every_query_branch(
    task_id: str,
    query_ids: tuple[str, ...],
    annotation_type: str,
    cardinality_matches_answer: bool,
) -> None:
    task = TASK_REGISTRY[task_id]()
    for index, query_id in enumerate(query_ids):
        output = task.generate(
            20260626 + index,
            params={"query_id": query_id},
            max_attempts=80,
        )
        trace = output.trace_payload

        assert output.scene_id == "cell_board"
        assert output.query_id == query_id
        assert output.answer_gt.type == "integer"
        assert isinstance(output.answer_gt.value, int)
        assert output.annotation_gt.type == annotation_type
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["params"]["query_id"] == query_id
        assert trace["query_spec"]["params"]["scene_id"] == "cell_board"
        assert "prompt_task_key" in trace["query_spec"]["params"]
        assert trace["render_spec"]["scene_id"] == "cell_board"
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == (
            "puzzles_cell_board_v1"
        )
        assert sorted(output.prompt_variants) == [
            "answer_and_annotation",
            "answer_only",
        ]
        assert output.image.size[0] >= int(trace["render_spec"]["tile_width_px"])
        assert output.image.size[1] >= int(trace["render_spec"]["tile_height_px"])
        tile_w = int(trace["render_spec"]["tile_width_px"])
        tile_h = int(trace["render_spec"]["tile_height_px"])
        assert max(tile_w, tile_h) / min(tile_w, tile_h) <= 1.25 + 1e-9
        assert trace["projected_annotation"]["type"] == annotation_type
        assert bool(cardinality_matches_answer)
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)


@pytest.mark.parametrize(
    ("task_id", "query_ids", "_annotation_type", "_cardinality"),
    _TASK_CASES,
)
def test_cell_board_tasks_are_deterministic(
    task_id: str,
    query_ids: tuple[str, ...],
    _annotation_type: str,
    _cardinality: bool,
) -> None:
    task = TASK_REGISTRY[task_id]()
    params = {"query_id": query_ids[0]}
    out_a = task.generate(998877, params=params, max_attempts=80)
    out_b = task.generate(998877, params=params, max_attempts=80)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["query_spec"] == out_b.trace_payload["query_spec"]
    assert out_a.image.size == out_b.image.size
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_largest_component_target_color_is_separated_from_fillers() -> None:
    task = TASK_REGISTRY["task_puzzles__cell_board__largest_component_size"]()

    for seed in range(2026070400, 2026070410):
        output = task.generate(seed, params={}, max_attempts=80)
        execution = output.trace_payload["execution_trace"]
        target_name = str(execution["query_color"])
        target_rgb = named_color(target_name)
        threshold = float(execution["target_filler_min_color_distance"])
        distance_space = str(execution["target_filler_color_distance_space"])
        color_names = {
            str(color_name)
            for row in execution["color_grid"]
            for color_name in row
            if str(color_name) != target_name
        }

        assert threshold >= 65.0
        assert color_names
        for color_name in color_names:
            assert (
                color_distance(
                    target_rgb,
                    named_color(str(color_name)),
                    distance_space=distance_space,
                )
                >= threshold
            )


def _has_path_without_edge(
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: set[tuple[int, int]],
    blocked_edge: tuple[tuple[int, int], tuple[int, int]],
    rows: int,
    cols: int,
) -> bool:
    """Return whether S and G remain connected after removing one path edge."""

    blocked = {blocked_edge, (blocked_edge[1], blocked_edge[0])}
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for neighbor in four_neighbors(current, rows=rows, cols=cols):
            if neighbor not in passable or neighbor in seen:
                continue
            if (current, neighbor) in blocked:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return False


def test_reachable_region_size_has_harder_passable_distractors() -> None:
    task = TASK_REGISTRY["task_puzzles__cell_board__reachable_region_size"]()

    for seed in range(20260627, 20260637):
        output = task.generate(seed, params={}, max_attempts=80)
        execution = output.trace_payload["execution_trace"]
        answer = int(output.answer_gt.value)
        reachable = [tuple(coord) for coord in execution["reachable_cells"]]
        distractors = [tuple(coord) for coord in execution["distractor_open_cells"]]
        diagonal = [tuple(coord) for coord in execution["diagonal_distractor_cells"]]

        assert 1 <= answer <= 8
        assert len(reachable) == answer
        assert len(distractors) >= 4
        assert int(execution["passable_cell_count"]) > answer
        assert diagonal
        assert set(reachable).isdisjoint(distractors)

        assert any(
            coord_distance(distractor, reachable_cell) == 2
            and distractor[0] != reachable_cell[0]
            and distractor[1] != reachable_cell[1]
            for distractor in diagonal
            for reachable_cell in reachable
        )


def test_shortest_path_length_uses_detour_and_distractors() -> None:
    task = TASK_REGISTRY["task_puzzles__cell_board__shortest_path_length_value"]()

    for seed in range(20260627, 20260637):
        output = task.generate(seed, params={}, max_attempts=80)
        execution = output.trace_payload["execution_trace"]
        answer = int(output.answer_gt.value)
        start = tuple(execution["start_cell"])
        goal = tuple(execution["goal_cell"])
        path = [tuple(coord) for coord in execution["shortest_path_cells"]]
        distractors = [tuple(coord) for coord in execution["distractor_open_cells"]]
        alternate = [tuple(coord) for coord in execution["alternate_path_cells"]]
        bypassed_edge = tuple(
            tuple(coord) for coord in execution["alternate_bypassed_edge"]
        )

        assert 4 <= answer <= 8
        assert len(path) == answer + 1
        assert path[0] == start
        assert path[-1] == goal
        assert int(execution["direct_manhattan_distance"]) < answer
        assert coord_distance(start, goal) < answer
        assert len(distractors) >= 4
        assert len(alternate) == 2
        assert set(path).isdisjoint(distractors)
        assert set(path).isdisjoint(alternate)
        assert _has_path_without_edge(
            start=start,
            goal=goal,
            passable=set(path) | set(alternate),
            blocked_edge=bypassed_edge,
            rows=int(execution["rows"]),
            cols=int(execution["cols"]),
        )


def test_symmetry_violation_count_uses_small_boards() -> None:
    task = TASK_REGISTRY["task_puzzles__cell_board__symmetry_violation_count"]()

    for seed in range(20260627, 20260637):
        output = task.generate(seed, params={}, max_attempts=80)
        execution = output.trace_payload["execution_trace"]
        rows = int(execution["rows"])
        cols = int(execution["cols"])

        assert 3 <= rows <= 5
        assert 3 <= cols <= 5
        assert 1 <= int(output.answer_gt.value) <= 5
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)
        assert execution["prompt_query_key"] == "symmetry_violation_count"
        assert output.annotation_gt.type == "segment_set"
        assert len(execution["violating_mirror_pairs"]) == int(output.answer_gt.value)
