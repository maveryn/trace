"""Contract tests for tower draughts board games tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.tower_draughts_board.controlled_stack_count import TASK_ID as CONTROLLED_STACK_TASK_ID
from trace_tasks.tasks.games.tower_draughts_board.marked_stack_capture_count import TASK_ID as MARKED_CAPTURE_TASK_ID
from trace_tasks.tasks.games.tower_draughts_board.shared.rules import capture_targets, destination_candidates, playable_coords
from trace_tasks.tasks.games.tower_draughts_board.shared.state import BLACK, RED, StackSpec
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.core.query_ids import SINGLE_QUERY_ID


TASK_IDS = (
    CONTROLLED_STACK_TASK_ID,
    MARKED_CAPTURE_TASK_ID,
)


def _player_value(name: str) -> int:
    return RED if str(name) == "red" else BLACK


def _trace_stacks(trace: dict) -> tuple[StackSpec, ...]:
    stacks: list[StackSpec] = []
    for item in trace["stacks"]:
        stacks.append(
            StackSpec(
                coord=(int(item["coord"][0]), int(item["coord"][1])),
                disks=tuple(_player_value(value) for value in item["disks"]),
                top_crowned=bool(item["top_crowned"]),
            )
        )
    return tuple(stacks)


def _generate_with_seed_search(task_id: str, *, target_answer: int, board_size: int) -> object:
    task = create_task(task_id)
    for seed in range(730000, 730250):
        try:
            return task.generate(
                seed + int(target_answer),
                params={"target_answer": int(target_answer), "board_size": int(board_size)},
                max_attempts=200,
            )
        except RuntimeError:
            continue
    raise AssertionError(f"could not generate {task_id} target={target_answer} board_size={board_size}")


def test_games_tower_draughts_board_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "tower_draughts_board")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert set(generation["style_variant_weights"].keys()) == {
        "wood_table",
        "ink_board",
        "felt_mat",
        "night_tokens",
        "parchment",
    }
    assert set(generation["target_player_weights"].keys()) == {"red", "black"}
    assert set(generation["marked_player_weights"].keys()) == {"red", "black"}
    assert set(generation["top_kind_weights"].keys()) == {"regular", "crowned"}
    assert list(generation["board_size_support"]) == [4, 5, 6]
    assert list(generation["controlled_stack_count_support"]) == list(range(11))
    assert list(generation["marked_capture_count_support"]) == [0, 1, 2, 3, 4]
    assert list(generation["stack_height_support"]) == [1, 2, 3, 4]
    assert int(rendering["cell_size_min_px"]) == 56
    assert int(rendering["cell_size_max_px"]) == 80
    assert str(prompt["bundle_id"]) == "games_tower_draughts_board_v1"


def test_games_tower_draughts_board_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/tower_draughts_board/games_tower_draughts_board_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(bundle["templates"]["query"].keys()) == {
        "controlled_stack_count",
        "marked_stack_capture_count",
    }
    assert bool(bundle["allow_empty_task_templates"])


def test_games_tower_draughts_board_registry_and_taxonomy() -> None:
    for task_id in TASK_IDS:
        task = create_task(task_id)
        assert task.task_id == task_id
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "tower_draughts_board"
        assert taxonomy.source_scene_id == ""


def test_games_tower_draughts_board_playable_square_counts() -> None:
    assert len(playable_coords(4)) == 8
    assert len(playable_coords(5)) == 12
    assert len(playable_coords(6)) == 18
    for board_size in (4, 5, 6):
        assert all((row + col) % 2 == 1 for row, col in playable_coords(board_size))


def test_games_tower_draughts_board_direction_rules() -> None:
    regular_red = destination_candidates(coord=(3, 2), owner=RED, crowned=False, board_size=6)
    regular_black = destination_candidates(coord=(2, 3), owner=BLACK, crowned=False, board_size=6)
    crowned_red = destination_candidates(coord=(3, 2), owner=RED, crowned=True, board_size=6)

    assert regular_red == ((2, 1), (2, 3))
    assert regular_black == ((3, 2), (3, 4))
    assert crowned_red == ((2, 1), (2, 3), (4, 1), (4, 3))


def test_games_tower_draughts_board_controlled_count_matches_trace() -> None:
    out = _generate_with_seed_search(CONTROLLED_STACK_TASK_ID, target_answer=6, board_size=6)
    trace = out.trace_payload["execution_trace"]
    target_player = _player_value(trace["target_player"])
    stacks = _trace_stacks(trace)
    expected_coords = tuple(sorted(stack.coord for stack in stacks if stack.owner == target_player))

    assert out.scene_id == "tower_draughts_board"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == len(expected_coords) == 6
    assert tuple(tuple(coord) for coord in trace["annotation_coords"]) == expected_coords
    assert len(out.annotation_gt.value) == len(expected_coords)
    assert out.annotation_gt.type == "bbox_set"
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"
    assert trace["prompt_query_key"] == "controlled_stack_count"


def test_games_tower_draughts_board_capture_count_matches_trace() -> None:
    out = _generate_with_seed_search(MARKED_CAPTURE_TASK_ID, target_answer=4, board_size=6)
    trace = out.trace_payload["execution_trace"]
    expected = capture_targets(
        stacks=_trace_stacks(trace),
        marked_coord=tuple(trace["marked_coord"]),  # type: ignore[arg-type]
        board_size=int(trace["board_size"]),
    )

    assert out.scene_id == "tower_draughts_board"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == len(expected) == 4
    assert tuple(tuple(coord) for coord in trace["annotation_coords"]) == expected
    assert tuple(tuple(coord) for coord in trace["captured_stacks"]) == expected
    assert len(out.annotation_gt.value) == len(expected)
    assert out.query_id == SINGLE_QUERY_ID
    assert out.annotation_gt.type == "bbox_set"
    assert trace["prompt_query_key"] == "marked_stack_capture_count"


def test_games_tower_draughts_board_support_endpoints_are_constructible() -> None:
    cases = (
        (CONTROLLED_STACK_TASK_ID, 0, 4),
        (CONTROLLED_STACK_TASK_ID, 10, 6),
        (MARKED_CAPTURE_TASK_ID, 0, 4),
        (MARKED_CAPTURE_TASK_ID, 4, 6),
    )
    for task_id, target, board_size in cases:
        out = _generate_with_seed_search(task_id, target_answer=target, board_size=board_size)
        assert int(out.answer_gt.value) == target
        assert len(out.annotation_gt.value) == target
