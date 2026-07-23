"""Contract tests for radial hunt board games tasks."""

from __future__ import annotations

from pathlib import Path
import json

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.radial_hunt_board.capture_move_count import TASK_ID as CAPTURE_MOVE_TASK_ID
from trace_tasks.tasks.games.radial_hunt_board.marked_piece_destination_count import TASK_ID as MARKED_DESTINATION_TASK_ID
from trace_tasks.tasks.games.radial_hunt_board.shared.rules import (
    CENTER,
    all_coords,
    all_possible_edges,
    capture_destinations,
    capture_paths,
    edge,
    legal_destinations,
    neighbors,
)
from trace_tasks.tasks.registry import create_task, list_default_task_ids
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_radial_hunt_board_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "radial_hunt_board")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert set(generation["scene_variant_weights"].keys()) == {
        "open_position",
        "mixed_position",
        "crowded_position",
    }
    assert set(generation["style_variant_weights"].keys()) == {
        "ink_rings",
        "carved_wood",
        "temple_cloth",
        "night_gold",
        "chalk_circle",
    }
    assert list(generation["target_answer_support"]) == list(range(7))
    assert int(rendering["max_board_size_px"]) == 560
    assert str(prompt["bundle_id"]) == "games_radial_hunt_board_v1"


def test_games_radial_hunt_board_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/radial_hunt_board/games_radial_hunt_board_v1.json").read_text(encoding="utf-8")
    )
    assert bundle["schema_version"] == "v1"
    assert set(bundle["templates"]["query"].keys()) == {
        "marked_piece_destination_count",
        "capture_move_count",
    }
    assert bool(bundle["allow_empty_task_templates"])


def test_games_radial_hunt_board_graph_shape() -> None:
    assert len(all_coords()) == 19
    assert len(all_possible_edges()) == 36
    assert len(neighbors(CENTER)) == 6

    for ring in (1, 2):
        for spoke in range(6):
            assert len(neighbors((ring, spoke))) == 4
    for spoke in range(6):
        assert len(neighbors((3, spoke))) == 3

    assert edge(CENTER, (1, 0)) in set(all_possible_edges())
    assert edge((1, 0), (1, 1)) in set(all_possible_edges())
    assert edge((2, 4), (3, 4)) in set(all_possible_edges())
    assert len(capture_paths(CENTER)) == 6


def test_games_radial_hunt_board_registry_and_taxonomy() -> None:
    default_ids = set(list_default_task_ids())
    assert MARKED_DESTINATION_TASK_ID in default_ids
    assert CAPTURE_MOVE_TASK_ID in default_ids
    for task_id in (MARKED_DESTINATION_TASK_ID, CAPTURE_MOVE_TASK_ID):
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "radial_hunt_board"
        assert taxonomy.source_scene_id == ""


def test_games_radial_hunt_board_destination_answer_matches_trace() -> None:
    out = create_task(MARKED_DESTINATION_TASK_ID).generate(
        551003,
        params={"target_answer": 5},
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    marked = tuple(trace["marked_coord"])
    occupied = tuple(tuple(coord) for coord in trace["occupied_coords"])
    expected = legal_destinations(
        marked_coord=marked,  # type: ignore[arg-type]
        occupied_coords=occupied,  # type: ignore[arg-type]
    )

    assert out.scene_id == "radial_hunt_board"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 5
    assert len(out.annotation_gt.value) == 5
    assert tuple(tuple(coord) for coord in trace["annotation_coords"]) == expected
    assert out.trace_payload["projected_annotation"]["type"] == "point_set"


def test_games_radial_hunt_board_capture_answer_matches_trace() -> None:
    out = create_task(CAPTURE_MOVE_TASK_ID).generate(
        552003,
        params={"target_answer": 5},
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    marked = tuple(trace["marked_coord"])
    occupied = tuple(tuple(coord) for coord in trace["occupied_coords"])
    expected = capture_destinations(
        marked_coord=marked,  # type: ignore[arg-type]
        occupied_coords=occupied,  # type: ignore[arg-type]
    )

    assert out.scene_id == "radial_hunt_board"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 5
    assert len(out.annotation_gt.value) == 5
    assert tuple(tuple(coord) for coord in trace["annotation_coords"]) == expected
    assert tuple(tuple(coord) for coord in trace["capture_destinations"]) == expected
    assert out.trace_payload["projected_annotation"]["type"] == "point_set"

    occupied_set = {tuple(coord) for coord in occupied}
    for destination in expected:
        matching = [
            captured
            for path_destination, captured in capture_paths(marked)  # type: ignore[arg-type]
            if tuple(path_destination) == tuple(destination)
        ]
        assert any(captured in occupied_set for captured in matching)
        assert destination not in occupied_set


def test_games_radial_hunt_board_support_endpoints_are_constructible() -> None:
    for task_id in (MARKED_DESTINATION_TASK_ID, CAPTURE_MOVE_TASK_ID):
        task = create_task(task_id)
        for target in (0, 6):
            out = task.generate(
                553000 + target + (100 if task_id == CAPTURE_MOVE_TASK_ID else 0),
                params={"target_answer": target},
                max_attempts=100,
            )
            assert int(out.answer_gt.value) == target
            assert len(out.annotation_gt.value) == target
