"""Contract tests for games Pac-Man maze tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.pacman.next_item_label import GamesPacmanNextItemLabelTask
from trace_tasks.tasks.games.pacman.pellet_count_before_ghost import GamesPacmanPelletCountBeforeGhostTask
from trace_tasks.tasks.games.pacman.route_score_value import GamesPacmanRouteScoreValueTask
from trace_tasks.tasks.games.pacman.shared.defaults import (
    PACMAN_ITEM_LABELS,
)
from trace_tasks.tasks.games.pacman.shared.state import (
    coord_from_entity_id,
    item_entity_id,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query", "answer_type"),
    (
        (GamesPacmanNextItemLabelTask, {"target_label": "E", "item_count": 6}, SINGLE_QUERY_ID, "string"),
        (GamesPacmanPelletCountBeforeGhostTask, {"target_answer": 4, "row_count": 9, "col_count": 13}, SINGLE_QUERY_ID, "integer"),
        (GamesPacmanRouteScoreValueTask, {"row_count": 9, "col_count": 13}, SINGLE_QUERY_ID, "integer"),
    ),
)
def test_games_pacman_public_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_query: str,
    answer_type: str,
) -> None:
    out = task_cls().generate(120000, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == answer_type
    assert out.query_id == expected_query
    assert out.scene_id == "pacman"
    assert trace["query_spec"]["query_id"] == expected_query
    assert trace["query_spec"]["params"]["query_id"] == expected_query
    assert execution["query_id"] == expected_query
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type
    assert "panel_scene_style" in trace["render_spec"]
    assert "text_style" in trace["render_spec"]
    if out.annotation_gt.type == "point":
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == 1
    elif out.annotation_gt.type == "point_set_map":
        assert trace["projected_annotation"]["point_set_map"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point_set_map"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)


def test_games_pacman_next_item_label_is_first_route_item() -> None:
    out = GamesPacmanNextItemLabelTask().generate(
        120020,
        params={"target_label": "F", "item_count": 6, "row_count": 8, "col_count": 11},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    route_order = {tuple(coord): index for index, coord in enumerate(execution["route_coords"])}
    item_steps = [
        (route_order[tuple(item["coord"])], str(item["label"]))
        for item in execution["items"]
        if tuple(item["coord"]) in route_order
    ]

    assert out.answer_gt.value == "F"
    assert out.annotation_gt.type == "point"
    assert execution["annotation_entity_ids"] == [item_entity_id("F")]
    assert min(item_steps)[1] == "F"


def test_games_pacman_pellet_count_before_ghost_stops_at_first_route_ghost() -> None:
    out = GamesPacmanPelletCountBeforeGhostTask().generate(
        120030,
        params={"target_answer": 5, "row_count": 9, "col_count": 13},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    route_order = {tuple(coord): index for index, coord in enumerate(execution["route_coords"])}
    stop_ghosts = [ghost for ghost in execution["ghosts"] if bool(ghost["is_stop_ghost"])]
    annotation_coords = {
        coord_from_entity_id(entity_id)
        for entity_id in execution["annotation_entity_ids"]
        if str(entity_id).startswith("pellet_r")
    }
    annotation_ghosts = [entity_id for entity_id in execution["annotation_entity_ids"] if str(entity_id).startswith("ghost_")]

    assert int(out.answer_gt.value) == 5
    assert len(stop_ghosts) == 1
    assert annotation_ghosts == [stop_ghosts[0]["entity_id"]]
    assert out.annotation_gt.type == "point_set_map"
    assert set(out.annotation_gt.value) == {"counted_pellets", "first_ghost"}
    stop_index = route_order[tuple(stop_ghosts[0]["coord"])]
    assert len(annotation_coords) == 5
    assert all(route_order[coord] < stop_index for coord in annotation_coords)


def test_games_pacman_route_score_value_annotation_recomputes_score() -> None:
    out = GamesPacmanRouteScoreValueTask().generate(
        120040,
        params={"row_count": 9, "col_count": 13},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    route = {tuple(coord) for coord in execution["route_coords"]}
    pellet_coords = {
        tuple(pellet["coord"])
        for pellet in execution["pellets"]
    }
    item_by_id = {str(item["entity_id"]): item for item in execution["items"]}
    scored_total = 0
    saw_pellet = False
    saw_bonus = False

    for entity_id in execution["annotation_entity_ids"]:
        entity_id = str(entity_id)
        if entity_id.startswith("pellet_r"):
            coord = coord_from_entity_id(entity_id)
            assert coord in route
            assert coord in pellet_coords
            scored_total += 1
            saw_pellet = True
            continue
        item = item_by_id[entity_id]
        assert tuple(item["coord"]) in route
        assert "score_value" in item
        scored_total += int(item["score_value"])
        saw_bonus = True

    off_route_items = [item for item in execution["items"] if tuple(item["coord"]) not in route]
    assert saw_pellet
    assert saw_bonus
    assert off_route_items
    assert int(out.answer_gt.value) == scored_total
    assert "Normal pellets score 1; bonus items score their printed value." in out.prompt
    assert "Ignore ghosts and all collectibles away from the highlighted route." in out.prompt


def test_games_pacman_query_cycle_covers_support() -> None:
    tasks = (GamesPacmanPelletCountBeforeGhostTask(),)
    queries: set[str] = set()
    rows: set[int] = set()
    cols: set[int] = set()
    counts: set[int] = set()

    for sampling_index in range(144):
        task = tasks[int(sampling_index) % len(tasks)]
        out = task.generate(
            120100 + sampling_index,
            params={},
            max_attempts=256,
        )
        execution = out.trace_payload["execution_trace"]
        queries.add(str(out.query_id))
        rows.add(int(execution["row_count"]))
        cols.add(int(execution["col_count"]))
        counts.add(int(out.answer_gt.value))

    assert queries == {SINGLE_QUERY_ID}
    assert rows == {7, 8, 9}
    assert cols == {9, 11, 13}
    assert counts == {1, 2, 3, 4, 5}


def test_games_pacman_next_item_label_cycle_covers_labels() -> None:
    task = GamesPacmanNextItemLabelTask()
    labels: set[str] = set()

    for sampling_index in range(72):
        out = task.generate(
            120300 + sampling_index,
            params={},
            max_attempts=256,
        )
        labels.add(str(out.answer_gt.value))

    assert labels == set(PACMAN_ITEM_LABELS)


def test_games_pacman_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__pacman"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__pacman",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__pacman__next_item_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__pacman__pellet_count_before_ghost", count=1, params={}),
            BuildTaskConfig(task_id="task_games__pacman__route_score_value", count=1, params={}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-pacman-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "pacman" for row in rows)
