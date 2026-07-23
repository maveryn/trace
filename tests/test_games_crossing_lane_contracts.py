"""Contract tests for games lane-crossing tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.crossing.first_exit_object_label import GamesCrossingFirstExitObjectLabelTask
from trace_tasks.tasks.games.crossing.hit_object_label import GamesCrossingHitObjectLabelTask
from trace_tasks.tasks.games.crossing.moving_object_direction_count import GamesCrossingMovingObjectDirectionCountTask
from trace_tasks.tasks.games.crossing.shared.rules import route_collision_vehicle_ids, vehicle_exit_tick
from trace_tasks.tasks.games.crossing.shared.state import (
    CrossingRouteOption,
    CrossingVehicle,
)
from tests.helpers import read_jsonl


def _vehicles(execution: dict) -> tuple[CrossingVehicle, ...]:
    return tuple(
        CrossingVehicle(
            vehicle_id=str(row["vehicle_id"]),
            row=int(row["row"]),
            start_col=int(row["start_col"]),
            direction=int(row["direction"]),
            color_index=int(row["color_index"]),
            option_label=None if row.get("option_label") is None else str(row["option_label"]),
        )
        for row in execution["vehicles"]
    )


def _routes(execution: dict) -> tuple[CrossingRouteOption, ...]:
    return tuple(
        CrossingRouteOption(
            route_id=str(row["route_id"]),
            label=str(row["label"]),
            path_cols=tuple(int(col) for col in row["path_cols"]),
            color_index=int(row["color_index"]),
        )
        for row in execution["route_options"]
    )


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query", "expected_answer", "expected_answer_type", "expected_annotation_type", "annotation_count"),
    (
        (
            GamesCrossingFirstExitObjectLabelTask,
            {"target_label": "B", "lane_count": 7, "row_count": 7, "style_variant": "paper"},
            "single",
            "B",
            "string",
            "point",
            1,
        ),
        (
            GamesCrossingHitObjectLabelTask,
            {"target_label": "C", "lane_count": 6, "row_count": 6, "style_variant": "retro"},
            "single",
            "C",
            "string",
            "point",
            1,
        ),
        (
            GamesCrossingMovingObjectDirectionCountTask,
            {"query_id": "left_moving_object_count", "target_answer": 4, "lane_count": 6, "row_count": 6},
            "left_moving_object_count",
            4,
            "integer",
            "point_set",
            4,
        ),
    ),
)
def test_games_crossing_public_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_query: str,
    expected_answer: int | str,
    expected_answer_type: str,
    expected_annotation_type: str,
    annotation_count: int,
) -> None:
    out = task_cls().generate(77100, params=params, max_attempts=512)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == expected_answer_type
    assert out.answer_gt.value == expected_answer
    assert out.annotation_gt.type == expected_annotation_type
    if expected_annotation_type == "point":
        assert len(out.annotation_gt.value) == 2
        assert annotation_count == 1
    else:
        assert len(out.annotation_gt.value) == annotation_count
    assert out.query_id == expected_query
    assert out.scene_id == "crossing"
    assert trace["query_spec"]["query_id"] == expected_query
    assert trace["query_spec"]["params"]["query_id"] == expected_query
    assert execution["query_id"] == expected_query
    assert trace["projected_annotation"]["type"] == expected_annotation_type
    assert trace["projected_annotation"][expected_annotation_type] == out.annotation_gt.value
    assert len(execution["annotation_entity_ids"]) == annotation_count


def test_games_crossing_hit_object_label_matches_trace() -> None:
    out = GamesCrossingHitObjectLabelTask().generate(
        77130,
        params={"target_label": "D", "lane_count": 8, "row_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    route = _routes(execution)[0]
    vehicles = _vehicles(execution)
    hit_ids = route_collision_vehicle_ids(route, vehicles, lane_count=int(execution["lane_count"]))
    labeled = {str(vehicle.option_label): str(vehicle.vehicle_id) for vehicle in vehicles if vehicle.option_label is not None}

    assert out.answer_gt.value == "D"
    assert set(labeled) == {"A", "B", "C", "D"}
    assert len(hit_ids) == 1
    assert labeled["D"] == hit_ids[0]
    assert tuple(execution["intersecting_vehicle_ids"]) == hit_ids
    assert tuple(execution["annotation_entity_ids"]) == hit_ids
    assert execution["target_object_label"] == "D"
    assert execution["target_label"] == "D"
    assert all(label.isdigit() for label in execution["start_labels"])
    assert len(set(route.path_cols)) == 1
    assert all(int(vehicle.start_col) != int(route.path_cols[int(vehicle.row)]) for vehicle in vehicles)


def test_games_crossing_first_exit_object_label_matches_trace() -> None:
    out = GamesCrossingFirstExitObjectLabelTask().generate(
        77132,
        params={"target_label": "A", "lane_count": 8, "row_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    vehicles = _vehicles(execution)
    labeled = {str(vehicle.option_label): str(vehicle.vehicle_id) for vehicle in vehicles if vehicle.option_label is not None}
    exit_ticks = {
        str(vehicle.vehicle_id): vehicle_exit_tick(vehicle, lane_count=int(execution["lane_count"]))
        for vehicle in vehicles
        if vehicle.option_label is not None
    }
    first_exit_tick = min(exit_ticks.values())
    first_exit_ids = tuple(
        sorted(vehicle_id for vehicle_id, tick in exit_ticks.items() if int(tick) == int(first_exit_tick))
    )

    assert out.answer_gt.value == "A"
    assert set(labeled) == {"A", "B", "C", "D"}
    assert len(first_exit_ids) == 1
    assert labeled["A"] == first_exit_ids[0]
    assert execution["route_options"] == []
    assert execution["start_labels"] == []
    assert tuple(execution["intersecting_vehicle_ids"]) == ()
    assert tuple(execution["annotation_entity_ids"]) == first_exit_ids
    assert execution["target_object_label"] == "A"
    assert execution["target_label"] == "A"
    assert execution["first_collision_tick"] is None
    assert execution["first_exit_tick"] == first_exit_tick


@pytest.mark.parametrize(
    ("query_id", "target_answer", "direction"),
    (
        ("left_moving_object_count", 6, -1),
        ("right_moving_object_count", 5, 1),
    ),
)
def test_games_crossing_direction_count_matches_trace(query_id: str, target_answer: int, direction: int) -> None:
    out = GamesCrossingMovingObjectDirectionCountTask().generate(
        77140 + int(target_answer),
        params={"query_id": query_id, "target_answer": target_answer, "lane_count": 7, "row_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    matching_ids = tuple(
        str(vehicle["vehicle_id"])
        for vehicle in execution["vehicles"]
        if int(vehicle["direction"]) == int(direction)
    )

    assert out.query_id == query_id
    assert int(out.answer_gt.value) == len(matching_ids) == int(target_answer)
    assert set(execution["annotation_entity_ids"]) == set(matching_ids)
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == int(target_answer)
    assert execution["route_options"] == []
    assert execution["marked_route_label"] is None


def test_games_crossing_direction_count_taxonomy() -> None:
    assert resolve_task_taxonomy("task_games__crossing__first_exit_object_label").scene_id == "crossing"
    assert resolve_task_taxonomy("task_games__crossing__hit_object_label").scene_id == "crossing"
    assert resolve_task_taxonomy("task_games__crossing__moving_object_direction_count").scene_id == "crossing"


def test_games_crossing_lane_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__crossing"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__crossing",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__crossing__first_exit_object_label", count=1, params={"target_label": "A"}),
            BuildTaskConfig(task_id="task_games__crossing__hit_object_label", count=1, params={"target_label": "B"}),
            BuildTaskConfig(
                task_id="task_games__crossing__moving_object_direction_count",
                count=1,
                params={"query_id": "right_moving_object_count", "target_answer": 3},
            ),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-crossing-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "crossing" for row in rows)
