"""Contract tests for games Mini-golf course tasks."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.games.minigolf.first_obstacle_label import GamesMinigolfFirstObstacleLabelTask
from trace_tasks.tasks.games.minigolf.shot_path_label import GamesMinigolfShotPathLabelTask
from trace_tasks.tasks.games.minigolf.shared.rules import distance, first_hit_obstacle_id, trace_shot_path
from trace_tasks.tasks.games.minigolf.shared.sampling import MIN_OBSTACLE_POINT_CLEARANCE_NORM
from trace_tasks.tasks.games.minigolf.shared.state import MinigolfObstacle
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "prompt_query_key", "expected_annotation_type"),
    (
        (
            GamesMinigolfFirstObstacleLabelTask,
            {"target_obstacle_label": "F", "obstacle_count": 6, "style_variant": "garden"},
            "first_obstacle_label",
            "point",
        ),
        (
            GamesMinigolfShotPathLabelTask,
            {"path_option_count": 6, "target_path_index": 4, "style_variant": "blueprint"},
            "shot_path_label",
            "segment",
        ),
    ),
)
def test_games_minigolf_public_tasks_emit_expected_contract(
    task_cls: type[GamesMinigolfFirstObstacleLabelTask | GamesMinigolfShotPathLabelTask],
    params: dict[str, int | str],
    prompt_query_key: str,
    expected_annotation_type: str,
) -> None:
    out = task_cls().generate(97000, params=params, max_attempts=512)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "string"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.scene_id == "minigolf"
    assert trace["query_spec"]["query_id"] == SINGLE_QUERY_ID
    assert trace["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
    assert trace["query_spec"]["params"]["prompt_query_key"] == prompt_query_key
    assert execution["query_id"] == SINGLE_QUERY_ID
    assert out.annotation_gt.type == expected_annotation_type
    assert trace["projected_annotation"]["type"] == expected_annotation_type
    if expected_annotation_type == "segment":
        assert len(out.annotation_gt.value) == 2
        assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_segment"] == out.annotation_gt.value
    else:
        assert len(out.annotation_gt.value) == 2
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
    assert "panel_scene_style" in trace["render_spec"]
    assert "text_style" in trace["render_spec"]
    assert len(execution["annotation_entity_ids"]) == 1


def test_games_minigolf_first_obstacle_matches_first_ray_hit() -> None:
    out = GamesMinigolfFirstObstacleLabelTask().generate(
        97010,
        params={"target_obstacle_label": "F", "obstacle_count": 6},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_obstacle_id"])
    shown_path = execution["hidden_paths_norm"]["shown_path"]
    start = tuple(float(value) for value in shown_path[0])
    end = tuple(float(value) for value in shown_path[-1])
    angle = math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    obstacles = tuple(
        MinigolfObstacle(
            obstacle_id=str(obstacle["obstacle_id"]),
            label=str(obstacle["label"]),
            kind=str(obstacle["kind"]),
            x_norm=float(obstacle["x_norm"]),
            y_norm=float(obstacle["y_norm"]),
            radius_norm=float(obstacle["radius_norm"]),
            color_index=0,
        )
        for obstacle in execution["obstacles"]
    )

    first_hit = first_hit_obstacle_id(origin=start, angle_rad=angle, obstacles=obstacles)
    assert str(first_hit) == target_id
    assert str(out.answer_gt.value) == str(execution["target_obstacle_label"])
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert len(execution["obstacles"]) in {4, 6}
    assert [str(obstacle["label"]) for obstacle in execution["obstacles"]] == list("ABCDEF")


@pytest.mark.parametrize(
    ("task_cls", "params"),
    (
        (GamesMinigolfFirstObstacleLabelTask, {"obstacle_count": 6}),
        (GamesMinigolfShotPathLabelTask, {"path_option_count": 6, "target_path_index": 4}),
    ),
)
def test_games_minigolf_obstacles_stay_clear_of_hole_and_ball(
    task_cls: type[GamesMinigolfFirstObstacleLabelTask | GamesMinigolfShotPathLabelTask],
    params: dict[str, int],
) -> None:
    """Guard against course objects visually overlapping the cup or ball."""

    seeds = (13980958204946, 97030, 97031, 97032, 97033, 97034)
    for seed in seeds:
        out = task_cls().generate(int(seed), params=dict(params), max_attempts=512)
        execution = out.trace_payload["execution_trace"]
        ball = tuple(float(value) for value in execution["ball_xy_norm"])
        hole = tuple(float(value) for value in execution["hole_xy_norm"])
        assert len(execution["obstacles"]) in {4, 6}
        assert [str(obstacle["label"]) for obstacle in execution["obstacles"]] == list("ABCDEF")[: len(execution["obstacles"])]
        for obstacle in execution["obstacles"]:
            center = (float(obstacle["x_norm"]), float(obstacle["y_norm"]))
            assert distance(center, hole) >= MIN_OBSTACLE_POINT_CLEARANCE_NORM - 1e-9
            assert distance(center, ball) >= MIN_OBSTACLE_POINT_CLEARANCE_NORM - 1e-9


def test_games_minigolf_shot_path_has_one_hole_reaching_option() -> None:
    out = GamesMinigolfShotPathLabelTask().generate(
        97020,
        params={"path_option_count": 6, "target_path_index": 5},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_path_id"])
    ball = tuple(float(value) for value in execution["ball_xy_norm"])
    hole = tuple(float(value) for value in execution["hole_xy_norm"])
    obstacles = tuple(
        MinigolfObstacle(
            obstacle_id=str(obstacle["obstacle_id"]),
            label=str(obstacle["label"]),
            kind=str(obstacle["kind"]),
            x_norm=float(obstacle["x_norm"]),
            y_norm=float(obstacle["y_norm"]),
            radius_norm=float(obstacle["radius_norm"]),
            color_index=0,
        )
        for obstacle in execution["obstacles"]
    )
    success_ids: list[str] = []
    for option in execution["shot_options"]:
        reaches_hole, _blocked_by, _path = trace_shot_path(
            ball_xy=ball,
            angle_rad=float(option["angle_rad"]),
            hole_xy=hole,
            obstacles=obstacles,
        )
        if reaches_hole:
            success_ids.append(str(option["path_id"]))

    assert success_ids == [target_id]
    assert str(out.answer_gt.value) == str(execution["target_path_label"])
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert out.annotation_gt.type == "segment"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["path_point_pairs_px"][target_id]
    assert 4 <= len(execution["shot_options"]) <= 6


def test_games_minigolf_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__minigolf"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__minigolf",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__minigolf__first_obstacle_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__minigolf__shot_path_label", count=1, params={}),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-minigolf-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "minigolf" for row in rows)
