"""Contract tests for games Pool-table tasks."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.pool.blocking_ball_count import GamesPoolBlockingBallCountTask
from trace_tasks.tasks.games.pool.group_ball_count import GamesPoolGroupBallCountTask
from trace_tasks.tasks.games.pool.shared.rules import (
    balls_on_segment,
    object_balls,
    pocket_by_id,
    sorted_ids,
)
from trace_tasks.tasks.games.pool.shared.state import POOL_POCKETS, PoolBall
from tests.helpers import read_jsonl


def _balls_from_execution(execution: dict) -> tuple[PoolBall, ...]:
    return tuple(
        PoolBall(
            ball_id=str(row["ball_id"]),
            number=int(row["number"]),
            group=str(row["group"]),
            center=(float(row["center"][0]), float(row["center"][1])),
            is_cue=bool(row["is_cue"]),
            is_marked=bool(row["is_marked"]),
        )
        for row in execution["balls"]
    )


def test_games_pool_table_scene_package_source_layout() -> None:
    expected_sources = {
        GamesPoolBlockingBallCountTask: Path("src/trace_tasks/tasks/games/pool/blocking_ball_count.py"),
        GamesPoolGroupBallCountTask: Path("src/trace_tasks/tasks/games/pool/group_ball_count.py"),
    }

    for task_cls, relative_path in expected_sources.items():
        source_path = Path(inspect.getsourcefile(task_cls) or "").resolve()
        assert source_path == (Path.cwd() / relative_path).resolve()


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_answer"),
    (
        (
            GamesPoolGroupBallCountTask,
            {
                "target_answer": 4,
                "object_ball_count": 9,
                "current_player_group": "solid",
                "style_variant": "classic",
            },
            4,
        ),
        (
            GamesPoolBlockingBallCountTask,
            {"target_answer": 2, "style_variant": "charcoal"},
            2,
        ),
    ),
)
def test_games_pool_table_public_tasks_emit_expected_contract(
    task_cls: type[GamesPoolGroupBallCountTask] | type[GamesPoolBlockingBallCountTask],
    params: dict[str, int | str],
    expected_answer: int,
) -> None:
    out = task_cls().generate(61100, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "point_set"
    assert out.query_id == "single"
    assert out.scene_id == "pool"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)


def test_games_pool_group_ball_count_matches_current_group() -> None:
    out = GamesPoolGroupBallCountTask().generate(
        61100,
        params={
            "target_answer": 5,
            "object_ball_count": 10,
            "current_player_group": "stripe",
        },
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    balls = _balls_from_execution(execution)
    expected_ids = sorted_ids(
        ball.ball_id
        for ball in object_balls(balls)
        if str(ball.group) == str(execution["current_player_group"])
    )

    assert str(execution["current_player_group"]) == "stripe"
    assert len(expected_ids) == int(out.answer_gt.value) == 5
    assert list(expected_ids) == execution["annotation_entity_ids"]


def test_games_pool_blocking_ball_count_matches_marked_two_segment_lane() -> None:
    out = GamesPoolBlockingBallCountTask().generate(
        61100,
        params={"target_answer": 3},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    balls = _balls_from_execution(execution)
    clearance = float(out.trace_payload["query_spec"]["params"]["line_clearance"])
    cue = next(ball for ball in balls if str(ball.ball_id) == str(execution["cue_ball_id"]))
    target = next(ball for ball in balls if str(ball.ball_id) == str(execution["marked_ball_id"]))
    pocket = pocket_by_id(POOL_POCKETS, str(execution["marked_pocket_id"]))
    first_segment = balls_on_segment(
        balls=balls,
        start=cue.center,
        end=target.center,
        ignore_ball_ids=(str(cue.ball_id), str(target.ball_id)),
        clearance=clearance,
    )
    second_segment = balls_on_segment(
        balls=balls,
        start=target.center,
        end=pocket.center,
        ignore_ball_ids=(str(target.ball_id),),
        clearance=clearance,
    )
    expected_ids = sorted_ids(ball.ball_id for ball in (*first_segment, *second_segment))

    assert len(expected_ids) == int(out.answer_gt.value) == 3
    assert list(expected_ids) == execution["blocking_ball_ids"]
    assert list(expected_ids) == execution["annotation_entity_ids"]


def test_games_pool_table_supports_requested_answer_ranges() -> None:
    cases = (
        (GamesPoolGroupBallCountTask, range(2, 7)),
        (GamesPoolBlockingBallCountTask, range(0, 5)),
    )
    for case_index, case in enumerate(cases):
        task_cls, support = case
        task = task_cls()
        for target_answer in support:
            params: dict[str, int | str] = {"target_answer": int(target_answer)}
            out = task.generate(
                61200 + (case_index * 100) + int(target_answer),
                params=params,
                max_attempts=256,
            )
            assert int(out.answer_gt.value) == int(target_answer)


def test_games_pool_table_is_deterministic() -> None:
    params = {
        "target_answer": 2,
        "style_variant": "light_rail",
    }
    task = GamesPoolBlockingBallCountTask()
    out_a = task.generate(61150, params=params, max_attempts=256)
    out_b = task.generate(61150, params=params, max_attempts=256)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_pool_table_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__pool"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__pool",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__pool__group_ball_count", count=2, params={"target_answer": 3}),
            BuildTaskConfig(task_id="task_games__pool__blocking_ball_count", count=1, params={"target_answer": 2}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-pool-table-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["scene_id"] == "pool" for row in rows)
    assert all(row.get("scene_id") for row in rows)
