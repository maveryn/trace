"""Contract tests for games Platformer level tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.platformer.collectible_count import (
    GamesPlatformerCollectibleCountTask,
)
from trace_tasks.tasks.games.platformer.jump_collectible_score_value import GamesPlatformerJumpCollectibleScoreValueTask
from trace_tasks.tasks.games.platformer.jump_landing_label import GamesPlatformerJumpLandingLabelTask
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query", "expected_type"),
    (
        (
            GamesPlatformerJumpLandingLabelTask,
            {"target_platform_label": "F", "platform_count": 7, "style_variant": "snow"},
            "single",
            ("string", "bbox"),
        ),
        (
            GamesPlatformerCollectibleCountTask,
            {"target_collectible_count": 6, "style_variant": "cave"},
            "single",
            ("integer", "point_set"),
        ),
        (
            GamesPlatformerJumpCollectibleScoreValueTask,
            {"style_variant": "neon"},
            "single",
            ("integer", "point_set"),
        ),
    ),
)
def test_games_platformer_public_tasks_emit_expected_contract(
    task_cls: type[Any],
    params: dict[str, int | str],
    expected_query: str,
    expected_type: tuple[str, str],
) -> None:
    out = task_cls().generate(97200, params=params, max_attempts=512)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    expected_answer_type, expected_annotation_type = expected_type
    assert out.answer_gt.type == expected_answer_type
    assert out.annotation_gt.type == expected_annotation_type
    assert out.query_id == expected_query
    assert out.scene_id == "platformer"
    assert trace["query_spec"]["query_id"] == expected_query
    assert trace["query_spec"]["params"]["query_id"] == expected_query
    assert execution["query_id"] == expected_query
    if expected_annotation_type == "bbox":
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == 1
    else:
        assert trace["projected_annotation"][expected_annotation_type] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)


def test_games_platformer_landing_answer_matches_target_platform() -> None:
    out = GamesPlatformerJumpLandingLabelTask().generate(
        97210,
        params={"target_platform_label": "H", "platform_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_platform_id"])
    target_platform = next(platform for platform in execution["platforms"] if str(platform["platform_id"]) == target_id)

    assert str(out.answer_gt.value) == str(target_platform["label"])
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert out.query_id == "single"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert 4 <= len(execution["platforms"]) <= 7
    path = execution["path_points_norm"]
    peak_index = min(range(len(path)), key=lambda index: float(path[int(index)][1]))
    peak_fraction = float(peak_index) / float(max(1, len(path) - 1))
    assert peak_fraction + 0.08 <= float(execution["visible_path_fraction"]) <= peak_fraction + 0.14


def test_games_platformer_collectible_count_matches_on_path_coins() -> None:
    out = GamesPlatformerCollectibleCountTask().generate(
        97230,
        params={"target_collectible_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    on_path_ids = [str(coin["collectible_id"]) for coin in execution["collectibles"] if bool(coin["on_path"])]

    assert int(out.answer_gt.value) == len(on_path_ids) == 7
    assert list(execution["annotation_entity_ids"]) == on_path_ids
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == 7


def test_games_platformer_score_value_matches_on_arc_collectibles() -> None:
    out = GamesPlatformerJumpCollectibleScoreValueTask().generate(
        97240,
        params={},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_ids = [str(collectible_id) for collectible_id in execution["target_collectible_ids"]]
    collectible_by_id = {str(coin["collectible_id"]): coin for coin in execution["collectibles"]}

    assert target_ids
    assert list(execution["annotation_entity_ids"]) == target_ids
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == len(target_ids)

    score = 0
    has_coin = False
    has_bonus = False
    for collectible_id in target_ids:
        collectible = collectible_by_id[collectible_id]
        assert bool(collectible["on_path"]) is True
        value = int(collectible["score_value"])
        score += value
        if str(collectible["kind"]) == "coin":
            has_coin = True
            assert value == 1
        else:
            has_bonus = True
            assert value > 1
            assert str(collectible["display_text"]) == str(value)

    assert has_coin is True
    assert has_bonus is True
    assert any(
        not bool(coin["on_path"]) and int(coin["score_value"]) > 1
        for coin in execution["collectibles"]
    )
    assert int(out.answer_gt.value) == score
    assert "Coins score 1; bonus items score their printed value." in out.prompt
    assert "Ignore hazards and all collectibles away from the dashed arc." in out.prompt


def test_games_platformer_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__platformer"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__platformer",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__platformer__jump_landing_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__platformer__collectible_count", count=1, params={}),
            BuildTaskConfig(task_id="task_games__platformer__jump_collectible_score_value", count=1, params={}),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-platformer-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "platformer" for row in rows)
