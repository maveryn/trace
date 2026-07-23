"""Contract tests for games Space-shooter playfield tasks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.space_shooter.enemy_ship_count import (
    GamesSpaceShooterEnemyShipCountTask,
)
from trace_tasks.tasks.games.space_shooter.enemy_ship_hit_count import (
    GamesSpaceShooterEnemyShipHitCountTask,
)
from trace_tasks.tasks.games.space_shooter.first_hit_enemy_ship_label import (
    GamesSpaceShooterFirstHitEnemyShipLabelTask,
)
from trace_tasks.tasks.games.space_shooter.hit_enemy_ship_label import (
    GamesSpaceShooterHitEnemyShipLabelTask,
)
from trace_tasks.tasks.games.space_shooter.safe_lane_count import (
    GamesSpaceShooterSafeLaneCountTask,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_prompt_query_key", "expected_answer"),
    (
        (
            GamesSpaceShooterEnemyShipCountTask,
            {"lane_count": 7, "enemy_count": 9, "style_variant": "deep_space"},
            "enemy_ship_count",
            9,
        ),
        (
            GamesSpaceShooterEnemyShipHitCountTask,
            {"target_answer": 6, "lane_count": 7, "enemy_count": 12, "style_variant": "vector"},
            "enemy_ship_hit_count",
            6,
        ),
        (
            GamesSpaceShooterSafeLaneCountTask,
            {"target_answer": 2, "lane_count": 6, "enemy_count": 10, "style_variant": "terminal"},
            "safe_lane_count",
            2,
        ),
    ),
)
def test_games_space_shooter_public_tasks_emit_expected_contract(
    task_cls,
    params: dict[str, int | str],
    expected_prompt_query_key: str,
    expected_answer: int,
) -> None:
    out = task_cls().generate(88100, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == "single"
    assert out.scene_id == "space_shooter"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_query_key"] == expected_prompt_query_key
    assert execution["query_id"] == "single"
    assert execution["prompt_query_key"] == expected_prompt_query_key
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["render_spec"]["panel_scene_style"]["treatment"]
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["panel_bbox_px"] is not None
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)


def test_games_space_shooter_enemy_ship_count_matches_trace() -> None:
    out = GamesSpaceShooterEnemyShipCountTask().generate(
        88105,
        params={"lane_count": 7, "enemy_count": 11},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    enemy_ids = [str(enemy["enemy_id"]) for enemy in execution["enemies"]]
    enemy_projectile_counts_by_lane = Counter(
        int(projectile["lane"])
        for projectile in execution["projectiles"]
        if str(projectile["owner"]) == "enemy"
    )

    assert int(out.answer_gt.value) == len(enemy_ids) == 11
    assert list(execution["annotation_entity_ids"]) == enemy_ids
    assert out.trace_payload["render_map"]["show_enemy_labels"] is False
    enemy_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_type"] == "enemy_ship"
    ]
    assert len(enemy_entities) == 11
    assert all(entity["text_visible"] is False for entity in enemy_entities)
    assert all(entity["display_text"] is None for entity in enemy_entities)
    assert len(out.annotation_gt.value) == len(enemy_ids)
    assert enemy_projectile_counts_by_lane
    assert max(enemy_projectile_counts_by_lane.values()) >= 2
    assert all(1 <= int(count) <= 3 for count in enemy_projectile_counts_by_lane.values())
    for bbox in out.annotation_gt.value:
        assert float(bbox[2]) - float(bbox[0]) >= 24.0
        assert float(bbox[3]) - float(bbox[1]) >= 24.0


def test_games_space_shooter_enemy_ship_hit_count_matches_trace() -> None:
    out = GamesSpaceShooterEnemyShipHitCountTask().generate(
        88122,
        params={"target_answer": 6, "lane_count": 7, "enemy_count": 12},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    enemies_by_lane = {}
    for enemy in execution["enemies"]:
        enemies_by_lane.setdefault(int(enemy["lane"]), []).append(enemy)
    player_projectile_counts_by_lane = Counter(
        int(projectile["lane"])
        for projectile in execution["projectiles"]
        if str(projectile["owner"]) == "player"
    )
    expected_ids = []
    for lane in sorted(enemies_by_lane):
        player_count = int(player_projectile_counts_by_lane.get(int(lane), 0))
        lane_enemies = sorted(enemies_by_lane.get(int(lane), []), key=lambda enemy: int(enemy["y_slot"]), reverse=True)
        expected_ids.extend(str(enemy["enemy_id"]) for enemy in lane_enemies[: min(player_count, len(lane_enemies))])

    assert int(out.answer_gt.value) == len(expected_ids) == 6
    assert list(execution["annotation_entity_ids"]) == expected_ids
    assert out.trace_payload["render_map"]["show_enemy_labels"] is False
    assert max(player_projectile_counts_by_lane.values()) >= 2
    assert len(out.annotation_gt.value) == len(expected_ids)
    for entity_id, bbox in zip(expected_ids, out.annotation_gt.value):
        assert entity_id in out.trace_payload["render_map"]["enemy_bboxes_px"]
        assert float(bbox[2]) - float(bbox[0]) >= 24.0
        assert float(bbox[3]) - float(bbox[1]) >= 24.0


def test_games_space_shooter_enemy_ship_hit_count_supports_zero_answer() -> None:
    out = GamesSpaceShooterEnemyShipHitCountTask().generate(
        88123,
        params={"target_answer": 0, "lane_count": 5, "enemy_count": 8},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    player_lanes = {
        int(projectile["lane"])
        for projectile in execution["projectiles"]
        if str(projectile["owner"]) == "player"
    }
    enemy_lanes = {int(enemy["lane"]) for enemy in execution["enemies"]}

    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert list(execution["annotation_entity_ids"]) == []
    assert not (player_lanes & enemy_lanes)


def test_games_space_shooter_hit_enemy_ship_label_matches_trace() -> None:
    out = GamesSpaceShooterHitEnemyShipLabelTask().generate(
        88127,
        params={"correct_option_index": 2, "lane_count": 7, "enemy_count": 9},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]
    hit_ids = set(str(enemy_id) for enemy_id in execution["hit_enemy_ids"])
    candidate_ids = tuple(str(enemy_id) for enemy_id in execution["candidate_enemy_ids"])
    selected_id = str(execution["selected_enemy_id"])
    visible_label_ids = tuple(str(enemy_id) for enemy_id in render_map["visible_enemy_label_ids"])
    visible_enemy_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_type"] == "enemy_ship" and bool(entity["text_visible"])
    ]

    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) == "C"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "hit_enemy_ship_label"
    assert selected_id in hit_ids
    assert list(execution["annotation_entity_ids"]) == [selected_id]
    assert len(candidate_ids) == 4
    assert visible_label_ids == candidate_ids
    assert sorted(str(entity["display_text"]) for entity in visible_enemy_entities) == ["A", "B", "C", "D"]
    assert sum(1 for enemy_id in candidate_ids if enemy_id in hit_ids) == 1
    assert execution["candidate_labels"][selected_id] == "C"
    assert out.annotation_gt.value == render_map["enemy_bboxes_px"][selected_id]
    assert float(out.annotation_gt.value[2]) - float(out.annotation_gt.value[0]) >= 24.0
    assert float(out.annotation_gt.value[3]) - float(out.annotation_gt.value[1]) >= 24.0


def test_games_space_shooter_hit_label_constructs_four_options_for_review_cursor() -> None:
    """The canonical cursor remains feasible when balanced enemy_count is three."""

    seed = 4600227129277623
    params = {"query_id": "single", "_sample_cursor": 5}
    first = GamesSpaceShooterHitEnemyShipLabelTask().generate(seed, params=params, max_attempts=100)
    second = GamesSpaceShooterHitEnemyShipLabelTask().generate(seed, params=params, max_attempts=100)
    execution = first.trace_payload["execution_trace"]
    hit_ids = set(str(value) for value in execution["hit_enemy_ids"])
    candidate_ids = tuple(str(value) for value in execution["candidate_enemy_ids"])

    assert len(execution["enemies"]) >= 4
    assert int(execution["minimum_non_hit_enemy_count"]) == 3
    assert len(execution["enemies"]) - len(hit_ids) >= 3
    assert len(candidate_ids) == 4
    assert sum(enemy_id in hit_ids for enemy_id in candidate_ids) == 1
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload == second.trace_payload
    assert first.image.tobytes() == second.image.tobytes()


def test_games_space_shooter_hit_enemy_ship_label_not_tied_to_player_lane() -> None:
    same_lane_count = 0
    sample_count = 80
    for seed in range(88220, 88220 + sample_count):
        out = GamesSpaceShooterHitEnemyShipLabelTask().generate(
            seed,
            params={},
            max_attempts=256,
        )
        execution = out.trace_payload["execution_trace"]
        selected_id = str(execution["selected_enemy_id"])
        selected_enemy = next(enemy for enemy in execution["enemies"] if str(enemy["enemy_id"]) == selected_id)
        same_lane_count += int(int(selected_enemy["lane"]) == int(execution["player_lane"]))

    assert same_lane_count <= 30


def test_games_space_shooter_first_hit_enemy_ship_label_matches_trace() -> None:
    out = GamesSpaceShooterFirstHitEnemyShipLabelTask().generate(
        88131,
        params={"correct_option_index": 1, "lane_count": 7, "enemy_count": 10},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]
    candidate_enemy_ids = tuple(str(value) for value in execution["candidate_enemy_ids"])
    selected_id = str(execution["selected_enemy_id"])
    visible_enemy_ids = tuple(str(value) for value in render_map["visible_enemy_label_ids"])
    visible_enemy_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_type"] == "enemy_ship" and bool(entity["text_visible"])
    ]
    projectiles_by_id = {str(projectile["projectile_id"]): projectile for projectile in execution["projectiles"]}
    enemies_by_id = {str(enemy["enemy_id"]): enemy for enemy in execution["enemies"]}
    computed_distances = {}
    for enemy_id in candidate_enemy_ids:
        enemy = enemies_by_id[str(enemy_id)]
        projectile = projectiles_by_id[str(execution["hit_projectile_by_enemy"][str(enemy_id)])]
        assert int(projectile["lane"]) == int(enemy["lane"])
        assert int(projectile["y_slot"]) > int(enemy["y_slot"])
        computed_distances[str(enemy_id)] = int(projectile["y_slot"]) - int(enemy["y_slot"])

    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) == "B"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "first_hit_enemy_ship_label"
    assert len(candidate_enemy_ids) == 4
    assert visible_enemy_ids == candidate_enemy_ids
    assert sorted(str(entity["display_text"]) for entity in visible_enemy_entities) == ["A", "B", "C", "D"]
    assert list(execution["annotation_entity_ids"]) == [selected_id]
    assert execution["candidate_labels"][selected_id] == "B"
    assert out.annotation_gt.value == render_map["enemy_bboxes_px"][selected_id]
    assert computed_distances == {
        str(key): int(value)
        for key, value in execution["first_hit_distance_by_enemy"].items()
    }
    selected_distance = int(computed_distances[selected_id])
    assert selected_distance == min(computed_distances.values())
    assert sum(1 for distance in computed_distances.values() if int(distance) == selected_distance) == 1
    assert float(out.annotation_gt.value[2]) - float(out.annotation_gt.value[0]) >= 24.0
    assert float(out.annotation_gt.value[3]) - float(out.annotation_gt.value[1]) >= 24.0


def test_games_space_shooter_safe_lane_count_matches_trace() -> None:
    out = GamesSpaceShooterSafeLaneCountTask().generate(
        88140,
        params={"target_answer": 3, "lane_count": 6, "enemy_count": 10},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    enemy_projectile_counts_by_lane = Counter(
        int(projectile["lane"])
        for projectile in execution["projectiles"]
        if str(projectile["owner"]) == "enemy"
    )
    threatened_lanes = set(enemy_projectile_counts_by_lane)
    expected_ids = [
        f"lane_{int(lane)}"
        for lane in range(int(execution["lane_count"]))
        if int(lane) not in threatened_lanes
    ]

    assert int(out.answer_gt.value) == len(expected_ids) == 3
    assert list(execution["annotation_entity_ids"]) == expected_ids
    assert out.trace_payload["render_map"]["show_enemy_labels"] is False
    assert enemy_projectile_counts_by_lane
    assert max(enemy_projectile_counts_by_lane.values()) >= 2
    assert all(1 <= int(count) <= 3 for count in enemy_projectile_counts_by_lane.values())


def test_games_space_shooter_non_lane_entities_do_not_share_lane_slots() -> None:
    out = GamesSpaceShooterSafeLaneCountTask().generate(
        88150,
        params={"target_answer": 5, "lane_count": 8, "enemy_count": 16},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    occupied: list[tuple[int, int]] = []
    for key in ("enemies", "projectiles"):
        occupied.extend((int(row["lane"]), int(row["y_slot"])) for row in execution[key])

    assert len(occupied) == len(set(occupied))
    assert "blockers" not in execution


def test_games_space_shooter_ships_and_projectiles_are_centered_on_lane_pads() -> None:
    out = GamesSpaceShooterSafeLaneCountTask().generate(
        88155,
        params={"target_answer": 3, "lane_count": 8, "enemy_count": 14},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]
    lane_bboxes = render_map["lane_bboxes_px"]
    enemy_bboxes = render_map["enemy_bboxes_px"]
    projectile_bboxes = render_map["projectile_bboxes_px"]
    for enemy in execution["enemies"]:
        enemy_bbox = enemy_bboxes[str(enemy["enemy_id"])]
        lane_bbox = lane_bboxes[f"lane_{int(enemy['lane'])}"]
        enemy_cx = 0.5 * (float(enemy_bbox[0]) + float(enemy_bbox[2]))
        lane_cx = 0.5 * (float(lane_bbox[0]) + float(lane_bbox[2]))
        assert abs(enemy_cx - lane_cx) <= 0.75
    for projectile in execution["projectiles"]:
        projectile_bbox = projectile_bboxes[str(projectile["projectile_id"])]
        lane_bbox = lane_bboxes[f"lane_{int(projectile['lane'])}"]
        projectile_cx = 0.5 * (float(projectile_bbox[0]) + float(projectile_bbox[2]))
        lane_cx = 0.5 * (float(lane_bbox[0]) + float(lane_bbox[2]))
        assert abs(projectile_cx - lane_cx) <= 0.75


@pytest.mark.parametrize(
    ("task_cls", "params"),
    (
        (GamesSpaceShooterEnemyShipCountTask, {"lane_count": 8, "enemy_count": 12}),
        (GamesSpaceShooterEnemyShipHitCountTask, {"target_answer": 4, "lane_count": 8, "enemy_count": 12}),
        (GamesSpaceShooterFirstHitEnemyShipLabelTask, {"lane_count": 8, "enemy_count": 12}),
        (GamesSpaceShooterHitEnemyShipLabelTask, {"lane_count": 8, "enemy_count": 12}),
        (GamesSpaceShooterSafeLaneCountTask, {"target_answer": 3, "lane_count": 7, "enemy_count": 12}),
    ),
)
def test_games_space_shooter_enemy_projectiles_have_visible_same_lane_shooter(task_cls, params) -> None:
    out = task_cls().generate(88165, params=params, max_attempts=256)
    execution = out.trace_payload["execution_trace"]
    enemies = tuple(execution["enemies"])
    for projectile in execution["projectiles"]:
        if str(projectile["owner"]) != "enemy":
            continue
        assert any(
            int(enemy["lane"]) == int(projectile["lane"])
            and int(enemy["y_slot"]) < int(projectile["y_slot"])
            for enemy in enemies
        )


@pytest.mark.parametrize(
    ("task_cls", "params"),
    (
        (GamesSpaceShooterEnemyShipCountTask, {"lane_count": 8, "enemy_count": 12}),
        (GamesSpaceShooterSafeLaneCountTask, {"target_answer": 3, "lane_count": 7, "enemy_count": 12}),
    ),
)
def test_games_space_shooter_projectiles_follow_gameplay_ordering(task_cls, params) -> None:
    for seed in range(88180, 88190):
        out = task_cls().generate(seed, params=params, max_attempts=256)
        execution = out.trace_payload["execution_trace"]
        enemies = tuple(execution["enemies"])
        enemy_projectiles = tuple(
            projectile for projectile in execution["projectiles"] if str(projectile["owner"]) == "enemy"
        )
        player_projectiles = tuple(
            projectile for projectile in execution["projectiles"] if str(projectile["owner"]) == "player"
        )

        for projectile in enemy_projectiles:
            assert int(projectile["y_slot"]) <= 4
        for projectile in player_projectiles:
            assert int(projectile["y_slot"]) == 5
            same_lane_enemies = [
                enemy for enemy in enemies if int(enemy["lane"]) == int(projectile["lane"])
            ]
            assert same_lane_enemies
            assert all(int(enemy["y_slot"]) < int(projectile["y_slot"]) for enemy in same_lane_enemies)
            assert all(
                int(enemy_projectile["y_slot"]) < int(projectile["y_slot"])
                for enemy_projectile in enemy_projectiles
                if int(enemy_projectile["lane"]) == int(projectile["lane"])
            )


def test_games_space_shooter_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__space_shooter"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__space_shooter",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__space_shooter__enemy_ship_count", count=1, params={"enemy_count": 7}),
            BuildTaskConfig(task_id="task_games__space_shooter__enemy_ship_hit_count", count=1, params={"target_answer": 3}),
            BuildTaskConfig(task_id="task_games__space_shooter__first_hit_enemy_ship_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__space_shooter__hit_enemy_ship_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__space_shooter__safe_lane_count", count=1, params={"target_answer": 4}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-space-shooter-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 5
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "space_shooter" for row in rows)
