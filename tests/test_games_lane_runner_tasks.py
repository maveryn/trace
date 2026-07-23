"""Contract tests for games lane-runner tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.lane_runner.shared.rules import (
    path_coin_collection,
    path_hits_hazard,
    path_option_entity_id,
)
from trace_tasks.tasks.games.lane_runner.shared.state import LaneRunnerCoin, LaneRunnerHazard
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


PATH_COIN_TASK_ID = "task_games__lane_runner__path_coin_count"
SAFE_PATH_TASK_ID = "task_games__lane_runner__safe_path_label"


def test_games_lane_runner_defaults_expose_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "lane_runner")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id=PATH_COIN_TASK_ID,
    )

    assert set(generation["scene_variant_weights"].keys()) == {"two_lane_track"}
    assert "query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == {
        "arcade_lane",
        "city_road",
        "forest_path",
        "neon_track",
        "paper_course",
    }
    assert list(generation["row_count_support"]) == [5, 6, 7, 8]
    assert list(generation["start_lane_support"]) == [0, 1]
    assert list(generation["target_answer_support"]) == [1, 2, 3, 4, 5, 6]
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert float(rendering["unit_size_scale_min"]) == 0.5
    assert float(rendering["unit_size_scale_max"]) == 1.0
    assert bool(rendering["layout_jitter_enabled"]) is True
    assert str(prompt["bundle_id"]) == "games_lane_runner_v1"

    safe_generation, _safe_rendering, safe_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id=SAFE_PATH_TASK_ID,
    )
    assert "query_id_weights" not in safe_generation
    assert list(safe_generation["option_count_support"]) == [4, 6]
    assert list(safe_generation["answer_option_index_support"]) == [0, 1, 2, 3, 4, 5]
    assert str(safe_prompt["bundle_id"]) == "games_lane_runner_v1"


def test_games_lane_runner_prompt_bundle_has_active_queries() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/lane_runner/games_lane_runner_v1.json").read_text(encoding="utf-8"))

    assert bundle["schema_version"] == "v1"
    assert set(bundle["templates"]["query"].keys()) == {"path_coin_count", "safe_path_label"}
    assert bundle["required_slots_by_key"]["query:path_coin_count"] == ["lane_runner_path_rule_text"]
    assert bundle["required_slots_by_key"]["query:safe_path_label"] == ["safe_path_rule_text"]
    assert len(bundle["templates"]["query"]["path_coin_count"]) == 5
    assert len(bundle["templates"]["query"]["safe_path_label"]) == 5
    assert "hazard cell" in str(bundle["code_prompt_defaults"]["safe_path_rule_text"])


def test_games_lane_runner_taxonomy_and_direct_registry() -> None:
    path_taxonomy = resolve_task_taxonomy(PATH_COIN_TASK_ID)
    assert path_taxonomy.domain == "games"
    assert path_taxonomy.scene_id == "lane_runner"
    assert path_taxonomy.source_domain == "games"
    assert create_task(PATH_COIN_TASK_ID).task_id == PATH_COIN_TASK_ID

    safe_taxonomy = resolve_task_taxonomy(SAFE_PATH_TASK_ID)
    assert safe_taxonomy.domain == "games"
    assert safe_taxonomy.scene_id == "lane_runner"
    assert safe_taxonomy.source_domain == "games"
    assert create_task(SAFE_PATH_TASK_ID).task_id == SAFE_PATH_TASK_ID


def test_games_lane_runner_tasks_reject_unsupported_public_query_id() -> None:
    with pytest.raises(ValueError, match="query_id"):
        create_task(PATH_COIN_TASK_ID).generate(
            1,
            params={"query_id": "path_coin_count"},
            max_attempts=10,
        )
    with pytest.raises(ValueError, match="query_id"):
        create_task(SAFE_PATH_TASK_ID).generate(
            1,
            params={"query_variant": "safe_path_label"},
            max_attempts=10,
        )


def test_games_lane_runner_path_coin_answer_matches_shown_path() -> None:
    out = create_task(PATH_COIN_TASK_ID).generate(
        92342,
        params={"row_count": 6, "target_answer": 3, "start_lane": 1, "query_id": "single"},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    coins = tuple(
        LaneRunnerCoin(
            coin_id=str(coin["coin_id"]),
            row=int(coin["row"]),
            lane=int(coin["lane"]),
        )
        for coin in execution["coins"]
    )
    answer, annotation_ids = path_coin_collection(
        coins=coins,
        shown_path_lanes=execution["shown_path_lanes"],
        row_count=int(execution["row_count"]),
        lane_count=int(execution["lane_count"]),
        start_lane=int(execution["start_lane"]),
    )
    coin_cells = {(int(coin.row), int(coin.lane)) for coin in coins}
    has_parallel_row = any(
        (row, int(lane)) in coin_cells and (row, 1 - int(lane)) in coin_cells
        for row, lane in enumerate(execution["shown_path_lanes"])
    )

    assert out.scene_id == "lane_runner"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "path_coin_count"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(answer) == 3
    assert len(coins) > int(answer)
    assert has_parallel_row is True
    assert execution["annotation_entity_ids"] == list(annotation_ids)
    assert execution["construction_mode"] == "sample_shown_path_with_parallel_coin_distractors"
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == int(answer)
    assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert out.trace_payload["render_map"]["shown_path_px"]["lanes_by_row"] == execution["shown_path_lanes"]
    for entity_id, point in zip(annotation_ids, out.annotation_gt.value):
        assert out.trace_payload["render_map"]["entity_points_px"][entity_id] == point


def test_games_lane_runner_safe_path_answer_matches_trace() -> None:
    out = create_task(SAFE_PATH_TASK_ID).generate(
        92741,
        params={"row_count": 6, "option_count": 6, "answer_option_index": 2, "start_lane": 0, "query_id": "single"},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    hazards = tuple(
        LaneRunnerHazard(
            hazard_id=str(hazard["hazard_id"]),
            row=int(hazard["row"]),
            lane=int(hazard["lane"]),
        )
        for hazard in execution["hazards"]
    )
    safe_labels = [
        str(option["label"])
        for option in execution["path_options"]
        if not path_hits_hazard(lanes_by_row=option["lanes_by_row"], hazards=hazards)
    ]
    answer = str(execution["answer"])
    answer_entity_id = path_option_entity_id(answer)

    assert out.scene_id == "lane_runner"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "safe_path_label"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert safe_labels == [answer] == ["C"]
    assert execution["answer_entity_id"] == answer_entity_id
    assert execution["annotation_entity_ids"] == [answer_entity_id]
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["projected_annotation"]["type"] == "bbox"
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.trace_payload["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert out.trace_payload["render_map"]["path_options_px"][answer]["card_bbox_px"] == out.annotation_gt.value
    assert out.trace_payload["render_map"]["show_board"] is False
    assert out.trace_payload["render_map"]["cell_bboxes_px"] == {}
    assert out.trace_payload["render_map"]["hazard_bboxes_px"] == {}
