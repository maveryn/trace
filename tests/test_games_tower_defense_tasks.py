"""Contract tests for games tower-defense tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


COVERED_PATH_TASK_ID = "task_games__tower_defense__covered_path_segment_count"
BEST_POSITION_TASK_ID = "task_games__tower_defense__best_tower_position_label"
NEAREST_EXIT_TASK_ID = "task_games__tower_defense__nearest_exit_enemy_label"


def _distance(point_a: list[float], point_b: list[float]) -> float:
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def test_games_tower_defense_defaults_expose_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "tower_defense")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert set(generation["scene_variant_weights"].keys()) == {"winding_path", "switchback_path"}
    assert "query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == {
        "grass_field",
        "desert_path",
        "blueprint_grid",
        "night_ops",
        "paper_map",
    }
    assert "target_answer_support" not in generation
    assert list(generation["covered_path_target_answer_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(generation["best_position_target_answer_support"]) == [2]
    assert list(generation["nearest_exit_option_count_support"]) == [6]
    assert list(generation["best_position_answer_option_index_support"]) == [0, 1, 2, 3]
    assert list(generation["nearest_exit_answer_option_index_support"]) == [0, 1, 2, 3, 4, 5]
    assert "tower_count_support" not in generation
    assert list(generation["covered_path_tower_count_support"]) == [3, 4, 5, 6]
    assert list(generation["best_position_candidate_count_support"]) == [4]
    assert list(generation["nearest_exit_tower_count_support"]) == [1, 2]
    assert int(rendering["map_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_tower_defense_v1"
    assert str(prompt["scene_key"]) == "tower_defense_map"
    assert str(prompt["task_key"]) == "tower_defense_query"


def test_games_tower_defense_prompt_bundle_has_coverage_queries() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/tower_defense/games_tower_defense_v1.json").read_text(encoding="utf-8"))
    assert set(bundle["templates"]["query"].keys()) == {
        "covered_path_segment_count",
        "best_tower_position_label",
        "nearest_exit_enemy_label",
    }
    assert bundle["required_slots_by_key"]["query:covered_path_segment_count"] == [
        "coverage_rule_text",
    ]
    assert bundle["required_slots_by_key"]["query:best_tower_position_label"] == [
        "coverage_rule_text",
    ]
    assert bundle["required_slots_by_key"]["query:nearest_exit_enemy_label"] == [
        "coverage_rule_text",
    ]


def test_games_tower_defense_covered_path_answer_matches_distance_trace() -> None:
    out = create_task(COVERED_PATH_TASK_ID).generate(
        92043,
        params={"target_answer": 6},
        max_attempts=500,
    )
    execution = out.trace_payload["execution_trace"]
    covered_ids = []
    for index, point in enumerate(execution["path_points_px_local"]):
        if any(_distance(tower["center_px_local"], point) <= float(tower["range_radius_px"]) + 1e-6 for tower in execution["towers"]):
            covered_ids.append(f"path_segment_{index:02d}")

    assert out.scene_id == "tower_defense"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "covered_path_segment_count"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 6
    assert covered_ids == list(execution["annotation_entity_ids"])
    assert len(out.annotation_gt.value) == 6
    assert out.annotation_gt.type == "point_set"
    assert out.trace_payload["projected_annotation"]["type"] == "point_set"
    assert execution["marked_enemy"] is None
    assert out.trace_payload["render_map"]["marked_enemy_id"] is None


def test_games_tower_defense_covered_path_supports_single_answer() -> None:
    out = create_task(COVERED_PATH_TASK_ID).generate(
        92044,
        params={"target_answer": 1},
        max_attempts=500,
    )

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 1
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == 1


def test_games_tower_defense_best_position_label_is_unique_maximum() -> None:
    out = create_task(BEST_POSITION_TASK_ID).generate(
        93107,
        params={"target_answer": 2, "answer_option_index": 2},
        max_attempts=800,
    )
    execution = out.trace_payload["execution_trace"]
    path_points = execution["path_points_px_local"]
    candidate_counts = {}
    candidate_radii = []
    for tower in execution["towers"]:
        tower_id = str(tower["tower_id"])
        if not tower_id.startswith("candidate_"):
            continue
        candidate_radii.append(round(float(tower["range_radius_px"]), 3))
        label = tower_id.removeprefix("candidate_")
        candidate_counts[label] = sum(
            1
            for point in path_points
            if _distance(tower["center_px_local"], point) <= float(tower["range_radius_px"]) + 1e-6
        )

    best_count = max(candidate_counts.values())
    best_labels = [label for label, count in candidate_counts.items() if int(count) == int(best_count)]

    assert out.scene_id == "tower_defense"
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert best_labels == ["C"]
    assert int(best_count) == 2
    assert len(set(candidate_radii)) == 1
    assert execution["sample_metadata"]["candidate_coverage_counts"] == candidate_counts
    assert execution["annotation_entity_ids"] == ["candidate_C"]
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["entity_points_px"]["candidate_C"]


def test_games_tower_defense_nearest_exit_enemy_label_follows_path_order() -> None:
    out = create_task(NEAREST_EXIT_TASK_ID).generate(
        94511,
        params={"answer_option_index": 5},
        max_attempts=500,
    )
    execution = out.trace_payload["execution_trace"]
    options = execution["labeled_path_enemy_options"]
    index_by_label = {str(option["label"]): int(option["path_index"]) for option in options}
    expected_label = max(index_by_label, key=lambda label: index_by_label[str(label)])
    expected_entity = f"path_segment_{index_by_label[str(expected_label)]:02d}"

    assert out.scene_id == "tower_defense"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "nearest_exit_enemy_label"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "F"
    assert expected_label == "F"
    assert sorted(index_by_label) == ["A", "B", "C", "D", "E", "F"]
    assert max(index_by_label.values()) < int(execution["exit_path_index"])
    assert execution["annotation_entity_ids"] == [expected_entity]
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["entity_points_px"][expected_entity]
    assert out.trace_payload["render_map"]["exit_marker_bbox_px"]


def test_games_tower_defense_taxonomy_mapping() -> None:
    path_taxonomy = resolve_task_taxonomy(COVERED_PATH_TASK_ID)
    assert path_taxonomy.domain == "games"
    assert path_taxonomy.scene_id == "tower_defense"
    assert path_taxonomy.source_domain == "games"

    best_position_taxonomy = resolve_task_taxonomy(BEST_POSITION_TASK_ID)
    assert best_position_taxonomy.domain == "games"
    assert best_position_taxonomy.scene_id == "tower_defense"
    assert best_position_taxonomy.source_domain == "games"

    nearest_exit_taxonomy = resolve_task_taxonomy(NEAREST_EXIT_TASK_ID)
    assert nearest_exit_taxonomy.domain == "games"
    assert nearest_exit_taxonomy.scene_id == "tower_defense"
    assert nearest_exit_taxonomy.source_domain == "games"
