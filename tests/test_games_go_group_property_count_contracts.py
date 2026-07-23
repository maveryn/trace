"""Contract tests for games Go group-count tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.go.group_adjacent_enemy_count import GamesGoGroupAdjacentEnemyCountTask
from trace_tasks.tasks.games.go.group_liberty_count import GamesGoGroupLibertyCountTask
from trace_tasks.tasks.games.go.marked_group_stone_count import GamesGoMarkedGroupStoneCountTask
from trace_tasks.tasks.games.go.shared.rendering import GO_MARKED_GROUP_RED_RGB
from trace_tasks.tasks.games.go.shared.rules import connected_group
from trace_tasks.tasks.games.shared.style import SUPPORTED_GO_STYLE_VARIANTS
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("query_id", "player_color", "target_answer", "annotation_coord_key"),
    (
        ("marked_group_liberty_count", "black", 2, "liberty_coords"),
        ("marked_group_liberty_count", "white", 6, "liberty_coords"),
        ("marked_group_shared_liberty_count", "black", 4, "shared_liberty_coords"),
    ),
)
def test_games_go_group_liberty_count_emits_expected_contract(
    query_id: str,
    player_color: str,
    target_answer: int,
    annotation_coord_key: str,
) -> None:
    out = GamesGoGroupLibertyCountTask().generate(
        34101,
        params={"query_id": query_id, "player_color": player_color, "target_answer": target_answer},
        max_attempts=192,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == query_id
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(target_answer)
    assert out.annotation_gt.type == "point_set"
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert trace["prompt_metadata"]["bundle_id"] == "games_go_v1"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert trace["render_spec"]["panel_scene_style"]
    assert trace["render_map"]["panel_scene_style"]
    marker_style = trace["render_map"]["marked_group_marker_style"]
    assert marker_style["role"] == "go_marked_stone_group"
    assert marker_style["inner_rgb"] == list(GO_MARKED_GROUP_RED_RGB)
    assert marker_style["passes"] is True
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value) == int(target_answer)
    assert len(execution[annotation_coord_key]) == int(target_answer)
    assert str(execution["marked_group_color"]) == str(execution["player_color"])
    assert "red outlined" in out.prompt.lower()


def test_games_go_group_adjacent_enemy_count_emits_expected_contract() -> None:
    out = GamesGoGroupAdjacentEnemyCountTask().generate(
        34121,
        params={"query_id": "single", "player_color": "white", "target_answer": 2},
        max_attempts=192,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 2
    assert out.annotation_gt.type == "point_set"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert len(execution["annotation_entity_ids"]) == 2
    assert len(execution["adjacent_enemy_coords"]) == 2
    assert str(execution["marked_group_color"]) == "white"
    assert "opponent stones" in out.prompt.lower()


def test_games_go_group_liberty_count_query_cycle_covers_answer_and_scene_support() -> None:
    task = GamesGoGroupLibertyCountTask()
    answers_by_query: dict[str, set[int]] = {
        "marked_group_liberty_count": set(),
        "marked_group_shared_liberty_count": set(),
    }
    scenes_by_query: dict[str, set[str]] = {key: set() for key in answers_by_query}
    colors_by_query: dict[str, set[str]] = {key: set() for key in answers_by_query}
    styles_by_query: dict[str, set[str]] = {key: set() for key in answers_by_query}
    for sampling_index in range(96):
        out = task.generate(
            34201 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=192,
        )
        execution = out.trace_payload["execution_trace"]
        query_id = str(out.query_id)
        answers_by_query[query_id].add(int(out.answer_gt.value))
        scenes_by_query[query_id].add(str(execution["scene_variant"]))
        colors_by_query[query_id].add(str(execution["player_color"]))
        styles_by_query[query_id].add(str(execution["style_variant"]))

    assert answers_by_query == {
        "marked_group_liberty_count": {1, 2, 3, 4, 6},
        "marked_group_shared_liberty_count": {1, 2, 3, 4, 5},
    }
    assert all(values == {"crowded_board", "open_board"} for values in scenes_by_query.values())
    assert all(values == {"black", "white"} for values in colors_by_query.values())
    assert all(values == set(SUPPORTED_GO_STYLE_VARIANTS) for values in styles_by_query.values())


def test_games_go_group_adjacent_enemy_count_query_cycle_covers_answer_support() -> None:
    task = GamesGoGroupAdjacentEnemyCountTask()
    answers: set[int] = set()
    scenes: set[str] = set()
    colors: set[str] = set()
    styles: set[str] = set()
    for sampling_index in range(72):
        out = task.generate(
            34281 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=192,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        answers.add(int(out.answer_gt.value))
        scenes.add(str(execution["scene_variant"]))
        colors.add(str(execution["player_color"]))
        styles.add(str(execution["style_variant"]))

    assert answers == {1, 2, 3, 4, 5, 6}
    assert scenes == {"crowded_board", "open_board"}
    assert colors == {"black", "white"}
    assert styles == set(SUPPORTED_GO_STYLE_VARIANTS)


def test_games_go_group_liberty_count_is_deterministic() -> None:
    params = {
        "query_id": "marked_group_shared_liberty_count",
        "player_color": "white",
        "target_answer": 4,
    }
    task = GamesGoGroupLibertyCountTask()
    out_a = task.generate(34111, params=params, max_attempts=192)
    out_b = task.generate(34111, params=params, max_attempts=192)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_go_prompt_bundle_requires_current_rule_texts() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/go/games_go_v1.json").read_text(encoding="utf-8"))
    required = bundle["required_slots_by_key"]
    assert required["query:marked_group_liberty_count"] == [
        "marked_group_rule_text",
        "group_rule_text",
        "liberty_rule_text",
        "player_color",
    ]
    assert required["query:marked_group_adjacent_enemy_count"] == [
        "marked_group_rule_text",
        "group_rule_text",
        "adjacent_enemy_rule_text",
        "player_color",
    ]
    assert required["query:marked_group_shared_liberty_count"] == [
        "marked_group_rule_text",
        "group_rule_text",
        "liberty_rule_text",
        "shared_liberty_rule_text",
        "player_color",
    ]
    assert required["query:marked_group_stone_count"] == [
        "marked_stone_rule_text",
        "group_rule_text",
        "player_color",
    ]


@pytest.mark.parametrize(("player_color", "target_answer"), (("black", 2), ("white", 4), ("black", 6)))
def test_games_go_marked_group_stone_count_emits_expected_contract(
    player_color: str,
    target_answer: int,
) -> None:
    out = GamesGoMarkedGroupStoneCountTask().generate(
        34301,
        params={"query_id": "single", "player_color": player_color, "target_answer": target_answer},
        max_attempts=192,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board_size = int(execution["board_size"])
    rows = [[0 for _ in range(board_size)] for _ in range(board_size)]
    for spec in execution["stone_specs"]:
        rows[int(spec["row"])][int(spec["col"])] = 1 if str(spec["color"]) == "black" else -1

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(target_answer)
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == int(target_answer)
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    marker_style = trace["render_map"]["marked_group_marker_style"]
    assert marker_style["role"] == "go_marked_stone_group"
    assert marker_style["inner_rgb"] == list(GO_MARKED_GROUP_RED_RGB)
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_query_key"] == "marked_group_stone_count"
    assert execution["query_id"] == "single"
    assert str(execution["marked_group_color"]) == str(player_color)
    assert len(execution["marked_group_coords"]) == int(target_answer)
    assert len(execution["annotation_entity_ids"]) == int(target_answer)
    marked_entities = [spec for spec in trace["scene_ir"]["entities"] if spec.get("is_marked_group")]
    assert len(marked_entities) == 1
    assert str(execution["marked_reference_point_id"]) in {
        str(spec["entity_id"]) for spec in marked_entities
    }
    marked_reference = tuple(
        (int(spec["row"]), int(spec["col"]))
        for spec in execution["stone_specs"]
        if str(spec["point_id"]) == str(execution["marked_reference_point_id"])
    )[0]
    assert connected_group(rows, marked_reference) == tuple(tuple(coord) for coord in execution["marked_group_coords"])
    assert set(execution["annotation_entity_ids"]) == set(execution["marked_group_point_ids"])
    assert "marked" in out.prompt.lower()
    assert "stones are in" in out.prompt.lower() or "count the stones" in out.prompt.lower()


def test_games_go_marked_group_stone_count_query_cycle_covers_answer_and_scene_support() -> None:
    task = GamesGoMarkedGroupStoneCountTask()
    answers: set[int] = set()
    colors: set[str] = set()
    scenes: set[str] = set()
    styles: set[str] = set()
    for sampling_index in range(96):
        out = task.generate(
            34321 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=192,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        answers.add(int(out.answer_gt.value))
        colors.add(str(execution["player_color"]))
        scenes.add(str(execution["scene_variant"]))
        styles.add(str(execution["style_variant"]))

    assert answers == {2, 3, 4, 5, 6}
    assert colors == {"black", "white"}
    assert scenes == {"crowded_board", "open_board"}
    assert styles == set(SUPPORTED_GO_STYLE_VARIANTS)


def test_games_go_public_taxonomy() -> None:
    for task_id in (
        "task_games__go__group_liberty_count",
        "task_games__go__group_adjacent_enemy_count",
        "task_games__go__marked_group_stone_count",
    ):
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "go"


def test_games_go_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "go_build_smoke"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_games_go",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__go__group_liberty_count", count=2, params={}),
            BuildTaskConfig(task_id="task_games__go__group_adjacent_enemy_count", count=2, params={}),
            BuildTaskConfig(task_id="task_games__go__marked_group_stone_count", count=2, params={}),
        ],
        strict_repro=False,
        max_attempts_per_instance=192,
        sampling_seed=92,
    )
    final_path = build_dataset(config, code_hash="games-go-scene-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 6
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "go" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_games__go__group_liberty_count"]) == 2
    assert int(build_report["accepted_counts_by_task"]["task_games__go__group_adjacent_enemy_count"]) == 2
    assert int(build_report["accepted_counts_by_task"]["task_games__go__marked_group_stone_count"]) == 2

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
