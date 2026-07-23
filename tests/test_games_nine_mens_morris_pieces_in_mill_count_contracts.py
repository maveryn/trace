"""Contract tests for the games Nine Men's Morris tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.nine_mens_morris.mill_completion_point_count import (
    GamesNineMensMorrisMillCompletionPointCountTask,
)
from trace_tasks.tasks.games.nine_mens_morris.pieces_in_mill_count import (
    GamesNineMensMorrisAllPiecesInMillCountTask,
)
from trace_tasks.tasks.games.nine_mens_morris.shared.state import MILL_POSITION_INDICES, POSITION_LAYOUT
from tests.helpers import read_jsonl


def _completion_node_labels(execution: dict, *, color: str) -> tuple[str, ...]:
    occupancy = {
        int(spec["node_index"]): str(spec["color"])
        for spec in execution["piece_specs"]
    }
    labels: list[str] = []
    for node_index, (node_label, _x_frac, _y_frac) in enumerate(POSITION_LAYOUT):
        if int(node_index) in occupancy:
            continue
        for mill in MILL_POSITION_INDICES:
            if int(node_index) not in mill:
                continue
            other_positions = [int(position) for position in mill if int(position) != int(node_index)]
            if all(str(occupancy.get(int(position), "")) == str(color) for position in other_positions):
                labels.append(str(node_label))
                break
    return tuple(labels)


def test_games_nine_mens_morris_pieces_in_mill_count_emits_expected_contract() -> None:
    out = GamesNineMensMorrisAllPiecesInMillCountTask().generate(
        29101,
        params={"target_answer": 9},
        max_attempts=64,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 9
    assert out.annotation_gt.type == "point_set"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_query_key"] == "all_pieces_in_mill_count"
    assert int(execution["target_answer"]) == 9
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert "panel_scene_style" in trace["render_spec"]
    assert "text_style" in trace["render_spec"]
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value) == 9
    assert len(execution["all_piece_ids_in_mill"]) == 9


@pytest.mark.parametrize(
    ("query_id", "target_answer", "color"),
    (
        ("white_mill_completion_point_count", 5, "white"),
        ("black_mill_completion_point_count", 4, "black"),
    ),
)
def test_games_nine_mens_morris_mill_completion_point_count_matches_trace(
    query_id: str,
    target_answer: int,
    color: str,
) -> None:
    out = GamesNineMensMorrisMillCompletionPointCountTask().generate(
        29141 + int(target_answer),
        params={
            "query_id": query_id,
            "target_answer": target_answer,
        },
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    expected_labels = _completion_node_labels(execution, color=color)
    expected_points = [list(trace["render_map"]["node_centers_px"][str(label)]) for label in expected_labels]

    assert out.query_id == query_id
    assert execution["player_color"] == color
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == len(expected_labels) == int(target_answer)
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == expected_points
    assert execution["annotation_map_key"] == "node_centers_px"
    assert tuple(execution["annotation_entity_ids"]) == expected_labels
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value


def test_games_nine_mens_morris_mill_completion_zero_answer_emits_empty_annotation() -> None:
    out = GamesNineMensMorrisMillCompletionPointCountTask().generate(
        29151,
        params={
            "query_id": "white_mill_completion_point_count",
            "target_answer": 0,
        },
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]

    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == []
    assert execution["annotation_entity_ids"] == []
    assert execution["white_mill_completion_node_labels"] == []


def test_games_nine_mens_morris_taxonomy() -> None:
    assert resolve_task_taxonomy("task_games__nine_mens_morris__pieces_in_mill_count").scene_id == "nine_mens_morris"
    assert resolve_task_taxonomy("task_games__nine_mens_morris__mill_completion_point_count").scene_id == "nine_mens_morris"


@pytest.mark.parametrize(
    ("task_cls", "query_id", "support"),
    (
        (
            GamesNineMensMorrisAllPiecesInMillCountTask,
            "single",
            (0, 3, 5, 6, 7, 8, 9),
        ),
        (
            GamesNineMensMorrisMillCompletionPointCountTask,
            "white_mill_completion_point_count",
            (0, 1, 2, 3, 4, 5),
        ),
        (
            GamesNineMensMorrisMillCompletionPointCountTask,
            "black_mill_completion_point_count",
            (0, 1, 2, 3, 4, 5),
        ),
    ),
)
def test_games_nine_mens_morris_public_tasks_cover_supported_targets(
    task_cls,
    query_id: str,
    support: tuple[int, ...],
) -> None:
    task = task_cls()
    for target_answer in support:
        out = task.generate(
            29301 + (17 * int(target_answer)),
            params={
                "query_id": str(query_id),
                "target_answer": int(target_answer),
            },
            max_attempts=256,
        )
        assert str(out.query_id) == str(query_id)
        assert int(out.answer_gt.value) == int(target_answer)


def test_games_nine_mens_morris_pieces_in_mill_count_zero_answer_emits_empty_annotation() -> None:
    out = GamesNineMensMorrisAllPiecesInMillCountTask().generate(
        29111,
        params={
            "target_answer": 0,
        },
        max_attempts=64,
    )
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert out.trace_payload["execution_trace"]["annotation_entity_ids"] == []


def test_games_nine_mens_morris_pieces_in_mill_count_rejects_legacy_branch_param() -> None:
    with pytest.raises(ValueError, match="unsupported query_id"):
        GamesNineMensMorrisAllPiecesInMillCountTask().generate(
            29112,
            params={"query_id": "all_pieces_in_mill_count"},
            max_attempts=64,
        )


def test_games_nine_mens_morris_pieces_in_mill_count_is_deterministic() -> None:
    params = {
        "target_answer": 9,
    }
    task = GamesNineMensMorrisAllPiecesInMillCountTask()
    out_a = task.generate(29131, params=params, max_attempts=64)
    out_b = task.generate(29131, params=params, max_attempts=64)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_nine_mens_morris_prompt_bundle_requires_rule_text() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/nine_mens_morris/games_nine_mens_morris_v1.json").read_text(encoding="utf-8"))
    required = bundle["required_slots_by_key"]
    assert required["query:all_pieces_in_mill_count"] == ["mill_rule_text"]
    assert required["query:white_mill_completion_point_count"] == ["mill_rule_text"]
    assert required["query:black_mill_completion_point_count"] == ["mill_rule_text"]


def test_games_nine_mens_morris_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__nine_mens_morris__pieces_in_mill_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__nine_mens_morris__pieces_in_mill_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__nine_mens_morris__pieces_in_mill_count",
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_games__nine_mens_morris__mill_completion_point_count",
                count=2,
                params={},
            ),
        ],
        strict_repro=False,
        max_attempts_per_instance=64,
        sampling_seed=81,
    )
    final_path = build_dataset(config, code_hash="games-nine-mens-morris-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 6
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "nine_mens_morris" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_games__nine_mens_morris__pieces_in_mill_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
