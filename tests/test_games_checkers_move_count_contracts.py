"""Contract tests for the games Checkers move-count task."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.checkers.max_capture_chain_length import (
    GamesCheckersMaxCaptureChainLengthTask,
)
from trace_tasks.tasks.games.checkers.move_count import GamesCheckersMoveCountPublicTask
from trace_tasks.tasks.games.checkers.piece_mobility_count import (
    GamesCheckersPieceMobilityCountTask,
)
from trace_tasks.tasks.games.checkers.piece_state_count import (
    GamesCheckersPieceStateCountTask,
)
from trace_tasks.tasks.games.checkers.shared.rules import (
    BLACK,
    BOARD_SIZE,
    RED,
    piece_to_entity_id,
    playable_coords,
)
from tests.helpers import read_jsonl


def _assert_bbox_annotation_matches_piece_ids(
    trace: dict, annotation: list[list[float]]
) -> None:
    execution = trace["execution_trace"]
    expected = [
        trace["render_map"]["piece_bboxes_px"][str(entity_id)]
        for entity_id in execution["annotation_entity_ids"]
    ]
    assert annotation == expected
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_set"] == annotation


def _assert_bbox_annotation_matches_entity_ids(
    trace: dict, annotation: list[list[float]]
) -> None:
    """Verify bbox annotations use cell boxes for cells and piece boxes for pieces."""

    execution = trace["execution_trace"]
    expected: list[list[float]] = []
    for entity_id in execution["annotation_entity_ids"]:
        entity_id = str(entity_id)
        if entity_id.startswith("piece_"):
            expected.append(trace["render_map"]["piece_bboxes_px"][entity_id])
        else:
            expected.append(trace["render_map"]["cell_bboxes_px"][entity_id])
    assert annotation == expected
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_set"] == annotation


def _expected_piece_state_coords(
    board_rows: list[list[int]], *, target: int, edge_only: bool
) -> set[tuple[int, int]]:
    expected: set[tuple[int, int]] = set()
    for row, col in playable_coords():
        if int(board_rows[int(row)][int(col)]) != int(target):
            continue
        if (
            edge_only
            and int(row) not in {0, BOARD_SIZE - 1}
            and int(col) not in {0, BOARD_SIZE - 1}
        ):
            continue
        expected.add((int(row), int(col)))
    return expected


@pytest.mark.parametrize(
    ("params", "expected_answer", "expected_annotation_count"),
    (
        (
            {
                "scene_variant": "midgame_board",
                "query_id": "legal_move_count",
                "target_answer": 3,
            },
            3,
            3,
        ),
        (
            {
                "scene_variant": "crowded_board",
                "query_id": "capture_move_count",
                "target_answer": 2,
            },
            2,
            2,
        ),
        (
            {
                "scene_variant": "midgame_board",
                "query_id": "max_capture_chain_length",
                "target_answer": 4,
            },
            4,
            4,
        ),
    ),
)
def test_games_checkers_move_count_emits_expected_contract(
    params: dict[str, int | str],
    expected_answer: int,
    expected_annotation_count: int,
) -> None:
    task_params = dict(params)
    task = GamesCheckersMoveCountPublicTask()
    if str(task_params.get("query_id")) == "max_capture_chain_length":
        task_params.pop("query_id", None)
        task = GamesCheckersMaxCaptureChainLengthTask()
    out = task.generate(33001, params=task_params, max_attempts=96)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "bbox_set"
    assert "dark square" not in out.prompt.lower()
    assert len(out.annotation_gt.value) == int(expected_annotation_count)
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert int(execution["target_answer"]) == int(expected_answer)
    assert len(execution["annotation_entity_ids"]) == int(expected_annotation_count)
    _assert_bbox_annotation_matches_entity_ids(trace, out.annotation_gt.value)
    if str(params["query_id"]) == "max_capture_chain_length":
        assert all(
            str(entity_id).startswith("piece_")
            for entity_id in execution["annotation_entity_ids"]
        )
        assert execution["annotation_kind"] == "piece"
        assert execution["max_capture_chain_length"] == int(expected_answer)
    else:
        assert all(
            str(entity_id).startswith("cell_r")
            for entity_id in execution["annotation_entity_ids"]
        )


def test_games_checkers_move_count_legal_annotation_tracks_unique_landing_squares() -> (
    None
):
    out = GamesCheckersMoveCountPublicTask().generate(
        33011,
        params={
            "scene_variant": "midgame_board",
            "query_id": "legal_move_count",
            "target_answer": 5,
        },
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    landing_coords = {tuple(move["landing"]) for move in execution["legal_move_specs"]}

    assert annotation_coords == landing_coords
    assert len(annotation_coords) == 5


def test_games_checkers_move_count_capture_annotation_tracks_capture_landings_only() -> (
    None
):
    out = GamesCheckersMoveCountPublicTask().generate(
        33021,
        params={
            "scene_variant": "crowded_board",
            "query_id": "capture_move_count",
            "target_answer": 4,
        },
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    capture_landings = {
        tuple(move["landing"])
        for move in execution["legal_move_specs"]
        if move["captured"] is not None
    }

    assert annotation_coords == capture_landings
    assert len(annotation_coords) == 4


def test_games_checkers_max_capture_chain_annotation_tracks_captured_pieces() -> None:
    out = GamesCheckersMaxCaptureChainLengthTask().generate(
        33041,
        params={
            "scene_variant": "crowded_board",
            "target_answer": 5,
        },
        max_attempts=160,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    chain = execution["max_capture_chain_specs"][0]
    captured_coords = {tuple(coord) for coord in chain["captured"]}

    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["query_id"] == "single"
    assert out.trace_payload["query_spec"]["params"]["query_id"] == "single"
    assert (
        out.trace_payload["query_spec"]["params"]["prompt_query_key"]
        == "max_capture_chain_length"
    )
    assert int(out.answer_gt.value) == 5
    assert annotation_coords == captured_coords
    assert len(annotation_coords) == 5


def test_games_checkers_piece_mobility_legal_annotation_tracks_source_pieces() -> None:
    out = GamesCheckersPieceMobilityCountTask().generate(
        33051,
        params={
            "scene_variant": "midgame_board",
            "query_id": "piece_with_legal_move_count",
            "target_answer": 4,
        },
        max_attempts=160,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    source_coords = {tuple(move["origin"]) for move in execution["legal_move_specs"]}

    assert out.scene_id == "checkers"
    assert out.query_id == "piece_with_legal_move_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 4
    assert annotation_coords == source_coords
    assert len(out.annotation_gt.value) == 4
    assert execution["annotation_kind"] == "piece_point"
    _assert_bbox_annotation_matches_piece_ids(trace, out.annotation_gt.value)


def test_games_checkers_piece_mobility_capture_annotation_tracks_source_pieces() -> (
    None
):
    out = GamesCheckersPieceMobilityCountTask().generate(
        33061,
        params={
            "scene_variant": "crowded_board",
            "query_id": "piece_with_capture_move_count",
            "target_answer": 3,
        },
        max_attempts=160,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    capture_sources = {
        tuple(move["origin"])
        for move in execution["legal_move_specs"]
        if move["captured"] is not None
    }

    assert out.query_id == "piece_with_capture_move_count"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 3
    assert annotation_coords == capture_sources
    assert len(out.annotation_gt.value) == 3
    assert execution["annotation_kind"] == "piece_point"
    _assert_bbox_annotation_matches_piece_ids(trace, out.annotation_gt.value)


def test_games_checkers_piece_mobility_zero_answer_uses_empty_point_set() -> None:
    out = GamesCheckersPieceMobilityCountTask().generate(
        33071,
        params={
            "scene_variant": "midgame_board",
            "query_id": "piece_with_capture_move_count",
            "target_answer": 0,
        },
        max_attempts=160,
    )
    execution = out.trace_payload["execution_trace"]

    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.type == "bbox_set"
    assert out.annotation_gt.value == []
    assert execution["annotation_entity_ids"] == []
    assert execution["annotation_coords"] == []
    assert out.trace_payload["projected_annotation"]["bbox_set"] == []
    assert out.trace_payload["projected_annotation"]["pixel_bbox_set"] == []


@pytest.mark.parametrize(
    ("target_player_name", "piece_state_kind", "target_answer"),
    (
        ("red", "all", 6),
        ("black", "all", 0),
        ("red", "edge", 5),
        ("black", "edge", 2),
    ),
)
def test_games_checkers_piece_state_annotation_tracks_matching_pieces(
    target_player_name: str, piece_state_kind: str, target_answer: int
) -> None:
    out = GamesCheckersPieceStateCountTask().generate(
        33081 + int(target_answer),
        params={
            "scene_variant": "crowded_board",
            "target_player": str(target_player_name),
            "piece_state_kind": str(piece_state_kind),
            "target_answer": int(target_answer),
        },
        max_attempts=192,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    edge_only = str(piece_state_kind) == "edge"
    target_player = RED if str(target_player_name) == "red" else BLACK
    expected_coords = _expected_piece_state_coords(
        execution["board_rows"], target=int(target_player), edge_only=bool(edge_only)
    )
    expected_entity_ids = {
        piece_to_entity_id(coord, player=int(target_player))
        for coord in expected_coords
    }

    assert out.scene_id == "checkers"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == len(expected_coords) == int(target_answer)
    assert {tuple(coord) for coord in execution["annotation_coords"]} == expected_coords
    assert set(execution["annotation_entity_ids"]) == expected_entity_ids
    assert execution["annotation_kind"] == "piece_point"
    assert execution["construction_mode"] == "piece_state_count_templates"
    assert execution["target_player"] == str(target_player_name)
    assert execution["piece_state_kind"] == str(piece_state_kind)
    assert bool(execution["edge_only"]) is bool(edge_only)
    expected_bboxes = [
        trace["render_map"]["piece_bboxes_px"][str(entity_id)]
        for entity_id in execution["annotation_entity_ids"]
    ]
    assert out.annotation_gt.value == expected_bboxes
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value


def test_games_checkers_piece_mobility_taxonomy() -> None:
    assert (
        resolve_task_taxonomy("task_games__checkers__piece_mobility_count").scene_id
        == "checkers"
    )
    assert (
        resolve_task_taxonomy("task_games__checkers__piece_state_count").scene_id
        == "checkers"
    )


def test_games_checkers_move_count_query_cycle_covers_legal_answer_support() -> None:
    observed: dict[str, set[int]] = {
        "legal_move_count": set(),
        "capture_move_count": set(),
        "max_capture_chain_length": set(),
        "piece_with_legal_move_count": set(),
        "piece_with_capture_move_count": set(),
        "piece_state_count": set(),
    }
    scenes_by_branch: dict[str, set[str]] = {key: set() for key in observed}
    styles_by_branch: dict[str, set[str]] = {key: set() for key in observed}
    piece_state_operands: set[tuple[str, str]] = set()

    task_runs = (
        (GamesCheckersMoveCountPublicTask(), 72),
        (GamesCheckersMaxCaptureChainLengthTask(), 36),
        (GamesCheckersPieceMobilityCountTask(), 72),
        (GamesCheckersPieceStateCountTask(), 140),
    )
    for task, count in task_runs:
        for sampling_index in range(count):
            out = task.generate(
                33101 + int(sampling_index),
                params={"_sample_cursor": sampling_index},
                max_attempts=160,
            )
            execution = out.trace_payload["execution_trace"]
            branch = str(execution.get("prompt_query_key") or out.query_id)
            observed[branch].add(int(out.answer_gt.value))
            scenes_by_branch[branch].add(str(execution["scene_variant"]))
            styles_by_branch[branch].add(str(execution["style_variant"]))
            if branch == "piece_state_count":
                piece_state_operands.add(
                    (
                        str(execution["target_player"]),
                        str(execution["piece_state_kind"]),
                    )
                )

    assert observed["legal_move_count"] == {0, 1, 2, 3, 4, 5}
    assert observed["capture_move_count"] == {0, 1, 2, 3, 4}
    assert observed["max_capture_chain_length"] == {1, 2, 3, 4, 5}
    assert observed["piece_with_legal_move_count"] == {0, 1, 2, 3, 4, 5}
    assert observed["piece_with_capture_move_count"] == {0, 1, 2, 3, 4}
    assert observed["piece_state_count"] == {0, 1, 2, 3, 4, 5, 6}
    assert piece_state_operands == {
        ("red", "all"),
        ("black", "all"),
        ("red", "edge"),
        ("black", "edge"),
    }
    assert all(
        values == {"crowded_board", "midgame_board"}
        for values in scenes_by_branch.values()
    )
    assert all(
        values
        == {"classic", "soft", "outlined", "wood_token", "blue_table", "charcoal"}
        for values in styles_by_branch.values()
    )


def test_games_checkers_move_count_is_deterministic() -> None:
    params = {
        "scene_variant": "crowded_board",
        "query_id": "capture_move_count",
        "target_answer": 1,
    }
    task = GamesCheckersMoveCountPublicTask()
    out_a = task.generate(33031, params=params, max_attempts=96)
    out_b = task.generate(33031, params=params, max_attempts=96)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert (
        out_a.trace_payload["query_spec"]["prompt_variant"]
        == out_b.trace_payload["query_spec"]["prompt_variant"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_checkers_move_count_prompt_bundle_requires_rule_text_for_query_specific_prompts() -> (
    None
):
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/checkers/games_checkers_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle["schema_version"] == "v1"
    required = bundle["required_slots_by_key"]
    assert required["query:legal_move_count"] == [
        "current_player_name",
        "movement_rule_text",
    ]
    assert required["query:capture_move_count"] == [
        "current_player_name",
        "movement_rule_text",
    ]
    assert required["query:max_capture_chain_length"] == [
        "current_player_name",
    ]
    static_slots = bundle["static_slots_by_key"]
    assert "capture_rule_text" in static_slots["query:legal_move_count"]
    assert "king_chain_rule_text" in static_slots["query:max_capture_chain_length"]
    assert "answer_hint" in static_slots["query:piece_state_count"]
    assert (
        "[x0, y0, x1, y1]" in static_slots["query:piece_state_count"]["annotation_hint"]
    )
    for slots in static_slots.values():
        capture_rule_text = str(slots.get("capture_rule_text", ""))
        assert "dark square" not in capture_rule_text.lower()
        annotation_hint = str(slots.get("annotation_hint", ""))
        if annotation_hint:
            assert "dark square" not in annotation_hint.lower()
            assert "pixel-space" in annotation_hint
            assert "box" in annotation_hint.lower()
        json_example = str(slots.get("json_example", ""))
        if json_example:
            example = json.loads(json_example)
            assert all(len(bbox) == 4 for bbox in example["annotation"])


def test_games_checkers_move_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__checkers__move_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__checkers__move_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__checkers__move_count",
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_games__checkers__piece_state_count",
                count=4,
                params={},
            ),
        ],
        strict_repro=False,
        max_attempts_per_instance=96,
        sampling_seed=83,
    )
    final_path = build_dataset(config, code_hash="games-checkers-move-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 8
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "checkers" for record in train_records)

    build_report = json.loads(
        (final_path / "build_report.json").read_text(encoding="utf-8")
    )
    assert (
        int(build_report["accepted_counts_by_task"]["task_games__checkers__move_count"])
        == 4
    )
    assert (
        int(
            build_report["accepted_counts_by_task"][
                "task_games__checkers__piece_state_count"
            ]
        )
        == 4
    )

    validation = json.loads(
        (final_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["total_errors"] == 0
