"""Contract tests for games Chess-board tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.chess.checkmate_move_label import GamesChessCheckmateMoveLabelTask
from trace_tasks.tasks.games.chess.colored_piece_kind_count import GamesChessColoredPieceKindCountTask
from trace_tasks.tasks.games.chess.king_escape_square_count import GamesChessKingEscapeSquareCountTask
from trace_tasks.tasks.games.chess.marked_piece_capture_count import GamesChessMarkedPieceCaptureCountTask
from trace_tasks.tasks.games.chess.marked_piece_destination_count import GamesChessMarkedPieceDestinationCountTask
from trace_tasks.tasks.games.chess.piece_kind_count import GamesChessPieceKindCountTask
from trace_tasks.tasks.games.chess.player_capture_piece_count import GamesChessPlayerCapturePieceCountTask
from trace_tasks.tasks.games.chess.target_square_attacker_count import GamesChessTargetSquareAttackerCountTask
from trace_tasks.tasks.games.shared.piece_board_rules import (
    ChessPiece,
    attackers_to_square,
    capturable_opponent_coords,
    coord_to_cell_id,
    freeze_board,
    king_escape_squares,
    move_checkmates,
    opponent,
    piece_attacks_square,
    piece_capture_targets,
    piece_move_destinations,
    piece_to_entity_id,
    validate_square_chess_material,
)
from trace_tasks.tasks.games.shared.piece_board_renderer import _FILLED_PIECE_CODEPOINTS, _fit_chess_symbol_font
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query"),
    (
        (
            GamesChessMarkedPieceDestinationCountTask,
            {"target_answer": 4, "scene_variant": "sparse_board"},
            "single",
        ),
        (
            GamesChessMarkedPieceCaptureCountTask,
            {"target_answer": 2, "scene_variant": "crowded_board"},
            "single",
        ),
        (
            GamesChessPlayerCapturePieceCountTask,
            {"target_answer": 3, "player_color": "white"},
            "single",
        ),
        (
            GamesChessTargetSquareAttackerCountTask,
            {"query_id": "king_square_attacker_count", "target_answer": 2},
            "king_square_attacker_count",
        ),
        (
            GamesChessKingEscapeSquareCountTask,
            {"target_answer": 3},
            "single",
        ),
        (
            GamesChessPieceKindCountTask,
            {"target_answer": 4, "target_piece_kind": "pawn"},
            "single",
        ),
        (
            GamesChessColoredPieceKindCountTask,
            {
                "target_answer": 2,
                "target_piece_kind": "knight",
                "target_piece_color": "black",
            },
            "single",
        ),
    ),
)
def test_games_chess_board_emits_expected_contract(
    task_cls: type[Any],
    params: dict[str, int | str],
    expected_query: str,
) -> None:
    out = task_cls().generate(50201, params=params, max_attempts=96)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == str(expected_query)
    assert trace["query_spec"]["query_id"] == str(expected_query)
    assert trace["query_spec"]["params"]["query_id"] == str(expected_query)
    assert execution["query_id"] == str(expected_query)
    assert int(execution["target_answer"]) == int(out.answer_gt.value)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)


def test_games_chess_marked_move_count_matches_rules() -> None:
    out = GamesChessMarkedPieceDestinationCountTask().generate(
        50211,
        params={"target_answer": 5},
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    destinations = tuple(
        sorted(
            coord
            for coord in piece_move_destinations(board, marked)
            if board[int(coord[0])][int(coord[1])] is None
        )
    )

    assert len(destinations) == int(out.answer_gt.value) == 5
    assert [list(coord) for coord in destinations] == sorted(execution["destination_coords"])
    assert set(execution["annotation_entity_ids"]) == {coord_to_cell_id(coord) for coord in destinations}
    assert execution["internal_query_id"] == "marked_piece_destination_count"


def test_games_chess_legal_moves_exclude_opponent_king_but_attacks_detect_it() -> None:
    mutable = [[None for _ in range(8)] for _ in range(8)]
    mutable[4][1] = ChessPiece(color="white", kind="rook")
    mutable[4][4] = ChessPiece(color="black", kind="king")
    board = freeze_board(mutable)

    assert (4, 4) not in piece_move_destinations(board, (4, 1))
    assert piece_attacks_square(board, (4, 1), (4, 4))
    assert attackers_to_square(board, (4, 4), "white") == ((4, 1),)


def test_games_chess_marked_capture_count_matches_rules() -> None:
    out = GamesChessMarkedPieceCaptureCountTask().generate(
        50221,
        params={"target_answer": 3},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    captures = tuple(sorted(piece_capture_targets(board, marked)))

    assert len(captures) == int(out.answer_gt.value) == 3
    assert [list(coord) for coord in captures] == sorted(execution["capture_coords"])
    assert set(execution["annotation_entity_ids"]) == {coord_to_cell_id(coord) for coord in captures}
    assert execution["internal_query_id"] == "marked_piece_capture_count"


@pytest.mark.parametrize(
    ("query_id", "marked_piece_kind", "target_answer"),
    (
        ("marked_piece_destination_count", "knight", 5),
        ("marked_piece_destination_count", "bishop", 5),
        ("marked_piece_destination_count", "rook", 6),
        ("marked_piece_destination_count", "queen", 6),
        ("marked_piece_capture_count", "knight", 3),
        ("marked_piece_capture_count", "bishop", 2),
        ("marked_piece_capture_count", "rook", 3),
        ("marked_piece_capture_count", "queen", 4),
    ),
)
def test_games_chess_marked_destination_count_supports_multiple_piece_kinds(
    query_id: str,
    marked_piece_kind: str,
    target_answer: int,
) -> None:
    task_cls = (
        GamesChessMarkedPieceCaptureCountTask
        if str(query_id) == "marked_piece_capture_count"
        else GamesChessMarkedPieceDestinationCountTask
    )
    out = task_cls().generate(
        61200 + int(target_answer) + len(str(marked_piece_kind)),
        params={
            "target_answer": int(target_answer),
            "marked_piece_kind": str(marked_piece_kind),
        },
        max_attempts=192,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    marked_piece = board[int(marked[0])][int(marked[1])]
    if str(query_id) == "marked_piece_capture_count":
        expected_coords = tuple(sorted(piece_capture_targets(board, marked)))
        trace_key = "capture_coords"
    else:
        expected_coords = tuple(
            sorted(
                coord
                for coord in piece_move_destinations(board, marked)
                if board[int(coord[0])][int(coord[1])] is None
            )
        )
        trace_key = "destination_coords"

    assert marked_piece is not None
    assert str(marked_piece.kind) == str(marked_piece_kind)
    assert execution["marked_piece_kind"] == str(marked_piece_kind)
    assert out.trace_payload["query_spec"]["params"]["marked_piece_kind"] == str(marked_piece_kind)
    assert len(expected_coords) == int(out.answer_gt.value) == int(target_answer)
    assert [list(coord) for coord in expected_coords] == sorted(execution[trace_key])
    assert set(execution["annotation_entity_ids"]) == {coord_to_cell_id(coord) for coord in expected_coords}
    assert out.query_id == "single"
    assert execution["internal_query_id"] == str(query_id)


def test_games_chess_player_capture_count_matches_rules() -> None:
    out = GamesChessPlayerCapturePieceCountTask().generate(
        50231,
        params={"target_answer": 4, "player_color": "black"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    captures = capturable_opponent_coords(board, str(execution["player_color"]))

    assert len(captures) == int(out.answer_gt.value) == 4
    assert [list(coord) for coord in captures] == sorted(execution["capture_coords"])


def test_games_chess_king_square_attacker_count_matches_rules() -> None:
    out = GamesChessTargetSquareAttackerCountTask().generate(
        50241,
        params={"query_id": "king_square_attacker_count", "target_answer": 3},
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    attacker_color = "black" if str(execution["player_color"]) == "white" else "white"
    attackers = attackers_to_square(board, marked, attacker_color)

    assert len(attackers) == int(out.answer_gt.value) == 3
    assert [list(coord) for coord in attackers] == sorted(execution["attacker_coords"])


@pytest.mark.parametrize(
    ("query_id", "attacker_color", "target_answer"),
    (
        ("white_piece_attacks_target_square_count", "white", 3),
        ("black_piece_attacks_target_square_count", "black", 2),
    ),
)
def test_games_chess_empty_target_square_attacker_count_matches_rules(
    query_id: str,
    attacker_color: str,
    target_answer: int,
) -> None:
    out = GamesChessTargetSquareAttackerCountTask().generate(
        50242 + int(target_answer),
        params={"query_id": str(query_id), "target_answer": int(target_answer)},
        max_attempts=160,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    attackers = attackers_to_square(board, marked, str(attacker_color))

    assert board[int(marked[0])][int(marked[1])] is None
    assert len(attackers) == int(out.answer_gt.value) == int(target_answer)
    assert [list(coord) for coord in attackers] == sorted(execution["attacker_coords"])
    assert set(execution["annotation_entity_ids"]) == {
        piece_to_entity_id(coord, board[int(coord[0])][int(coord[1])])
        for coord in attackers
    }


def test_games_chess_king_escape_square_count_matches_rules() -> None:
    out = GamesChessKingEscapeSquareCountTask().generate(
        50246,
        params={"target_answer": 4},
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(value) for value in execution["marked_coord"])
    escapes = tuple(sorted(king_escape_squares(board, marked)))

    assert len(escapes) == int(out.answer_gt.value) == 4
    assert [list(coord) for coord in escapes] == sorted(execution["destination_coords"])
    assert set(execution["annotation_entity_ids"]) == {coord_to_cell_id(coord) for coord in escapes}


def test_games_chess_king_escape_sampler_uses_opponent_attacks_not_only_blockers() -> None:
    for target_answer in range(0, 6):
        out = GamesChessKingEscapeSquareCountTask().generate(
            60246 + int(target_answer),
            params={"target_answer": int(target_answer)},
            max_attempts=256,
        )
        execution = out.trace_payload["execution_trace"]
        board = _board_from_execution(execution)
        marked = tuple(int(value) for value in execution["marked_coord"])
        king = board[int(marked[0])][int(marked[1])]
        assert king is not None
        king_color = str(king.color)
        escapes = {tuple(int(value) for value in coord) for coord in execution["destination_coords"]}
        adjacent = {
            (int(marked[0]) + dr, int(marked[1]) + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (int(dr) == 0 and int(dc) == 0)
        }

        attacked_open_unsafe = []
        for coord in sorted(adjacent - escapes):
            occupant = board[int(coord[0])][int(coord[1])]
            if occupant is not None and str(occupant.color) == king_color:
                continue
            moved = [list(row) for row in board]
            moved[int(marked[0])][int(marked[1])] = None
            moved[int(coord[0])][int(coord[1])] = king
            if attackers_to_square(freeze_board(moved), coord, opponent(king_color)):
                attacked_open_unsafe.append(coord)
        assert attacked_open_unsafe

        for row in range(8):
            for col in range(8):
                coord = (row, col)
                piece = board[row][col]
                if piece is None or str(piece.color) != king_color or str(piece.kind) not in {"queen", "rook", "bishop"}:
                    continue
                assert not any(piece_attacks_square(board, coord, escape_coord) for escape_coord in escapes)


def test_games_chess_piece_type_count_matches_visible_pieces() -> None:
    out = GamesChessColoredPieceKindCountTask().generate(
        50261,
        params={
            "target_answer": 3,
            "target_piece_kind": "bishop",
            "target_piece_color": "white",
            "piece_count_distractor_count": 8,
        },
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)
    matches = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece is not None and str(piece.color) == "white" and str(piece.kind) == "bishop":
                matches.append((row, col))

    assert int(out.answer_gt.value) == 3
    assert matches == [tuple(coord) for coord in execution["annotation_coords"]]
    assert len(out.annotation_gt.value) == 3


def test_games_chess_piece_type_count_zero_answer_uses_empty_annotation() -> None:
    out = GamesChessPieceKindCountTask().generate(
        50262,
        params={
            "target_answer": 0,
            "target_piece_kind": "queen",
            "piece_count_distractor_count": 5,
        },
        max_attempts=96,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board_from_execution(execution)

    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert not any(piece is not None and str(piece.kind) == "queen" for row in board for piece in row)


def test_games_chess_checkmate_move_label_has_unique_mating_option() -> None:
    out = GamesChessCheckmateMoveLabelTask().generate(
        50263,
        params={"option_count": 6, "answer_option_label": "D", "player_color": "white"},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board = _board_from_execution(execution)

    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "point_map"
    assert out.answer_gt.value == "D"
    assert execution["answer_option_label"] == out.answer_gt.value
    assert trace["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    assert set(out.annotation_gt.value) == {"from", "to", "king"}
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
    assert trace["render_map"]["coordinate_label_bboxes_px"]["files"]
    assert set(trace["render_map"]["move_option_panel"]["option_bboxes_px"]) == {"A", "B", "C", "D", "E", "F"}

    mating_labels = []
    for option in execution["move_options"]:
        assert "White " not in str(option["text"])
        assert "Black " not in str(option["text"])
        source = tuple(int(value) for value in option["source_coord"])
        destination = tuple(int(value) for value in option["destination_coord"])
        is_mate = move_checkmates(board, source, destination)
        assert bool(option["is_checkmate"]) is bool(is_mate)
        if is_mate:
            mating_labels.append(str(option["label"]))

    assert mating_labels == [str(out.answer_gt.value)]


@pytest.mark.parametrize(
    ("task_cls", "params"),
    (
        (GamesChessMarkedPieceDestinationCountTask, {"target_answer": 4}),
        (GamesChessMarkedPieceCaptureCountTask, {"target_answer": 2}),
        (GamesChessPlayerCapturePieceCountTask, {"target_answer": 3, "player_color": "white"}),
        (GamesChessTargetSquareAttackerCountTask, {"query_id": "king_square_attacker_count", "target_answer": 2}),
        (GamesChessTargetSquareAttackerCountTask, {"query_id": "white_piece_attacks_target_square_count", "target_answer": 2}),
        (GamesChessKingEscapeSquareCountTask, {"target_answer": 3}),
    ),
)
def test_games_chess_movement_rule_boards_are_material_plausible(
    task_cls: type[Any],
    params: dict[str, int | str],
) -> None:
    for offset in range(8):
        out = task_cls().generate(62200 + int(offset), params=params, max_attempts=256)
        board = _board_from_execution(out.trace_payload["execution_trace"])
        assert validate_square_chess_material(board)


def test_games_chess_checkmate_board_is_material_plausible() -> None:
    for offset in range(6):
        out = GamesChessCheckmateMoveLabelTask().generate(
            62300 + int(offset),
            params={"option_count": 4 if int(offset) % 2 == 0 else 6},
            max_attempts=160,
        )
        board = _board_from_execution(out.trace_payload["execution_trace"])
        assert validate_square_chess_material(board)


@pytest.mark.parametrize(
    ("task_cls", "query_id", "support"),
    (
        (GamesChessMarkedPieceDestinationCountTask, "marked_piece_destination_count", (1, 2, 3, 4, 5, 6)),
        (GamesChessMarkedPieceCaptureCountTask, "marked_piece_capture_count", (0, 1, 2, 3, 4)),
        (GamesChessPlayerCapturePieceCountTask, "player_capture_piece_count", (1, 2, 3, 4, 5, 6)),
        (GamesChessTargetSquareAttackerCountTask, "king_square_attacker_count", (0, 1, 2, 3, 4)),
        (GamesChessTargetSquareAttackerCountTask, "white_piece_attacks_target_square_count", (0, 1, 2, 3, 4)),
        (GamesChessTargetSquareAttackerCountTask, "black_piece_attacks_target_square_count", (0, 1, 2, 3, 4)),
        (GamesChessKingEscapeSquareCountTask, "king_escape_square_count", (0, 1, 2, 3, 4, 5)),
        (GamesChessPieceKindCountTask, "piece_kind_count", (0, 1, 2, 3, 4, 5, 6)),
        (GamesChessColoredPieceKindCountTask, "colored_piece_kind_count", (0, 1, 2, 3, 4, 5, 6)),
    ),
)
def test_games_chess_public_tasks_cover_declared_integer_answer_support(
    task_cls: type[Any],
    query_id: str,
    support: tuple[int, ...],
) -> None:
    seen = set()
    for index, target_answer in enumerate(support):
        params: dict[str, Any] = {"target_answer": int(target_answer)}
        if str(query_id) not in {
            "marked_piece_destination_count",
            "marked_piece_capture_count",
            "player_capture_piece_count",
            "king_escape_square_count",
            "piece_kind_count",
            "colored_piece_kind_count",
        }:
            params["query_id"] = str(query_id)
        if str(query_id) == "piece_kind_count":
            params["target_piece_kind"] = "pawn"
        if str(query_id) == "colored_piece_kind_count":
            params["target_piece_kind"] = "bishop"
            params["target_piece_color"] = "white"
        out = task_cls().generate(50301 + (37 * int(index)), params=params, max_attempts=256)
        expected_public_query = "single" if "query_id" not in params else str(query_id)
        assert out.query_id == expected_public_query
        assert out.trace_payload["query_spec"]["params"]["query_id"] == expected_public_query
        seen.add(int(out.answer_gt.value))

    assert seen == set(int(value) for value in support)


def test_games_chess_board_is_deterministic() -> None:
    params = {
        "query_id": "white_piece_attacks_target_square_count",
        "target_answer": 2,
        "scene_variant": "crowded_board",
        "style_variant": "outlined",
    }
    task = GamesChessTargetSquareAttackerCountTask()
    out_a = task.generate(50251, params=params, max_attempts=96)
    out_b = task.generate(50251, params=params, max_attempts=96)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_chess_piece_symbols_ignore_sampled_readout_font() -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)

    with temporary_default_font_family("georama"):
        font = _fit_chess_symbol_font(
            draw,
            glyph="♔",
            max_width=84,
            max_height=84,
            min_size_px=18,
            max_size_px=78,
            fill_ratio=0.98,
        )

    assert getattr(font, "getname")()[0] == "DejaVu Sans"


def test_games_chess_uses_filled_piece_symbol_set_for_both_sides() -> None:
    assert _FILLED_PIECE_CODEPOINTS == {
        "king": 0x265A,
        "queen": 0x265B,
        "rook": 0x265C,
        "bishop": 0x265D,
        "knight": 0x265E,
        "pawn": 0x265F,
    }


def test_games_chess_board_prompt_bundle_requires_rule_texts() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/chess/games_chess_v1.json").read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "v1"
    assert bundle["required_slots_by_key"] == {}
    static = bundle["static_slots_by_key"]
    assert "normal chess" in static["query:marked_piece_destination_count"]["standard_rule_text"].lower()
    assert "red outlined square" in static["query:marked_piece_destination_count"]["marked_piece_rule_text"].lower()
    assert "empty destination" in static["query:marked_piece_destination_count"]["answer_hint"].lower()
    assert "opponent-occupied" in static["query:marked_piece_capture_count"]["standard_rule_text"].lower()
    for query_key in ("white_piece_attacks_target_square_count", "black_piece_attacks_target_square_count"):
        assert all("attackers" not in str(template).lower() for template in bundle["templates"]["query"][query_key])
        assert any("pieces attack" in str(template).lower() for template in bundle["templates"]["query"][query_key])
    assert "from" in static["query:checkmate_move_label"]["annotation_hint"]
    for key, slots in static.items():
        annotation_hint = str(slots.get("annotation_hint", ""))
        if annotation_hint:
            if key == "query:checkmate_move_label":
                assert "pixel-space point" in annotation_hint
                assert '"from"' in annotation_hint
                assert '"to"' in annotation_hint
                assert '"king"' in annotation_hint
            else:
                assert "pixel-space" in annotation_hint
                assert "boxes" in annotation_hint
        json_example = str(slots.get("json_example", ""))
        if json_example:
            example = json.loads(json_example)
            annotation = example["annotation"]
            if isinstance(annotation, list):
                assert all(len(bbox) == 4 for bbox in annotation)
            else:
                assert set(annotation) == {"from", "to", "king"}
                assert all(len(point) == 2 for point in annotation.values())
    dynamic = bundle["dynamic_slots"]
    assert set(dynamic) >= {
        "player_color_name",
        "opponent_color_name",
        "defender_color_name",
        "target_color_name",
        "target_piece_kind",
        "target_piece_kind_plural",
    }


def test_games_chess_board_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__chess__marked_piece_destination_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__chess__marked_piece_destination_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__chess__marked_piece_destination_count",
                count=4,
                params={},
            )
        ],
        max_attempts_per_instance=96,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-chess-board-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")
    report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))

    assert int(report["accepted_counts_by_task"]["task_games__chess__marked_piece_destination_count"]) == 4
    assert len(rows) == 4
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["task"] == "task_games__chess__marked_piece_destination_count" for row in rows)
    assert all(row["query_id"] == "single" for row in rows)


def test_games_chess_retries_transient_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import trace_tasks.tasks.games.chess._lifecycle as lifecycle_module
    import trace_tasks.tasks.games.chess.marked_piece_destination_count as task_module

    real_sampler = lifecycle_module.sample_marked_piece_destination_scene
    calls = {"count": 0}

    def flaky_sampler(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("forced transient construction failure")
        return real_sampler(*args, **kwargs)

    monkeypatch.setattr(lifecycle_module, "sample_marked_piece_destination_scene", flaky_sampler)

    out = task_module.GamesChessMarkedPieceDestinationCountTask().generate(
        50201,
        params={"target_answer": 4, "scene_variant": "sparse_board"},
        max_attempts=96,
    )

    assert calls["count"] >= 2
    assert out.answer_gt.type == "integer"
    assert out.query_id == "single"
    assert out.trace_payload["execution_trace"]["internal_query_id"] == "marked_piece_destination_count"


def _board_from_execution(execution: dict):
    from trace_tasks.tasks.games.shared.piece_board_rules import ChessPiece

    rows = []
    for row in execution["board_rows"]:
        parsed_row = []
        for cell in row:
            if cell is None:
                parsed_row.append(None)
            else:
                color, kind = str(cell).split("_", 1)
                parsed_row.append(ChessPiece(color=color, kind=kind))
        rows.append(parsed_row)
    return tuple(tuple(row) for row in rows)
