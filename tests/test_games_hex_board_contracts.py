"""Contract tests for games Hex-board tasks."""

from __future__ import annotations

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.hex.candidate_neighbor_count import (
    GamesHexCandidateNeighborCountTask,
)
from trace_tasks.tasks.games.hex.connection_gap_count import GamesHexConnectionGapCountTask
from trace_tasks.tasks.games.hex.shared.rules import (
    BLUE,
    EMPTY,
    RED,
    coord_to_cell_id,
    immediate_winning_moves,
    minimum_connection_gap_sets,
    minimum_connection_path,
    neighbors,
    sorted_coords,
    winning_path_after_move,
)
from trace_tasks.tasks.games.hex.winning_move_cell_label import (
    GamesHexWinningMoveCellLabelTask,
)
from trace_tasks.tasks.games.shared.style import SUPPORTED_HEX_STYLE_VARIANTS
from tests.helpers import read_jsonl


def _coords(values: list[list[int]]) -> tuple[tuple[int, int], ...]:
    """Return trace coordinate lists as stable coordinate tuples."""

    return tuple((int(row), int(col)) for row, col in values)


def test_games_hex_winning_move_cell_label_emits_expected_contract() -> None:
    out = GamesHexWinningMoveCellLabelTask().generate(
        51201,
        params={
            "target_label": "D",
            "candidate_count": 6,
            "board_size": 6,
            "player_color": "red",
        },
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "point"
    assert out.query_id == "single"
    assert out.scene_id == "hex"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
    assert trace["render_spec"]["panel_scene_style"]["treatment"]
    assert trace["render_spec"]["text_style"]["font_family"]
    assert len(out.annotation_gt.value) == 2
    assert len(execution["annotation_entity_ids"]) == 1


def test_games_hex_winning_move_cell_label_has_unique_immediate_winning_cell() -> None:
    out = GamesHexWinningMoveCellLabelTask().generate(
        51211,
        params={
            "target_label": "F",
            "candidate_count": 6,
            "board_size": 7,
            "player_color": "blue",
        },
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    board = tuple(tuple(int(value) for value in row) for row in execution["board_rows"])
    player_value = int(execution["player_value"])
    winning_coord = tuple(int(value) for value in execution["winning_move_coord"])
    answer_candidates = [
        spec for spec in execution["candidate_specs"] if bool(spec["is_answer"])
    ]
    annotation_coords = _coords(execution["annotation_coords"])
    completed_path_coords = _coords(execution["completed_winning_path_coords"])

    assert int(board[winning_coord[0]][winning_coord[1]]) == EMPTY
    assert immediate_winning_moves(board, player_value=player_value) == (winning_coord,)
    assert len(answer_candidates) == 1
    assert answer_candidates[0]["label"] == out.answer_gt.value == "F"
    assert tuple(answer_candidates[0]["coord"]) == winning_coord
    assert annotation_coords == (winning_coord,)
    assert completed_path_coords == winning_path_after_move(
        board,
        player_value=player_value,
        move_coord=winning_coord,
    )
    assert set(execution["annotation_entity_ids"]) == {coord_to_cell_id(winning_coord)}


def test_games_hex_connection_gap_count_emits_expected_contract() -> None:
    out = GamesHexConnectionGapCountTask().generate(
        51221,
        params={"target_answer": 4, "board_size": 7, "player_color": "red"},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.annotation_gt.type == "point_set"
    assert out.query_id == "single"
    assert out.scene_id == "hex"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value


def test_games_hex_connection_gap_count_matches_shortest_path_cost() -> None:
    out = GamesHexConnectionGapCountTask().generate(
        51231,
        params={"target_answer": 5, "board_size": 8, "player_color": "blue"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    board = tuple(tuple(int(value) for value in row) for row in execution["board_rows"])
    player_value = int(execution["player_value"])
    gap_count, path = minimum_connection_path(board, player_value=player_value)
    gap_search = minimum_connection_gap_sets(
        board, player_value=player_value, max_sets=2
    )
    annotation_coords = _coords(execution["annotation_coords"])
    empty_on_path = tuple(
        coord for coord in path if int(board[coord[0]][coord[1]]) == EMPTY
    )

    assert int(out.answer_gt.value) == int(gap_count) == 5
    assert gap_search.exhaustive
    assert len(gap_search.gap_sets) == 1
    assert sorted_coords(empty_on_path) == gap_search.gap_sets[0]
    assert sorted_coords(annotation_coords) == gap_search.gap_sets[0]
    assert len(annotation_coords) == int(out.answer_gt.value)
    assert set(execution["annotation_entity_ids"]) == {
        coord_to_cell_id(coord) for coord in annotation_coords
    }


def test_games_hex_candidate_neighbor_count_emits_expected_contract() -> None:
    out = GamesHexCandidateNeighborCountTask().generate(
        51251,
        params={"neighbor_target_state": "red", "target_answer": 4, "board_size": 6},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.annotation_gt.type == "point_set"
    assert out.query_id == "single"
    assert out.scene_id == "hex"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert execution["neighbor_target_state"] == "red"
    assert execution["reference_cell_id"]
    assert trace["render_map"]["reference_cell_ids"] == [execution["reference_cell_id"]]
    assert "green" in out.prompt.lower()
    assert "cell C" not in out.prompt
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value


def test_games_hex_candidate_neighbor_count_matches_adjacent_cell_states() -> None:
    state_to_value = {
        "red": RED,
        "blue": BLUE,
        "empty": EMPTY,
    }
    for index, (target_state, target_answer) in enumerate(
        (
            ("red", 0),
            ("blue", 6),
            ("empty", 3),
        )
    ):
        out = GamesHexCandidateNeighborCountTask().generate(
            51261 + index,
            params={
                "neighbor_target_state": target_state,
                "target_answer": target_answer,
                "board_size": 6,
            },
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        board = tuple(
            tuple(int(value) for value in row) for row in execution["board_rows"]
        )
        reference_coord = tuple(int(value) for value in execution["reference_coord"])
        assert int(board[reference_coord[0]][reference_coord[1]]) == EMPTY
        expected = sorted_coords(
            coord
            for coord in neighbors(
                reference_coord, board_size=int(execution["board_size"])
            )
            if int(board[coord[0]][coord[1]]) == int(state_to_value[target_state])
        )
        annotation_coords = _coords(execution["annotation_coords"])

        assert int(out.answer_gt.value) == len(expected) == int(target_answer)
        assert (
            len(neighbors(reference_coord, board_size=int(execution["board_size"])))
            == 6
        )
        assert sorted_coords(annotation_coords) == expected
        assert sorted_coords(_coords(execution["neighbor_match_coords"])) == expected
        assert set(execution["annotation_entity_ids"]) == {
            coord_to_cell_id(coord) for coord in annotation_coords
        }
        assert len(out.annotation_gt.value) == int(target_answer)
        assert out.query_id == "single"
        assert execution["neighbor_target_state"] == target_state


def test_games_hex_candidate_neighbor_count_query_cycle_covers_support() -> None:
    task = GamesHexCandidateNeighborCountTask()
    queries: set[str] = set()
    target_states: set[str] = set()
    counts: set[int] = set()
    boards: set[int] = set()
    styles: set[str] = set()
    for sampling_index in range(96):
        out = task.generate(
            51281 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        queries.add(str(out.query_id))
        target_states.add(str(execution["neighbor_target_state"]))
        counts.add(int(out.answer_gt.value))
        boards.add(int(execution["board_size"]))
        styles.add(str(execution["style_variant"]))

    assert queries == {"single"}
    assert target_states == {"red", "blue", "empty"}
    assert counts == {0, 1, 2, 3, 4, 5, 6}
    assert boards == {5, 6, 7, 8}
    assert styles == set(SUPPORTED_HEX_STYLE_VARIANTS)


def test_games_hex_connection_gap_count_cycle_covers_support() -> None:
    task = GamesHexConnectionGapCountTask()
    gap_counts: set[int] = set()
    players: set[str] = set()
    styles: set[str] = set()
    for sampling_index in range(80):
        out = task.generate(
            51301 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        gap_counts.add(int(out.answer_gt.value))
        players.add(str(execution["player_color"]))
        styles.add(str(execution["style_variant"]))

    assert gap_counts == {1, 2, 3, 4, 5}
    assert players == {"red", "blue"}
    assert styles == set(SUPPORTED_HEX_STYLE_VARIANTS)


def test_games_hex_winning_move_cell_label_cycle_covers_support() -> None:
    task = GamesHexWinningMoveCellLabelTask()
    labels: set[str] = set()
    boards: set[int] = set()
    players: set[str] = set()
    styles: set[str] = set()
    for sampling_index in range(96):
        out = task.generate(
            51391 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        labels.add(str(out.answer_gt.value))
        boards.add(int(execution["board_size"]))
        players.add(str(execution["player_color"]))
        styles.add(str(execution["style_variant"]))

    assert labels == set("ABCDEF")
    assert boards == {5, 6, 7, 8}
    assert players == {"red", "blue"}
    assert styles == set(SUPPORTED_HEX_STYLE_VARIANTS)


def test_games_hex_tasks_are_deterministic() -> None:
    cases = (
        (
            GamesHexWinningMoveCellLabelTask(),
            {"query_id": "single", "target_label": "C", "board_size": 6},
        ),
        (
            GamesHexConnectionGapCountTask(),
            {"query_id": "single", "target_answer": 3, "board_size": 6},
        ),
        (
            GamesHexCandidateNeighborCountTask(),
            {"neighbor_target_state": "empty", "target_answer": 2, "board_size": 6},
        ),
    )
    for task, params in cases:
        out_a = task.generate(51241, params=params, max_attempts=128)
        out_b = task.generate(51241, params=params, max_attempts=128)
        assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
        assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
        assert (
            out_a.trace_payload["execution_trace"]
            == out_b.trace_payload["execution_trace"]
        )
        assert (
            out_a.trace_payload["query_spec"]["prompt_variant"]
            == out_b.trace_payload["query_spec"]["prompt_variant"]
        )
        assert out_a.prompt == out_b.prompt
        assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_hex_board_build_smoke(tmp_path) -> None:
    output_root = tmp_path / "task_games__hex__winning_move_cell_label"
    cfg = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games_hex_board",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__hex__winning_move_cell_label",
                count=1,
                params={"query_id": "single", "target_label": "B", "board_size": 6},
            ),
            BuildTaskConfig(
                task_id="task_games__hex__connection_gap_count",
                count=1,
                params={"query_id": "single", "target_answer": 3, "board_size": 6},
            ),
            BuildTaskConfig(
                task_id="task_games__hex__candidate_neighbor_count",
                count=1,
                params={
                    "neighbor_target_state": "empty",
                    "target_answer": 2,
                    "board_size": 6,
                },
            ),
        ],
        max_attempts_per_instance=128,
        workers=1,
    )
    final_path = build_dataset(cfg, code_hash="games-hex-board-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "hex" for row in rows)
