"""Contract tests for games Battleship-grid tasks."""

from __future__ import annotations

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.battleship.last_ship_cell_label import GamesBattleshipLastShipCellLabelTask
from trace_tasks.tasks.games.battleship.remaining_ship_shape_label import GamesBattleshipRemainingShipShapeLabelTask
from trace_tasks.tasks.games.battleship.ship_cell_status_count import GamesBattleshipShipCellStatusCountTask, SHIP_CELL_STATUS_COUNT_QUERY_IDS
from trace_tasks.tasks.games.battleship.ship_status_count import GamesBattleshipShipStatusCountTask
from trace_tasks.tasks.games.battleship.shared.rules import (
    fleet_shape_by_id,
    matching_fleet_shape_ids,
)
from trace_tasks.tasks.games.battleship.shared.state import (
    FLEET_SHAPES,
    SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS,
)
from trace_tasks.tasks.games.shared.style import SUPPORTED_BATTLESHIP_STYLE_VARIANTS
from tests.helpers import read_jsonl


def _coords(values: list[list[int]]) -> tuple[tuple[int, int], ...]:
    """Return trace coordinate lists as stable coordinate tuples."""

    return tuple((int(row), int(col)) for row, col in values)


def _bbox_center(bbox: list[float]) -> list[float]:
    """Return the center of a pixel bbox using the task's rounding."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _assert_hit_marker_bbox_map_annotation(trace: dict, annotation: dict[str, list[list[float]]]) -> None:
    """Assert ship-status bbox-set-map annotations mark every hit-marker cell for each ship."""

    execution = trace["execution_trace"]
    assert trace["projected_annotation"]["type"] == "bbox_set_map"
    assert trace["projected_annotation"]["bbox_set_map"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_set_map"] == annotation
    assert set(annotation.keys()) == set(execution["annotation_ship_name_to_ship_id"].keys())
    assert set(annotation.keys()) == set(execution["annotation_hit_cell_ids_by_ship_name"].keys())
    for ship_name, bboxes in annotation.items():
        hit_cell_ids = execution["annotation_hit_cell_ids_by_ship_name"][ship_name]
        expected = [
            trace["render_map"]["cell_bboxes_px"][str(cell_id)]
            for cell_id in hit_cell_ids
        ]
        assert bboxes == expected


def _assert_bbox_annotation_matches_cells(trace: dict, annotation: list[list[float]]) -> None:
    """Assert homogeneous bbox-set annotation marks the recorded cell ids."""

    execution = trace["execution_trace"]
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_set"] == annotation
    expected = [
        trace["render_map"]["cell_bboxes_px"][str(cell_id)]
        for cell_id in execution["annotation_cell_ids"]
    ]
    assert annotation == expected
    assert execution["annotation_entity_ids"] == execution["annotation_cell_ids"]


def _assert_point_annotation_matches_cell(trace: dict, annotation: list[float]) -> None:
    """Assert scalar point annotation marks the recorded cell id."""

    execution = trace["execution_trace"]
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == annotation
    assert trace["projected_annotation"]["pixel_point"] == annotation
    assert annotation == _bbox_center(trace["render_map"]["cell_bboxes_px"][execution["annotation_cell_ids"][0]])
    assert execution["annotation_entity_ids"] == execution["annotation_cell_ids"]


def _assert_bbox_annotation_matches_shape_option(trace: dict, answer: str, annotation: list[float]) -> None:
    """Assert scalar bbox annotation marks the selected panel answer choice."""

    execution = trace["execution_trace"]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == annotation
    assert trace["projected_annotation"]["pixel_bbox"] == annotation
    assert trace["render_map"]["shape_option_bboxes_px"][str(answer)] == annotation
    assert execution["annotation_cell_ids"] == []
    assert execution["annotation_entity_ids"] == [f"shape_option_{str(answer)}"]


def test_games_battleship_fleet_uses_five_ship_scene() -> None:
    assert [shape.shape_id for shape in FLEET_SHAPES] == [
        "line5",
        "line4",
        "line3",
        "square4",
        "elbow3",
    ]
    assert "line2" not in {shape.shape_id for shape in FLEET_SHAPES}
    assert set(SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS) == {shape.shape_id for shape in FLEET_SHAPES}


def test_games_battleship_sunk_ship_count_emits_expected_contract() -> None:
    out = GamesBattleshipShipStatusCountTask().generate(
        74101,
        params={"target_answer": 3, "board_size": 9, "style_variant": "navy", "query_id": "sunk_ship_count"},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "bbox_set_map"
    assert out.query_id == "sunk_ship_count"
    assert out.scene_id == "battleship"
    assert trace["query_spec"]["query_id"] == "sunk_ship_count"
    assert trace["query_spec"]["params"]["query_id"] == "sunk_ship_count"
    assert execution["query_id"] == "sunk_ship_count"
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["font_family"] == trace["render_spec"]["text_style"]["font_family"]
    _assert_hit_marker_bbox_map_annotation(trace, out.annotation_gt.value)
    assert "Battleship tracking grid" in out.prompt


def test_games_battleship_sunk_ship_count_places_each_ship_once_and_counts_sunk_ships() -> None:
    out = GamesBattleshipShipStatusCountTask().generate(
        74111,
        params={"target_answer": 4, "board_size": 10, "style_variant": "paper", "query_id": "sunk_ship_count"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    hit_coords = set(_coords(execution["hit_coords"]))
    miss_coords = set(_coords(execution["miss_coords"]))
    annotation_coords = set(_coords(execution["annotation_coords"]))
    ships = execution["ship_placements"]
    sunk_ships = [ship for ship in ships if bool(ship["is_sunk"])]
    ship_cells = {coord for ship in ships for coord in _coords(ship["coords"])}

    assert len(ships) == len(FLEET_SHAPES)
    assert {ship["shape_id"] for ship in ships} == {shape.shape_id for shape in FLEET_SHAPES}
    assert hit_coords <= ship_cells
    assert not (miss_coords & ship_cells)
    assert len(sunk_ships) == int(out.answer_gt.value) == 4
    assert annotation_coords == {
        coord
        for ship in sunk_ships
        for coord in _coords(ship["hit_coords"])
    }
    assert set(execution["annotation_ship_ids"]) == {str(ship["ship_id"]) for ship in sunk_ships}
    assert execution["annotation_entity_ids"] == execution["annotation_ship_ids"]
    assert len(out.annotation_gt.value) == len(sunk_ships)
    assert set(out.annotation_gt.value.keys()) == {str(ship["display_name"]) for ship in sunk_ships}
    for ship in ships:
        coords = _coords(ship["coords"])
        ship_hits = set(_coords(ship["hit_coords"]))
        matches = matching_fleet_shape_ids(coords)
        assert str(ship["shape_id"]) in matches
        assert ship_hits <= set(coords)
        assert bool(ship["is_sunk"]) == (ship_hits == set(coords))


def test_games_battleship_partial_ship_count_emits_expected_contract() -> None:
    out = GamesBattleshipShipStatusCountTask().generate(
        74131,
        params={"target_answer": 3, "board_size": 9, "style_variant": "classic", "query_id": "partial_ship_count"},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "bbox_set_map"
    assert out.query_id == "partial_ship_count"
    assert out.scene_id == "battleship"
    assert trace["query_spec"]["query_id"] == "partial_ship_count"
    assert trace["query_spec"]["params"]["query_id"] == "partial_ship_count"
    assert execution["query_id"] == "partial_ship_count"
    assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["font_family"] == trace["render_spec"]["text_style"]["font_family"]
    _assert_hit_marker_bbox_map_annotation(trace, out.annotation_gt.value)


def test_games_battleship_partial_ship_count_places_each_ship_once_and_counts_partial_ships() -> None:
    out = GamesBattleshipShipStatusCountTask().generate(
        74141,
        params={"target_answer": 4, "board_size": 10, "style_variant": "outlined", "query_id": "partial_ship_count"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    hit_coords = set(_coords(execution["hit_coords"]))
    miss_coords = set(_coords(execution["miss_coords"]))
    annotation_coords = set(_coords(execution["annotation_coords"]))
    ships = execution["ship_placements"]
    partial_ships = [
        ship
        for ship in ships
        if bool(ship["hit_coords"]) and not bool(ship["is_sunk"])
    ]
    ship_cells = {coord for ship in ships for coord in _coords(ship["coords"])}

    assert len(ships) == len(FLEET_SHAPES)
    assert {ship["shape_id"] for ship in ships} == {shape.shape_id for shape in FLEET_SHAPES}
    assert hit_coords <= ship_cells
    assert not (miss_coords & ship_cells)
    assert len(partial_ships) == int(out.answer_gt.value) == 4
    assert annotation_coords == {
        coord
        for ship in partial_ships
        for coord in _coords(ship["hit_coords"])
    }
    assert set(execution["annotation_ship_ids"]) == {str(ship["ship_id"]) for ship in partial_ships}
    assert execution["annotation_entity_ids"] == execution["annotation_ship_ids"]
    assert len(out.annotation_gt.value) == len(partial_ships)
    assert set(out.annotation_gt.value.keys()) == {str(ship["display_name"]) for ship in partial_ships}
    for ship in ships:
        coords = _coords(ship["coords"])
        ship_hits = set(_coords(ship["hit_coords"]))
        matches = matching_fleet_shape_ids(coords)
        assert str(ship["shape_id"]) in matches
        assert ship_hits <= set(coords)
        assert bool(ship["is_sunk"]) == (ship_hits == set(coords))


def test_games_battleship_named_hit_cell_count_emits_expected_contract() -> None:
    out = GamesBattleshipShipCellStatusCountTask().generate(
        74701,
        params={
            "query_id": "named_ship_hit_cell_count",
            "target_ship_id": "line5",
            "target_answer": 3,
            "board_size": 9,
        },
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    target_ship = next(ship for ship in execution["ship_placements"] if ship["ship_id"] == "line5")

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == "named_ship_hit_cell_count"
    assert out.scene_id == "battleship"
    assert trace["query_spec"]["query_id"] == "named_ship_hit_cell_count"
    assert trace["query_spec"]["params"]["query_id"] == "named_ship_hit_cell_count"
    assert execution["query_id"] == "named_ship_hit_cell_count"
    assert execution["target_ship_id"] == "line5"
    assert execution["target_ship_display_name"] == "Line 5"
    assert execution["target_ship_shape_id"] == "line5"
    assert execution["target_cell_status"] == "hit"
    assert execution["target_ship_cell_ids"] == [f"r{row}_c{col}" for row, col in _coords(target_ship["coords"])]
    assert set(_coords(execution["annotation_coords"])) == set(_coords(target_ship["hit_coords"]))
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    _assert_bbox_annotation_matches_cells(trace, out.annotation_gt.value)


def test_games_battleship_named_unhit_cell_count_emits_expected_contract() -> None:
    out = GamesBattleshipShipCellStatusCountTask().generate(
        74711,
        params={
            "query_id": "named_ship_unhit_cell_count",
            "target_ship_id": "square4",
            "target_answer": 2,
            "board_size": 10,
        },
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    target_ship = next(ship for ship in execution["ship_placements"] if ship["ship_id"] == "square4")
    target_coords = set(_coords(target_ship["coords"]))
    hit_coords = set(_coords(target_ship["hit_coords"]))

    assert int(out.answer_gt.value) == 2
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == "named_ship_unhit_cell_count"
    assert execution["target_ship_id"] == "square4"
    assert execution["target_ship_display_name"] == "Square 2x2"
    assert execution["target_cell_status"] == "unhit"
    assert set(_coords(execution["annotation_coords"])) == target_coords - hit_coords
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    _assert_bbox_annotation_matches_cells(trace, out.annotation_gt.value)


def test_games_battleship_grid_query_cycle_covers_answer_board_and_style_support() -> None:
    tasks = (
        GamesBattleshipShipStatusCountTask(),
        GamesBattleshipShipCellStatusCountTask(),
        GamesBattleshipLastShipCellLabelTask(),
        GamesBattleshipRemainingShipShapeLabelTask(),
    )
    query_ids: set[str] = set()
    boards: set[int] = set()
    styles: set[str] = set()

    for task_index, task in enumerate(tasks):
        for sampling_index in range(96):
            out = task.generate(
                74201 + (1000 * int(task_index)) + int(sampling_index),
                params={},
                max_attempts=128,
            )
            execution = out.trace_payload["execution_trace"]
            query_ids.add(str(out.query_id))
            boards.add(int(execution["board_size"]))
            styles.add(str(execution["style_variant"]))

    assert query_ids == {
        "partial_ship_count",
        "sunk_ship_count",
        "named_ship_hit_cell_count",
        "named_ship_unhit_cell_count",
        "single",
    }
    assert boards == {8, 9, 10}
    assert styles == set(SUPPORTED_BATTLESHIP_STYLE_VARIANTS)

    for query_id in ("partial_ship_count", "sunk_ship_count"):
        for target_answer in (0, 1, 2, 3, 4, 5):
            out = GamesBattleshipShipStatusCountTask().generate(
                74400 + (10 * target_answer),
                params={"query_id": query_id, "target_answer": target_answer},
                max_attempts=128,
            )
            assert int(out.answer_gt.value) == target_answer

    for query_id in SHIP_CELL_STATUS_COUNT_QUERY_IDS:
        for target_ship_id in SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS:
            ship_size = len(fleet_shape_by_id()[str(target_ship_id)].offsets)
            for target_answer in (0, ship_size):
                out = GamesBattleshipShipCellStatusCountTask().generate(
                    74790 + ship_size + target_answer,
                    params={
                        "query_id": query_id,
                        "target_ship_id": target_ship_id,
                        "target_answer": target_answer,
                    },
                    max_attempts=128,
                )
                assert int(out.answer_gt.value) == target_answer


def test_games_battleship_zero_count_uses_empty_annotation() -> None:
    task = GamesBattleshipShipStatusCountTask()
    for query_id in ("partial_ship_count", "sunk_ship_count"):
        out = task.generate(
            74600,
            params={"query_id": query_id, "target_answer": 0},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]

        assert int(out.answer_gt.value) == 0
        assert out.annotation_gt.type == "bbox_set_map"
        assert out.annotation_gt.value == {}
        assert out.trace_payload["projected_annotation"]["bbox_set_map"] == {}
        assert out.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == {}
        assert execution["annotation_entity_ids"] == []
        assert execution["annotation_ship_ids"] == []
        assert execution["annotation_coords"] == []


def test_games_battleship_named_cell_zero_count_uses_empty_bbox_set() -> None:
    task = GamesBattleshipShipCellStatusCountTask()
    cases = (
        ("named_ship_hit_cell_count", "line4", 0),
        ("named_ship_unhit_cell_count", "line4", 0),
    )
    for query_id, target_ship_id, target_answer in cases:
        out = task.generate(
            74731,
            params={
                "query_id": query_id,
                "target_ship_id": target_ship_id,
                "target_answer": target_answer,
            },
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]

        assert int(out.answer_gt.value) == 0
        assert out.annotation_gt.type == "bbox_set"
        assert out.annotation_gt.value == []
        assert out.trace_payload["projected_annotation"]["bbox_set"] == []
        assert out.trace_payload["projected_annotation"]["pixel_bbox_set"] == []
        assert execution["annotation_entity_ids"] == []
        assert execution["annotation_cell_ids"] == []
        assert execution["annotation_coords"] == []
        assert execution["target_ship_id"] == target_ship_id


def test_games_battleship_named_cell_status_samples_supported_target_ships() -> None:
    task = GamesBattleshipShipCellStatusCountTask()
    query_ids: set[str] = set()
    target_ship_ids: set[str] = set()

    for sampling_index in range(160):
        out = task.generate(
            74801 + int(sampling_index),
            params={},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        query_id = str(out.query_id)
        target_ship_id = str(execution["target_ship_id"])
        query_ids.add(query_id)
        target_ship_ids.add(target_ship_id)
        ship_size = len(fleet_shape_by_id()[target_ship_id].offsets)

        assert query_id in SHIP_CELL_STATUS_COUNT_QUERY_IDS
        assert target_ship_id in SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS
        assert 0 <= int(out.answer_gt.value) <= int(ship_size)
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)

    assert query_ids == set(SHIP_CELL_STATUS_COUNT_QUERY_IDS)
    assert target_ship_ids == set(SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS)


def test_games_battleship_grid_is_deterministic() -> None:
    params = {"query_id": "sunk_ship_count", "target_answer": 2, "board_size": 8, "style_variant": "radar"}
    task = GamesBattleshipShipStatusCountTask()
    out_a = task.generate(74121, params=params, max_attempts=128)
    out_b = task.generate(74121, params=params, max_attempts=128)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]


def test_games_battleship_last_ship_cell_label_emits_expected_contract() -> None:
    out = GamesBattleshipLastShipCellLabelTask().generate(
        74901,
        params={"target_ship_id": "line3", "board_size": 9, "style_variant": "radar"},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    answer = str(out.answer_gt.value)
    answer_option = next(option for option in execution["candidate_options"] if bool(option["is_answer"]))
    target_ship = next(ship for ship in execution["ship_placements"] if ship["ship_id"] == execution["target_ship_id"])
    target_hits = set(_coords(target_ship["hit_coords"]))
    target_coords = set(_coords(target_ship["coords"]))
    missing_coord = tuple(int(value) for value in execution["target_missing_coord"])

    assert out.answer_gt.type == "string"
    assert answer in {"A", "B", "C", "D", "E", "F"}
    assert out.annotation_gt.type == "point"
    assert out.query_id == "single"
    assert out.scene_id == "battleship"
    assert trace["query_spec"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["render_map"]["show_ship_bodies"] is False
    assert answer_option["label"] == answer
    assert tuple(answer_option["coord"]) == missing_coord
    assert target_coords - target_hits == {missing_coord}
    assert set(_coords(execution["annotation_coords"])) == {missing_coord}
    _assert_point_annotation_matches_cell(trace, out.annotation_gt.value)
    assert trace["render_map"]["candidate_label_cell_ids"][answer] == execution["annotation_cell_ids"][0]
    assert trace["render_map"]["candidate_label_points_px"][answer] == out.annotation_gt.value
    assert all(
        bool(ship["is_sunk"])
        for ship in execution["ship_placements"]
        if ship["ship_id"] != execution["target_ship_id"]
    )
    valid_labels = []
    for option in execution["candidate_options"]:
        completed = target_hits | {tuple(option["coord"])}
        if str(target_ship["shape_id"]) in matching_fleet_shape_ids(tuple(completed)):
            valid_labels.append(str(option["label"]))
    assert valid_labels == [answer]


def test_games_battleship_remaining_ship_shape_label_emits_expected_contract() -> None:
    out = GamesBattleshipRemainingShipShapeLabelTask().generate(
        74951,
        params={"target_ship_id": "square4", "target_answer": 2, "board_size": 9, "style_variant": "paper"},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    answer = str(out.answer_gt.value)
    options = execution["shape_options"]
    answer_option = next(option for option in options if bool(option["is_answer"]))
    ships = execution["ship_placements"]
    untouched_ships = [ship for ship in ships if not bool(ship["hit_coords"])]
    sunk_ships = [ship for ship in ships if bool(ship["is_sunk"])]

    assert out.answer_gt.type == "string"
    assert answer in {"A", "B", "C", "D", "E"}
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert out.scene_id == "battleship"
    assert trace["query_spec"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["render_map"]["show_ship_bodies"] is False
    assert trace["render_map"]["candidate_label_cell_ids"] == {}
    assert set(trace["render_map"]["shape_option_bboxes_px"].keys()) == {"A", "B", "C", "D", "E"}
    assert len(options) == 5
    assert [option["label"] for option in options] == ["A", "B", "C", "D", "E"]
    assert answer_option["label"] == answer
    assert answer_option["shape_id"] == "square4"
    assert execution["target_ship_id"] == "square4"
    assert execution["target_ship_display_name"] == "Square 2x2"
    assert execution["target_cell_status"] == "untouched"
    assert execution["target_answer"] == 2
    assert len(untouched_ships) == 1
    assert untouched_ships[0]["ship_id"] == "square4"
    assert len(sunk_ships) == len(FLEET_SHAPES) - 1
    assert execution["fleet_sunk_total"] == len(FLEET_SHAPES) - 1
    assert execution["fleet_partial_total"] == 0
    assert execution["fleet_untouched_total"] == 1
    assert set(execution["target_ship_cell_ids"]) == {f"r{row}_c{col}" for row, col in _coords(untouched_ships[0]["coords"])}
    assert execution["annotation_ship_ids"] == ["square4"]
    _assert_bbox_annotation_matches_shape_option(trace, answer, out.annotation_gt.value)


def test_games_battleship_grid_build_dataset_smoke(tmp_path) -> None:
    output_root = tmp_path / "task_games__battleship__ship_status_count"
    cfg = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games_battleship_grid",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__battleship__ship_status_count",
                count=2,
                params={"board_size": 9, "target_answer": 2},
            ),
        ],
        max_attempts_per_instance=128,
        workers=1,
    )
    final_path = build_dataset(cfg, code_hash="games-battleship-grid-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["task"] == "task_games__battleship__ship_status_count" for row in rows)
    assert all(row["answer_gt"]["type"] == "integer" for row in rows)
    assert all(row["annotation_gt"]["type"] == "bbox_set_map" for row in rows)


def test_games_battleship_named_cell_status_build_dataset_smoke(tmp_path) -> None:
    output_root = tmp_path / "task_games__battleship__ship_cell_status_count"
    cfg = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games_battleship_cell_status",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__battleship__ship_cell_status_count",
                count=2,
                params={"board_size": 9, "target_ship_id": "line5", "target_answer": 2},
            ),
        ],
        max_attempts_per_instance=128,
        workers=1,
    )
    final_path = build_dataset(cfg, code_hash="games-battleship-cell-status-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["task"] == "task_games__battleship__ship_cell_status_count" for row in rows)
    assert all(row["answer_gt"]["type"] == "integer" for row in rows)
    assert all(row["annotation_gt"]["type"] == "bbox_set" for row in rows)
