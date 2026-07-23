"""Contract tests for games match-3 tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.games.match3.shared.defaults import GEM_KEYS
from trace_tasks.tasks.games.match3.shared.rules import (
    cell_entity_id,
    external_same_color_neighbors_for_clear,
    simulate_move,
)
from trace_tasks.tasks.games.match3.shared.state import SwapMove
from trace_tasks.tasks.shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.named_colors import named_color


def test_games_match3_defaults_expose_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "match3")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
    )

    assert set(generation["scene_variant_weights"].keys()) == {
        "square_board",
        "wide_board",
        "tall_board",
    }
    assert set(generation["style_variant_weights"].keys()) == {
        "faceted_jewels",
        "round_candies",
        "beveled_tiles",
        "diamond_gems",
        "orb_tokens",
    }
    assert list(generation["option_count_support"]) == [4]
    assert list(generation["gem_count_answer_support"]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert int(rendering["canvas_width"]) == 760
    assert int(rendering["canvas_height"]) == 720
    assert (
        float(rendering["unit_size_scale_max"])
        / float(rendering["unit_size_scale_min"])
        >= 2.0
    )
    assert str(prompt["bundle_id"]) == "games_match3_v1"


def test_games_match3_palette_uses_high_contrast_colors() -> None:
    assert GEM_KEYS == ("red", "blue", "green", "yellow", "purple", "cyan")
    assert "maroon" not in GEM_KEYS
    assert "magenta" not in GEM_KEYS


def test_games_match3_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/match3/games_match3_v1.json").read_text(encoding="utf-8")
    )
    assert bundle["schema_version"] == "v1"
    assert set(bundle["templates"]["query"].keys()) == {
        "column_color_gem_count",
        "grid_color_gem_count",
        "max_clear_swap_label",
        "row_color_gem_count",
        "swap_clear_count",
    }
    assert (
        "target_color_label"
        in bundle["required_slots_by_key"]["query:grid_color_gem_count"]
    )
    assert "cascades" in str(bundle["code_prompt_defaults"]["match3_rule_text"])


def test_games_match3_swap_tasks_use_easier_task_overrides() -> None:
    cfg = get_scene_defaults("games", "match3")
    generation, rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__match3__max_clear_swap_label",
    )

    assert list(generation["row_count_support"]) == [5]
    assert list(generation["col_count_support"]) == [5]
    assert list(generation["gem_type_count_support"]) == [5]
    assert list(generation["option_count_support"]) == [4]
    assert int(rendering["arrow_width_px"]) == 9

    clear_generation, _clear_rendering, _prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_games__match3__swap_clear_count",
        )
    )
    assert list(clear_generation["row_count_support"]) == [5]
    assert list(clear_generation["col_count_support"]) == [5]
    assert list(clear_generation["gem_type_count_support"]) == [5]
    assert list(clear_generation["swap_clear_count_answer_support"]) == [
        0,
        3,
        4,
        5,
        6,
    ]


def test_games_match3_max_clear_label_has_unique_answer() -> None:
    out = create_task("task_games__match3__max_clear_swap_label").generate(
        71231,
        params={"query_id": "single", "option_count": 6},
        max_attempts=300,
    )
    options = out.trace_payload["execution_trace"]["swap_options"]
    max_clear = max(int(option["clear_count"]) for option in options)
    answers = [
        str(option["label"])
        for option in options
        if int(option["clear_count"]) == int(max_clear)
    ]

    assert out.answer_gt.type == "option_letter"
    assert answers == [str(out.answer_gt.value)]
    assert out.scene_id == "match3"
    assert out.query_id == "single"
    assert (
        out.trace_payload["query_spec"]["params"]["prompt_query_key"]
        == "max_clear_swap_label"
    )
    assert len(options) == 4
    assert out.trace_payload["query_spec"]["params"]["option_count"] == 4
    assert out.annotation_gt.type == "point"
    assert out.trace_payload["projected_annotation"]["point"] == out.annotation_gt.value


def test_games_match3_swap_clear_count_annotates_cleared_cells() -> None:
    out = create_task("task_games__match3__swap_clear_count").generate(
        71321,
        params={"target_answer": 6},
        max_attempts=800,
    )
    params = out.trace_payload["query_spec"]["params"]
    options = out.trace_payload["execution_trace"]["swap_options"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 6
    assert out.scene_id == "match3"
    assert out.query_id == "single"
    assert params["prompt_query_key"] == "swap_clear_count"
    assert params["answer_support"] == [0, 3, 4, 5, 6]
    assert len(options) == 1
    assert options[0]["label"] == "A"
    assert int(options[0]["clear_count"]) == int(out.answer_gt.value)
    assert int(params["marked_swap_clear_count"]) == int(out.answer_gt.value)
    assert out.annotation_gt.type == "bbox_set"
    assert out.trace_payload["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert out.trace_payload["execution_trace"]["annotation_entity_ids"] == [
        cell_entity_id((int(row) - 1, int(col) - 1))
        for row, col in options[0]["cleared_cells"]
    ]


def test_games_match3_swap_clear_count_rejects_connected_branch_ambiguity() -> None:
    board = (
        ("red", "yellow", "blue", "blue", "yellow"),
        ("blue", "blue", "yellow", "blue", "blue"),
        ("blue", "yellow", "red", "red", "yellow"),
        ("red", "cyan", "blue", "cyan", "cyan"),
        ("red", "cyan", "cyan", "red", "purple"),
    )
    ambiguous = simulate_move(board, SwapMove(a=(0, 2), b=(1, 2)))

    assert int(ambiguous.clear_count) == 5
    assert external_same_color_neighbors_for_clear(board, ambiguous) == ((0, 3), (2, 0))

    out = create_task("task_games__match3__swap_clear_count").generate(
        339289371382477,
        params={"target_answer": 5},
        max_attempts=1000,
    )
    trace = out.trace_payload["execution_trace"]
    option = trace["swap_options"][0]
    generated_board = tuple(tuple(row) for row in trace["board_before"])
    generated_outcome = simulate_move(
        generated_board,
        SwapMove(
            a=(int(option["from_cell"][0]) - 1, int(option["from_cell"][1]) - 1),
            b=(int(option["to_cell"][0]) - 1, int(option["to_cell"][1]) - 1),
        ),
    )

    assert int(generated_outcome.clear_count) == int(out.answer_gt.value)
    assert external_same_color_neighbors_for_clear(generated_board, generated_outcome) == ()


def test_games_match3_swap_clear_count_zero_answer_has_empty_bbox_set() -> None:
    out = create_task("task_games__match3__swap_clear_count").generate(
        71325,
        params={"target_answer": 0},
        max_attempts=1000,
    )

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.type == "bbox_set"
    assert out.annotation_gt.value == []
    assert out.trace_payload["projected_annotation"]["bbox_set"] == []


def test_games_match3_gem_count_uses_canonical_named_colors() -> None:
    out = create_task("task_games__match3__gem_count").generate(
        71251,
        params={"query_id": "grid_color_gem_count", "target_answer": 4},
        max_attempts=500,
    )
    trace = out.trace_payload
    target_color = str(trace["query_spec"]["params"]["target_color_name"])
    target_rgb = tuple(
        int(value) for value in trace["query_spec"]["params"]["target_color_rgb"]
    )

    assert target_rgb == named_color(target_color)
    assert "[" in str(out.prompt) and "#" in str(out.prompt)
    assert out.answer_gt.type == "integer"
    assert out.scene_id == "match3"
    assert out.query_id == "grid_color_gem_count"
    assert out.annotation_gt.type == "bbox_set"

    matching = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if entity.get("entity_type") == "match3_gem"
        and entity.get("color_name") == target_color
    ]
    assert int(out.answer_gt.value) == len(matching)
    assert len(out.annotation_gt.value) == len(matching)
    for entity in matching:
        assert tuple(int(value) for value in entity["color_rgb"]) == named_color(
            str(entity["color_name"])
        )


def test_games_match3_row_and_column_gem_count_scopes() -> None:
    for query_id, scope_key in (
        ("row_color_gem_count", "row_index"),
        ("column_color_gem_count", "col_index"),
    ):
        out = create_task("task_games__match3__gem_count").generate(
            71300 + len(query_id),
            params={"query_id": query_id, "target_answer": 2},
            max_attempts=500,
        )
        trace = out.trace_payload
        params = trace["query_spec"]["params"]
        target_color = str(params["target_color_name"])
        scope_index = int(params[scope_key])
        matching = []
        for entity in trace["scene_ir"]["entities"]:
            if (
                entity.get("entity_type") != "match3_gem"
                or entity.get("color_name") != target_color
            ):
                continue
            if query_id == "row_color_gem_count" and int(entity["row"]) != scope_index:
                continue
            if (
                query_id == "column_color_gem_count"
                and int(entity["col"]) != scope_index
            ):
                continue
            matching.append(entity)
        assert int(out.answer_gt.value) == len(matching)
        assert len(out.annotation_gt.value) == len(matching)
