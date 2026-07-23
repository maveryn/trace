"""Config regression tests for games Battleship-grid defaults."""

from __future__ import annotations

from trace_tasks.core.prompts import load_scene_prompt_bundle
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.battleship.last_ship_cell_label import LAST_SHIP_CELL_OPTION_COUNT_SUPPORT
from trace_tasks.tasks.games.battleship.remaining_ship_shape_label import REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT
from trace_tasks.tasks.games.battleship.ship_status_count import PARTIAL_SHIP_COUNT_SUPPORT, SUNK_SHIP_COUNT_SUPPORT
from trace_tasks.tasks.games.shared.style import SUPPORTED_BATTLESHIP_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_battleship_grid_defaults_expose_scene_target_board_and_style_axes() -> None:
    cfg = get_scene_defaults("games", "battleship")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__battleship__ship_status_count",
    )
    cell_generation, _cell_rendering, cell_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__battleship__ship_cell_status_count",
    )
    last_generation, _last_rendering, last_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__battleship__last_ship_cell_label",
    )
    remaining_generation, _remaining_rendering, remaining_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__battleship__remaining_ship_shape_label",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_board_size_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"standard_fleet"}
    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert "target_ship_id_weights" not in generation
    assert "balanced_target_ship_id_sampling" not in generation
    assert bool(cell_generation["balanced_target_ship_answer_pair_sampling"]) is True
    assert bool(cell_generation["balanced_target_ship_id_sampling"]) is True
    assert bool(last_generation["balanced_target_ship_id_sampling"]) is True
    assert bool(remaining_generation["balanced_target_ship_id_sampling"]) is True
    assert bool(remaining_generation["balanced_target_answer_sampling"]) is True
    assert set(cell_generation["target_ship_id_weights"].keys()) == {
        "line5",
        "line4",
        "line3",
        "square4",
        "elbow3",
    }
    assert set(last_generation["target_ship_id_weights"].keys()) == set(cell_generation["target_ship_id_weights"].keys())
    assert set(remaining_generation["target_ship_id_weights"].keys()) == set(cell_generation["target_ship_id_weights"].keys())
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_BATTLESHIP_STYLE_VARIANTS)
    assert list(generation["board_size_support"]) == [8, 9, 10]
    assert list(generation["sunk_ship_count_support"]) == list(SUNK_SHIP_COUNT_SUPPORT)
    assert list(generation["partial_ship_count_support"]) == list(PARTIAL_SHIP_COUNT_SUPPORT)
    assert list(last_generation["last_ship_cell_option_count_support"]) == list(LAST_SHIP_CELL_OPTION_COUNT_SUPPORT)
    assert list(remaining_generation["remaining_ship_shape_label_index_support"]) == list(REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT)
    assert int(rendering["canvas_width"]) == 1100
    assert int(rendering["canvas_height"]) == 820
    assert int(rendering["fleet_panel_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_battleship_v1"
    assert str(cell_prompt["bundle_id"]) == "games_battleship_v1"
    assert str(last_prompt["bundle_id"]) == "games_battleship_v1"
    assert str(remaining_prompt["bundle_id"]) == "games_battleship_v1"
    assert {"bundle_id", "scene_key", "task_key"}.issubset(set(prompt))
    assert not any(key.startswith(("annotation_hint", "answer_hint", "json_example")) for key in prompt)
    bundle = load_scene_prompt_bundle("games", "battleship", "games_battleship_v1")
    assert bundle.schema_version == "v1"
    assert bundle.source_hash
