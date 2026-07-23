"""Config regression tests for games Pac-Man defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import resolve_scene_section_defaults


def test_games_pacman_defaults_expose_scene_answer_and_style_axes() -> None:
    cfg = get_scene_defaults("games", "pacman")
    before_ghost_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__pacman__pellet_count_before_ghost",
    )
    rendering = resolve_scene_section_defaults(
        cfg,
        "rendering",
        task_id="task_games__pacman__pellet_count_before_ghost",
    )
    prompt = resolve_scene_section_defaults(
        cfg,
        "prompt",
        task_id="task_games__pacman__pellet_count_before_ghost",
    )
    next_item_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__pacman__next_item_label",
    )
    score_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__pacman__route_score_value",
    )

    assert "query_id_weights" not in before_ghost_generation
    assert set(before_ghost_generation["scene_variant_weights"].keys()) == {"compact_maze", "wide_maze"}
    assert set(before_ghost_generation["style_variant_weights"].keys()) == {"classic", "neon", "paper", "terminal", "pastel"}
    assert before_ghost_generation["row_count_support"] == [7, 8, 9]
    assert before_ghost_generation["col_count_support"] == [9, 11, 13]
    assert before_ghost_generation["pellet_count_before_ghost_support"] == [1, 2, 3, 4, 5]
    assert score_generation["route_score_on_route_pellet_count_support"] == [1, 2, 3, 4]
    assert score_generation["route_score_on_route_bonus_count_support"] == [1, 2]
    assert score_generation["route_score_off_route_bonus_count_support"] == [1, 2, 3]
    assert score_generation["route_score_bonus_value_support"] == [2, 3, 4]
    assert next_item_generation["item_count_support"] == [4, 5, 6]
    assert rendering["ghost_radius_px"] == 18
    assert rendering["canvas_width"] == 980
    assert rendering["canvas_height"] == 760
    assert str(prompt["bundle_id"]) == "games_pacman_v1"
