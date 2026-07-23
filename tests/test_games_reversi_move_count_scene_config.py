"""Config regression tests for games Reversi defaults."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.style import SUPPORTED_REVERSI_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)


def test_games_reversi_move_count_defaults_expose_scene_query_and_answer_axes() -> None:
    cfg = get_scene_defaults("games", "reversi")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__reversi__legal_destination_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {
        "compact_board",
        "classic_board",
    }
    assert set(generation["style_variant_weights"].keys()) == set(
        SUPPORTED_REVERSI_STYLE_VARIANTS
    )
    assert list(generation["legal_move_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(generation["flip_count_support"]) == [2, 3, 4, 5, 6]
    assert list(generation["frontier_disc_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert int(rendering["max_board_size_px"]) > 0
    assert int(rendering["player_badge_height_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_reversi_v1"
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/reversi/games_reversi_v1.json").read_text(encoding="utf-8")
    )
    code_defaults = bundle["code_prompt_defaults"]
    assert "6 by 6" in str(code_defaults["object_description_compact_board"])
    assert (
        "pixel point"
        in str(code_defaults["annotation_hint_flip_count_for_marked_move"]).lower()
    )
    assert (
        "flip" in str(code_defaults["answer_hint_flip_count_for_marked_move"]).lower()
    )
    assert "frontier" in str(code_defaults["frontier_rule_text"]).lower()
    assert "frontier" in str(code_defaults["answer_hint_frontier_disc_count"]).lower()
    assert "{query_player}" in str(code_defaults["answer_hint_frontier_disc_count"])
