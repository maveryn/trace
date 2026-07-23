"""Config regression tests for games Space-shooter defaults."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.space_shooter.shared.state import (
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_space_shooter_defaults_present() -> None:
    cfg = get_scene_defaults("games", "space_shooter")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__space_shooter__enemy_ship_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_lane_count_sampling"]) is True
    assert bool(generation["balanced_enemy_count_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert "query_id_weights" not in generation
    assert set(generation["scene_variant_weights"].keys()) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_STYLE_VARIANTS)
    assert list(generation["lane_count_support"]) == [4, 5, 6, 7, 8]
    assert list(generation["enemy_count_support"]) == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert list(generation["safe_lane_count_support"]) == [1, 2, 3, 4, 5]
    assert list(generation["enemy_ship_hit_count_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(generation["hit_enemy_ship_label_option_support"]) == [0, 1, 2, 3]
    assert list(generation["first_hit_enemy_ship_label_option_support"]) == [0, 1, 2, 3]
    assert bool(generation["balanced_correct_option_sampling"]) is True
    assert list(generation["enemy_projectile_per_lane_support"]) == [1, 2, 3]
    assert int(rendering["canvas_width"]) == 1060
    assert int(rendering["canvas_height"]) == 820
    assert int(rendering["enemy_width_px"]) > 0
    assert int(rendering["projectile_width_px"]) == 24
    assert int(rendering["projectile_height_px"]) == 36
    assert "blocker_width_px" not in rendering
    assert "blocker_height_px" not in rendering
    assert set(generation["style_variant_weights"].keys()) == {
        "neon",
        "deep_space",
        "vector",
        "amber",
        "terminal",
    }
    assert str(prompt["bundle_id"]) == "games_space_shooter_v1"
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/space_shooter/games_space_shooter_v1.json").read_text(encoding="utf-8"))
    code_defaults = bundle["code_prompt_defaults"]
    assert "bottom lane pads" in str(code_defaults["space_shooter_lane_rule_text"]).lower()
    assert "shield" not in json.dumps(bundle).lower()
    assert "asteroid" not in json.dumps(bundle).lower()
    assert "visible enemy ships" in str(code_defaults["answer_hint_enemy_ship_count"]).lower()
    assert "every visible enemy ship" in str(code_defaults["annotation_hint_enemy_ship_count"]).lower()
    assert "destroyed by the current blue shots" in str(code_defaults["answer_hint_enemy_ship_hit_count"]).lower()
    assert "enemy ships that can be destroyed" in str(code_defaults["annotation_hint_enemy_ship_hit_count"]).lower()
    assert "selected enemy ship label" in str(code_defaults["answer_hint_hit_enemy_ship_label"]).lower()
    assert "labeled enemy ship selected" in str(code_defaults["annotation_hint_hit_enemy_ship_label"]).lower()
    assert "selected enemy ship label" in str(code_defaults["answer_hint_first_hit_enemy_ship_label"]).lower()
    assert "labeled enemy ship selected" in str(code_defaults["annotation_hint_first_hit_enemy_ship_label"]).lower()
    assert "bounding boxes" in str(code_defaults["annotation_hint_safe_lane_count"])
