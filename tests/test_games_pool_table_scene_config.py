"""Config regression tests for games Pool-table scene defaults."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.tasks.games.shared.style import SUPPORTED_POOL_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


def test_games_pool_table_defaults_present() -> None:
    generation, rendering, prompt = load_scene_generation_rendering_prompt_defaults(
        "games",
        "pool",
        task_id="task_games__pool__group_ball_count",
    )
    blocking_generation, _blocking_rendering, _blocking_prompt = load_scene_generation_rendering_prompt_defaults(
        "games",
        "pool",
        task_id="task_games__pool__blocking_ball_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_object_ball_count_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"standard_table"}
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_POOL_STYLE_VARIANTS)
    assert list(generation["object_ball_count_support"]) == [7, 8, 9, 10]
    assert list(generation["current_group_ball_count_support"]) == [2, 3, 4, 5, 6]
    assert list(blocking_generation["blocking_ball_count_support"]) == [0, 1, 2, 3, 4]
    assert int(rendering["canvas_width"]) == 1120
    assert int(rendering["canvas_height"]) == 760
    assert int(rendering["ball_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_pool_v1"
    prompt_asset = json.loads(Path("src/trace_tasks/resources/prompts/games/pool/games_pool_v1.json").read_text(encoding="utf-8"))
    code_defaults = prompt_asset["code_prompt_defaults"]
    assert "two straight segments" in str(code_defaults["marked_shot_rule_text"])
    assert "current player" in str(code_defaults["answer_hint_current_group_ball_count"])
    assert "pixel point" in str(code_defaults["annotation_hint_blocking_ball_count"])
