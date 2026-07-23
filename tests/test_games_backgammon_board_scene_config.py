"""Config regression tests for games Backgammon defaults."""

from __future__ import annotations

from trace_tasks.core.prompts import load_scene_prompt_bundle
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.backgammon.destination_count import (
    BLOCKED_COUNT_SUPPORT,
    HIT_COUNT_SUPPORT,
    LEGAL_COUNT_SUPPORT,
)
from trace_tasks.tasks.games.backgammon.pip_count_value import PIP_COUNT_SUPPORT
from trace_tasks.tasks.games.backgammon.point_state_count import POINT_STATE_COUNT_SUPPORT
from trace_tasks.tasks.games.backgammon.shared.state import SUPPORTED_BACKGAMMON_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_backgammon_defaults_present() -> None:
    cfg = get_scene_defaults("games", "backgammon")
    destination_generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__backgammon__destination_count",
    )
    point_generation, _point_rendering, point_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__backgammon__point_state_count",
    )
    pip_generation, _pip_rendering, pip_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__backgammon__pip_count_value",
    )

    assert bool(destination_generation["balanced_scene_variant_sampling"]) is True
    assert "balanced_query_id_sampling" not in destination_generation
    assert bool(destination_generation["balanced_style_variant_sampling"]) is True
    assert bool(destination_generation["balanced_active_player_sampling"]) is True
    assert bool(destination_generation["balanced_target_answer_sampling"]) is True
    assert set(destination_generation["scene_variant_weights"].keys()) == {"standard_board"}
    assert "query_id_weights" not in destination_generation
    assert set(destination_generation["style_variant_weights"].keys()) == set(SUPPORTED_BACKGAMMON_STYLE_VARIANTS)
    assert set(destination_generation["active_player_weights"].keys()) == {"black", "white"}
    assert list(destination_generation["legal_count_support"]) == list(LEGAL_COUNT_SUPPORT)
    assert list(destination_generation["hit_count_support"]) == list(HIT_COUNT_SUPPORT)
    assert list(destination_generation["blocked_count_support"]) == list(BLOCKED_COUNT_SUPPORT)
    assert "query_id_weights" not in point_generation
    assert "balanced_query_id_sampling" not in point_generation
    assert list(point_generation["point_state_count_support"]) == list(POINT_STATE_COUNT_SUPPORT)
    assert "query_id_weights" not in pip_generation
    assert "balanced_query_id_sampling" not in pip_generation
    assert list(pip_generation["pip_count_support"]) == list(PIP_COUNT_SUPPORT)
    assert int(rendering["canvas_width"]) == 1000
    assert int(rendering["canvas_height"]) == 720
    assert float(rendering["unit_size_scale_min"]) == 0.5
    assert float(rendering["unit_size_scale_max"]) == 1.0
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert int(rendering["checker_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_backgammon_v1"
    assert str(point_prompt["bundle_id"]) == "games_backgammon_v1"
    assert str(pip_prompt["bundle_id"]) == "games_backgammon_v1"
    assert {"bundle_id", "scene_key", "task_key"}.issubset(set(prompt))
    assert not any(key.startswith(("annotation_hint", "answer_hint", "json_example")) for key in prompt)
    bundle = load_scene_prompt_bundle("games", "backgammon", "games_backgammon_v1")
    assert bundle.schema_version == "v1"
    assert bundle.source_hash
