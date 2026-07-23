"""Config regression tests for games Rhythm-lanes defaults."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.rhythm.shared.state import SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_rhythm_lanes_defaults_present() -> None:
    cfg = get_scene_defaults("games", "rhythm")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__rhythm__lane_note_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_lane_count_sampling"]) is True
    assert bool(generation["balanced_row_count_sampling"]) is True
    assert bool(generation["balanced_beat_window_sampling"]) is True
    assert bool(generation["balanced_note_count_sampling"]) is True
    assert bool(generation["balanced_score_total_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == set(SUPPORTED_SCENE_VARIANTS)
    assert "query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_STYLE_VARIANTS)
    assert list(generation["lane_count_support"]) == [5, 6, 7, 8]
    assert list(generation["row_count_support"]) == [10, 11, 12, 13, 14]
    assert list(generation["beat_window_support"]) == [5, 6, 7]
    assert list(generation["note_count_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(generation["score_total_support"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert int(rendering["canvas_width"]) == 900
    assert int(rendering["canvas_height"]) == 900
    assert str(prompt["bundle_id"]) == "games_rhythm_v1"

    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/rhythm/games_rhythm_v1.json").read_text(encoding="utf-8"))
    prompt_defaults = bundle["code_prompt_defaults"]
    assert "one row per beat" in str(prompt_defaults["rhythm_motion_rule_text"])
    assert "bounding boxes" in str(prompt_defaults["annotation_hint_lane_note_count"])
    assert "POINTS palette" in str(prompt_defaults["score_palette_rule_text"])
    assert "bounding box [x0, y0, x1, y1]" in str(prompt_defaults["annotation_hint_earliest_hit_lane_label"])
