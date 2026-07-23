"""Regression tests for scene default config loading."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_charts_table_defaults_loaded_for_public_tasks() -> None:
    cfg = get_scene_defaults("charts", "table")
    for section in ("generation", "rendering", "prompt"):
        assert isinstance(cfg.get(section), dict)

    generation_shared = cfg["generation"]["shared"]
    assert int(generation_shared["row_count_min"]) >= 5
    assert int(generation_shared["row_count_max"]) == 10
    assert int(generation_shared["numeric_column_count_min"]) >= 3
    assert int(generation_shared["numeric_column_count_max"]) == 5
    assert sorted(generation_shared["scene_variant_weights"]) == [
        "card_table",
        "ledger",
        "spreadsheet",
        "zebra",
    ]
    assert "query_id_weights" not in generation_shared

    rendering_shared = cfg["rendering"]["shared"]
    assert int(rendering_shared["canvas_width"]) == 940
    assert int(rendering_shared["canvas_height"]) == 640
    assert int(rendering_shared["table_margin_left_px"]) > 0
    assert int(rendering_shared["table_margin_bottom_px"]) > 0

    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_charts__table__absolute_difference_between_rows_over_year_interval",
    )
    assert int(generation_defaults["numeric_column_count_min"]) == 8
    assert int(generation_defaults["numeric_column_count_max"]) == 12
    assert int(generation_defaults["interval_length_min"]) == 4
    assert int(generation_defaults["interval_length_max"]) == 5
    assert int(rendering_defaults["canvas_width"]) == 1360
    assert int(rendering_defaults["canvas_height"]) == 872
    assert str(prompt_defaults["bundle_id"]) == "charts_table_temporal_v1"

    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_charts__table__filtered_column_mean",
    )
    assert int(generation_defaults["selected_row_count_min"]) == 6
    assert int(generation_defaults["selected_row_count_max"]) == 7
    assert "query_id_weights" not in generation_defaults
    assert int(rendering_defaults["canvas_height"]) == 900
    assert str(prompt_defaults["bundle_id"]) == "charts_table_statistics_v1"
