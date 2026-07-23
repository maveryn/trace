"""Config regression tests for geometry function-graph defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_geometry_function_graph_config_exposes_generation_support_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("geometry", "function_graph")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert list(generation["piecewise_turning_support"]) == [2, 3, 4, 5, 6]
    assert list(generation["sinusoid_turning_support"]) == [3, 4]
    assert list(generation["sinusoid_local_extremum_support"]) == [2]
    assert list(generation["piecewise_local_extremum_support"]) == [2, 3, 4, 5, 6]
    assert list(generation["average_rate_support"]) == [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
    assert int(rendering["line_width"]) > 0
    assert str(prompt["bundle_id"]) == "geometry_graphing_v0"
    assert str(prompt["scene_key"]) == "graphing_scene"
    assert str(prompt["task_key"]) == "graphing_query"
