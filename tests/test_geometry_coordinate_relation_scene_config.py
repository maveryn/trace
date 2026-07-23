"""Config regression tests for geometry coordinate-plane defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_geometry_coordinate_plane_defaults_expose_task_generation_knobs_without_query_routing() -> None:
    cfg = get_scene_defaults("geometry", "coordinate_plane")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_geometry__coordinate_plane__segment_relation_count",
    )

    assert "balanced_scene_variant_sampling" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert "scene_variant_weights" not in generation
    assert "query_id_weights" not in generation
    assert int(generation["segment_candidate_count"]) == 6
    assert int(generation["segment_endpoint_abs_max"]) == 8
    assert list(generation["segment_target_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert int(generation["collinear_candidate_count"]) == 8
    assert int(generation["collinear_point_abs_max"]) == 8
    assert list(generation["collinear_target_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(generation["same_quadrant_target_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(generation["point_in_shape_target_support"]) == [1, 2, 3, 4, 5, 6]
    assert str(prompt["bundle_id"]) == "geometry_coordinate_v1"
    assert str(prompt["scene_key"]) == "coordinate_graph_scene"
    assert str(prompt["task_key"]) == "coordinate_query"
    assert int(rendering["line_width"]) > 0
