"""Regression tests for scene default config loading."""

from __future__ import annotations
import json
import pytest
from trace_tasks.core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
    resolve_scene_section_defaults,
)
from trace_tasks.core.prompts import load_scene_prompt_bundle
from trace_tasks.tasks.shared.config_defaults import (
    required_group_default,
    required_group_defaults,
    resolve_optional_int_bounds,
    resolve_required_float_bounds,
    resolve_required_int_bounds,
    split_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS

FULL_NODE_LINK_LAYOUT_VARIANTS = set(SUPPORTED_LAYOUT_VARIANTS)


def test_icons_single_transform_options_scene_defaults_loaded() -> None:
    task_id = "task_icons__single_transform_options__geometric_transform_result_label"
    inverse_task_id = (
        "task_icons__single_transform_options__inverse_geometric_transform_source_label"
    )
    cfg = get_scene_defaults("icons", "single_transform_options")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg, task_id=task_id
    )
    assert str(generation["pool_manifest"]) == "non_symmetry.txt"
    assert int(generation["object_count_min"]) == 6
    assert int(generation["object_count_max"]) == 6
    assert "query_id_weights" not in generation
    assert int(rendering["scene_icon_size_min_px"]) == 96
    assert int(rendering["scene_icon_size_max_px"]) == 112
    assert int(rendering["reference_icon_size_px"]) == 112
    assert int(rendering["reference_icon_size_px"]) == int(
        rendering["scene_icon_size_max_px"]
    )
    assert str(prompt["bundle_id"]) == "icons_single_transform_options_v1"
    assert str(prompt["scene_key"]) == "single_transform_options_transformation"
    assert str(prompt["task_key"]) == "transformation_query"
    assert resolve_scene_section_defaults(cfg, "prompt", task_id=task_id) == prompt
    inverse_generation, inverse_rendering, inverse_prompt = (
        split_generation_rendering_prompt_defaults(cfg, task_id=inverse_task_id)
    )
    assert str(inverse_generation["pool_manifest"]) == "non_symmetry.txt"
    assert int(inverse_generation["object_count_min"]) == 4
    assert int(inverse_generation["object_count_max"]) == 4
    assert int(inverse_rendering["scene_icon_size_min_px"]) == 96
    assert int(inverse_rendering["scene_icon_size_max_px"]) == 112
    assert int(inverse_rendering["reference_icon_size_px"]) == 112
    assert str(inverse_prompt["bundle_id"]) == "icons_single_transform_options_v1"
    assert (
        str(inverse_prompt["scene_key"])
        == "single_transform_options_inverse_transformation"
    )
    assert str(inverse_prompt["task_key"]) == "transformation_query"
    bundle = load_scene_prompt_bundle(
        "icons", "single_transform_options", "icons_single_transform_options_v1"
    )
    assert bundle.bundle_id == "icons_single_transform_options_v1"
    assert bundle.schema_version == "v1"
    assert set(bundle.scene_templates.keys()) == {
        "single_transform_options_transformation",
        "single_transform_options_inverse_transformation",
    }
    assert set(bundle.query_templates.keys()) == {
        "flip_horizontal_result_label",
        "flip_horizontal_source_label",
        "flip_vertical_result_label",
        "flip_vertical_source_label",
        "rotate_180_result_label",
        "rotate_180_source_label",
        "rotate_90_clockwise_result_label",
        "rotate_90_clockwise_source_label",
        "rotate_90_counterclockwise_result_label",
        "rotate_90_counterclockwise_source_label",
    }


def test_icons_reference_canvas_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "reference_canvas")
    type_generation, _, type_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__reference_canvas__reference_type_match_count",
    )
    color_generation, color_rendering, color_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__reference_canvas__reference_color_match_count",
        )
    )
    rotation_generation, _, rotation_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__reference_canvas__reference_rotation_match_count",
        )
    )
    binding_generation, binding_rendering, binding_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__reference_canvas__reference_type_color_rotation_match_count",
        )
    )
    assert "query_id_weights" not in type_generation
    assert (
        str(
            type_generation["variant_generation_params"]["match_type"]["pool_manifest"]
        ).strip()
        == "all_icons.txt"
    )
    assert (
        str(
            color_generation["variant_generation_params"]["match_color"][
                "pool_manifest"
            ]
        ).strip()
        == "all_icons.txt"
    )
    assert (
        str(
            rotation_generation["variant_generation_params"]["match_rotation"][
                "pool_manifest"
            ]
        ).strip()
        == "non_symmetry.txt"
    )
    assert list(
        rotation_generation["variant_generation_params"]["match_rotation"][
            "rotation_candidates_degrees"
        ]
    ) == [0, 90, 180, 270]
    assert (
        str(
            binding_generation["variant_generation_params"][
                "match_type_color_rotation"
            ]["pool_manifest"]
        ).strip()
        == "non_symmetry.txt"
    )
    assert (
        float(
            color_rendering["variant_render_params"]["match_color"][
                "min_color_distance"
            ]
        )
        == 60.0
    )
    assert (
        int(color_rendering["variant_render_params"]["match_color"]["palette_size_min"])
        == 3
    )
    assert (
        int(color_rendering["variant_render_params"]["match_color"]["palette_size_max"])
        == 4
    )
    assert (
        int(
            binding_rendering["variant_render_params"]["match_type_color_rotation"][
                "palette_size_min"
            ]
        )
        == 3
    )
    for prompt_defaults in (type_prompt, color_prompt, rotation_prompt, binding_prompt):
        assert str(prompt_defaults["bundle_id"]) == "icons_reference_canvas_v1"
        assert str(prompt_defaults["scene_key"]) == "reference_canvas_counting"
    metric_task_id = "task_icons__reference_canvas__reference_metric_relation_count"
    metric_generation, metric_rendering, metric_prompt = (
        split_generation_rendering_prompt_defaults(cfg, task_id=metric_task_id)
    )
    assert "query_id_weights" not in metric_generation
    assert str(metric_prompt["bundle_id"]) == "icons_reference_canvas_v1"
    metric_generation_smaller = metric_generation["variant_generation_params"][
        "size_smaller"
    ]
    assert str(metric_generation_smaller["pool_manifest"]).strip() == "all_icons.txt"
    assert list(metric_generation_smaller["rotation_candidates_degrees"]) == [
        0,
        90,
        180,
        270,
    ]
    assert int(metric_generation_smaller["size_relation_min_delta_px"]) == 18
    assert int(metric_generation_smaller["object_count_max"]) == 14
    assert int(metric_generation_smaller["target_count_max"]) == 5
    assert int(metric_generation_smaller["distractor_count_max"]) == 6
    assert (
        int(
            metric_rendering["variant_render_params"]["size_smaller"][
                "scene_icon_size_max_px"
            ]
        )
        == 120
    )
    assert (
        int(
            metric_rendering["variant_render_params"]["size_smaller"][
                "reference_icon_size_min_px"
            ]
        )
        == 64
    )
    assert (
        int(
            metric_rendering["variant_render_params"]["size_smaller"][
                "reference_icon_size_max_px"
            ]
        )
        == 96
    )
    anchor_task_id = "task_icons__reference_canvas__anchor_position_count"
    anchor_generation, anchor_rendering, anchor_prompt = (
        split_generation_rendering_prompt_defaults(cfg, task_id=anchor_task_id)
    )
    assert str(anchor_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert dict(anchor_generation["direction_weights"]) == {
        "left": 1.0,
        "right": 1.0,
        "above": 1.0,
        "below": 1.0,
    }
    assert float(anchor_rendering["scene_max_overlap_fraction"]) == 0.05
    assert int(anchor_rendering["anchor_gap_px_directional"]) == 8
    assert str(anchor_prompt["bundle_id"]) == "icons_reference_canvas_v1"
    assert str(anchor_prompt["scene_key"]) == "reference_canvas_anchor"
    bundle = load_scene_prompt_bundle(
        "icons", "reference_canvas", "icons_reference_canvas_v1"
    )
    assert bundle.bundle_id == "icons_reference_canvas_v1"
    assert bundle.schema_version == "v1"
    assert set(bundle.scene_templates.keys()) == {
        "reference_canvas_anchor",
        "reference_canvas_counting",
    }


def test_icons_icon_field_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "icon_field")
    singleton_generation, rendering, prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__icon_field__singleton_type_count",
        )
    )
    most_frequent_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__icon_field__most_frequent_type_count",
    )
    frequency_extreme_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__icon_field__frequency_extreme_type_label",
    )
    assert str(singleton_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert "variant_generation_params" not in singleton_generation
    assert int(singleton_generation["object_count_min"]) == 5
    assert int(singleton_generation["object_count_max"]) == 10
    assert int(singleton_generation["target_count_min"]) == 0
    assert int(singleton_generation["target_count_max"]) == 4
    assert int(singleton_generation["repeated_type_count_min"]) == 1
    assert int(singleton_generation["repeated_type_count_max"]) == 4
    assert int(singleton_generation["repeated_type_multiplicity_min"]) == 2
    assert int(singleton_generation["repeated_type_multiplicity_max"]) == 4
    assert str(most_frequent_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert "variant_generation_params" not in most_frequent_generation
    assert int(most_frequent_generation["object_count_min"]) == 7
    assert int(most_frequent_generation["object_count_max"]) == 12
    assert int(most_frequent_generation["target_count_min"]) == 2
    assert int(most_frequent_generation["target_count_max"]) == 6
    assert int(most_frequent_generation["other_repeated_type_count_max"]) == 3
    assert int(frequency_extreme_generation["object_count_min"]) == 8
    assert int(frequency_extreme_generation["object_count_max"]) == 18
    assert int(frequency_extreme_generation["frequency_min"]) == 1
    assert int(frequency_extreme_generation["frequency_max"]) == 5
    assert list(frequency_extreme_generation["option_count_support"]) == [4, 6]
    assert int(rendering["canvas_width"]) == 960
    assert int(rendering["canvas_height"]) == 544
    assert int(rendering["scene_icon_size_min_px"]) == 64
    assert int(rendering["scene_icon_size_max_px"]) == 96
    assert list(rendering["rotation_candidates_degrees"]) == [0, 90, 180, 270]
    assert str(prompt["bundle_id"]) == "icons_icon_field_v1"
    assert str(prompt["scene_key"]) == "single_scene_counting"
    assert str(prompt["task_key"]) == "type_frequency_query"
    bundle = load_scene_prompt_bundle("icons", "icon_field", "icons_icon_field_v1")
    assert bundle.bundle_id == "icons_icon_field_v1"
    assert set(bundle.scene_templates.keys()) == {"single_scene_counting"}
    assert set(bundle.query_templates.keys()) == {
        "least_frequent_type_label",
        "most_frequent_type_label",
        "most_frequent_type_count",
        "singleton_type_count",
    }


def test_icons_icon_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "icon_grid")
    type_generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__icon_grid__distinct_type_count",
    )
    color_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__icon_grid__distinct_color_count",
    )
    assert str(type_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert int(type_generation["object_count_min"]) == 6
    assert int(type_generation["object_count_max"]) == 12
    assert int(type_generation["target_count_min"]) == 1
    assert int(type_generation["target_count_max"]) == 5
    assert int(color_generation["target_count_min"]) == 1
    assert int(color_generation["target_count_max"]) == 5
    assert "type_count_min" not in color_generation
    assert "type_count_max" not in color_generation
    assert int(rendering["canvas_width"]) == 860
    assert int(rendering["canvas_height"]) == 660
    assert int(rendering["grid_cell_max_size_px"]) == 112
    assert int(rendering["grid_border_width_px"]) == 3
    assert list(rendering["rotation_candidates_degrees"]) == [0, 90, 180, 270]
    assert str(prompt["bundle_id"]) == "icons_icon_grid_v1"
    assert str(prompt["scene_key"]) == "visible_icon_grid"
    assert str(prompt["task_key"]) == "category_count_query"
    bundle = load_scene_prompt_bundle("icons", "icon_grid", "icons_icon_grid_v1")
    assert bundle.bundle_id == "icons_icon_grid_v1"
    assert set(bundle.scene_templates.keys()) == {"visible_icon_grid"}
    assert set(bundle.query_templates.keys()) == {
        "distinct_color_count",
        "distinct_type_count",
    }


def test_icons_named_field_grid_ring_defaults_loaded() -> None:
    named_cfg = get_scene_defaults("icons", "named_field")
    named_grid_cfg = get_scene_defaults("icons", "named_grid")
    named_ring_cfg = get_scene_defaults("icons", "named_ring")
    assert (
        "task_icons__named_field__closer_to_reference_count"
        in named_cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__named_grid__scoped_attribute_count"
        in named_grid_cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__named_grid__row_column_shape_extreme_number"
        in named_grid_cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__named_grid__group_predicate_count"
        in named_grid_cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__named_grid__line_adjacency_pair_count"
        in named_grid_cfg["generation"]["task_overrides"]
    )
    grid_generation, grid_rendering, grid_prompt = (
        split_generation_rendering_prompt_defaults(
            named_grid_cfg, task_id="task_icons__named_grid__scoped_attribute_count"
        )
    )
    assert int(grid_generation["target_count_min"]) == 1
    assert int(grid_generation["target_count_max"]) == 5
    assert "query_id_weights" not in grid_generation
    assert [list(value) for value in grid_generation["grid_size_support"]] == [
        [4, 4],
        [4, 5],
        [4, 6],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 4],
        [6, 5],
        [6, 6],
    ]
    assert int(grid_rendering["canvas_width"]) == 880
    assert int(grid_rendering["canvas_height"]) == 680
    assert int(grid_rendering["grid_cell_max_size_px"]) == 104
    assert int(grid_rendering["axis_label_font_size_px"]) == 24
    assert str(grid_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(grid_prompt["question_text_row_shape_count"]).strip()
    assert str(grid_prompt["question_text_column_shape_count"]).strip()
    assert str(grid_prompt["annotation_hint"]).strip()
    assert str(grid_prompt["answer_hint"]).strip()
    assert str(grid_prompt["json_example"]).strip()
    assert str(grid_prompt["json_example_answer_only"]).strip()
    grid_extreme_generation, grid_extreme_rendering, grid_extreme_prompt = (
        split_generation_rendering_prompt_defaults(
            named_grid_cfg,
            task_id="task_icons__named_grid__row_column_shape_extreme_number",
        )
    )
    assert int(grid_extreme_generation["answer_line_number_min"]) == 1
    assert int(grid_extreme_generation["answer_line_number_max"]) == 6
    assert "query_id_weights" not in grid_extreme_generation
    assert [list(value) for value in grid_extreme_generation["grid_size_support"]] == [
        [4, 4],
        [4, 5],
        [4, 6],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 4],
        [6, 5],
        [6, 6],
    ]
    assert int(grid_extreme_rendering["canvas_width"]) == 880
    assert int(grid_extreme_rendering["canvas_height"]) == 680
    assert str(grid_extreme_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(grid_extreme_prompt["question_text_row_most_shape_number"]).strip()
    assert str(grid_extreme_prompt["question_text_row_fewest_shape_number"]).strip()
    assert str(grid_extreme_prompt["question_text_column_most_shape_number"]).strip()
    assert str(grid_extreme_prompt["question_text_column_fewest_shape_number"]).strip()
    assert str(grid_extreme_prompt["annotation_hint"]).strip()
    assert str(grid_extreme_prompt["answer_hint"]).strip()
    assert str(grid_extreme_prompt["json_example"]).strip()
    assert str(grid_extreme_prompt["json_example_answer_only"]).strip()
    grid_line_generation, grid_line_rendering, grid_line_prompt = (
        split_generation_rendering_prompt_defaults(
            named_grid_cfg, task_id="task_icons__named_grid__group_predicate_count"
        )
    )
    assert int(grid_line_generation["answer_count_min"]) == 0
    assert int(grid_line_generation["answer_count_max"]) == 5
    assert int(grid_line_generation["at_least_threshold_min"]) == 2
    assert int(grid_line_generation["at_least_threshold_max"]) == 3
    assert int(grid_line_generation["exactly_threshold_min"]) == 1
    assert int(grid_line_generation["exactly_threshold_max"]) == 3
    assert "query_id_weights" not in grid_line_generation
    assert [list(value) for value in grid_line_generation["grid_size_support"]] == [
        [4, 4],
        [4, 5],
        [4, 6],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 4],
        [6, 5],
        [6, 6],
    ]
    assert int(grid_line_rendering["canvas_width"]) == 880
    assert int(grid_line_rendering["canvas_height"]) == 680
    assert str(grid_line_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(grid_line_prompt["question_text_row_at_least_shape_count"]).strip()
    assert str(grid_line_prompt["question_text_column_at_least_shape_count"]).strip()
    assert str(grid_line_prompt["question_text_row_exactly_shape_count"]).strip()
    assert str(grid_line_prompt["question_text_column_exactly_shape_count"]).strip()
    assert str(grid_line_prompt["question_text_row_no_shape_count"]).strip()
    assert str(grid_line_prompt["question_text_column_no_shape_count"]).strip()
    assert str(grid_line_prompt["annotation_hint"]).strip()
    assert str(grid_line_prompt["answer_hint"]).strip()
    assert str(grid_line_prompt["json_example"]).strip()
    assert str(grid_line_prompt["json_example_answer_only"]).strip()
    grid_pair_generation, grid_pair_rendering, grid_pair_prompt = (
        split_generation_rendering_prompt_defaults(
            named_grid_cfg,
            task_id="task_icons__named_grid__line_adjacency_pair_count",
        )
    )
    assert int(grid_pair_generation["answer_count_min"]) == 1
    assert int(grid_pair_generation["answer_count_max"]) == 5
    assert "query_id_weights" not in grid_pair_generation
    assert [list(value) for value in grid_pair_generation["grid_size_support"]] == [
        [4, 4],
        [4, 5],
        [4, 6],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 4],
        [6, 5],
        [6, 6],
    ]
    assert int(grid_pair_rendering["canvas_width"]) == 880
    assert int(grid_pair_rendering["canvas_height"]) == 680
    assert str(grid_pair_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(grid_pair_prompt["question_text_row_unordered_adjacent_pair_count"]).strip()
    assert str(grid_pair_prompt["question_text_column_unordered_adjacent_pair_count"]).strip()
    assert str(grid_pair_prompt["annotation_hint"]).strip()
    assert str(grid_pair_prompt["answer_hint"]).strip()
    assert str(grid_pair_prompt["json_example"]).strip()
    assert str(grid_pair_prompt["json_example_answer_only"]).strip()
    ring_generation, ring_rendering, ring_prompt = (
        split_generation_rendering_prompt_defaults(
            named_ring_cfg,
            task_id="task_icons__named_ring__scoped_attribute_count",
        )
    )
    assert int(ring_generation["ring_icon_count_min"]) == 12
    assert int(ring_generation["ring_icon_count_max"]) == 22
    assert int(ring_generation["answer_count_min"]) == 0
    assert int(ring_generation["answer_count_max"]) == 6
    assert int(ring_generation["arc_span_min"]) == 3
    assert int(ring_generation["arc_span_max"]) == 12
    assert int(ring_rendering["canvas_width"]) == 880
    assert int(ring_rendering["canvas_height"]) == 680
    assert int(ring_rendering["ring_margin_px"]) == 86
    assert int(ring_rendering["marker_label_radius_px"]) == 18
    assert str(ring_prompt["bundle_id"]).strip() == "icons_named_ring_v1"
    assert str(ring_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(ring_prompt["task_key"]).strip() == "counting_query"
    ring_prompt_defaults = required_group_defaults(
        ring_prompt,
        (
            "object_description",
            "question_text_clockwise_arc_shape_count",
            "question_text_counterclockwise_arc_shape_count",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="named_ring prompt defaults",
    )
    assert str(ring_prompt_defaults["question_text_clockwise_arc_shape_count"]).strip()
    assert str(
        ring_prompt_defaults["question_text_counterclockwise_arc_shape_count"]
    ).strip()
    assert str(ring_prompt_defaults["annotation_hint"]).strip()
    assert str(ring_prompt_defaults["answer_hint"]).strip()
    assert str(ring_prompt_defaults["json_example"]).strip()
    assert str(ring_prompt_defaults["json_example_answer_only"]).strip()
    closer_generation, closer_rendering, closer_prompt = (
        split_generation_rendering_prompt_defaults(
            named_cfg, task_id="task_icons__named_field__closer_to_reference_count"
        )
    )
    assert int(closer_generation["target_icon_count_min"]) == 4
    assert int(closer_generation["target_icon_count_max"]) == 8
    assert int(closer_generation["target_answer_min"]) == 0
    assert int(closer_generation["target_answer_max"]) == 4
    assert dict(closer_generation["queried_reference_label_weights"]) == {
        "A": 1.0,
        "B": 1.0,
    }
    assert list(closer_generation["reference_axis_degrees"]) == [0, 35, 90, 145]
    assert int(closer_rendering["canvas_width"]) > 0
    assert int(closer_rendering["canvas_height"]) > 0
    assert int(closer_rendering["distance_margin_px"]) == 42
    assert str(closer_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(closer_prompt["question_text_closer_to_reference_count"]).strip()
    assert str(closer_prompt["annotation_hint"]).strip()
    assert str(closer_prompt["answer_hint"]).strip()
    assert str(closer_prompt["json_example"]).strip()
    assert str(closer_prompt["json_example_answer_only"]).strip()


def test_icons_venn_field_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "venn_field")
    assert (
        "task_icons__venn_field__scoped_attribute_count"
        in cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__venn_field__same_region_as_reference_count"
        in cfg["generation"]["task_overrides"]
    )
    venn_generation, venn_rendering, venn_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__venn_field__scoped_attribute_count",
        )
    )
    assert int(venn_generation["object_count_min"]) == 8
    assert int(venn_generation["object_count_max"]) == 13
    assert int(venn_generation["target_count_min"]) == 1
    assert int(venn_generation["target_count_max"]) == 5
    assert "named_venn_query_ids" not in venn_generation
    assert "query_id_weights" not in venn_generation
    assert sorted(venn_generation["target_attribute_mode_weights"].keys()) == [
        "color_shape",
        "shape_only",
    ]
    assert int(venn_rendering["canvas_width"]) > 0
    assert int(venn_rendering["canvas_height"]) > 0
    assert int(venn_rendering["venn_boundary_margin_px"]) == 12
    assert str(venn_prompt["bundle_id"]).strip() == "icons_venn_field_v1"
    assert str(venn_prompt["scene_key"]).strip() == "venn_field_scene"
    assert str(venn_prompt["task_key"]).strip() == "scoped_attribute_count"
    _, _, same_region_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__venn_field__same_region_as_reference_count",
    )
    assert (
        str(same_region_prompt["task_key"]).strip() == "same_region_as_reference_count"
    )
    bundle = load_scene_prompt_bundle("icons", "venn_field", "icons_venn_field_v1")
    assert bundle.bundle_id == "icons_venn_field_v1"
    assert bundle.schema_version == "v1"
    assert set(bundle.scene_templates.keys()) == {"venn_field_scene"}
    assert set(bundle.query_templates.keys()) == {
        "single",
        "inside_both_circles_count",
        "inside_either_circle_count",
        "inside_exactly_one_circle_count",
        "outside_both_circles_count",
    }


def test_icons_pair_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "pair_grid")
    generation_shared = cfg["generation"]["shared"]
    assert int(generation_shared["option_count"]) == 6
    assert (
        "task_icons__pair_grid__reference_color_pair_match_label"
        in cfg["generation"]["task_overrides"]
    )
    assert (
        "task_icons__pair_grid__reference_transform_match_label"
        in cfg["generation"]["task_overrides"]
    )
    render_shared = cfg["rendering"]["shared"]
    assert int(render_shared["canvas_width"]) > 0
    assert int(render_shared["canvas_height"]) > 0
    assert int(render_shared["reference_panel_width_px"]) > 0
    assert int(render_shared["scene_icon_size_min_px"]) > 0
    assert int(render_shared["scene_icon_size_max_px"]) >= int(
        render_shared["scene_icon_size_min_px"]
    )
    assert int(render_shared["cell_padding_px"]) > 0
    assert int(render_shared["cell_label_font_size_px"]) > 0
    assert int(render_shared["pair_arrow_stroke_px"]) > 0
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]) == "icons_pair_grid_v1"
    assert str(prompt_shared["scene_key"]) == "reference_pair_grid"
    assert str(prompt_shared["task_key"]) == "relation_match_label_query"
    assert str(prompt_shared["json_output_contract"]).strip()
    assert str(prompt_shared["json_output_contract_answer_only"]).strip()
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg, task_id="task_icons__pair_grid__reference_transform_match_label"
    )
    assert str(generation["pool_manifest"]).strip() == "non_symmetry.txt"
    assert int(generation["option_count"]) == 4
    assert list(generation["transform_ids"]) == [
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
    ]
    assert int(generation["transform_check_size_px"]) > 0
    assert int(rendering["canvas_width"]) > 0
    assert int(rendering["reference_panel_width_px"]) > 0
    prompt_defaults = required_group_defaults(
        prompt,
        (
            "object_description",
            "question_text",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="pair_grid reference-transform prompt defaults",
    )
    assert str(prompt_defaults["object_description"]).strip()
    assert str(prompt_defaults["question_text"]).strip()
    assert str(prompt_defaults["annotation_hint"]).strip()
    assert str(prompt_defaults["answer_hint"]).strip()
    assert str(prompt_defaults["json_example"]).strip()
    assert str(prompt_defaults["json_example_answer_only"]).strip()
    color_generation, color_rendering, color_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_icons__pair_grid__reference_color_pair_match_label"
        )
    )
    assert str(color_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert int(color_generation["option_count"]) == 6
    assert int(color_rendering["reference_icon_size_px"]) > 0
    assert int(color_rendering["palette_size_min"]) >= 4
    assert int(color_rendering["palette_size_max"]) >= int(
        color_rendering["palette_size_min"]
    )
    color_prompt_defaults = required_group_defaults(
        color_prompt,
        (
            "object_description",
            "question_text",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="pair_grid color-pair prompt defaults",
    )
    assert str(color_prompt_defaults["object_description"]).strip()
    assert str(color_prompt_defaults["question_text"]).strip()
    assert str(color_prompt_defaults["annotation_hint"]).strip()
    assert str(color_prompt_defaults["answer_hint"]).strip()
    assert str(color_prompt_defaults["json_example"]).strip()
    assert str(color_prompt_defaults["json_example_answer_only"]).strip()
    bundle = load_scene_prompt_bundle("icons", "pair_grid", "icons_pair_grid_v1")
    assert bundle.bundle_id == "icons_pair_grid_v1"
    assert set(bundle.scene_templates.keys()) == {"reference_pair_grid"}
    assert set(bundle.task_templates.keys()) == {"relation_match_label_query"}


def test_icons_paired_canvas_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "paired_canvas")
    assert sorted(cfg["generation"]["task_overrides"].keys()) == [
        "task_icons__paired_canvas__color_change_count",
        "task_icons__paired_canvas__panel_set_relation_count",
        "task_icons__paired_canvas__rotation_change_count",
    ]
    assert sorted(cfg["rendering"]["task_overrides"].keys()) == []
    assert sorted(cfg["prompt"]["task_overrides"].keys()) == [
        "task_icons__paired_canvas__color_change_count",
        "task_icons__paired_canvas__panel_set_relation_count",
        "task_icons__paired_canvas__rotation_change_count",
    ]
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]) == "icons_paired_canvas_v0"
    assert str(prompt_shared["scene_key"]) == "paired_canvas_set_relation"
    assert str(prompt_shared["task_key"]) == "paired_canvas_query"
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg, task_id="task_icons__paired_canvas__color_change_count"
    )
    assert int(generation["object_count_min"]) == 5
    assert int(generation["object_count_max"]) == 10
    assert "query_id_weights" not in generation
    assert "attribute_change_query_weights" not in generation
    assert "variant_generation_params" not in generation
    assert int(rendering["reference_panel_width_px"]) == 516
    assert str(prompt["scene_key"]) == "paired_canvas_attribute_change"
    assert str(prompt["question_text_color_changed_count"]).strip()
    bundle = load_scene_prompt_bundle(
        "icons", "paired_canvas", "icons_paired_canvas_v0"
    )
    assert bundle.bundle_id == "icons_paired_canvas_v0"
    assert set(bundle.scene_templates.keys()) == {
        "paired_canvas_attribute_change",
        "paired_canvas_set_relation",
    }


def test_icons_mirror_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "mirror_grid")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__mirror_grid__mirror_symmetry_match_label",
    )
    assert str(generation["pool_manifest"]).strip() == "non_symmetry.txt"
    assert list(generation["option_count_choices"]) == [4, 6]
    assert int(rendering["canvas_width"]) == 1104
    assert int(rendering["canvas_height"]) == 640
    assert int(rendering["reference_panel_width_px"]) == 296
    assert list(rendering["symmetric_icon_count_choices"]) == [2, 4, 6]
    assert list(rendering["both_axes_icon_count_choices"]) == [4]
    assert list(rendering["nonsymmetric_icon_count_choices"]) == [2, 4, 6]
    assert int(rendering["patch_inner_margin_px"]) == 8
    assert int(rendering["patch_min_gap_px"]) == 6
    assert str(prompt["bundle_id"]) == "icons_mirror_grid_v1"
    assert str(prompt["scene_key"]).strip() == "reference_mirror_grid"
    assert str(prompt["task_key"]).strip() == "mirror_symmetry_match_label"


def test_icons_overlap_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "overlap_grid")
    generation_shared = cfg["generation"]["shared"]
    assert int(generation_shared["object_count_min"]) == 2
    assert int(generation_shared["object_count_max"]) == 8
    assert int(generation_shared["target_count_min"]) == 0
    assert int(generation_shared["target_count_max"]) == 4
    assert int(generation_shared["distractor_count_min"]) == 1
    assert int(generation_shared["distractor_count_max"]) == 5
    assert int(generation_shared["distractor_margin_over_target"]) == 0
    assert bool(generation_shared["balanced_sampling"]) is True
    assert str(generation_shared["pool_manifest"]).strip() == "all_icons.txt"
    named_cfg = get_scene_defaults("icons", "named_field")
    assert (
        "task_icons__named_field__reference_distance_rank_label"
        in named_cfg["generation"]["task_overrides"]
    )
    render_shared = cfg["rendering"]["shared"]
    assert int(render_shared["canvas_width"]) == 1104
    assert int(render_shared["canvas_height"]) == 640
    assert int(render_shared["reference_panel_width_px"]) == 296
    assert int(render_shared["scene_icon_size_min_px"]) > 0
    assert int(render_shared["scene_icon_size_max_px"]) >= int(
        render_shared["scene_icon_size_min_px"]
    )
    assert int(render_shared["reference_icon_size_px"]) == 110
    assert float(render_shared["min_color_distance"]) == 40.0
    assert float(render_shared["pair_min_color_distance"]) == 80.0
    assert list(render_shared["overlap_ratio_range"]) == [0.4, 0.6]
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]).strip() == "icons_overlap_grid_v1"
    assert str(prompt_shared["scene_key"]).strip() == "overlap_grid_occlusion_order"
    assert str(prompt_shared["task_key"]).strip() == "occlusion_order_count"
    assert str(prompt_shared["json_output_contract"]).strip()
    assert str(prompt_shared["json_output_contract_answer_only"]).strip()
    occlusion_generation, occlusion_rendering, occlusion_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_icons__overlap_grid__occlusion_order_count"
        )
    )
    assert str(occlusion_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert int(occlusion_generation["object_count_min"]) == 2
    assert int(occlusion_generation["object_count_max"]) == 8
    assert int(occlusion_generation["target_count_max"]) == 4
    assert int(occlusion_generation["distractor_count_max"]) == 5
    assert int(occlusion_generation["distractor_margin_over_target"]) == 0
    assert int(occlusion_rendering["canvas_width"]) == 1104
    assert int(occlusion_rendering["reference_panel_width_px"]) == 296
    assert float(occlusion_rendering["min_color_distance"]) == 40.0
    assert float(occlusion_rendering["pair_min_color_distance"]) == 80.0
    assert list(occlusion_rendering["overlap_ratio_range"]) == [0.4, 0.6]
    assert str(occlusion_prompt["bundle_id"]).strip() == "icons_overlap_grid_v1"
    assert str(occlusion_prompt["scene_key"]).strip() == "overlap_grid_occlusion_order"
    assert str(occlusion_prompt["task_key"]).strip() == "occlusion_order_count"
    prompt_required = required_group_defaults(
        occlusion_prompt,
        (
            "object_description",
            "question_text",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="overlap-grid prompt defaults",
    )
    assert str(prompt_required["object_description"]).strip()
    assert str(prompt_required["question_text"]).strip()
    assert str(prompt_required["annotation_hint"]).strip()
    assert str(prompt_required["answer_hint"]).strip()
    assert str(prompt_required["json_example"]).strip()
    assert str(prompt_required["json_example_answer_only"]).strip()
    distance_generation, distance_rendering, distance_prompt = (
        split_generation_rendering_prompt_defaults(
            named_cfg, task_id="task_icons__named_field__reference_distance_rank_label"
        )
    )
    assert int(distance_generation["candidate_count"]) == 6
    assert int(distance_generation["distractor_count_min"]) == 4
    assert int(distance_generation["distractor_count_max"]) == 8
    assert "distance_rank_query_weights" not in distance_generation
    assert list(distance_generation["named_icon_fill_style_support"]) == [
        "solid",
        "striped",
        "dotted",
    ]
    assert int(distance_rendering["canvas_width"]) == 960
    assert int(distance_rendering["canvas_height"]) == 560
    assert int(distance_rendering["scene_icon_size_min_px"]) == 48
    assert int(distance_rendering["scene_icon_size_max_px"]) == 72
    assert int(distance_rendering["distance_rank_margin_px"]) == 24
    assert int(distance_rendering["candidate_label_font_size_px"]) == 24
    assert str(distance_prompt["scene_key"]).strip() == "single_scene_counting"
    assert str(distance_prompt["object_description"]).strip()
    assert str(
        distance_prompt["question_text_closest_to_named_reference_label"]
    ).strip()
    assert str(
        distance_prompt["question_text_second_closest_to_named_reference_label"]
    ).strip()
    assert str(
        distance_prompt["question_text_farthest_from_named_reference_label"]
    ).strip()
    assert str(distance_prompt["annotation_hint"]).strip()
    assert str(distance_prompt["answer_hint"]).strip()
    assert str(distance_prompt["json_example"]).strip()
    assert str(distance_prompt["json_example_answer_only"]).strip()


def test_icons_named_path_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "named_path")
    path_generation, path_rendering, path_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_icons__named_path__path_neighbor_label",
        )
    )
    assert int(path_generation["candidate_count"]) == 6
    assert int(path_generation["distractor_count_min"]) == 4
    assert int(path_generation["distractor_count_max"]) == 8
    assert int(path_generation["target_occurrence_count_min"]) == 2
    assert int(path_generation["target_occurrence_count_max"]) == 4
    assert list(path_generation["named_icon_fill_style_support"]) == [
        "solid",
        "striped",
        "dotted",
    ]
    assert int(path_rendering["canvas_width"]) == 1280
    assert int(path_rendering["canvas_height"]) == 720
    assert int(path_rendering["scene_icon_size_min_px"]) == 52
    assert int(path_rendering["scene_icon_size_max_px"]) == 78
    assert int(path_rendering["path_stroke_width_px"]) == 7
    assert int(path_rendering["path_line_alpha"]) == 145
    assert int(path_rendering["candidate_label_font_size_px"]) == 24
    assert str(path_prompt["bundle_id"]).strip() == "icons_named_path_v1"
    assert str(path_prompt["scene_key"]).strip() == "named_path_relation"
    assert str(path_prompt["task_key"]).strip() == "path_neighbor_query"
    prompt_defaults = required_group_defaults(
        path_prompt,
        (
            "object_description",
            "question_text_after_first_shape_label",
            "question_text_before_first_shape_label",
            "question_text_after_last_shape_label",
            "question_text_before_last_shape_label",
            "question_text_after_second_shape_label",
            "question_text_before_second_shape_label",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="named_path prompt defaults",
    )
    assert str(prompt_defaults["object_description"]).strip()
    for suffix in (
        "after_first_shape_label",
        "before_first_shape_label",
        "after_last_shape_label",
        "before_last_shape_label",
        "after_second_shape_label",
        "before_second_shape_label",
    ):
        assert str(prompt_defaults[f"question_text_{suffix}"]).strip()
    assert str(prompt_defaults["annotation_hint"]).strip()
    assert str(prompt_defaults["answer_hint"]).strip()
    assert str(prompt_defaults["json_example"]).strip()
    assert str(prompt_defaults["json_example_answer_only"]).strip()


def test_icons_named_strip_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "named_strip")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__named_strip__shape_run_length",
    )
    assert int(generation["strip_length_min"]) == 12
    assert int(generation["strip_length_max"]) == 16
    assert int(generation["longest_run_length_min"]) == 2
    assert int(generation["longest_run_length_max"]) == 6
    assert int(generation["shortest_run_length_min"]) == 1
    assert int(generation["shortest_run_length_max"]) == 5
    assert int(generation["run_count_min"]) == 1
    assert int(generation["run_count_max"]) == 4
    assert list(generation["named_icon_fill_style_support"]) == [
        "solid",
        "striped",
        "dotted",
    ]
    assert "query_id_weights" not in generation
    assert int(rendering["scene_icon_size_min_px"]) == 42
    assert int(rendering["scene_icon_size_max_px"]) == 58
    assert int(rendering["cell_box_width_min_px"]) == 58
    assert int(rendering["cell_box_width_max_px"]) == 72
    assert int(rendering["cell_box_height_min_px"]) == 88
    assert int(rendering["cell_box_height_max_px"]) == 108
    assert int(rendering["cell_padding_px"]) == 4
    assert int(rendering["cell_icon_padding_px"]) == 8
    assert float(rendering["scene_max_overlap_fraction"]) == pytest.approx(
        0.0, rel=1e-09
    )
    assert str(prompt["bundle_id"]).strip() == "icons_named_strip_v1"
    assert str(prompt["scene_key"]).strip() == "named_strip_run_length"
    assert str(prompt["task_key"]).strip() == "shape_run_length"
    run_count_generation, _, run_count_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__named_strip__shape_run_count",
    )
    assert int(run_count_generation["run_count_min"]) == 1
    assert int(run_count_generation["run_count_max"]) == 4
    assert str(run_count_prompt["task_key"]).strip() == "shape_run_count"


def test_icons_sequence_strip_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "sequence_strip")
    count_task_id = "task_icons__sequence_strip__count_progression_completion_label"
    rotation_task_id = (
        "task_icons__sequence_strip__rotation_progression_completion_label"
    )
    size_task_id = "task_icons__sequence_strip__size_progression_completion_label"
    assert count_task_id in cfg["generation"]["task_overrides"]
    assert rotation_task_id in cfg["generation"]["task_overrides"]
    assert size_task_id in cfg["generation"]["task_overrides"]
    count_generation, count_rendering, count_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id=count_task_id,
        )
    )
    assert str(count_generation["pool_manifest"]).strip() == "all_icons.txt"
    assert list(count_generation["count_step_candidates"]) == [-2, -1, 1, 2]
    assert int(count_rendering["scene_icon_size_min_px"]) == 20
    assert int(count_rendering["scene_icon_size_max_px"]) == 38
    assert str(count_prompt["bundle_id"]) == "icons_sequence_strip_v1"
    assert str(count_prompt["scene_key"]) == "sequence_completion_options"
    assert str(count_prompt["task_key"]) == "completion_option_label"
    rotation_generation, rotation_rendering, rotation_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id=rotation_task_id,
        )
    )
    assert str(rotation_generation["pool_manifest"]).strip() == "non_symmetry.txt"
    assert list(rotation_generation["rotation_step_candidates_degrees"]) == [
        45,
        90,
        270,
        315,
    ]
    assert int(rotation_rendering["scene_icon_size_min_px"]) == 62
    assert int(rotation_rendering["scene_icon_size_max_px"]) == 72
    assert str(rotation_prompt["bundle_id"]) == "icons_sequence_strip_v1"
    size_generation, size_rendering, _ = split_generation_rendering_prompt_defaults(
        cfg, task_id=size_task_id
    )
    assert list(size_generation["size_step_candidates_px"]) == [-16, -12, 12, 16]
    assert int(size_rendering["scene_icon_size_max_px"]) == 92
    bundle = load_scene_prompt_bundle(
        "icons", "sequence_strip", "icons_sequence_strip_v1"
    )
    assert bundle.bundle_id == "icons_sequence_strip_v1"
    assert bundle.schema_version == "v1"
    assert set(bundle.scene_templates.keys()) == {"sequence_completion_options"}
    assert set(bundle.query_templates.keys()) == {
        "count_progression_completion_label",
        "rotation_progression_completion_label",
        "size_progression_completion_label",
    }


def test_icons_wallpaper_panels_defaults_loaded() -> None:
    cfg = get_scene_defaults("icons", "wallpaper_panels")
    generation_shared = cfg["generation"]["shared"]
    assert list(generation_shared["option_count_choices"]) == [4]
    assert int(generation_shared["lattice_rows"]) == 4
    assert int(generation_shared["lattice_cols"]) == 4
    assert str(generation_shared["pool_manifest"]) == "non_symmetry.txt"
    assert len(generation_shared["wallpaper_group_ids"]) >= 6
    render_shared = cfg["rendering"]["shared"]
    assert int(render_shared["canvas_width"]) == 1104
    assert int(render_shared["canvas_height"]) == 640
    assert int(render_shared["lattice_rows"]) == 4
    assert int(render_shared["lattice_cols"]) == 4
    assert int(render_shared["scene_icon_size_min_px"]) == 40
    assert int(render_shared["scene_icon_size_max_px"]) == 44
    assert list(render_shared["icon_noise_edit_count_range"]) == [0, 0]
    _, reference_rendering, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_icons__wallpaper_panels__same_pattern_as_reference_label",
    )
    assert int(reference_rendering["canvas_width"]) == 1104
    assert int(reference_rendering["canvas_height"]) == 960
    assert (
        int(reference_rendering["canvas_width"])
        * int(reference_rendering["canvas_height"])
        < 1200000
    )
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]).strip() == "icons_wallpaper_panels_v1"
    prompt_overrides = cfg["prompt"]["task_overrides"]
    assert str(
        prompt_overrides["task_icons__wallpaper_panels__motif_violation_label"][
            "task_key"
        ]
    ) == ("motif_violation_label")
    assert str(
        prompt_overrides[
            "task_icons__wallpaper_panels__same_pattern_as_reference_label"
        ]["task_key"]
    ) == ("same_pattern_as_reference_label")
