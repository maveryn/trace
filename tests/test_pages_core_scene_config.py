"""Regression tests for scene default config loading."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    required_group_default,
    required_group_defaults,
    resolve_optional_int_bounds,
    resolve_required_float_bounds,
    resolve_required_int_bounds,
    split_generation_rendering_prompt_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.pages.shared.render_audit_defaults import (
    wrap_pages_generation,
    wrap_pages_scene_generation,
)
from trace_tasks.tasks.shared.visual_style.information_scene import INFORMATION_SCENE_TREATMENT_IDS


FULL_NODE_LINK_LAYOUT_VARIANTS = set(SUPPORTED_LAYOUT_VARIANTS)


def _dummy_pages_output() -> TaskOutput:
    return TaskOutput(
        prompt="demo",
        answer_gt=TypedValue(type="integer", value=1),
        annotation_gt=TypedValue(type="bbox_set", value=[]),
        image=Image.new("RGB", (120, 80), "white"),
        image_id="img0",
        trace_payload={"render_spec": {}, "render_map": {}, "scene_ir": {"entities": []}},
        task_versions={},
        scene_id="demo_scene",
        query_id="default",
    )


def test_pages_scene_render_wrapper_uses_scene_metadata_without_scene_id() -> None:
    def _generate(_self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        assert instance_seed == 17
        assert max_attempts == 1
        assert params["pages_context_text_enabled"] is False
        return _dummy_pages_output()

    wrapped = wrap_pages_scene_generation(
        _generate,
        task_id="task_pages__demo_scene__lookup_value",
        scene_id="demo_scene",
    )
    output = wrapped(
        object(),
        17,
        params={"pages_context_text_enabled": False, "pages_font_family": "georama"},
        max_attempts=1,
    )

    font_assets = output.trace_payload["render_spec"]["font_assets"]
    assert font_assets["task_id"] == "task_pages__demo_scene__lookup_value"
    assert font_assets["scene_id"] == "demo_scene"
    assert output.trace_payload["render_spec"]["context_text_layer"]["enabled"] is False


def test_pages_legacy_render_wrapper_keeps_scene_id_metadata() -> None:
    def _generate(_self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        return _dummy_pages_output()

    wrapped = wrap_pages_generation(
        _generate,
        task_id="task_pages__demo_scene__lookup_value",
        scene_id="demo_group",
    )
    output = wrapped(
        object(),
        19,
        params={"pages_context_text_enabled": False, "pages_font_family": "georama"},
        max_attempts=1,
    )

    font_assets = output.trace_payload["render_spec"]["font_assets"]
    assert font_assets["scene_id"] == "demo_group"


def test_pages_calendar_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "calendar")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert sorted(generation_defaults["marked_day_class_weights"].keys()) == ["weekday", "weekend"]
    assert list(generation_defaults["weekend_weekday_indices"]) == [5, 6]
    assert list(generation_defaults["date_occurrence_support"]) == [1, 2, 3, 4, 5]
    assert list(generation_defaults["marked_weekend_count_support"]) == [0, 1, 2, 3, 4]
    assert list(generation_defaults["marked_weekday_count_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(generation_defaults["marked_weekday_distractor_support"]) == [1, 2, 3, 4]
    assert list(generation_defaults["marked_weekend_distractor_support"]) == [1, 2, 3, 4]
    assert list(generation_defaults["workday_offset_support"]) == [2, 3, 4, 5, 6, 7]

    assert int(rendering_defaults["canvas_width"]) == 860
    assert int(rendering_defaults["canvas_height"]) == 760
    assert bool(rendering_defaults["information_scene_allow_dark"]) is True
    assert list(rendering_defaults["information_scene_treatments"]) == list(INFORMATION_SCENE_TREATMENT_IDS)
    assert str(rendering_defaults["pages_context_profile"]) == "report_paragraph"
    assert dict(rendering_defaults["pages_context_mode_weights"]) == {
        "clean": 0.24,
        "minimal": 0.2,
        "paragraph_box": 0.56,
    }
    assert int(rendering_defaults["title_font_size_px"]) == 30
    assert int(rendering_defaults["date_font_size_px"]) == 22

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_calendar_v1"

def test_pages_calendar_event_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "calendar_event_grid")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["slot_id_weights"].keys()) == ["end", "mid", "top"]
    assert list(generation_defaults["target_count_support"]) == [2, 3, 4, 5, 6]
    assert len(generation_defaults["event_category_labels"]) == 10
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 780
    assert str(rendering_defaults["pages_context_profile"]) == "report_paragraph"
    assert dict(rendering_defaults["pages_context_mode_weights"]) == {
        "clean": 0.24,
        "minimal": 0.2,
        "paragraph_box": 0.56,
    }
    assert str(prompt_defaults["bundle_id"]).strip() == "pages_calendar_event_grid_v1"


def test_pages_process_flow_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "process_flow")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_weights" not in generation_defaults
    assert "balanced_query_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "editorial_review",
        "incident_response",
        "lab_sample",
        "model_release",
        "order_fulfillment",
        "support_ticket",
    ]
    assert sorted(generation_defaults["layout_variant_weights"].keys()) == ["horizontal_swimlane"]
    assert int(rendering_defaults["canvas_width"]) == 1200
    assert int(rendering_defaults["canvas_height"]) == 900
    assert str(prompt_defaults["bundle_id"]).strip() == "pages_process_flow_v1"
    assert "object_description" not in prompt_defaults


def test_pages_schedule_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "schedule")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert list(generation_defaults["event_count_support"]) == [7, 8, 9, 10]
    assert list(generation_defaults["overlap_count_support"]) == [1, 2, 3, 4, 5]
    assert list(generation_defaults["maximum_non_overlapping_support"]) == [2, 3, 4, 5]
    assert int(generation_defaults["slot_minutes"]) == 30
    assert int(generation_defaults["max_lane_count"]) == 5

    assert int(rendering_defaults["canvas_width"]) == 920
    assert int(rendering_defaults["canvas_height"]) == 820
    assert int(rendering_defaults["header_height_px"]) == 92
    assert int(rendering_defaults["time_axis_width_px"]) == 94
    assert bool(rendering_defaults["show_reference_time_band"]) is False

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_schedule_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults
    assert "object_description_overlap_count" not in prompt_defaults

def test_pages_timeline_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "timeline")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert "interval_relation_weights" not in generation_defaults
    assert list(generation_defaults["event_count_support"]) == [6, 7, 8, 9, 10, 11, 12]
    assert list(generation_defaults["between_count_support"]) == [1, 2, 3, 4]
    assert list(generation_defaults["outside_count_support"]) == [1, 2, 3, 4]
    assert list(generation_defaults["threshold_count_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(generation_defaults["relative_offset_support"]) == [1, 2, 3, 4]

    assert int(rendering_defaults["canvas_width"]) == 1120
    assert int(rendering_defaults["canvas_height"]) == 700
    assert int(rendering_defaults["card_width_px"]) == 106
    assert int(rendering_defaults["marker_radius_px"]) == 10
    assert str(rendering_defaults["pages_context_profile"]) == "report_paragraph"
    assert dict(rendering_defaults["pages_context_mode_weights"]) == {
        "clean": 0.24,
        "minimal": 0.2,
        "paragraph_box": 0.56,
    }
    assert int(rendering_defaults["pages_context_text_max_elements"]) == 7

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_timeline_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults


def test_pages_infographic_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "infographic")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_pages__infographic__sum_named_metrics_value",
    )
    sectioned_cfg = get_scene_defaults("pages", "sectioned_infographic")
    sectioned_generation, sectioned_rendering, sectioned_prompt = split_generation_rendering_prompt_defaults(
        sectioned_cfg,
    )

    assert list(generation_defaults["rank_position_support"]) == [2, 3]
    assert int(generation_defaults["card_count_min"]) == 20
    assert int(generation_defaults["card_count_max"]) == 30
    assert int(generation_defaults["section_count_min"]) == 4
    assert int(generation_defaults["section_count_max"]) == 6
    assert "balanced_query_id_sampling" not in generation_defaults
    assert "query_id_weights" not in generation_defaults

    assert int(rendering_defaults["canvas_width"]) == 912
    assert int(rendering_defaults["canvas_height"]) == 1416
    assert int(rendering_defaults["section_header_height_px"]) == 34

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_infographic_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults

    assert "sectioned_query_id_weights" not in sectioned_generation
    assert "balanced_query_id_sampling" not in sectioned_generation
    assert sorted(sectioned_generation["scene_variant_weights"].keys()) == [
        "bullet_columns",
        "checklist_bands",
        "topic_cards",
    ]
    assert bool(sectioned_generation["balanced_scene_variant_sampling"]) is True
    assert list(sectioned_generation["section_count_support"]) == [3, 4, 5]
    assert list(sectioned_generation["item_count_support"]) == [3, 4, 5, 6, 7]
    assert int(sectioned_rendering["canvas_width"]) == 1040
    assert int(sectioned_rendering["canvas_height"]) == 980
    assert str(sectioned_prompt["bundle_id"]).strip() == "pages_sectioned_infographic_v1"
    assert "scene_key_sectioned" not in sectioned_prompt
    assert "task_key_sectioned" not in sectioned_prompt
    assert "annotation_hint_section_item_count" not in sectioned_prompt
    assert "answer_hint_section_filtered_item" not in sectioned_prompt


def test_pages_mixed_infographic_page_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "mixed_infographic_page")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        cfg,
    )

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_mixed_infographic_page_v1"
    assert generation_defaults["rank_position_support"] == [2, 3]
    assert int(rendering_defaults["canvas_width"]) == 1040
    assert int(rendering_defaults["canvas_height"]) == 1320
    assert bool(rendering_defaults["pages_context_text_enabled"]) is False


def test_pages_schema_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "schema")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_weights" not in generation_defaults
    assert "balanced_query_sampling" not in generation_defaults
    assert int(generation_defaults["table_count_min"]) == 5
    assert int(generation_defaults["table_count_max"]) == 8
    assert int(generation_defaults["relationship_count_min"]) == 5
    assert int(generation_defaults["relationship_count_max"]) == 9
    assert bool(generation_defaults["balanced_context_sampling"]) is True

    assert int(rendering_defaults["canvas_width"]) == 1300
    assert int(rendering_defaults["canvas_height"]) == 950
    assert int(rendering_defaults["marker_font_size_px"]) == 12

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_schema_v1"
    assert "scene_key" not in prompt_defaults
    assert "annotation_hint_relationship_cardinality" not in prompt_defaults


def test_pages_step_list_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "step_list")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "horizontal_cards",
        "two_column_cards",
        "vertical_cards",
    ]
    assert list(generation_defaults["step_count_support"]) == [10, 11, 12, 13, 14, 15, 16]
    assert list(generation_defaults["relative_offset_support"]) == [2, 3]
    assert list(generation_defaults["between_count_support"]) == [2, 3, 4, 5, 6, 7, 8]

    assert int(rendering_defaults["canvas_width"]) == 1120
    assert int(rendering_defaults["canvas_height"]) == 980
    assert int(rendering_defaults["number_badge_size_px"]) == 30
    assert int(rendering_defaults["step_meta_font_size_px"]) == 11

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_step_list_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults
    assert "annotation_hint" not in prompt_defaults
    assert "object_description_vertical_cards" not in prompt_defaults


def test_pages_instruction_panel_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "instruction_panel")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "checklist_table",
        "manual_cards",
        "side_legend_sheet",
    ]
    assert list(generation_defaults["step_count_support"]) == [5, 6, 7, 8]
    assert list(generation_defaults["controls_per_step_support"]) == [2, 3]
    assert list(generation_defaults["control_count_support"]) == [9, 10, 11, 12]
    assert list(generation_defaults["step_set_size_support"]) == [2, 3]
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True

    assert int(rendering_defaults["canvas_width"]) == 1100
    assert int(rendering_defaults["canvas_height"]) == 880
    assert int(rendering_defaults["control_chip_height_px"]) == 30
    assert int(rendering_defaults["number_badge_size_px"]) == 34

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_instruction_panel_v1"

def test_pages_profile_card_grid_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "profile_card_grid")
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == ["compact_cards", "directory_grid"]
    assert list(generation_defaults["card_count_support"]) == [9, 12]
    assert list(generation_defaults["rank_position_support"]) == [2, 3]
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert int(rendering_defaults["canvas_width"]) == 1120
    assert int(rendering_defaults["canvas_height"]) == 860
    assert int(rendering_defaults["label_font_size_px"]) == 15
    assert str(prompt_defaults["bundle_id"]).strip() == "pages_profile_card_grid_v1"


def test_pages_category_grid_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "category_grid")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "card_grid",
        "column_groups",
        "compact_index",
    ]
    assert list(generation_defaults["category_count_support"]) == [3, 4]
    assert list(generation_defaults["subcategory_count_support"]) == [2, 3]
    assert list(generation_defaults["item_count_support"]) == [2, 3, 4, 5, 6]
    assert int(rendering_defaults["canvas_height"]) == 900
    assert str(prompt_defaults["bundle_id"]).strip() == "pages_category_grid_v1"


def test_pages_concept_map_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "concept_map")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "query_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert "balanced_query_sampling" not in generation_defaults
    assert "task_overrides" not in generation_defaults
    assert sorted(generation_defaults["context_weights"].keys()) == [
        "career_options",
        "climate_actions",
        "community_groups",
        "science_topics",
        "shopping_tips",
        "travel_plans",
    ]
    assert sorted(generation_defaults["layout_weights"].keys()) == [
        "clustered_map",
        "left_right_map",
        "radial_mind_map",
    ]
    assert sorted(generation_defaults["style_weights"].keys()) == [
        "bright_notes",
        "ink_outline",
        "soft_cards",
        "technical_pastel",
    ]
    assert int(generation_defaults["branch_count_min"]) == 5
    assert int(generation_defaults["branch_count_max"]) == 7
    assert int(generation_defaults["child_count_min"]) == 2
    assert int(generation_defaults["child_count_max"]) == 8
    assert int(rendering_defaults["canvas_width"]) == 1344
    assert int(rendering_defaults["canvas_height"]) == 944
    assert int(rendering_defaults["canvas_width"]) * int(rendering_defaults["canvas_height"]) <= 1_280_000
    assert str(prompt_defaults["bundle_id"]).strip() == "pages_concept_map_v1"


def test_pages_cycle_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "cycle")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert "query_relationship_weights" not in generation_defaults
    assert "balanced_query_relationship_sampling" not in generation_defaults
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == ["cycle_ring"]
    assert sorted(generation_defaults["cycle_direction_weights"].keys()) == ["clockwise", "counterclockwise"]
    assert int(generation_defaults["stage_count_min"]) == 5
    assert int(generation_defaults["stage_count_max"]) == 12

    assert int(rendering_defaults["canvas_width"]) == 1600
    assert int(rendering_defaults["canvas_height"]) == 1250
    assert int(rendering_defaults["outer_margin_px"]) == 220
    assert int(rendering_defaults["node_width_px"]) == 122
    assert int(rendering_defaults["ring_radius_x_px"]) == 360
    assert str(rendering_defaults["pages_context_profile"]) == "report_paragraph"
    assert str(rendering_defaults["pages_context_mode"]) == "paragraph_box"
    assert int(rendering_defaults["pages_context_simple_count"]) == 2
    assert int(rendering_defaults["pages_context_paragraph_box_count"]) == 2
    assert int(rendering_defaults["pages_context_text_max_elements"]) == 6

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_cycle_v1"

def test_pages_form_section_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "form_section")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "form_sheet",
        "invoice_sheet",
        "receipt_sheet",
    ]
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True

    assert int(rendering_defaults["canvas_width"]) == 1256
    assert int(rendering_defaults["canvas_height"]) == 1000
    assert int(rendering_defaults["sheet_page_width_px"]) == 960
    assert int(rendering_defaults["sheet_page_height_px"]) == 860
    assert int(rendering_defaults["receipt_page_width_px"]) == 520
    assert int(rendering_defaults["receipt_page_height_px"]) == 920
    assert int(rendering_defaults["section_font_size_px"]) == 24

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_form_section_v1"

def test_pages_paired_forms_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "paired_forms")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == ["purchase_receipt_pair"]
    assert int(generation_defaults["item_count_min"]) == 4
    assert int(generation_defaults["item_count_max"]) == 6
    assert int(generation_defaults["quantity_min"]) == 10
    assert int(generation_defaults["quantity_max"]) == 49
    assert int(generation_defaults["unit_value_min"]) == 2
    assert int(generation_defaults["unit_value_max"]) == 12
    assert int(generation_defaults["discrepancy_min"]) == 1
    assert int(generation_defaults["discrepancy_max"]) == 6
    assert int(generation_defaults["mismatch_count_min"]) == 2
    assert int(generation_defaults["mismatch_count_max"]) == 3
    assert int(generation_defaults["direction_count_min"]) == 1

    assert int(rendering_defaults["canvas_width"]) == 1392
    assert int(rendering_defaults["canvas_height"]) == 848
    assert int(rendering_defaults["panel_gap_px"]) == 28
    assert int(rendering_defaults["cell_font_size_px"]) == 18

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_paired_forms_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults

def test_pages_hierarchy_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "hierarchy")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == ["org_chart"]
    assert int(generation_defaults["tree_node_count_min"]) == 16
    assert int(generation_defaults["tree_node_count_max"]) == 30
    assert int(generation_defaults["manager_total_reports_min"]) == 6
    assert int(generation_defaults["manager_total_reports_max"]) == 14
    assert int(generation_defaults["manager_direct_reports_min"]) == 3
    assert int(generation_defaults["manager_direct_reports_max"]) == 5

    assert int(rendering_defaults["canvas_width"]) == 1464
    assert int(rendering_defaults["canvas_height"]) == 848
    assert int(rendering_defaults["node_width_px"]) == 84
    assert int(rendering_defaults["label_font_size_px"]) == 14
    assert str(rendering_defaults["pages_context_profile"]) == "report_paragraph"
    assert dict(rendering_defaults["pages_context_mode_weights"]) == {
        "clean": 0.24,
        "minimal": 0.2,
        "paragraph_box": 0.56,
    }
    assert int(rendering_defaults["pages_context_text_max_elements"]) == 7

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_hierarchy_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults

def test_pages_map_defaults_loaded() -> None:
    cfg = get_scene_defaults("pages", "map")
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(cfg)

    assert "query_id_weights" not in generation_defaults
    assert "balanced_query_id_sampling" not in generation_defaults
    assert "task_overrides" not in generation_defaults
    assert int(generation_defaults["landmark_count_min"]) == 8
    assert int(generation_defaults["landmark_count_max"]) == 12
    assert int(generation_defaults["direction_step_count_min"]) == 2
    assert int(generation_defaults["direction_step_count_max"]) == 4
    assert int(generation_defaults["highlighted_route_step_min"]) == 2
    assert int(generation_defaults["highlighted_route_step_max"]) == 5

    assert int(rendering_defaults["canvas_width"]) == 1280
    assert int(rendering_defaults["canvas_height"]) == 900
    assert int(rendering_defaults["landmark_width_px"]) == 126
    assert int(rendering_defaults["highlighted_path_width_px"]) == 18

    assert str(prompt_defaults["bundle_id"]).strip() == "pages_map_v1"
    assert "scene_key" not in prompt_defaults
    assert "task_key" not in prompt_defaults
