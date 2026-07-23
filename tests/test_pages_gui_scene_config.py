"""Regression tests for scene default config loading."""
from __future__ import annotations
import json
import pytest
from trace_tasks.core.scene_config import get_domain_defaults, get_scene_defaults, resolve_scene_section_defaults
from trace_tasks.tasks.shared.config_defaults import required_group_default, required_group_defaults, resolve_optional_int_bounds, resolve_required_float_bounds, resolve_required_int_bounds, split_generation_rendering_prompt_defaults
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
FULL_NODE_LINK_LAYOUT_VARIANTS = set(SUPPORTED_LAYOUT_VARIANTS)

def _assert_pages_information_style_defaults(rendering_defaults) -> None:
    assert bool(rendering_defaults['information_scene_allow_dark']) is True
    assert bool(rendering_defaults['information_scene_allow_colored_surface']) is True
    treatments = list(rendering_defaults['information_scene_treatments'])
    assert len(treatments) == 25
    assert 'dark_console_panel' in treatments
    assert 'clean_default' in treatments

def test_pages_record_table_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'record_table')
    row_generation_defaults, rendering_defaults, row_prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert 'query_id_weights' not in row_generation_defaults
    assert 'balanced_query_id_sampling' not in row_generation_defaults
    assert bool(row_generation_defaults['balanced_scene_variant_sampling']) is True
    assert bool(row_generation_defaults['balanced_style_variant_sampling']) is True
    assert list(row_generation_defaults['row_count_support']) == [9, 10, 11, 12, 13, 14, 15]
    assert list(row_generation_defaults['section_count_support']) == [2, 3]
    assert list(row_generation_defaults['answer_count_support']) == [2, 3, 4, 5, 6, 7]
    assert list(row_generation_defaults['size_threshold_support']) == [25, 35, 45, 55, 65]
    assert int(rendering_defaults['canvas_width']) == 1280
    assert int(rendering_defaults['canvas_height']) == 800
    assert int(rendering_defaults['row_height_px']) == 28
    assert str(row_prompt_defaults['bundle_id']).strip() == 'pages_record_table_v1'


def test_pages_control_board_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'control_board')
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert set(generation_defaults['scene_variant_weights'].keys()) == {'office_document', 'creative_workspace', 'developer_ide', 'cad_workspace', 'scientific_plotter', 'os_file_manager'}
    assert 'style_variant_weights' not in generation_defaults
    assert bool(generation_defaults['balanced_scene_variant_sampling']) is True
    assert 'workspace_variant_weights' not in generation_defaults
    assert 'balanced_workspace_variant_sampling' not in generation_defaults
    assert 'balanced_style_variant_sampling' not in generation_defaults
    assert list(generation_defaults['group_name_pool']) == ['Layout', 'Editing', 'Review', 'Output']
    assert list(generation_defaults['state_count_support']) == [2, 3, 4, 5, 6, 7]
    assert list(generation_defaults['extremum_state_count_support']) == [3, 4, 5]
    assert len(generation_defaults['candidate_label_pool']) == 26
    assert int(rendering_defaults['canvas_width']) == 1280
    assert int(rendering_defaults['canvas_height']) == 800
    assert int(rendering_defaults['badge_size_px']) == 28
    _assert_pages_information_style_defaults(rendering_defaults)
    assert str(prompt_defaults['bundle_id']).strip() == 'pages_control_board_v1'


def test_pages_navigation_flow_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'navigation_flow')
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert set(generation_defaults['scene_variant_weights'].keys()) == {'office_document', 'creative_workspace', 'developer_ide', 'cad_workspace', 'scientific_plotter', 'os_file_manager'}
    assert 'style_variant_weights' not in generation_defaults
    assert 'query_id_weights' not in generation_defaults
    assert 'balanced_query_id_sampling' not in generation_defaults
    assert bool(generation_defaults['balanced_scene_variant_sampling']) is True
    assert 'balanced_style_variant_sampling' not in generation_defaults
    assert int(generation_defaults['menu_command_count_min']) == 3
    assert int(generation_defaults['menu_command_count_max']) == 4
    assert int(generation_defaults['ribbon_tab_count_min']) == 3
    assert int(generation_defaults['ribbon_tab_count_max']) == 5
    assert int(generation_defaults['ribbon_group_count_min']) == 2
    assert int(generation_defaults['ribbon_group_count_max']) == 3
    assert int(generation_defaults['ribbon_command_count_min']) == 3
    assert int(generation_defaults['ribbon_command_count_max']) == 4
    assert list(generation_defaults['nav_menu_pool']) == ['File', 'Edit', 'View']
    assert list(generation_defaults['nav_submenu_pool']) == ['Arrange', 'Inspect']
    assert list(generation_defaults['nav_menu_group_pool']) == ['Primary', 'Advanced']
    assert list(generation_defaults['nav_command_pool']) == ['Align', 'Duplicate', 'Export', 'Preview']
    assert list(generation_defaults['nav_sidebar_section_pool']) == ['Workspace', 'Assets', 'Settings', 'Reports']
    assert list(generation_defaults['nav_sidebar_item_pool']) == ['Overview', 'Timeline', 'Details']
    assert list(generation_defaults['nav_ribbon_tab_pool']) == ['Home', 'Insert', 'Review', 'Analyze', 'Share']
    assert int(rendering_defaults['canvas_width']) == 1280
    assert int(rendering_defaults['canvas_height']) == 800
    assert int(rendering_defaults['badge_size_px']) == 30
    _assert_pages_information_style_defaults(rendering_defaults)
    assert str(prompt_defaults['bundle_id']).strip() == 'pages_navigation_flow_v1'


def test_pages_cycle_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'cycle')
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert set(generation_defaults['cycle_direction_weights'].keys()) == {'clockwise', 'counterclockwise'}
    assert set(generation_defaults['scene_variant_weights'].keys()) == {'cycle_ring'}
    assert bool(generation_defaults['balanced_cycle_direction_sampling']) is True
    assert 'query_id_weights' not in generation_defaults
    assert 'balanced_query_id_sampling' not in generation_defaults
    assert 'query_relationship_weights' not in generation_defaults
    assert 'balanced_query_relationship_sampling' not in generation_defaults
    assert int(generation_defaults['stage_count_min']) == 5
    assert int(generation_defaults['stage_count_max']) == 12
    assert int(rendering_defaults['canvas_width']) == 1600
    assert int(rendering_defaults['canvas_height']) == 1250
    assert int(rendering_defaults['node_width_px']) == 122
    assert str(prompt_defaults['bundle_id']).strip() == 'pages_cycle_v1'

def test_pages_web_action_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'web_action')
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert 'query_id_weights' not in generation_defaults
    assert 'balanced_query_id_sampling' not in generation_defaults
    assert set(generation_defaults['scene_variant_weights'].keys()) == {
        'content_cms',
        'finance_portal',
        'learning_portal',
        'shop_catalog',
        'support_center',
        'travel_booking',
    }
    assert 'style_variant_weights' not in generation_defaults
    assert bool(generation_defaults['balanced_scene_variant_sampling']) is True
    assert 'balanced_style_variant_sampling' not in generation_defaults
    assert int(generation_defaults['web_click_item_count_min']) == 4
    assert int(generation_defaults['web_click_item_count_max']) == 6
    assert int(generation_defaults['web_type_section_count_min']) == 3
    assert int(generation_defaults['web_select_option_count_max']) == 4
    assert list(generation_defaults['web_click_action_pool']) == ['Details', 'Compare', 'Save', 'Open']
    assert int(rendering_defaults['canvas_width']) == 1280
    assert int(rendering_defaults['canvas_height']) == 800
    assert int(rendering_defaults['browser_margin_px']) == 34
    assert int(rendering_defaults['instruction_height_px']) == 60
    _assert_pages_information_style_defaults(rendering_defaults)
    assert str(prompt_defaults['bundle_id']).strip() == 'pages_web_action_v1'


def test_pages_workspace_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults('pages', 'workspace')
    generation_defaults, rendering_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(cfg)
    assert 'query_id_weights' not in generation_defaults
    assert 'balanced_query_id_sampling' not in generation_defaults
    assert set(generation_defaults['scene_variant_weights'].keys()) == {
        'office_document',
        'creative_workspace',
        'developer_ide',
        'cad_workspace',
        'scientific_plotter',
        'os_file_manager',
    }
    assert 'style_variant_weights' not in generation_defaults
    assert bool(generation_defaults['balanced_scene_variant_sampling']) is True
    assert 'balanced_style_variant_sampling' not in generation_defaults
    assert int(generation_defaults['context_count_min']) == 3
    assert int(generation_defaults['context_count_max']) == 4
    assert int(rendering_defaults['canvas_width']) == 1280
    assert int(rendering_defaults['canvas_height']) == 800
    assert int(rendering_defaults['badge_size_px']) == 34
    assert int(rendering_defaults['label_font_size_px']) == 18
    _assert_pages_information_style_defaults(rendering_defaults)
    assert str(prompt_defaults['bundle_id']).strip() == 'pages_workspace_v1'
