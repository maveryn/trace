"""Regression tests for chart scene default config loading."""
from __future__ import annotations
import pytest
from trace_tasks.core.scene_config import get_scene_defaults, resolve_scene_section_defaults
from trace_tasks.tasks.shared.config_defaults import resolve_optional_int_bounds, resolve_required_float_bounds, resolve_required_int_bounds, split_generation_rendering_prompt_defaults

@pytest.mark.parametrize(('scene_id', 'task_id'), [('single_series', 'task_charts__single_series__observed_threshold_crossing_label'), ('histogram', 'task_charts__histogram__interval_mass'), ('boxplot', 'task_charts__boxplot__iqr_extremum_label'), ('composition_panels', 'task_charts__composition_panels__conditioned_panel_sum_from_percent'), ('region_map', 'task_charts__region_map__numeric_threshold_region_count'), ('scatter_cluster', 'task_charts__scatter_cluster__cluster_spread_extremum_label'), ('surface_3d', 'task_charts__surface_3d__reference_nearest_label'), ('multiseries', 'task_charts__multiseries__category_total_extremum_label')])
def test_chart_scene_defaults_loaded(scene_id: str, task_id: str) -> None:
    cfg = get_scene_defaults('charts', scene_id)
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
    assert generation
    assert rendering
    assert prompt
    assert int(rendering['canvas_width']) > 0
    assert int(rendering['canvas_height']) > 0
    assert str(prompt['bundle_id']).strip()
    if 'scene_key' in prompt:
        assert str(prompt['scene_key']).strip()

def test_chart_scene_task_overrides_are_resolved() -> None:
    cfg = get_scene_defaults('charts', 'single_series')
    generation = resolve_scene_section_defaults(cfg, 'generation', task_id='task_charts__single_series__observed_threshold_crossing_label')
    assert generation
    assert generation == cfg['generation']['shared']
    assert 'query_id_weights' not in generation
    assert int(generation['mark_count_min']) <= int(generation['mark_count_max'])

def test_chart_scene_required_bound_helpers_work_on_scene_defaults() -> None:
    cfg = get_scene_defaults('charts', 'multiseries')
    generation = resolve_scene_section_defaults(cfg, 'generation', task_id='task_charts__multiseries__category_total_extremum_label')
    rendering = resolve_scene_section_defaults(cfg, 'rendering')
    category_bounds = resolve_required_int_bounds(generation, {}, min_key='category_count_min', max_key='category_count_max', fallback_min=1, fallback_max=1, context='category count')
    value_bounds = resolve_required_float_bounds(generation, {}, min_key='value_min', max_key='value_max', fallback_min=0.0, fallback_max=1.0, context='value')
    optional_bounds = resolve_optional_int_bounds(rendering, {}, min_key='label_font_size_px_min', max_key='label_font_size_px_max', context='label font size')
    assert int(category_bounds[0]) <= int(category_bounds[1])
    assert float(value_bounds[0]) <= float(value_bounds[1])
    if optional_bounds[0] is not None and optional_bounds[1] is not None:
        assert int(optional_bounds[0]) >= 0
        assert int(optional_bounds[1]) >= 0
