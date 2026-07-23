"""Behavior tests for hexbin-density chart tasks."""
from __future__ import annotations
from collections import Counter
import pytest
from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.hexbin_density.threshold_bin_count import SUPPORTED_DENSITY_PALETTE_SCHEMES, SUPPORTED_QUERY_IDS, TASK_ID, ChartsHexbinDensityThresholdBinCountTask
from trace_tasks.tasks.registry import create_task

def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height

@pytest.mark.parametrize('query_index,query_id', list(enumerate(SUPPORTED_QUERY_IDS)))
def test_charts_hexbin_density_task_matches_contract(query_index: int, query_id: str) -> None:
    task = ChartsHexbinDensityThresholdBinCountTask()
    out = task.generate(hash64(20260606, 'hexbin_density', query_index), params={'query_id': query_id}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    assert create_task(TASK_ID).__class__.task_id == TASK_ID
    assert not hasattr(create_task(TASK_ID).__class__, 'scene_id')
    assert out.scene_id == 'hexbin_density'
    assert out.query_id == query_id
    assert str(execution['query_id']) == query_id
    assert str(execution['question_format']) == 'hexbin_density_threshold_query'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    assert str(render['density_palette_scheme']) in SUPPORTED_DENSITY_PALETTE_SCHEMES
    assert len(render['density_palette_rgb']) == 5
    assert float(render['density_palette_contrast_policy']['level_one_distance_from_white']) >= 52.0
    assert 5 <= int(execution['row_count']) <= 7
    assert 7 <= int(execution['column_count']) <= 10
    assert 24 <= int(execution['occupied_bin_count']) <= 42
    threshold = int(execution['density_threshold_level'])
    density_by_bin = {str(key): int(value) for key, value in execution['density_level_by_bin_id'].items()}
    if query_id == 'above_threshold_bin_count':
        expected_ids = [bin_id for bin_id, level in density_by_bin.items() if int(level) >= threshold]
    else:
        expected_ids = [bin_id for bin_id, level in density_by_bin.items() if int(level) < threshold]
    expected_ids = sorted(expected_ids, key=lambda bin_id: (int(execution['bins'][str(bin_id)]['row_index']), int(execution['bins'][str(bin_id)]['column_index'])))
    assert int(out.answer_gt.value) == len(expected_ids)
    assert execution['answer'] == len(expected_ids)
    assert execution['matching_bin_ids'] == expected_ids
    assert execution['annotation_bin_ids'] == expected_ids
    expected_points = [trace['render_map']['bin_centers_px'][bin_id] for bin_id in expected_ids]
    assert out.annotation_gt.value == expected_points
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == expected_points
    assert trace['projected_annotation']['pixel_point_set'] == expected_points
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    for point in out.annotation_gt.value:
        _assert_point_inside_canvas([float(value) for value in point], width=int(render['canvas_width']), height=int(render['canvas_height']))

def test_charts_hexbin_density_prompt_examples_match_contract() -> None:
    for index, query_id in enumerate(SUPPORTED_QUERY_IDS):
        out = ChartsHexbinDensityThresholdBinCountTask().generate(hash64(20260606, 'hexbin_density_prompt', index), params={'query_id': query_id}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert set(answer_only.keys()) == {'answer'}
        assert isinstance(answer_only['answer'], int)
        assert isinstance(answer_and_annotation['answer'], int)
        assert isinstance(answer_and_annotation['annotation'], list)
        assert answer_and_annotation['annotation']
        assert len(answer_and_annotation['annotation'][0]) == 2

def test_charts_hexbin_density_balanced_sampling_covers_branches_and_counts() -> None:
    queries: Counter[str] = Counter()
    answers: Counter[int] = Counter()
    thresholds: Counter[int] = Counter()
    rows: Counter[int] = Counter()
    columns: Counter[int] = Counter()
    palette_schemes: Counter[str] = Counter()
    for index in range(80):
        out = ChartsHexbinDensityThresholdBinCountTask().generate(hash64(20260606, 'hexbin_density_balanced', index), params={'_sample_cursor': index}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        queries[str(execution['query_id'])] += 1
        answers[int(out.answer_gt.value)] += 1
        thresholds[int(execution['density_threshold_level'])] += 1
        rows[int(execution['row_count'])] += 1
        columns[int(execution['column_count'])] += 1
        palette_schemes[str(out.trace_payload['render_spec']['density_palette_scheme'])] += 1
    assert set(queries) == set(SUPPORTED_QUERY_IDS)
    assert len(answers) >= 8
    assert set(thresholds).issubset({2, 3, 4, 5})
    assert len(rows) >= 3
    assert len(columns) >= 4
    assert len(palette_schemes) >= 6

def test_charts_hexbin_density_generation_is_deterministic() -> None:
    params = {'query_id': 'above_threshold_bin_count', 'row_count': 6, 'column_count': 9}
    seed = hash64(20260606, 'hexbin_density_deterministic')
    first = ChartsHexbinDensityThresholdBinCountTask().generate(seed, params=params, max_attempts=80)
    second = ChartsHexbinDensityThresholdBinCountTask().generate(seed, params=params, max_attempts=80)
    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload['execution_trace'] == second.trace_payload['execution_trace']
    assert first.image.size == second.image.size

def test_charts_hexbin_density_registry_config_and_prompt_defaults_exist() -> None:
    defaults = get_scene_defaults('charts', 'hexbin_density')
    assert defaults['prompt']['shared']['bundle_id'] == 'charts_hexbin_density_v1'
    assert defaults['prompt']['shared']['scene_key'] == 'hexbin_density_scene'
    assert defaults['prompt']['shared']['task_key'] == 'hexbin_density_threshold_query'
    assert 'query_id_weights' not in defaults['generation']['shared']
    assert sorted(defaults['rendering']['shared']['density_palette_scheme_weights']) == sorted(SUPPORTED_DENSITY_PALETTE_SCHEMES)
