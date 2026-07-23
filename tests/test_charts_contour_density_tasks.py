"""Behavior tests for contour-density chart tasks."""
from __future__ import annotations
from collections import Counter
import pytest
from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.contour_density.density_extremum_region_label import ChartsContourDensityDensityExtremumRegionLabelTask
from trace_tasks.tasks.charts.contour_density.density_threshold_region_count import ChartsContourDensityDensityThresholdRegionCountTask
from trace_tasks.tasks.charts.contour_density.reference_distance_extremum_label import ChartsContourDensityReferenceDistanceExtremumLabelTask
from trace_tasks.tasks.charts.contour_density.shared.defaults import SUPPORTED_DENSITY_EXTREMA, SUPPORTED_DENSITY_THRESHOLD_DIRECTIONS, SUPPORTED_DISTANCE_EXTREMA, SUPPORTED_REFERENCE_KINDS, SUPPORTED_SCENE_VARIANTS, SUPPORTED_SPREAD_EXTREMA
from trace_tasks.tasks.charts.contour_density.spread_extremum_region_label import ChartsContourDensitySpreadExtremumRegionLabelTask
from trace_tasks.tasks.registry import create_task
TASK_CASES = (
    (ChartsContourDensityDensityExtremumRegionLabelTask, 'highest_density_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityDensityExtremumRegionLabelTask, 'lowest_density_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'point_nearest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'point_farthest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'vertical_line_nearest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'vertical_line_farthest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'horizontal_line_nearest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityReferenceDistanceExtremumLabelTask, 'horizontal_line_farthest_region_label', 'string', 'bbox', set()),
    (ChartsContourDensityDensityThresholdRegionCountTask, 'density_at_least_threshold_region_count', 'integer', 'bbox_set', set()),
    (ChartsContourDensityDensityThresholdRegionCountTask, 'density_below_threshold_region_count', 'integer', 'bbox_set', set()),
    (ChartsContourDensitySpreadExtremumRegionLabelTask, 'widest_spread_region_label', 'string', 'bbox', set()),
    (ChartsContourDensitySpreadExtremumRegionLabelTask, 'narrowest_spread_region_label', 'string', 'bbox', set()),
)
SUPPORTED_QUERY_IDS = tuple(str(query_id) for _task_cls, query_id, _answer_type, _annotation_type, _annotation_keys in TASK_CASES)
DENSITY_QUERY_IDS = {'highest_density_region_label', 'lowest_density_region_label'}
REFERENCE_DISTANCE_QUERY_IDS = {
    'point_nearest_region_label',
    'point_farthest_region_label',
    'vertical_line_nearest_region_label',
    'vertical_line_farthest_region_label',
    'horizontal_line_nearest_region_label',
    'horizontal_line_farthest_region_label',
}
SPREAD_QUERY_IDS = {'widest_spread_region_label', 'narrowest_spread_region_label'}

def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _assert_rendered_scene_bboxes_inside_canvas(trace: dict) -> None:
    render = trace['render_spec']
    width = int(render['canvas_width'])
    height = int(render['canvas_height'])
    render_map = trace['render_map']
    for bbox in render_map.get('region_bboxes_px', {}).values():
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=width, height=height)
    for bbox in render_map.get('option_bboxes_px', {}).values():
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=width, height=height)
    for bbox in render_map.get('reference_bboxes_px', {}).values():
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=width, height=height)


def _expected_answer(execution: dict) -> str | int:
    query_id = str(execution.get('internal_query_id') or execution['query_id'])
    if query_id in DENSITY_QUERY_IDS:
        densities = {str(label): float(value) for label, value in execution['density_by_region_label'].items()}
        if str(execution['density_extremum']) == 'highest':
            return max(densities, key=lambda label: (densities[label], label))
        return min(densities, key=lambda label: (densities[label], label))
    if query_id in REFERENCE_DISTANCE_QUERY_IDS:
        distances = {str(label): float(value) for label, value in execution['distances_by_region_label'].items()}
        if str(execution['distance_extremum']) == 'nearest':
            return min(distances, key=lambda label: (distances[label], label))
        return max(distances, key=lambda label: (distances[label], label))
    if query_id in {'density_at_least_threshold_region_count', 'density_below_threshold_region_count'}:
        threshold = int(execution['density_threshold_level'])
        levels = {str(label): int(value) for label, value in execution['density_level_by_region_label'].items()}
        if str(execution['density_threshold_direction']) == 'at_least':
            return sum((1 for value in levels.values() if int(value) >= threshold))
        return sum((1 for value in levels.values() if int(value) < threshold))
    if query_id in SPREAD_QUERY_IDS:
        spreads = {str(label): float(value) for label, value in execution['footprint_area_by_region_label'].items()}
        if str(execution['spread_extremum']) == 'widest':
            return max(spreads, key=lambda label: (spreads[label], label))
        return min(spreads, key=lambda label: (spreads[label], label))
    raise AssertionError(f'unsupported query id: {query_id}')

@pytest.mark.parametrize(('task_cls', 'query_id', 'answer_type', 'annotation_type', 'annotation_keys'), TASK_CASES)
def test_charts_contour_density_tasks_match_contract(task_cls: type, query_id: str, answer_type: str, annotation_type: str, annotation_keys: set[str]) -> None:
    seed_index = SUPPORTED_QUERY_IDS.index(str(query_id))
    public_query_id = str(query_id)
    params = {'query_id': public_query_id}
    out = task_cls().generate(hash64(20260605, 'contour_density', seed_index), params=params, max_attempts=120)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    assert out.scene_id == 'contour_density'
    assert out.query_id == public_query_id
    assert str(execution['query_id']) == public_query_id
    assert str(execution['question_format']) == 'contour_density_field_query'
    assert str(execution['scene_variant']) in SUPPORTED_SCENE_VARIANTS
    assert out.answer_gt.type == answer_type
    assert out.annotation_gt.type == annotation_type
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    _assert_rendered_scene_bboxes_inside_canvas(trace)
    assert int(execution['region_count']) == len(execution['region_labels'])
    assert 5 <= int(execution['region_count']) <= 7
    expected = _expected_answer(execution)
    assert out.answer_gt.value == expected
    assert execution['answer'] == expected
    if annotation_type == 'bbox':
        assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
        assert trace['projected_annotation']['pixel_bbox'] == out.annotation_gt.value
        _assert_bbox_inside_canvas([float(value) for value in out.annotation_gt.value], width=int(render['canvas_width']), height=int(render['canvas_height']))
    elif annotation_type == 'bbox_set':
        assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
        if answer_type == 'integer':
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            assert len(execution['matching_region_labels']) == int(out.answer_gt.value)
        else:
            assert len(out.annotation_gt.value) == 1
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render['canvas_width']), height=int(render['canvas_height']))
    elif annotation_type == 'point_map':
        assert set(out.annotation_gt.value.keys()) == annotation_keys
        assert trace['projected_annotation']['point_map'] == out.annotation_gt.value
        for point in out.annotation_gt.value.values():
            _assert_point_inside_canvas([float(value) for value in point], width=int(render['canvas_width']), height=int(render['canvas_height']))
    else:
        assert set(out.annotation_gt.value.keys()) == annotation_keys
        assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
        for bbox in out.annotation_gt.value.values():
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render['canvas_width']), height=int(render['canvas_height']))

def test_charts_contour_density_prompt_examples_match_contract() -> None:
    for seed_index, (task_cls, _query_id, answer_type, annotation_type, annotation_keys) in enumerate(TASK_CASES):
        public_query_id = str(_query_id)
        params = {'query_id': public_query_id}
        out = task_cls().generate(hash64(20260605, 'contour_density_prompt', seed_index), params=params, max_attempts=120)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        if annotation_type == 'bbox':
            assert isinstance(answer_and_annotation['annotation'], list)
            assert len(answer_and_annotation['annotation']) == 4
        elif annotation_type == 'bbox_set':
            assert isinstance(answer_and_annotation['annotation'], list)
            assert answer_and_annotation['annotation']
        else:
            assert set(answer_and_annotation['annotation'].keys()) == annotation_keys
        assert set(answer_only.keys()) == {'answer'}
        if answer_type == 'integer':
            assert isinstance(answer_and_annotation['answer'], int)
            assert isinstance(answer_only['answer'], int)
        else:
            assert isinstance(answer_and_annotation['answer'], str)
            assert isinstance(answer_only['answer'], str)
            assert len(answer_and_annotation['answer']) > 1

def test_charts_contour_density_balanced_sampling_covers_axes() -> None:
    scene_variants: Counter[str] = Counter()
    density_query_ids: Counter[str] = Counter()
    density_extrema: Counter[str] = Counter()
    reference_query_ids: Counter[str] = Counter()
    distance_extrema: Counter[str] = Counter()
    reference_kinds: Counter[str] = Counter()
    threshold_query_ids: Counter[str] = Counter()
    threshold_directions: Counter[str] = Counter()
    threshold_counts: Counter[int] = Counter()
    spread_query_ids: Counter[str] = Counter()
    spread_extrema: Counter[str] = Counter()
    for index in range(90):
        density = ChartsContourDensityDensityExtremumRegionLabelTask().generate(hash64(20260605, 'contour_density_density', index), params={}, max_attempts=120)
        scene_variants[str(density.trace_payload['execution_trace']['scene_variant'])] += 1
        density_query_ids[str(density.query_id)] += 1
        density_extrema[str(density.trace_payload['execution_trace']['density_extremum'])] += 1
        distance = ChartsContourDensityReferenceDistanceExtremumLabelTask().generate(hash64(20260605, 'contour_density_distance', index), params={}, max_attempts=120)
        scene_variants[str(distance.trace_payload['execution_trace']['scene_variant'])] += 1
        reference_query_ids[str(distance.query_id)] += 1
        distance_extrema[str(distance.trace_payload['execution_trace']['distance_extremum'])] += 1
        reference_kinds[str(distance.trace_payload['execution_trace']['reference_kind'])] += 1
        threshold = ChartsContourDensityDensityThresholdRegionCountTask().generate(hash64(20260605, 'contour_density_threshold', index), params={}, max_attempts=120)
        scene_variants[str(threshold.trace_payload['execution_trace']['scene_variant'])] += 1
        threshold_query_ids[str(threshold.query_id)] += 1
        threshold_directions[str(threshold.trace_payload['execution_trace']['density_threshold_direction'])] += 1
        threshold_counts[int(threshold.answer_gt.value)] += 1
        spread = ChartsContourDensitySpreadExtremumRegionLabelTask().generate(hash64(20260605, 'contour_density_spread', index), params={}, max_attempts=120)
        scene_variants[str(spread.trace_payload['execution_trace']['scene_variant'])] += 1
        spread_query_ids[str(spread.query_id)] += 1
        spread_extrema[str(spread.trace_payload['execution_trace']['spread_extremum'])] += 1
    assert set(scene_variants) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(density_query_ids) == DENSITY_QUERY_IDS
    assert set(density_extrema) == set(SUPPORTED_DENSITY_EXTREMA)
    assert set(reference_query_ids) == REFERENCE_DISTANCE_QUERY_IDS
    assert set(distance_extrema) == set(SUPPORTED_DISTANCE_EXTREMA)
    assert set(reference_kinds) == set(SUPPORTED_REFERENCE_KINDS)
    assert set(threshold_query_ids) == {'density_at_least_threshold_region_count', 'density_below_threshold_region_count'}
    assert set(threshold_directions) == set(SUPPORTED_DENSITY_THRESHOLD_DIRECTIONS)
    assert set(threshold_counts).issubset({1, 2, 3, 4, 5})
    assert len(threshold_counts) >= 4
    assert set(spread_query_ids) == SPREAD_QUERY_IDS
    assert set(spread_extrema) == set(SUPPORTED_SPREAD_EXTREMA)

def test_charts_contour_density_is_deterministic() -> None:
    params = {'scene_variant': 'scatter_contour', 'query_id': 'vertical_line_farthest_region_label'}
    out_a = ChartsContourDensityReferenceDistanceExtremumLabelTask().generate(202606051, params=params, max_attempts=120)
    out_b = ChartsContourDensityReferenceDistanceExtremumLabelTask().generate(202606051, params=params, max_attempts=120)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload['execution_trace'] == out_b.trace_payload['execution_trace']
    assert out_a.trace_payload['render_map'] == out_b.trace_payload['render_map']


def test_charts_contour_density_spread_regions_stay_inside_canvas_regression() -> None:
    out = ChartsContourDensitySpreadExtremumRegionLabelTask().generate(
        6616146116377272,
        params={'query_id': 'widest_spread_region_label'},
        max_attempts=120,
    )
    _assert_rendered_scene_bboxes_inside_canvas(out.trace_payload)


def test_charts_contour_density_registry_and_config_are_wired() -> None:
    for seed_index, (task_cls, query_id, _answer_type, _annotation_type, _annotation_keys) in enumerate(TASK_CASES):
        task = create_task(str(task_cls.task_id))
        assert isinstance(task, task_cls)
        public_query_id = str(query_id)
        params = {'query_id': public_query_id}
        out = task.generate(hash64(20260605, 'contour_density_registry', seed_index), params=params, max_attempts=120)
        assert out.query_id == public_query_id
    defaults = get_scene_defaults('charts', 'contour_density')
    assert 'density_extremum_weights' not in defaults['generation']['shared']
    assert 'distance_extremum_weights' not in defaults['generation']['shared']
    assert 'reference_kind_weights' not in defaults['generation']['shared']
    assert 'spread_extremum_weights' not in defaults['generation']['shared']
    prompt = defaults['prompt']['shared']
    assert str(prompt['bundle_id']) == 'charts_contour_density_v1'
