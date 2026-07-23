"""Behavior tests for radar chart tasks."""
from __future__ import annotations
import pytest
from tests.helpers import extract_prompt_json_example
from trace_tasks.tasks.charts.radar.highlighted_metric_threshold_panel_count import ChartsRadarHighlightedMetricThresholdPanelCountTask
from trace_tasks.tasks.charts.radar.matching_condition_panel_count import ChartsRadarMatchingConditionPanelCountTask
from trace_tasks.tasks.charts.radar.profile_advantage_count import ChartsRadarProfileAdvantageCountTask
from trace_tasks.tasks.charts.radar.threshold_metric_count_for_panel import ChartsRadarThresholdMetricCountForPanelTask
CASES = ((ChartsRadarHighlightedMetricThresholdPanelCountTask, 'bbox_set'), (ChartsRadarMatchingConditionPanelCountTask, 'bbox_set'), (ChartsRadarThresholdMetricCountForPanelTask, 'point_set'), (ChartsRadarProfileAdvantageCountTask, 'segment_set'))

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

def _annotation_len(value) -> int:
    return len(value)

@pytest.mark.parametrize(('task_cls', 'annotation_type'), CASES)
def test_charts_radar_tasks_match_contract(task_cls: type, annotation_type: str) -> None:
    out = task_cls().generate(94600, params={'query_id': 'single'}, max_attempts=120)
    trace = out.trace_payload
    render = trace['render_spec']
    projected = trace['projected_annotation']
    execution = trace['execution_trace']
    width = int(render['canvas_width'])
    height = int(render['canvas_height'])
    assert out.scene_id == 'radar'
    assert out.query_id == 'single'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == annotation_type
    assert projected['type'] == annotation_type
    assert int(out.answer_gt.value) == int(execution['answer'])
    if annotation_type == 'bbox_set':
        assert projected['bbox_set'] == out.annotation_gt.value
        assert int(out.answer_gt.value) == _annotation_len(out.annotation_gt.value)
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=width, height=height)
    elif annotation_type == 'point_set':
        assert projected['point_set'] == out.annotation_gt.value
        assert projected['pixel_point_set'] == out.annotation_gt.value
        assert int(out.answer_gt.value) == _annotation_len(out.annotation_gt.value)
        for point in out.annotation_gt.value:
            _assert_point_inside_canvas([float(value) for value in point], width=width, height=height)
    else:
        assert annotation_type == 'segment_set'
        assert projected['segment_set'] == out.annotation_gt.value
        assert int(out.answer_gt.value) == _annotation_len(out.annotation_gt.value)
        for pair in out.annotation_gt.value:
            assert len(pair) == 2
            for point in pair:
                _assert_point_inside_canvas([float(value) for value in point], width=width, height=height)

def test_charts_radar_prompt_examples_match_annotation_contracts() -> None:
    for task_cls, annotation_type in CASES:
        out = task_cls().generate(94700, params={'query_id': 'single'}, max_attempts=120)
        example = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert isinstance(example['answer'], int)
        assert isinstance(answer_only['answer'], int)
        if annotation_type == 'bbox_set':
            assert isinstance(example['annotation'], list)
            assert len(example['annotation'][0]) == 4
        elif annotation_type == 'point_set':
            assert isinstance(example['annotation'], list)
            assert len(example['annotation'][0]) == 2
        else:
            assert isinstance(example['annotation'], list)
            assert len(example['annotation'][0]) == 2
            assert len(example['annotation'][0][0]) == 2
