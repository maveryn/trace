"""Tests for named-ring directed arc icon counting."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
TASK_ID = 'task_icons__named_ring__scoped_attribute_count'
QUERY_IDS = ('clockwise_arc_shape_count', 'counterclockwise_arc_shape_count')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def _arc_indices(start_index: int, end_index: int, *, count: int, direction: str) -> list[int]:
    step = 1 if direction == 'clockwise' else -1
    values: list[int] = []
    cursor = (int(start_index) + step) % int(count)
    while int(cursor) != int(end_index):
        values.append(int(cursor))
        cursor = (int(cursor) + step) % int(count)
    return values

def _target_count_on_arc(execution: dict) -> int:
    target = str(execution['target_shape_id'])
    shape_ids = [str(value) for value in execution['clockwise_order_shape_ids']]
    return sum((1 for index in execution['arc_indices'] if shape_ids[int(index)] == target))

def test_icons_counting_named_ring_clockwise_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-ring-clockwise', 0), params={'query_id': 'clockwise_arc_shape_count', 'target_shape_id': 'star', 'answer_count': 3, 'ring_icon_count': 14, 'arc_span_count': 7, 'start_index': 2, 'off_arc_target_count': 2}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    entity_by_id = {str(entity['instance_id']): entity for entity in entities}
    counted_ids = list(trace['render_map']['counted_instance_ids'])
    expected_arc = _arc_indices(int(execution['start_index']), int(execution['end_index']), count=int(execution['ring_icon_count']), direction='clockwise')
    assert out.scene_id == 'named_ring'
    assert out.query_id == 'clockwise_arc_shape_count'
    assert out.answer_gt.type == 'integer'
    assert out.answer_gt.value == 3
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert trace['scene_ir']['scene_kind'] == 'icons_named_ring'
    assert execution['question_format'] == 'count_named_shape_icons_strictly_between_ring_markers'
    assert execution['direction'] == 'clockwise'
    assert execution['arc_indices'] == expected_arc
    assert _target_count_on_arc(execution) == 3
    assert len(counted_ids) == 3
    assert all((str(entity_by_id[instance_id]['shape_id']) == 'star' for instance_id in counted_ids))
    assert all((int(entity_by_id[instance_id]['ring_index']) in set(execution['arc_indices']) for instance_id in counted_ids))
    assert sorted(out.annotation_gt.value) == sorted((entity_by_id[instance_id]['bbox_xyxy'] for instance_id in counted_ids))
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    style = trace['render_spec']['style']
    assert 'marker_label_stroke_rgb' in style
    assert style['text_legibility']['required_role_count'] >= 2
    assert style['text_legibility']['failure_count'] == 0
    assert '"star"' in out.prompt
    assert 'clockwise' in out.prompt
    assert 'strictly between A and B' in out.prompt
    assert set(trace['render_map']['marker_label_bboxes_px']) == {'A', 'B'}

def test_icons_counting_named_ring_counterclockwise_zero_answer_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-ring-counterclockwise', 0), params={'query_id': 'counterclockwise_arc_shape_count', 'target_shape_id': 'guitar', 'answer_count': 0, 'ring_icon_count': 16, 'arc_span_count': 5, 'start_index': 9, 'off_arc_target_count': 2}, max_attempts=100)
    execution = out.trace_payload['execution_trace']
    expected_arc = _arc_indices(int(execution['start_index']), int(execution['end_index']), count=int(execution['ring_icon_count']), direction='counterclockwise')
    target = str(execution['target_shape_id'])
    shape_ids = [str(value) for value in execution['clockwise_order_shape_ids']]
    assert out.query_id == 'counterclockwise_arc_shape_count'
    assert out.answer_gt.value == 0
    assert out.annotation_gt.value == []
    assert execution['direction'] == 'counterclockwise'
    assert execution['arc_indices'] == expected_arc
    assert _target_count_on_arc(execution) == 0
    assert all((shape_ids[int(index)] != target for index in execution['arc_indices']))
    assert len(execution['off_arc_target_indices']) == 2
    assert all((shape_ids[int(index)] == target for index in execution['off_arc_target_indices']))
    assert 'counterclockwise' in out.prompt
    assert '"guitar"' in out.prompt

def test_icons_counting_named_ring_prompt_example_matches_contract() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-ring-prompt', 0), params={'query_id': 'clockwise_arc_shape_count', 'target_shape_id': 'bell', 'answer_count': 2}, max_attempts=100)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert all((len(bbox) == 4 for bbox in answer_and_annotation['annotation']))
    assert answer_and_annotation['answer'] == 2

def test_icons_counting_named_ring_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    ring_counts: Counter[int] = Counter()
    for index in range(180):
        out = task.generate(hash64(20260528, 'named-ring-sampling', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_id = str(out.query_id)
        expected_arc = _arc_indices(int(execution['start_index']), int(execution['end_index']), count=int(execution['ring_icon_count']), direction=str(execution['direction']))
        assert query_id in QUERY_IDS
        assert 0 <= int(out.answer_gt.value) <= 6
        assert 12 <= int(execution['ring_icon_count']) <= 22
        assert 3 <= int(execution['arc_span_count']) <= 12
        assert execution['arc_indices'] == expected_arc
        assert _target_count_on_arc(execution) == int(out.answer_gt.value)
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        query_counts[query_id] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        ring_counts[int(execution['ring_icon_count'])] += 1
    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts) == set(range(0, 7))
    assert len(ring_counts) >= 6
