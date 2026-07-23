"""Tests for named-grid row/column icon counting."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
TASK_ID = 'task_icons__named_grid__scoped_attribute_count'
QUERY_IDS = ('row_shape_count', 'column_shape_count')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def _count_target_in_line(shape_grid: list[list[str]], *, query_id: str, line_index: int, target_shape_id: str) -> int:
    if query_id == 'row_shape_count':
        return sum((1 for shape_id in shape_grid[int(line_index)] if str(shape_id) == str(target_shape_id)))
    return sum((1 for row in shape_grid if str(row[int(line_index)]) == str(target_shape_id)))

def test_icons_counting_named_grid_row_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-row', 0), params={'query_id': 'row_shape_count', 'target_shape_id': 'star', 'target_count': 4, 'grid_rows': 4, 'grid_cols': 5, 'target_row_number': 3, 'off_line_target_count': 2}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    counted_ids = set(trace['render_map']['counted_instance_ids'])
    entity_by_id = {str(entity['instance_id']): entity for entity in entities}
    assert out.scene_id == 'named_grid'
    assert out.query_id == 'row_shape_count'
    assert out.answer_gt.type == 'integer'
    assert out.answer_gt.value == 4
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 4
    assert trace['scene_ir']['scene_kind'] == 'icons_named_grid_row_column_shape_count'
    assert execution['question_format'] == 'count_named_shape_in_grid_row_or_column'
    assert int(execution['grid_rows']) == 4
    assert int(execution['grid_cols']) == 5
    assert int(execution['queried_number']) == 3
    assert str(execution['queried_axis']) == 'row'
    assert _count_target_in_line(execution['shape_ids_by_cell'], query_id='row_shape_count', line_index=2, target_shape_id='star') == 4
    assert len(entities) == 20
    assert all((str(entity_by_id[instance_id]['shape_id']) == 'star' for instance_id in counted_ids))
    assert all((int(entity_by_id[instance_id]['row_number']) == 3 for instance_id in counted_ids))
    assert sorted(out.annotation_gt.value) == sorted((entity_by_id[instance_id]['bbox_xyxy'] for instance_id in counted_ids))
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    style = trace['render_spec']['style']
    assert 'axis_label_stroke_rgb' in style
    assert style['text_legibility']['required_role_count'] >= 2
    assert style['text_legibility']['failure_count'] == 0
    assert '"star"' in out.prompt
    assert 'row 3' in out.prompt

def test_icons_counting_named_grid_column_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-column', 0), params={'query_id': 'column_shape_count', 'target_shape_id': 'bell', 'target_count': 5, 'grid_rows': 5, 'grid_cols': 4, 'target_column_number': 2, 'off_line_target_count': 1}, max_attempts=100)
    execution = out.trace_payload['execution_trace']
    entities = out.trace_payload['scene_ir']['entities']
    counted_ids = set(out.trace_payload['render_map']['counted_instance_ids'])
    entity_by_id = {str(entity['instance_id']): entity for entity in entities}
    assert out.query_id == 'column_shape_count'
    assert out.answer_gt.value == 5
    assert len(out.annotation_gt.value) == 5
    assert str(execution['queried_axis']) == 'column'
    assert int(execution['queried_number']) == 2
    assert _count_target_in_line(execution['shape_ids_by_cell'], query_id='column_shape_count', line_index=1, target_shape_id='bell') == 5
    assert all((str(entity_by_id[instance_id]['shape_id']) == 'bell' for instance_id in counted_ids))
    assert all((int(entity_by_id[instance_id]['column_number']) == 2 for instance_id in counted_ids))
    assert '"bell"' in out.prompt
    assert 'column 2' in out.prompt

def test_icons_counting_named_grid_prompt_example_matches_contract() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-prompt', 0), params={'query_id': 'row_shape_count', 'target_shape_id': 'guitar', 'target_count': 3}, max_attempts=100)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 3}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 3
    assert all((len(bbox) == 4 for bbox in answer_and_annotation['annotation']))
    assert answer_and_annotation['answer'] == 3

def test_icons_counting_named_grid_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    grid_sizes: Counter[str] = Counter()
    for index in range(100):
        out = task.generate(hash64(20260528, 'named-grid-sampling', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_id = str(out.query_id)
        target_shape_id = str(execution['target_shape_id'])
        line_index = int(execution['queried_index'])
        shape_grid = execution['shape_ids_by_cell']
        assert query_id in QUERY_IDS
        assert 1 <= int(out.answer_gt.value) <= 5
        assert _count_target_in_line(shape_grid, query_id=query_id, line_index=int(line_index), target_shape_id=target_shape_id) == int(out.answer_gt.value)
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert 4 <= int(execution['grid_rows']) <= 6
        assert 4 <= int(execution['grid_cols']) <= 6
        query_counts[query_id] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        grid_sizes[f"{int(execution['grid_rows'])}x{int(execution['grid_cols'])}"] += 1
    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts) == set(range(1, 6))
    assert len(grid_sizes) >= 4
