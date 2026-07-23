"""Tests for named-grid row/column line-condition counting."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
TASK_ID = 'task_icons__named_grid__group_predicate_count'
QUERY_IDS = ('row_at_least_shape_count', 'column_at_least_shape_count', 'row_exactly_shape_count', 'column_exactly_shape_count', 'row_no_shape_count', 'column_no_shape_count')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def _line_counts(shape_grid: list[list[str]], *, axis: str, target_shape_id: str) -> list[int]:
    if axis == 'row':
        return [sum((1 for value in row if str(value) == str(target_shape_id))) for row in shape_grid]
    return [sum((1 for row in shape_grid if str(row[col]) == str(target_shape_id))) for col in range(len(shape_grid[0]))]

def _qualifies(count: int, *, condition: str, threshold: int) -> bool:
    if condition == 'at_least':
        return int(count) >= int(threshold)
    if condition == 'exactly':
        return int(count) == int(threshold)
    if condition == 'none':
        return int(count) == 0
    raise AssertionError(f'unsupported condition: {condition}')

def test_icons_counting_named_grid_row_at_least_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-line-row-at-least', 0), params={'query_id': 'row_at_least_shape_count', 'target_shape_id': 'star', 'grid_rows': 5, 'grid_cols': 4, 'answer_count': 2, 'threshold': 3}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    counts = _line_counts(execution['shape_ids_by_cell'], axis='row', target_shape_id=str(execution['target_shape_id']))
    qualifying_indices = [index for index, count in enumerate(counts) if int(count) >= 3]
    expected_bboxes = sorted(trace['render_map']['qualifying_line_region_bboxes_px'].values())
    assert out.scene_id == 'named_grid'
    assert out.query_id == 'row_at_least_shape_count'
    assert out.answer_gt.type == 'integer'
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 2
    assert trace['scene_ir']['scene_kind'] == 'icons_named_grid_line_condition_count'
    assert execution['question_format'] == 'count_grid_lines_satisfying_named_shape_count_condition'
    assert str(execution['queried_axis']) == 'row'
    assert str(execution['condition']) == 'at_least'
    assert int(execution['threshold']) == 3
    assert counts == execution['row_target_counts']
    assert qualifying_indices == execution['qualifying_line_indices']
    assert len(qualifying_indices) == 2
    assert sorted(out.annotation_gt.value) == expected_bboxes
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    style = trace['render_spec']['style']
    assert 'axis_label_stroke_rgb' in style
    assert style['text_legibility']['required_role_count'] >= 2
    assert style['text_legibility']['failure_count'] == 0
    assert '"star"' in out.prompt
    assert 'at least 3' in out.prompt

def test_icons_counting_named_grid_column_no_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-line-column-no', 0), params={'query_id': 'column_no_shape_count', 'target_shape_id': 'guitar', 'grid_rows': 5, 'grid_cols': 6, 'answer_count': 2}, max_attempts=100)
    execution = out.trace_payload['execution_trace']
    counts = _line_counts(execution['shape_ids_by_cell'], axis='column', target_shape_id=str(execution['target_shape_id']))
    qualifying_indices = [index for index, count in enumerate(counts) if int(count) == 0]
    assert out.query_id == 'column_no_shape_count'
    assert out.answer_gt.value == 2
    assert len(out.annotation_gt.value) == 2
    assert str(execution['queried_axis']) == 'column'
    assert str(execution['condition']) == 'none'
    assert int(execution['threshold']) == 0
    assert counts == execution['column_target_counts']
    assert qualifying_indices == execution['qualifying_line_indices']
    assert 'no "guitar"' in out.prompt

def test_icons_counting_named_grid_line_condition_prompt_example_matches_contract() -> None:
    task = create_task(TASK_ID)
    out = task.generate(hash64(20260528, 'named-grid-line-prompt', 0), params={'query_id': 'row_exactly_shape_count', 'target_shape_id': 'bell', 'answer_count': 2, 'threshold': 2}, max_attempts=100)
    assert '"bell"' in out.prompt
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert all((len(bbox) == 4 for bbox in answer_and_annotation['annotation']))
    assert answer_and_annotation['answer'] == 2

def test_icons_counting_named_grid_line_condition_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    grid_sizes: Counter[str] = Counter()
    for index in range(180):
        out = task.generate(hash64(20260528, 'named-grid-line-sampling', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_id = str(out.query_id)
        axis = str(execution['queried_axis'])
        target_shape_id = str(execution['target_shape_id'])
        condition = str(execution['condition'])
        threshold = int(execution['threshold'])
        counts = _line_counts(execution['shape_ids_by_cell'], axis=axis, target_shape_id=target_shape_id)
        qualifying_indices = [index for index, count in enumerate(counts) if _qualifies(int(count), condition=condition, threshold=int(threshold))]
        assert query_id in QUERY_IDS
        assert 0 <= int(out.answer_gt.value) <= 5
        assert len(qualifying_indices) == int(out.answer_gt.value)
        assert qualifying_indices == execution['qualifying_line_indices']
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert 4 <= int(execution['grid_rows']) <= 6
        assert 4 <= int(execution['grid_cols']) <= 6
        if condition == 'at_least':
            assert 2 <= threshold <= 3
        elif condition == 'exactly':
            assert 1 <= threshold <= 3
        else:
            assert threshold == 0
        query_counts[query_id] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        grid_sizes[f"{int(execution['grid_rows'])}x{int(execution['grid_cols'])}"] += 1
    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts) == set(range(0, 6))
    assert len(grid_sizes) >= 6
