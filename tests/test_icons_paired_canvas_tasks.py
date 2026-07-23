"""Tests for paired-canvas icon tasks."""
from __future__ import annotations
import json
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
PAIRED_TASKS = (
    'task_icons__paired_canvas__panel_set_relation_count',
    'task_icons__paired_canvas__color_change_count',
    'task_icons__paired_canvas__rotation_change_count',
)

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def _panel_entities(out, panel: str) -> list[dict]:
    return [dict(entity) for entity in out.trace_payload['scene_ir']['entities'] if str(entity.get('panel')) == str(panel)]

def test_icons_paired_canvas_added_removed_contracts() -> None:
    for query_id, annotation_panel in (('added_in_right_count', 'right'), ('missing_from_right_count', 'left')):
        out = create_task('task_icons__paired_canvas__panel_set_relation_count').generate(hash64(20260519002, query_id), params={'query_id': query_id, 'target_count': 2, 'distractor_count': 2}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        panel_entities = _panel_entities(out, annotation_panel)
        indices = execution['matching_right_indices'] if annotation_panel == 'right' else execution['matching_left_indices']
        assert out.query_id == query_id
        assert execution['annotation_panel'] == annotation_panel
        assert int(out.answer_gt.value) == 2
        assert len(out.annotation_gt.value) == 2
        assert sorted(out.annotation_gt.value) == sorted([panel_entities[index]['bbox_xyxy'] for index in indices])
        assert out.trace_payload['projected_annotation']['type'] == 'bbox_set'

def test_icons_paired_canvas_attribute_change_contracts() -> None:
    cases = (
        ('task_icons__paired_canvas__color_change_count', 'color_changed_count', 'color'),
        ('task_icons__paired_canvas__rotation_change_count', 'rotation_changed_count', 'rotation'),
    )
    for task_id, internal_query_id, attribute in cases:
        out = create_task(task_id).generate(hash64(20260519003, task_id), params={'target_count': 2, 'distractor_count': 3}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        right = _panel_entities(out, 'right')
        assert out.query_id == 'single'
        assert execution['internal_query_id'] == internal_query_id
        assert execution['active_attribute'] == attribute
        assert int(out.answer_gt.value) == 2
        assert len(out.annotation_gt.value) == 2
        assert out.trace_payload['projected_annotation']['type'] == 'bbox_set'
        for index, entity in enumerate(right):
            has_attribute = attribute in set((str(value) for value in entity.get('changed_attributes', [])))
            assert has_attribute is (index in set(execution['matching_right_indices']))

def test_icons_paired_canvas_prompt_examples_and_balanced_queries() -> None:
    out = create_task('task_icons__paired_canvas__panel_set_relation_count').generate(20260519005, params={'query_id': 'added_in_right_count'}, max_attempts=200)
    assert _extract_prompt_json_example(out.prompt_variants['answer_only']) == {'answer': 2}
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert isinstance(answer_and_annotation['answer'], int)
    expected_queries = {
        'task_icons__paired_canvas__panel_set_relation_count': {'added_in_right_count', 'missing_from_right_count'},
        'task_icons__paired_canvas__color_change_count': {'single'},
        'task_icons__paired_canvas__rotation_change_count': {'single'},
    }
    for task_id, expected in expected_queries.items():
        observed: set[str] = set()
        for index, query_id in enumerate(sorted(expected)):
            out = create_task(task_id).generate(
                hash64(20260519006, f'{task_id}:{query_id}', index),
                params={'query_id': query_id} if query_id != 'single' else {},
                max_attempts=200,
            )
            observed.add(str(out.query_id))
            assert out.scene_id == 'paired_canvas'
            assert out.trace_payload['query_spec']['query_id'] == query_id
            assert out.trace_payload['query_spec']['params']['query_id'] == query_id
        assert observed == expected
