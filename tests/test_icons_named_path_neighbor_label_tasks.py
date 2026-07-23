"""Behavior tests for named-icon path-neighbor labels."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.named_path.path_neighbor_label import IconsNamedPathPathNeighborLabelTask
QUERY_IDS = ('after_first_shape_label', 'before_first_shape_label', 'after_last_shape_label', 'before_last_shape_label', 'after_second_shape_label', 'before_second_shape_label')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def test_icons_named_path_neighbor_contract_matches_scene() -> None:
    task = IconsNamedPathPathNeighborLabelTask()
    out = task.generate(2026052801, params={'query_id': 'after_second_shape_label', 'answer_label': 'E', 'distractor_count': 5, 'target_occurrence_count': 3}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    assert out.scene_id == 'named_path'
    assert out.query_id == 'after_second_shape_label'
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'E'
    assert out.annotation_gt.type == 'bbox'
    assert len(out.annotation_gt.value) == 4
    assert trace['scene_ir']['scene_kind'] == 'icons_named_path_neighbor'
    assert execution['question_format'] == 'select_labeled_neighbor_of_named_icon_along_start_to_end_path'
    labels = [str(entity['label']) for entity in entities if str(entity['label'])]
    assert sorted(labels) == list('ABCDEF')
    assert int(execution['stop_count']) == 14
    assert int(execution['distractor_count']) == 5
    assert int(execution['target_occurrence_count']) == 3
    target_positions = [int(value) for value in execution['target_positions']]
    assert len(target_positions) == 3
    assert all((0 < position < int(execution['stop_count']) - 1 for position in target_positions))
    assert all((right - left > 1 for left, right in zip(target_positions, target_positions[1:])))
    query_position = int(execution['query_position_index'])
    answer_position = int(execution['answer_position_index'])
    assert query_position == target_positions[1]
    assert answer_position == query_position + 1
    assert str(execution['labels_by_position'][str(answer_position)]) == 'E'
    query_entity = next((entity for entity in entities if int(entity['position_index']) == query_position))
    answer_entity = next((entity for entity in entities if int(entity['position_index']) == answer_position))
    assert bool(query_entity['is_query_occurrence']) is True
    assert bool(answer_entity['is_answer_neighbor']) is True
    assert str(query_entity['shape_id']) == str(execution['target_shape_id'])
    assert str(query_entity['label']) == ''
    assert str(answer_entity['label']) == 'E'
    assert str(answer_entity['shape_id']) != str(execution['target_shape_id'])
    expected_annotation = answer_entity['bbox_xyxy']
    assert out.annotation_gt.value == expected_annotation
    assert trace['projected_annotation']['type'] == 'bbox'
    assert trace['projected_annotation']['bbox'] == expected_annotation
    assert trace['projected_annotation']['pixel_bbox'] == expected_annotation
    style = trace['render_spec']['style']
    assert 'candidate_label_stroke_rgb' in style
    assert 'endpoint_label_stroke_rgb' in style
    assert style['text_legibility']['required_role_count'] >= 3
    assert style['text_legibility']['failure_count'] == 0
    assert len(trace['scene_ir']['frames']['path']['points_xy']) == int(execution['stop_count'])

def test_icons_named_path_neighbor_query_variants() -> None:
    task = IconsNamedPathPathNeighborLabelTask()
    for index, query_id in enumerate(QUERY_IDS):
        out = task.generate(2026052802 + index, params={'query_id': query_id, 'answer_label': 'B', 'distractor_count': 4, 'target_occurrence_count': 3}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        target_positions = [int(value) for value in execution['target_positions']]
        query_position = int(execution['query_position_index'])
        answer_position = int(execution['answer_position_index'])
        if query_id == 'after_first_shape_label':
            assert query_position == target_positions[0]
            assert answer_position == query_position + 1
        elif query_id == 'before_first_shape_label':
            assert query_position == target_positions[0]
            assert answer_position == query_position - 1
        elif query_id == 'after_last_shape_label':
            assert query_position == target_positions[-1]
            assert answer_position == query_position + 1
        elif query_id == 'before_last_shape_label':
            assert query_position == target_positions[-1]
            assert answer_position == query_position - 1
        elif query_id == 'after_second_shape_label':
            assert query_position == target_positions[1]
            assert answer_position == query_position + 1
        else:
            assert query_position == target_positions[1]
            assert answer_position == query_position - 1
        assert out.answer_gt.value == 'B'
        assert execution['labels_by_position'][str(answer_position)] == 'B'

def test_icons_named_path_neighbor_prompt_example_matches_contract() -> None:
    task = IconsNamedPathPathNeighborLabelTask()
    out = task.generate(2026052803, params={'query_id': 'before_last_shape_label', 'answer_index': 4}, max_attempts=100)
    assert '"' in out.prompt
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 'E'}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == [546, 246, 606, 306]
    assert answer_and_annotation['answer'] == 'E'

def test_icons_named_path_neighbor_sampling_smoke() -> None:
    task = IconsNamedPathPathNeighborLabelTask()
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    distractor_counts: Counter[int] = Counter()
    occurrence_counts: Counter[int] = Counter()
    for index in range(60):
        out = task.generate(hash64(2026052804, 'icons_named_path_neighbor', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        entities = out.trace_payload['scene_ir']['entities']
        assert str(out.query_id) == str(execution['query_id'])
        assert sorted((str(entity['label']) for entity in entities if str(entity['label']))) == list('ABCDEF')
        assert execution['labels_by_position'][str(execution['answer_position_index'])] == str(out.answer_gt.value)
        assert 4 <= int(execution['distractor_count']) <= 8
        assert 2 <= int(execution['target_occurrence_count']) <= 4
        assert int(execution['stop_count']) == 6 + int(execution['distractor_count']) + int(execution['target_occurrence_count'])
        query_counts[str(out.query_id)] += 1
        answer_counts[str(out.answer_gt.value)] += 1
        distractor_counts[int(execution['distractor_count'])] += 1
        occurrence_counts[int(execution['target_occurrence_count'])] += 1
    assert set(query_counts) == set(QUERY_IDS)
    assert len(answer_counts) == 6
    assert len(distractor_counts) >= 4
    assert len(occurrence_counts) == 3
