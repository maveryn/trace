"""Behavior tests for named-reference icon distance-rank labels."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.named_field.reference_distance_rank_label import IconsRelationNamedReferenceDistanceRankLabelTask

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def test_icons_relation_named_reference_distance_rank_contract_matches_scene() -> None:
    task = IconsRelationNamedReferenceDistanceRankLabelTask()
    out = task.generate(2026052401, params={'distance_rank_query': 'second_closest_to_named_reference_label', 'answer_label': 'D', 'distractor_count': 4}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    references = [entity for entity in entities if str(entity.get('role')) == 'reference']
    candidates = [entity for entity in entities if str(entity.get('role')) == 'candidate']
    distractors = [entity for entity in entities if str(entity.get('role')) == 'distractor']
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'D'
    assert out.annotation_gt.type == 'bbox_map'
    assert len(out.annotation_gt.value) == 2
    assert out.scene_id == 'named_field'
    assert out.query_id == 'second_closest_to_named_reference_label'
    assert trace['scene_ir']['scene_kind'] == 'icons_named_field_distance_rank'
    assert execution['question_format'] == 'select_labeled_named_icon_by_distance_rank_from_unique_named_reference'
    assert len(references) == 1
    assert len(candidates) == 6
    assert len(distractors) == 4
    assert sorted((str(entity['label']) for entity in candidates)) == list('ABCDEF')
    assert all((str(entity.get('label', '')) == '' for entity in distractors))
    assert int(execution['answer_rank']) == 1
    assert execution['sorted_candidate_labels_by_distance'][1] == 'D'
    reference = references[0]
    matching_reference_combo = [entity for entity in entities if str(entity['shape_id']) == str(reference['shape_id']) and str(entity['color_name']) == str(reference['color_name'])]
    assert matching_reference_combo == [reference]
    answer_entity = next((entity for entity in candidates if str(entity['label']) == 'D'))
    expected_annotation = {
        'reference_icon': reference['bbox_xyxy'],
        'selected_candidate': answer_entity['bbox_xyxy'],
    }
    assert out.annotation_gt.value == expected_annotation
    assert trace['projected_annotation']['type'] == 'bbox_map'
    assert trace['projected_annotation']['bbox_map'] == expected_annotation
    assert trace['projected_annotation']['pixel_bbox_map'] == expected_annotation
    assert trace['render_spec']['style']['text_legibility']['required_role_count'] >= 2
    assert trace['render_spec']['style']['text_legibility']['failure_count'] == 0
    assert 'candidate_label_stroke_rgb' in trace['render_spec']['style']

def test_icons_relation_named_reference_distance_rank_prompt_example_matches_contract() -> None:
    task = IconsRelationNamedReferenceDistanceRankLabelTask()
    out = task.generate(2026052402, params={'distance_rank_query': 'closest_to_named_reference_label', 'answer_index': 3}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 'D'}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == {
        'reference_icon': [178, 244, 238, 304],
        'selected_candidate': [614, 186, 674, 246],
    }
    assert answer_and_annotation['answer'] == 'D'

def test_icons_relation_named_reference_distance_rank_sampling_smoke() -> None:
    task = IconsRelationNamedReferenceDistanceRankLabelTask()
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    distractor_counts: Counter[int] = Counter()
    for index in range(48):
        out = task.generate(hash64(2026052403, 'icons_relation_named_reference_distance_rank', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        assert str(out.query_id) == str(execution['query_id'])
        assert execution['sorted_candidate_labels_by_distance'][int(execution['answer_rank'])] == str(out.answer_gt.value)
        assert set((str(label) for label in execution['candidate_labels'])) == set('ABCDEF')
        assert 4 <= int(execution['distractor_count']) <= 8
        query_counts[str(out.query_id)] += 1
        answer_counts[str(out.answer_gt.value)] += 1
        distractor_counts[int(execution['distractor_count'])] += 1
    assert set(query_counts) == {'closest_to_named_reference_label', 'second_closest_to_named_reference_label', 'farthest_from_named_reference_label'}
    assert len(answer_counts) == 6
    assert len(distractor_counts) >= 4
