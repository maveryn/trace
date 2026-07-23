"""Behavior tests for icon counting size-relation task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.reference_canvas.reference_metric_relation_count import IconsReferenceCanvasReferenceMetricRelationCountTask

def _overlap_fraction_smaller(left: list[int], right: list[int]) -> float:
    ix0 = max(int(left[0]), int(right[0]))
    iy0 = max(int(left[1]), int(right[1]))
    ix1 = min(int(left[2]), int(right[2]))
    iy1 = min(int(left[3]), int(right[3]))
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    left_area = max(1, int(left[2]) - int(left[0])) * max(1, int(left[3]) - int(left[1]))
    right_area = max(1, int(right[2]) - int(right[0])) * max(1, int(right[3]) - int(right[1]))
    return float(inter) / float(min(left_area, right_area))

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_icons_counting_size_relation_contract_matches_scene() -> None:
    task = IconsReferenceCanvasReferenceMetricRelationCountTask()
    out = task.generate(14610, params={'target_count': 3, 'distractor_count': 4, 'size_relation': 'smaller'}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity['panel']) == 'scene']
    reference_entity = trace['render_map']['anchors']['reference_icon']
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert out.annotation_gt.value == sorted(out.annotation_gt.value, key=lambda box: (box[1], box[0], box[3], box[2]))
    assert trace['scene_ir']['scene_kind'] == 'icons_reference_counting_size_relation'
    assert out.query_id == 'size_smaller'
    assert execution['query_id'] == 'size_smaller'
    assert execution['question_format'] == 'count_matching_scene_icons_by_size_relation'
    assert execution['size_relation'] == 'smaller'
    assert int(execution['size_relation_min_delta_px']) == 18
    assert int(execution['reference_nominal_size_px']) == int(reference_entity['nominal_size_px'])
    assert int(reference_entity['nominal_size_px']) >= 64
    assert int(reference_entity['nominal_size_px']) <= 96
    assert int(execution['object_count']) == 7
    assert int(execution['target_count']) == 3
    assert int(execution['distractor_count']) == 4
    assert len(scene_entities) == 7
    sampled_palette = [tuple((int(channel) for channel in color)) for color in trace['render_spec']['style']['sampled_palette_rgb']]
    reference_icon_id = str(execution['reference_icon_id'])
    matching_indices = {int(value) for value in execution['matching_scene_indices']}
    assert len(matching_indices) == 3
    reference_size = int(reference_entity['nominal_size_px'])
    for index, entity in enumerate(scene_entities):
        assert str(entity['icon_id']) == reference_icon_id
        assert 40 <= int(entity['nominal_size_px']) <= 120
        assert tuple((int(channel) for channel in entity['tint_rgb'])) in sampled_palette
        assert isinstance(entity['noise_edits'], list)
        size_delta = abs(int(entity['nominal_size_px']) - int(reference_size))
        assert size_delta >= 18
        if int(index) in matching_indices:
            assert int(entity['nominal_size_px']) <= int(reference_size) - 18
        else:
            assert int(entity['nominal_size_px']) >= int(reference_size) + 18
    for left_index, left in enumerate(scene_entities):
        for right in scene_entities[left_index + 1:]:
            assert _overlap_fraction_smaller(left['bbox_xyxy'], right['bbox_xyxy']) <= 0.1 + 1e-06

def test_icons_counting_size_relation_supports_zero_matches() -> None:
    task = IconsReferenceCanvasReferenceMetricRelationCountTask()
    out = task.generate(14611, params={'target_count_min': 0, 'target_count': 0, 'distractor_count': 5, 'size_relation': 'larger'}, max_attempts=200)
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []

def test_icons_counting_size_relation_prompt_example_matches_contract() -> None:
    task = IconsReferenceCanvasReferenceMetricRelationCountTask()
    out = task.generate(14612, params={'target_count': 2, 'distractor_count': 3, 'size_relation': 'larger'}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 2
    assert answer_and_annotation['answer'] == 2

def test_icons_counting_size_relation_balanced_sampling_defaults() -> None:
    task = IconsReferenceCanvasReferenceMetricRelationCountTask()
    object_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    distractor_counts: Counter[int] = Counter()
    size_relations: Counter[str] = Counter()
    for index in range(60):
        out = task.generate(hash64(14613, 'icons_counting_size_relation', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        object_count = int(execution['object_count'])
        target_count = int(execution['target_count'])
        distractor_count = int(execution['distractor_count'])
        object_counts[object_count] += 1
        target_counts[target_count] += 1
        distractor_counts[distractor_count] += 1
        size_relations[str(execution['size_relation'])] += 1
        assert 1 <= target_count <= 5
        assert 1 <= distractor_count <= 6
        assert int(object_count) == int(target_count) + int(distractor_count)
    assert min(object_counts.keys()) >= 1
    assert max(object_counts.keys()) <= 14
    assert set(target_counts.keys()) == set(range(1, 6))
    assert set(distractor_counts.keys()) == set(range(1, 7))
    assert set(size_relations.keys()) == {'larger', 'smaller'}
