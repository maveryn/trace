"""Behavior tests for icon relation relative-position type task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.reference_canvas.anchor_position_count import IconsRelationRelativePositionTypeTask

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

def _satisfies_relation(box: list[int], anchor_box: list[int], relation_id: str, gap_px: int) -> bool:
    if str(relation_id) == 'left_of_anchor':
        return int(box[2]) <= int(anchor_box[0]) - int(gap_px)
    if str(relation_id) == 'right_of_anchor':
        return int(box[0]) >= int(anchor_box[2]) + int(gap_px)
    if str(relation_id) == 'above_anchor':
        return int(box[3]) <= int(anchor_box[1]) - int(gap_px)
    if str(relation_id) == 'below_anchor':
        return int(box[1]) >= int(anchor_box[3]) + int(gap_px)
    raise ValueError(f'unsupported relation_id: {relation_id}')

def _fraction_in_relation(box: list[int], anchor_box: list[int], relation_id: str, gap_px: int) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    ax0, ay0, ax1, ay1 = [float(v) for v in anchor_box]
    width = max(1e-06, x1 - x0)
    height = max(1e-06, y1 - y0)
    area = width * height
    gap = float(gap_px)
    if str(relation_id) == 'left_of_anchor':
        overlap = max(0.0, min(x1, ax0 - gap) - x0) * height
        return overlap / area
    if str(relation_id) == 'right_of_anchor':
        overlap = max(0.0, x1 - max(x0, ax1 + gap)) * height
        return overlap / area
    if str(relation_id) == 'above_anchor':
        overlap = width * max(0.0, min(y1, ay0 - gap) - y0)
        return overlap / area
    if str(relation_id) == 'below_anchor':
        overlap = width * max(0.0, y1 - max(y0, ay1 + gap))
        return overlap / area
    raise ValueError(f'unsupported relation_id: {relation_id}')

def test_icons_relation_relative_position_type_contract_matches_scene() -> None:
    task = IconsRelationRelativePositionTypeTask()
    out = task.generate(14610, params={'query_id': 'right_of_anchor', 'target_count': 2, 'distractor_count': 3}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'scene']
    candidate_entities = [entity for entity in scene_entities if str(entity.get('role')) == 'candidate']
    anchor_entities = [entity for entity in scene_entities if str(entity.get('role')) == 'anchor']
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 2
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 2
    assert out.annotation_gt.value == sorted(out.annotation_gt.value, key=lambda box: (box[1], box[0], box[3], box[2]))
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    assert trace['scene_ir']['scene_kind'] == 'icons_reference_anchor_relation_type'
    assert execution['question_format'] == 'count_matching_scene_icons_by_reference_and_anchor_relation'
    assert out.query_id == 'right_of_anchor'
    assert execution['query_id'] == 'right_of_anchor'
    assert execution['direction'] == 'right'
    assert int(execution['object_count']) == 5
    assert int(execution['target_count']) == 2
    assert int(execution['distractor_count']) == 3
    assert int(execution['distractor_margin_over_target']) == 1
    assert int(execution['same_type_distractor_count']) >= 1
    assert int(execution['same_type_nonspatial_distractor_count']) >= 1
    assert int(execution['different_type_spatial_distractor_count']) >= 1
    assert float(trace['render_spec']['style']['same_type_distractor_opposite_fraction_min']) == 0.75
    assert len(candidate_entities) == 5
    assert len(anchor_entities) == 1
    assert float(trace['render_spec']['style']['scene_max_overlap_fraction']) == 0.05
    assert int(trace['render_spec']['style']['anchor_gap_px_directional']) == 8
    reference_icon_id = str(execution['reference_icon_id'])
    matching_indices = {int(value) for value in execution['matching_scene_indices']}
    anchor_box = list(anchor_entities[0]['bbox_xyxy'])
    same_type_wrong_side_count = 0
    different_type_right_side_count = 0
    for index, entity in enumerate(candidate_entities):
        assert _overlap_fraction_smaller(entity['bbox_xyxy'], anchor_box) <= 0.05 + 1e-06
        for other in candidate_entities[index + 1:]:
            assert _overlap_fraction_smaller(entity['bbox_xyxy'], other['bbox_xyxy']) <= 0.05 + 1e-06
        is_match = bool(entity['is_match'])
        assert is_match == (int(index) in matching_indices)
        spatial_match = _satisfies_relation(entity['bbox_xyxy'], anchor_box, 'right_of_anchor', 8)
        assert bool(entity['spatial_match']) == bool(spatial_match)
        if is_match:
            assert str(entity['icon_id']) == reference_icon_id
            assert spatial_match
        else:
            assert not (str(entity['icon_id']) == reference_icon_id and spatial_match)
            if str(entity['icon_id']) == reference_icon_id and (not spatial_match):
                assert _fraction_in_relation(entity['bbox_xyxy'], anchor_box, 'right_of_anchor', 8) <= 0.25 + 1e-06
                same_type_wrong_side_count += 1
            if str(entity['icon_id']) != reference_icon_id and spatial_match:
                different_type_right_side_count += 1
    assert same_type_wrong_side_count >= 1
    assert different_type_right_side_count >= 1

def test_icons_relation_relative_position_type_supports_zero_matches() -> None:
    task = IconsRelationRelativePositionTypeTask()
    out = task.generate(14611, params={'query_id': 'above_anchor', 'target_count': 0, 'distractor_count': 4}, max_attempts=200)
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []

def test_icons_relation_relative_position_type_prompt_example_matches_contract() -> None:
    task = IconsRelationRelativePositionTypeTask()
    out = task.generate(14612, params={'target_count': 2, 'distractor_count': 3}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 2
    assert answer_and_annotation['answer'] == 2

def test_icons_relation_relative_position_type_balanced_sampling_defaults() -> None:
    task = IconsRelationRelativePositionTypeTask()
    target_counts: Counter[int] = Counter()
    distractor_counts: Counter[int] = Counter()
    direction_counts: Counter[str] = Counter()
    direction_target_counts: dict[str, Counter[int]] = {'left': Counter(), 'right': Counter(), 'above': Counter(), 'below': Counter()}
    for index in range(96):
        out = task.generate(hash64(14613, 'icons_relation_relative_position_type', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        target_count = int(execution['target_count'])
        distractor_count = int(execution['distractor_count'])
        target_counts[target_count] += 1
        distractor_counts[distractor_count] += 1
        assert str(execution['query_id']) in {'left_of_anchor', 'right_of_anchor', 'above_anchor', 'below_anchor'}
        direction = str(execution['direction'])
        direction_counts[direction] += 1
        direction_target_counts[direction][target_count] += 1
        assert 0 <= target_count <= 5
        assert max(1, target_count + 1) <= distractor_count <= 10
        assert int(execution['object_count']) == int(target_count) + int(distractor_count)
    assert set(target_counts.keys()) == set(range(0, 6))
    assert set(direction_counts.keys()) == {'left', 'right', 'above', 'below'}
    assert all((direction_target_counts[direction] for direction in direction_target_counts))
    assert sum(target_counts.values()) == 96
    assert sum(direction_counts.values()) == 96
    for target_count in range(0, 6):
        observed = []
        for index in range(120):
            out = task.generate(hash64(14614, 'icons_relation_relative_position_type_feasible_distractors', index), params={'target_count': target_count}, max_attempts=200)
            observed.append(int(out.trace_payload['execution_trace']['distractor_count']))
        feasible = set(range(max(1, target_count + 1), 11))
        assert set(observed) == feasible
