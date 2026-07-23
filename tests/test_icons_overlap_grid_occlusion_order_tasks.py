"""Behavior tests for icon overlap-grid occlusion-order task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.overlap_grid.occlusion_order_count import IconsOverlapGridOcclusionOrderCountTask

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_icons_overlap_grid_occlusion_order_contract_matches_scene() -> None:
    task = IconsOverlapGridOcclusionOrderCountTask()
    out = task.generate(14710, params={'object_count': 8, 'target_count': 3}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'scene']
    reference_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'reference']
    assert len(reference_entities) == 1
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert all((len(bbox) == 4 for bbox in out.annotation_gt.value))
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    assert trace['scene_ir']['scene_kind'] == 'icons_reference_grid_occlusion_order_count'
    assert execution['question_format'] == 'count_scene_cells_matching_reference_occlusion_order'
    assert out.scene_id == 'overlap_grid'
    assert out.query_id == 'single'
    assert execution['query_id'] == 'single'
    assert execution['fixed_relation_id'] == 'same_front_to_back_order'
    assert trace['query_spec']['params']['fixed_relation_id'] == 'same_front_to_back_order'
    assert int(execution['object_count']) == 8
    assert int(execution['target_count']) == 3
    assert int(execution['distractor_count']) == 5
    reference_pair = reference_entities[0]
    reference_order_id = str(execution['reference_order_id'])
    assert str(reference_pair['order_id']) == reference_order_id
    assert len(scene_entities) == 8
    assert len({str(entity['label']) for entity in scene_entities}) == 8
    matching_labels = set((str(value) for value in execution['matching_cell_labels']))
    assert trace['witness_symbolic']['matching_cell_labels'] == execution['matching_cell_labels']
    annotation_by_label = {str(entity['label']): list(entity['cell_bbox_xyxy']) for entity in scene_entities}
    expected_annotation = [annotation_by_label[str(label)] for label in trace['witness_symbolic']['matching_cell_labels_top_left']]
    assert out.annotation_gt.value == expected_annotation
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    style = trace['render_spec']['style']
    assert int(style['text_legibility']['failure_count']) == 0
    assert {str(record['role']) for record in style['text_legibility']['records']} >= {'icon_panel_header_text', 'icon_cell_label_text'}
    assert 'cell_label_stroke_rgb' in style
    sampled_palette = [tuple((int(channel) for channel in color)) for color in trace['render_spec']['style']['sampled_palette_rgb']]
    assert len(sampled_palette) >= 2
    overlap_min, overlap_max = trace['render_spec']['style']['overlap_ratio_range']
    for entity in scene_entities:
        assert str(entity['icon_a_id']) == str(execution['icon_a_id'])
        assert str(entity['icon_b_id']) == str(execution['icon_b_id'])
        assert overlap_min <= float(entity['overlap_ratio']) <= overlap_max
        assert tuple((int(channel) for channel in entity['icon_a_tint_rgb'])) in sampled_palette
        assert tuple((int(channel) for channel in entity['icon_b_tint_rgb'])) in sampled_palette
        assert tuple((int(channel) for channel in entity['icon_a_tint_rgb'])) != tuple((int(channel) for channel in entity['icon_b_tint_rgb']))
        is_match = bool(entity['is_match'])
        assert is_match == (str(entity['label']) in matching_labels)
        if is_match:
            assert str(entity['order_id']) == reference_order_id
        else:
            assert str(entity['order_id']) != reference_order_id

def test_icons_overlap_grid_occlusion_order_supports_zero_matches() -> None:
    task = IconsOverlapGridOcclusionOrderCountTask()
    out = task.generate(14711, params={'target_count': 0, 'distractor_count': 5}, max_attempts=200)
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []

def test_icons_overlap_grid_occlusion_order_prompt_example_matches_contract() -> None:
    task = IconsOverlapGridOcclusionOrderCountTask()
    out = task.generate(14712, params={'object_count': 8, 'target_count': 3}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 3}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == [[336, 104, 506, 274], [532, 104, 702, 274], [728, 104, 898, 274]]
    assert answer_and_annotation['answer'] == 3

def test_icons_overlap_grid_occlusion_order_balanced_sampling_defaults() -> None:
    task = IconsOverlapGridOcclusionOrderCountTask()
    object_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    distractor_counts: Counter[int] = Counter()
    for index in range(42):
        out = task.generate(hash64(14713, 'icons_overlap_grid_occlusion_order', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        object_count = int(execution['object_count'])
        target_count = int(execution['target_count'])
        distractor_count = int(execution['distractor_count'])
        object_counts[object_count] += 1
        target_counts[target_count] += 1
        distractor_counts[distractor_count] += 1
        assert 0 <= target_count <= 4
        assert 1 <= distractor_count <= 5
        assert 2 <= object_count <= 8
        assert object_count == target_count + distractor_count
    assert set(target_counts.keys()) == set(range(0, 5))
    assert set(distractor_counts.keys()) == set(range(1, 6))
    assert set(object_counts.keys()) == set(range(2, 9))
    assert sum(target_counts.values()) == 42
