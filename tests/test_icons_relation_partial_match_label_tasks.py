"""Behavior tests for the icon partial-match label task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.icon_cutout.partial_match_label import IconsIconCutoutPartialMatchLabelTask

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def test_icons_relation_partial_match_contract_matches_scene() -> None:
    task = IconsIconCutoutPartialMatchLabelTask()
    out = task.generate(2026052801, params={'answer_index': 2}, max_attempts=120)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render_style = trace['render_spec']['style']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'scene']
    reference_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'reference']
    assert len(reference_entities) == 1
    assert len(scene_entities) == 6
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'C'
    assert out.annotation_gt.type == 'bbox_map'
    assert sorted(out.annotation_gt.value.keys()) == ['selected_option', 'source_fragment']
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.query_id == 'single'
    assert trace['query_spec']['params']['internal_query_id'] == 'partial_icon_match_label'
    assert out.scene_id == 'icon_cutout'
    assert trace['scene_ir']['scene_kind'] == 'icons_icon_cutout_partial_match_label'
    assert execution['question_format'] == 'select_full_icon_option_matching_partial_source_fragment'
    assert int(execution['object_count']) == 6
    reference = reference_entities[0]
    assert str(reference['source_icon_id']) == str(execution['correct_icon_id'])
    assert str(reference['fragment_window_style']) in {'rectangle', 'rounded', 'ellipse'}
    assert 0.0 < float(reference['fragment_visible_alpha_ratio']) < 1.0
    assert 0.0 < float(reference['fragment_alpha_density']) <= 1.0
    answer_label = str(out.answer_gt.value)
    matching = [entity for entity in scene_entities if bool(entity['is_match'])]
    assert len(matching) == 1
    assert str(matching[0]['label']) == answer_label
    assert str(matching[0]['icon_id']) == str(execution['correct_icon_id'])
    assert str(execution['option_icon_ids_by_label'][answer_label]) == str(execution['correct_icon_id'])
    assert len({str(entity['icon_id']) for entity in scene_entities}) == 6
    assert out.annotation_gt.value == {'source_fragment': list(reference['fragment_bbox_xyxy']), 'selected_option': list(matching[0]['cell_bbox_xyxy'])}
    assert trace['projected_annotation']['type'] == 'bbox_map'
    assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
    assert trace['witness_symbolic']['selected_option_label'] == answer_label
    assert render_style['text_legibility']['required_role_count'] >= 2
    assert render_style['text_legibility']['failure_count'] == 0
    assert 'cell_label_stroke_rgb' in render_style

def test_icons_relation_partial_match_prompt_example_matches_contract() -> None:
    task = IconsIconCutoutPartialMatchLabelTask()
    out = task.generate(2026052802, params={'answer_label': 'C'}, max_attempts=120)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 'C'}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert sorted(answer_and_annotation['annotation'].keys()) == ['selected_option', 'source_fragment']
    assert answer_and_annotation['answer'] == 'C'

def test_icons_relation_partial_match_default_sampling() -> None:
    task = IconsIconCutoutPartialMatchLabelTask()
    answer_counts: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    for index in range(18):
        out = task.generate(hash64(2026052803, 'icons_relation_partial_match', index), params={}, max_attempts=120)
        execution = out.trace_payload['execution_trace']
        assert str(out.query_id) == 'single'
        assert str(out.trace_payload['query_spec']['params']['internal_query_id']) == 'partial_icon_match_label'
        assert int(execution['object_count']) == 6
        answer_counts[str(out.answer_gt.value)] += 1
        styles[str(execution['fragment_window_style'])] += 1
    assert len(answer_counts) >= 4
    assert set(styles).issubset({'rectangle', 'rounded', 'ellipse'})
