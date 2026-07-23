"""Behavior tests for icon reference-predicate counting."""
from __future__ import annotations
import json
from collections import Counter
import pytest
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.reference_canvas.reference_color_match_count import IconsReferenceCanvasReferenceColorMatchCountTask
from trace_tasks.tasks.icons.reference_canvas.reference_rotation_match_count import IconsReferenceCanvasReferenceRotationMatchCountTask
from trace_tasks.tasks.icons.reference_canvas.reference_type_color_rotation_match_count import IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask
from trace_tasks.tasks.icons.reference_canvas.reference_type_match_count import IconsReferenceCanvasReferenceTypeMatchCountTask
from trace_tasks.tasks.shared.visual_style.panel import PANEL_SCENE_TREATMENTS
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.registry import list_default_task_ids
EXPECTED_ICON_CANVAS_TREATMENTS = set(PANEL_SCENE_TREATMENTS)

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

@pytest.mark.parametrize(
    ('task_cls', 'internal_query_id', 'expected_scene_kind'),
    (
        (IconsReferenceCanvasReferenceTypeMatchCountTask, 'match_type', 'icons_reference_type_match_count'),
        (IconsReferenceCanvasReferenceColorMatchCountTask, 'match_color', 'icons_reference_color_match_count'),
        (IconsReferenceCanvasReferenceRotationMatchCountTask, 'match_rotation', 'icons_reference_rotation_match_count'),
        (
            IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask,
            'match_type_color_rotation',
            'icons_reference_type_color_rotation_match_count',
        ),
    ),
)
def test_icons_counting_attribute_match_count_tracks_consolidated_trace(task_cls, internal_query_id: str, expected_scene_kind: str) -> None:
    task = task_cls()
    out = task.generate(24010, params={'object_count': 8, 'target_count': 3}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity['panel']) == 'scene']
    reference = trace['render_map']['anchors']['reference_icon']
    assert trace['scene_ir']['scene_kind'] == expected_scene_kind
    assert 'source_task_id' not in execution
    assert 'source_query_id' not in execution
    assert execution['scene_variant'] == 'reference_scene'
    assert execution['query_id'] == 'single'
    assert execution['internal_query_id'] == internal_query_id
    assert trace['query_spec']['template_id'] == 'icons_reference_canvas_v1'
    assert trace['query_spec']['params']['scene_variant'] == 'reference_scene'
    assert trace['query_spec']['params']['query_id'] == 'single'
    assert trace['query_spec']['params']['internal_query_id'] == internal_query_id
    assert out.query_id == 'single'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    assert execution['query_id_probabilities'] == trace['query_spec']['params']['query_id_probabilities']
    render_style = trace['render_spec']['style']
    assert bool(render_style['icon_canvas_style']['enabled']) is True
    assert str(render_style['icon_canvas_style']['treatment']) in EXPECTED_ICON_CANVAS_TREATMENTS
    matching_indices = {int(value) for value in execution['matching_scene_indices']}
    assert len(matching_indices) == 3
    assert len(scene_entities) == 8
    if internal_query_id == 'match_type':
        reference_icon_id = str(execution['reference_icon_id'])
        for index, entity in enumerate(scene_entities):
            assert bool(entity['is_match']) == (int(index) in matching_indices)
            if int(index) in matching_indices:
                assert str(entity['icon_id']) == reference_icon_id
    elif internal_query_id == 'match_color':
        reference_icon_id = str(execution['reference_icon_id'])
        reference_tint = tuple((int(channel) for channel in execution['reference_tint_rgb']))
        assert all((str(entity['icon_id']) == reference_icon_id for entity in scene_entities))
        for index, entity in enumerate(scene_entities):
            entity_tint = tuple((int(channel) for channel in entity['tint_rgb']))
            assert bool(entity['is_match']) == (int(index) in matching_indices)
            if int(index) in matching_indices:
                assert entity_tint == reference_tint
            else:
                assert entity_tint != reference_tint
    elif internal_query_id == 'match_rotation':
        base_icon_id = str(execution['base_icon_id'])
        reference_rotation = int(execution['reference_rotation_degrees'])
        assert str(reference['icon_id']) == base_icon_id
        for index, entity in enumerate(scene_entities):
            assert str(entity['icon_id']) == base_icon_id
            assert bool(entity['is_match']) == (int(index) in matching_indices)
            if int(index) in matching_indices:
                assert int(entity['rotation_degrees']) == reference_rotation
    else:
        reference_icon_id = str(execution['reference_icon_id'])
        reference_tint = tuple((int(channel) for channel in execution['reference_tint_rgb']))
        reference_rotation = int(execution['reference_rotation_degrees'])
        for index, entity in enumerate(scene_entities):
            assert bool(entity['is_match']) == (int(index) in matching_indices)
            if int(index) in matching_indices:
                assert str(entity['icon_id']) == reference_icon_id
                assert tuple((int(channel) for channel in entity['tint_rgb'])) == reference_tint
                assert int(entity['rotation_degrees']) == reference_rotation

def test_icons_counting_attribute_match_count_prompt_example_matches_contract() -> None:
    task = IconsReferenceCanvasReferenceTypeMatchCountTask()
    out = task.generate(24011, params={'object_count': 8, 'target_count': 3}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 2
    assert answer_and_annotation['answer'] == 2

def test_icons_counting_single_attribute_match_count_balances_variants_by_default() -> None:
    task_classes = (
        IconsReferenceCanvasReferenceTypeMatchCountTask,
        IconsReferenceCanvasReferenceColorMatchCountTask,
        IconsReferenceCanvasReferenceRotationMatchCountTask,
        IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask,
    )
    counts: Counter[str] = Counter()
    for index, task_cls in enumerate(task_classes):
        out = task_cls().generate(hash64(24012, task_cls.task_id, index), params={}, max_attempts=200)
        counts[str(out.trace_payload['execution_trace']['internal_query_id'])] += 1
        assert out.query_id == 'single'
    assert set(counts.keys()) == {'match_type', 'match_color', 'match_rotation', 'match_type_color_rotation'}

def test_icons_counting_multi_attribute_match_count_is_harder_than_type_for_same_counts() -> None:
    type_out = IconsReferenceCanvasReferenceTypeMatchCountTask().generate(
        24016,
        params={'object_count': 9, 'target_count': 3},
        max_attempts=200,
    )
    binding_out = IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask().generate(
        24016,
        params={'object_count': 9, 'target_count': 3},
        max_attempts=200,
    )

def test_icons_registry_contains_only_active_icons_tasks() -> None:
    active_icons = {task_id for task_id in TASK_REGISTRY if task_id.startswith('task_icons__')}
    expected_icons = {task_id for task_id in list_default_task_ids() if resolve_task_taxonomy(task_id).domain == 'icons'}
    assert active_icons == expected_icons
