"""Behavior tests for pages instruction-panel control lookup tasks."""
from __future__ import annotations
import json
import pytest
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.pages.instruction_panel import (
    CONTROL_PAIR_PROMPT_QUERY_KEY,
    SCENE_VARIANTS,
    SHARED_CONTROL_PROMPT_QUERY_KEY,
    PagesInstructionPanelSharedControlForStepSetLabelTask,
    PagesInstructionPanelStepForControlPairLabelTask,
)

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height

def _assert_bbox_map_inside_canvas(annotation: dict, *, width: int, height: int) -> None:
    assert annotation
    for bbox in annotation.values():
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(width), height=int(height))

def _assert_bbox_set_inside_canvas(annotation: list, *, width: int, height: int) -> None:
    assert annotation
    for bbox in annotation:
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(width), height=int(height))

def _assert_bbox_set_map_inside_canvas(annotation: dict, *, width: int, height: int) -> None:
    assert annotation
    for bboxes in annotation.values():
        assert bboxes
        for bbox in bboxes:
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(width), height=int(height))

def _step_control_ids(step: dict) -> set[str]:
    return {str(control['control_id']) for control in step['controls']}

def _step_by_index(execution: dict, index: int) -> dict:
    return dict(execution['steps'][int(index)])

def test_pages_instruction_panel_shared_control_contract() -> None:
    task = PagesInstructionPanelSharedControlForStepSetLabelTask()
    out = task.generate(81231, params={'scene_variant': 'manual_cards', 'step_count': 6, 'controls_per_step': 3, 'control_count': 12, 'step_set_size': 3, 'target_step_indices': [0, 2, 4], 'target_control_index': 0, 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = dict(execution['target'])
    assert out.scene_id == 'instruction_panel'
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['prompt_query_key'] == SHARED_CONTROL_PROMPT_QUERY_KEY
    assert trace['query_spec']['prompt_variant']['prompt_schema_version'] == 'v1'
    assert trace['query_spec']['params']['source_query_id'] == SHARED_CONTROL_PROMPT_QUERY_KEY
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox_set'
    assert str(out.answer_gt.value) == str(target['target_control_label'])
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert set(trace['projected_annotation']['bbox_set_map']) == {'step_numbers', 'shared_control_chips'}
    selected_indices = [int(value) for value in target['target_step_indices']]
    selected_control_sets = [_step_control_ids(_step_by_index(execution, index)) for index in selected_indices]
    assert set.intersection(*selected_control_sets) == {str(target['target_control_id'])}
    render_map = trace['render_map']
    expected_number_bboxes = [render_map['step_number_bboxes_px'][str(_step_by_index(execution, index)['step_id'])] for index in selected_indices]
    expected_control_bboxes = [render_map['control_chip_bboxes_px'][str(_step_by_index(execution, index)['step_id'])][str(target['target_control_id'])] for index in selected_indices]
    assert out.annotation_gt.value == expected_control_bboxes
    assert trace['projected_annotation']['bbox_set_map']['step_numbers'] == expected_number_bboxes
    assert trace['projected_annotation']['bbox_set_map']['shared_control_chips'] == expected_control_bboxes
    _assert_bbox_set_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['annotation'], list)
    assert isinstance(example['answer'], str)

def test_pages_instruction_panel_control_pair_contract() -> None:
    task = PagesInstructionPanelStepForControlPairLabelTask()
    out = task.generate(81252, params={'scene_variant': 'checklist_table', 'step_count': 7, 'controls_per_step': 3, 'control_count': 12, 'target_step_index': 3, 'target_pair_control_labels': ['Save', 'Print'], 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = dict(execution['target'])
    assert out.scene_id == 'instruction_panel'
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['prompt_query_key'] == CONTROL_PAIR_PROMPT_QUERY_KEY
    assert trace['query_spec']['prompt_variant']['prompt_schema_version'] == 'v1'
    assert trace['query_spec']['params']['source_query_id'] == CONTROL_PAIR_PROMPT_QUERY_KEY
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    assert int(out.answer_gt.value) == int(target['target_step_number'])
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert set(trace['projected_annotation']['bbox_map']) == {'first_control', 'second_control', 'target_step_number'}
    pair_ids = {str(target['first_control_id']), str(target['second_control_id'])}
    matching_steps = [step for step in execution['steps'] if pair_ids.issubset({str(control['control_id']) for control in step['controls']})]
    assert len(matching_steps) == 1
    assert int(matching_steps[0]['step_number']) == int(out.answer_gt.value)
    target_step_id = str(matching_steps[0]['step_id'])
    render_map = trace['render_map']
    expected_bboxes = [
        render_map['control_chip_bboxes_px'][target_step_id][str(target['first_control_id'])],
        render_map['control_chip_bboxes_px'][target_step_id][str(target['second_control_id'])],
        render_map['step_number_bboxes_px'][target_step_id],
    ]
    assert out.annotation_gt.value == expected_bboxes
    assert trace['projected_annotation']['bbox_map']['first_control'] == expected_bboxes[0]
    assert trace['projected_annotation']['bbox_map']['second_control'] == expected_bboxes[1]
    assert trace['projected_annotation']['bbox_map']['target_step_number'] == expected_bboxes[2]
    _assert_bbox_set_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['annotation'], list)
    assert isinstance(example['answer'], int)

@pytest.mark.parametrize('scene_variant', SCENE_VARIANTS)
@pytest.mark.parametrize('task_cls,params', [(PagesInstructionPanelSharedControlForStepSetLabelTask, {'step_set_size': 2, 'target_step_indices': [1, 5]}), (PagesInstructionPanelStepForControlPairLabelTask, {'target_step_index': 2, 'target_pair_control_labels': ['Search', 'Lock']})])
def test_pages_instruction_panel_scene_variants_render_inside_canvas(task_cls, params: dict, scene_variant: str) -> None:
    task = task_cls()
    out = task.generate(81330 + SCENE_VARIANTS.index(scene_variant), params={'scene_variant': scene_variant, 'step_count': 8, 'controls_per_step': 3, 'control_count': 12, 'pages_context_text_enabled': False, **params}, max_attempts=10)
    trace = out.trace_payload
    render = trace['render_spec']
    assert str(render['scene_variant']) == scene_variant
    for step in trace['execution_trace']['steps']:
        for key in ('step_bbox_px', 'number_bbox_px', 'title_bbox_px'):
            _assert_bbox_inside_canvas(step[key], width=int(render['canvas_width']), height=int(render['canvas_height']))
        for control in step['controls']:
            _assert_bbox_inside_canvas(control['chip_bbox_px'], width=int(render['canvas_width']), height=int(render['canvas_height']))
            _assert_bbox_inside_canvas(control['label_bbox_px'], width=int(render['canvas_width']), height=int(render['canvas_height']))

def test_pages_instruction_panel_is_deterministic() -> None:
    task = PagesInstructionPanelStepForControlPairLabelTask()
    params = {'scene_variant': 'side_legend_sheet', 'step_count': 7, 'controls_per_step': 3, 'control_count': 12, 'target_step_index': 4, 'target_pair_control_labels': ['Filter', 'Review'], 'pages_context_text_enabled': False}
    out_a = task.generate(81991, params=params, max_attempts=10)
    out_b = task.generate(81991, params=params, max_attempts=10)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload['execution_trace'] == out_b.trace_payload['execution_trace']

def test_pages_instruction_panel_context_text_disabled_by_default() -> None:
    out = PagesInstructionPanelStepForControlPairLabelTask().generate(82411, params={}, max_attempts=10)
    context_layer = out.trace_payload['render_spec'].get('context_text_layer', {})
    assert context_layer.get('enabled') is False
    assert out.trace_payload['render_map'].get('context_text_bboxes_px') == {}
