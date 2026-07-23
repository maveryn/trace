"""Behavior tests for symbolic abacus readout tasks."""
from __future__ import annotations
from collections import Counter
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.symbolic.abacus.displayed_value_readout import TASK_ID, SymbolicAbacusDisplayedValueReadoutTask
from trace_tasks.tasks.symbolic.abacus.place_digit_readout import TASK_ID as PLACE_DIGIT_TASK_ID
from trace_tasks.tasks.symbolic.abacus.place_digit_readout import SymbolicAbacusPlaceDigitReadoutTask
from trace_tasks.tasks.symbolic.abacus.target_value_match_label import TASK_ID as MATCH_TASK_ID
from trace_tasks.tasks.symbolic.abacus.target_value_match_label import SymbolicAbacusTargetValueMatchTask
from tests.helpers import extract_prompt_json_example

def _active_count_for_digit(digit: int) -> int:
    return (1 if int(digit) >= 5 else 0) + int(digit) % 5

def _vertical_gap(upper_bbox: list[float], lower_bbox: list[float]) -> float:
    return round(float(lower_bbox[1]) - float(upper_bbox[3]), 3)

def _bbox_center(bbox: list[float]) -> list[float]:
    return [round((float(bbox[0]) + float(bbox[2])) * 0.5, 3), round((float(bbox[1]) + float(bbox[3])) * 0.5, 3)]

def test_symbolic_abacus_displayed_value_contract_matches_trace() -> None:
    task = SymbolicAbacusDisplayedValueReadoutTask()
    cases = (0, 5, 40, 207, 603, 999)
    for value in cases:
        out = task.generate(50100 + value, params={'answer_value': value, 'scene_variant': 'clean_card'}, max_attempts=5)
        trace = out.trace_payload
        execution = trace['execution_trace']
        assert out.answer_gt.type == 'option_letter'
        assert out.annotation_gt.type == 'bbox'
        assert out.scene_id == 'abacus'
        assert out.query_id == SINGLE_QUERY_ID
        assert str(execution['query_id']) == SINGLE_QUERY_ID
        assert str(execution['internal_query_id']) == 'displayed_value_readout'
        assert str(execution['question_format']) == 'displayed_value_readout'
        assert int(execution['answer_value']) == value
        assert str(execution['correct_label']) == str(out.answer_gt.value)
        assert int(execution['option_values_by_label'][str(out.answer_gt.value)]) == value
        assert list(trace['projected_annotation']['bbox']) == out.annotation_gt.value
        assert list(trace['projected_annotation']['pixel_bbox']) == out.annotation_gt.value
        assert trace['render_map']['selected_option_card_bbox_px'] == out.annotation_gt.value
        assert trace['render_spec']['abacus_style']['renderer'] == 'abacus_single_board_v1'
        active_points_by_column = trace['render_map']['active_bead_points_by_column_px']
        for role, digit in execution['digits_by_role'].items():
            key = f'{role}_active_beads'
            assert len(active_points_by_column[key]) == _active_count_for_digit(int(digit))

def test_symbolic_abacus_lower_inactive_beads_are_visually_separated_from_active_group() -> None:
    task = SymbolicAbacusDisplayedValueReadoutTask()
    out = task.generate(50316, params={'answer_value': 316, 'scene_variant': 'worksheet'}, max_attempts=5)
    trace = out.trace_payload
    bead_bboxes = trace['render_map']['bead_bboxes_px']
    active_by_column = trace['render_map']['active_bead_ids_by_column']
    assert out.image.size == (980, 860)
    assert active_by_column['tens'] == ['column_tens_lower_1']
    assert active_by_column['hundreds'] == ['column_hundreds_lower_1', 'column_hundreds_lower_2', 'column_hundreds_lower_3']
    assert _vertical_gap(bead_bboxes['column_tens_lower_1'], bead_bboxes['column_tens_lower_2']) >= 60.0
    assert _vertical_gap(bead_bboxes['column_hundreds_lower_3'], bead_bboxes['column_hundreds_lower_4']) >= 60.0

def test_symbolic_abacus_prompt_examples_match_contract() -> None:
    task = SymbolicAbacusDisplayedValueReadoutTask()
    out = task.generate(50120, params={'answer_value': 603, 'scene_variant': 'wood_frame'}, max_attempts=5)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
    assert set(answer_and_annotation.keys()) == {'annotation', 'answer'}
    assert isinstance(answer_and_annotation['annotation'], list)
    assert answer_and_annotation['answer'] == 'C'
    assert answer_only == {'answer': 'C'}

def test_symbolic_abacus_balanced_sampling_defaults_cover_scene_axis() -> None:
    task = SymbolicAbacusDisplayedValueReadoutTask()
    scene_variants: Counter[str] = Counter()
    answers: set[int] = set()
    for index in range(36):
        out = task.generate(hash64(50140, TASK_ID, index), params={}, max_attempts=5)
        execution = out.trace_payload['execution_trace']
        scene_variants[str(execution['scene_variant'])] += 1
        answers.add(int(execution['answer_value']))
        assert int(execution['option_values_by_label'][str(execution['correct_label'])]) == int(execution['answer_value'])
        assert str(execution['query_id']) == SINGLE_QUERY_ID
        assert str(execution['internal_query_id']) == 'displayed_value_readout'
        assert 0 <= int(execution['answer_value']) <= 999
    assert set(scene_variants.keys()) == {'clean_card', 'wood_frame', 'worksheet'}
    assert len(answers) > 20

def test_symbolic_abacus_place_digit_contract_matches_trace() -> None:
    task = SymbolicAbacusPlaceDigitReadoutTask()
    out = task.generate(52100, params={'displayed_value': 603, 'target_column_role': 'hundreds', 'scene_variant': 'clean_card'}, max_attempts=5)
    trace = out.trace_payload
    execution = trace['execution_trace']
    assert out.answer_gt.type == 'option_letter'
    assert out.annotation_gt.type == 'bbox'
    assert out.scene_id == 'abacus'
    assert out.query_id == SINGLE_QUERY_ID
    assert str(execution['query_id']) == SINGLE_QUERY_ID
    assert str(execution['internal_query_id']) == 'place_digit_readout'
    assert str(execution['question_format']) == 'place_digit_readout'
    assert str(execution['target_column_role']) == 'hundreds'
    assert str(execution['target_place_label']) == '100'
    assert int(execution['target_place_value']) == 100
    assert int(execution['displayed_value']) == 603
    assert int(execution['answer_digit']) == 6
    assert str(execution['correct_label']) == str(out.answer_gt.value)
    assert int(execution['option_values_by_label'][str(out.answer_gt.value)]) == 6
    assert execution['digits_by_role'] == {'hundreds': 6, 'tens': 0, 'ones': 3}
    assert execution['annotation_key'] == 'hundreds_active_beads'
    expected_points = trace['render_map']['active_bead_points_by_column_px']['hundreds_active_beads']
    assert trace['render_map']['target_active_bead_points_px'] == expected_points
    assert list(trace['projected_annotation']['bbox']) == out.annotation_gt.value
    assert list(trace['projected_annotation']['pixel_bbox']) == out.annotation_gt.value
    assert trace['render_map']['selected_option_card_bbox_px'] == out.annotation_gt.value
    assert trace['render_spec']['abacus_style']['renderer'] == 'abacus_single_board_v1'
    assert len(expected_points) == _active_count_for_digit(6)
    assert '100' in out.prompt

def test_symbolic_abacus_place_digit_zero_uses_empty_point_set() -> None:
    task = SymbolicAbacusPlaceDigitReadoutTask()
    out = task.generate(52101, params={'displayed_value': 603, 'column_role': 'tens', 'scene_variant': 'worksheet'}, max_attempts=5)
    trace = out.trace_payload
    execution = trace['execution_trace']
    assert out.annotation_gt.type == 'bbox'
    assert int(execution['answer_digit']) == 0
    assert int(execution['option_values_by_label'][str(out.answer_gt.value)]) == 0
    assert trace['render_map']['target_active_bead_points_px'] == []
    assert str(execution['target_column_role']) == 'tens'
    assert str(execution['target_place_label']) == '10'
    assert execution['annotation_key'] == 'tens_active_beads'

def test_symbolic_abacus_place_digit_prompt_examples_match_contract() -> None:
    task = SymbolicAbacusPlaceDigitReadoutTask()
    out = task.generate(52120, params={'displayed_value': 742, 'target_column_role': 'ones'}, max_attempts=5)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
    assert set(answer_and_annotation.keys()) == {'annotation', 'answer'}
    assert isinstance(answer_and_annotation['annotation'], list)
    assert answer_and_annotation['answer'] == 'C'
    assert answer_only == {'answer': 'C'}
    assert '1' in out.prompt

def test_symbolic_abacus_place_digit_balanced_sampling_defaults_cover_axes() -> None:
    task = SymbolicAbacusPlaceDigitReadoutTask()
    scene_variants: Counter[str] = Counter()
    target_columns: Counter[str] = Counter()
    answer_digits: set[int] = set()
    for index in range(72):
        out = task.generate(hash64(52140, PLACE_DIGIT_TASK_ID, index), params={}, max_attempts=5)
        execution = out.trace_payload['execution_trace']
        scene_variants[str(execution['scene_variant'])] += 1
        target_columns[str(execution['target_column_role'])] += 1
        answer_digits.add(int(execution['answer_digit']))
        assert int(execution['option_values_by_label'][str(execution['correct_label'])]) == int(execution['answer_digit'])
        assert 0 <= int(execution['answer_digit']) <= 9
        assert int(execution['answer_digit']) == int(execution['digits_by_role'][str(execution['target_column_role'])])
    assert set(scene_variants.keys()) == {'clean_card', 'wood_frame', 'worksheet'}
    assert set(target_columns.keys()) == {'hundreds', 'tens', 'ones'}
    assert len(answer_digits) >= 8

def test_symbolic_abacus_option_panel_contract_matches_trace() -> None:
    task = SymbolicAbacusTargetValueMatchTask()
    out = task.generate(61100, params={'target_value': 316, 'answer_label': 'E', 'scene_variant': 'worksheet'}, max_attempts=5)
    trace = out.trace_payload
    execution = trace['execution_trace']
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'E'
    assert out.annotation_gt.type == 'bbox'
    assert len(out.annotation_gt.value) == 4
    assert out.scene_id == 'abacus'
    assert out.query_id == SINGLE_QUERY_ID
    assert trace['scene_ir']['scene_kind'] == 'symbolic_abacus_option_panel'
    assert str(execution['query_id']) == SINGLE_QUERY_ID
    assert str(execution['question_format']) == 'target_value_match_label'
    assert int(execution['target_value']) == 316
    assert execution['target_digits'] == [3, 1, 6]
    assert str(execution['correct_label']) == 'E'
    assert int(execution['option_values_by_label']['E']) == 316
    assert list(trace['projected_annotation']['bbox']) == out.annotation_gt.value
    assert list(trace['projected_annotation']['pixel_bbox']) == out.annotation_gt.value
    assert trace['render_map']['selected_option_card_bbox_px'] == out.annotation_gt.value
    assert trace['render_spec']['abacus_style']['renderer'] == 'abacus_option_panel_v1'
    assert trace['render_spec']['abacus_style']['active_inactive_bead_color_shared'] is True
    assert trace['render_spec']['abacus_style']['active_bead_rgb'] == trace['render_spec']['abacus_style']['inactive_bead_rgb']
    assert trace['render_spec']['post_image_noise']['apply_prob'] == 0.18
    assert out.image.size == (1200, 760)
    assert len(execution['option_labels']) == 6
    assert len(set(execution['option_values_by_label'].values())) == 6

def test_symbolic_abacus_option_panel_prompt_examples_match_contract() -> None:
    task = SymbolicAbacusTargetValueMatchTask()
    out = task.generate(61120, params={'target_value': 742, 'answer_label': 'B'}, max_attempts=5)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
    assert answer_and_annotation == {'annotation': [430, 74, 770, 354], 'answer': 'B'}
    assert answer_only == {'answer': 'B'}
    assert '742' in out.prompt

def test_symbolic_abacus_option_panel_balanced_sampling_defaults_cover_axes() -> None:
    task = SymbolicAbacusTargetValueMatchTask()
    scene_variants: Counter[str] = Counter()
    answer_labels: Counter[str] = Counter()
    target_values: set[int] = set()
    for index in range(72):
        out = task.generate(hash64(61140, MATCH_TASK_ID, index), params={}, max_attempts=5)
        execution = out.trace_payload['execution_trace']
        scene_variants[str(execution['scene_variant'])] += 1
        answer_labels[str(execution['correct_label'])] += 1
        target_values.add(int(execution['target_value']))
        assert int(execution['option_values_by_label'][str(execution['correct_label'])]) == int(execution['target_value'])
        assert list(execution['option_values_by_label'].values()).count(int(execution['target_value'])) == 1
    assert set(scene_variants.keys()) == {'clean_card', 'wood_frame', 'worksheet'}
    assert set(answer_labels.keys()) == {'A', 'B', 'C', 'D', 'E', 'F'}
    assert len(target_values) > 50
