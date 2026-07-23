"""Behavior tests for pages infographic arithmetic tasks."""
from __future__ import annotations
import importlib
import json
import pytest
from trace_tasks.core.seed import hash64
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.pages.hero_callout_infographic.callout_condition_count import ABOVE_THRESHOLD_QUERY_ID as HERO_CALLOUT_ABOVE_THRESHOLD_QUERY_ID, BELOW_THRESHOLD_QUERY_ID as HERO_CALLOUT_BELOW_THRESHOLD_QUERY_ID, TASK_ID as HERO_CALLOUT_CONDITION_COUNT_TASK_ID, PagesHeroCalloutConditionCountTask
from trace_tasks.tasks.pages.hero_callout_infographic.callout_composite_metric_extremum_label import HIGHEST_COMPOSITE_QUERY_ID as HERO_CALLOUT_HIGHEST_COMPOSITE_QUERY_ID, LOWEST_COMPOSITE_QUERY_ID as HERO_CALLOUT_LOWEST_COMPOSITE_QUERY_ID, TASK_ID as HERO_CALLOUT_COMPOSITE_EXTREMUM_TASK_ID, PagesHeroCalloutCompositeMetricExtremumLabelTask
from trace_tasks.tasks.pages.hero_callout_infographic.callout_metric_extremum_label import HIGHEST_FIELD_VALUE_QUERY_ID as HERO_CALLOUT_HIGHEST_FIELD_VALUE_QUERY_ID, LOWEST_FIELD_VALUE_QUERY_ID as HERO_CALLOUT_LOWEST_FIELD_VALUE_QUERY_ID, TASK_ID as HERO_CALLOUT_METRIC_EXTREMUM_TASK_ID, PagesHeroCalloutMetricExtremumLabelTask
from trace_tasks.tasks.pages.infographic.global_metric_ranked_item_label import PagesInfographicGlobalMetricRankedItemLabelTask
from trace_tasks.tasks.pages.infographic.section_extrema_arithmetic_value import PagesInfographicSectionExtremaArithmeticValueTask
from trace_tasks.tasks.pages.infographic.section_icon_extremum_label import PagesInfographicSectionIconExtremumLabelTask
from trace_tasks.tasks.pages.infographic.section_icon_total_difference_value import PagesInfographicSectionIconTotalDifferenceValueTask
from trace_tasks.tasks.pages.infographic.section_icon_total_value import PagesInfographicSectionIconTotalValueTask
from trace_tasks.tasks.pages.infographic.section_metric_ranked_item_label import PagesInfographicSectionMetricRankedItemLabelTask
from trace_tasks.tasks.pages.infographic.section_ranked_total_label import PagesInfographicSectionRankedTotalLabelTask
from trace_tasks.tasks.pages.infographic.section_total_except_named_value import PagesInfographicSectionTotalExceptNamedValueTask
from trace_tasks.tasks.pages.infographic.section_total_extrema_difference_value import PagesInfographicSectionTotalExtremaDifferenceValueTask
from trace_tasks.tasks.pages.infographic.sum_named_metrics_value import PagesInfographicSumNamedMetricsValueTask
from trace_tasks.tasks.pages.shared.infographic_metric_common import COLUMN_PROFILE_COMPARISON_VARIANTS, FILTERED_SECTION_EXTREMUM_VARIANTS, FILTERED_METRIC_TOTAL_VARIANTS, GLOBAL_METRIC_RANKED_ITEM_VARIANTS, SECTION_METRIC_RANKED_ITEM_VARIANTS, SECTION_RANKED_TOTAL_VARIANTS, SUPPORTED_QUERY_IDS
from trace_tasks.tasks.pages.mixed_infographic_page import CONDITION_COUNT_QUERY_ID as MIXED_INFOGRAPHIC_CONDITION_COUNT_QUERY_ID, FIELD_RANKED_QUERY_ID as MIXED_INFOGRAPHIC_FIELD_RANKED_QUERY_ID, FIELD_TOTAL_QUERY_ID as MIXED_INFOGRAPHIC_FIELD_TOTAL_QUERY_ID, PAGE_FIELD_EXTREMUM_QUERY_ID as MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_QUERY_ID, TWO_FIELD_CONDITION_QUERY_ID as MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_QUERY_ID, TWO_MODULE_TOTAL_COMPARISON_QUERY_ID as MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_QUERY_ID, MIXED_INFOGRAPHIC_CONDITION_COUNT_TASK_ID, MIXED_INFOGRAPHIC_FIELD_RANKED_TASK_ID, MIXED_INFOGRAPHIC_FIELD_TOTAL_TASK_ID, MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_TASK_ID, MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_TASK_ID, MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_TASK_ID, MIXED_INFOGRAPHIC_TASK_ID, NATIVE_LAYOUT_MODES as MIXED_INFOGRAPHIC_NATIVE_LAYOUT_MODES, QUERY_ID as MIXED_INFOGRAPHIC_QUERY_ID, SCENE_VARIANTS as MIXED_INFOGRAPHIC_SCENE_VARIANTS, PagesMixedInfographicModuleConditionItemCountTask, PagesMixedInfographicModuleFieldRankedItemLabelTask, PagesMixedInfographicModuleFieldTotalValueTask, PagesMixedInfographicModuleFieldValueLabelTask, PagesMixedInfographicModuleTwoFieldConditionItemLabelTask, PagesMixedInfographicPageFieldExtremumModuleLabelTask, PagesMixedInfographicTwoModuleFieldTotalComparisonModuleLabelTask
from trace_tasks.tasks.pages.mixed_infographic_page import _lifecycle as mixed_infographic_lifecycle
from trace_tasks.tasks.pages.sectioned_infographic.section_filtered_item_label import PROMPT_QUERY_KEY as SECTIONED_FILTERED_PROMPT_QUERY_KEY, TASK_ID as SECTIONED_INFOGRAPHIC_FILTERED_ITEM_TASK_ID, PagesSectionedInfographicSectionFilteredItemLabelTask
from trace_tasks.tasks.pages.sectioned_infographic.section_item_count import PROMPT_QUERY_KEY as SECTIONED_PROMPT_QUERY_KEY, SCENE_VARIANTS as SECTIONED_SCENE_VARIANTS, TASK_ID as SECTIONED_INFOGRAPHIC_ITEM_COUNT_TASK_ID, PagesSectionedInfographicSectionItemCountTask
METRIC_VALUE_QUERY_IDS = (*SUPPORTED_QUERY_IDS, *FILTERED_METRIC_TOTAL_VARIANTS, *COLUMN_PROFILE_COMPARISON_VARIANTS)
METRIC_VALUE_BBOX_SET_QUERY_IDS = {
    'sum_named_metrics',
    'section_total_except_named',
    'section_icon_total_value',
}
METRIC_VALUE_BBOX_SET_MAP_QUERY_IDS = {
    'section_total_extrema_difference',
    'section_icon_total_difference_value',
}
METRIC_VALUE_BBOX_SET_MAP_GROUP_KEYS = {
    'section_total_extrema_difference': ('highest_total_section', 'lowest_total_section'),
    'section_icon_total_difference_value': ('section_a_filtered_icon_cards', 'section_b_filtered_icon_cards'),
}
SECTION_RANK_QUERY_IDS = (*SECTION_RANKED_TOTAL_VARIANTS, *FILTERED_SECTION_EXTREMUM_VARIANTS)
GLOBAL_METRIC_RANKED_ITEM_QUERY_IDS = GLOBAL_METRIC_RANKED_ITEM_VARIANTS
SECTION_METRIC_RANKED_ITEM_QUERY_IDS = SECTION_METRIC_RANKED_ITEM_VARIANTS
METRIC_RANKED_ITEM_QUERY_IDS = (*GLOBAL_METRIC_RANKED_ITEM_QUERY_IDS, *SECTION_METRIC_RANKED_ITEM_QUERY_IDS)

def _metric_value_task_for_query(query_id: str):
    return {'sum_named_metrics': PagesInfographicSumNamedMetricsValueTask, 'section_extrema_arithmetic': PagesInfographicSectionExtremaArithmeticValueTask, 'section_total_extrema_difference': PagesInfographicSectionTotalExtremaDifferenceValueTask, 'section_total_except_named': PagesInfographicSectionTotalExceptNamedValueTask, 'section_icon_total_value': PagesInfographicSectionIconTotalValueTask, 'section_icon_total_difference_value': PagesInfographicSectionIconTotalDifferenceValueTask}[str(query_id)]()

def _section_rank_task_for_query(query_id: str):
    return {'section_ranked_total_label': PagesInfographicSectionRankedTotalLabelTask, 'section_icon_extremum_label': PagesInfographicSectionIconExtremumLabelTask}[str(query_id)]()

def _metric_ranked_item_task_for_query(query_id: str):
    if str(query_id) in set(GLOBAL_METRIC_RANKED_ITEM_QUERY_IDS):
        return PagesInfographicGlobalMetricRankedItemLabelTask()
    if str(query_id) in set(SECTION_METRIC_RANKED_ITEM_QUERY_IDS):
        return PagesInfographicSectionMetricRankedItemLabelTask()
    if str(query_id) not in set(METRIC_RANKED_ITEM_QUERY_IDS):
        raise ValueError(f'unsupported metric ranked item query: {query_id}')
    raise AssertionError(f'unreachable metric ranked item query: {query_id}')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _expected_answer(trace: dict) -> int | str:
    values_by_label = {str(label): int(value) for label, value in trace['values_by_label'].items()}
    target_labels = [str(label) for label in trace['target_labels']]
    variant = str(trace.get('internal_query_id') or trace.get('source_query_id') or trace.get('query_id') or trace['query_id'])
    if variant == 'sum_named_metrics':
        return int(sum((values_by_label[label] for label in target_labels)))
    if variant == 'section_extrema_arithmetic':
        value_a = values_by_label[str(trace['target_groups']['section_a_extremum'][0])]
        value_b = values_by_label[str(trace['target_groups']['section_b_extremum'][0])]
        if str(trace['extrema_operation']) == 'sum':
            return int(value_a + value_b)
        return int(abs(value_a - value_b))
    if variant == 'section_total_extrema_difference':
        group_a = [str(label) for label in trace['target_groups']['highest_total_section']]
        group_b = [str(label) for label in trace['target_groups']['lowest_total_section']]
        return int(sum((values_by_label[label] for label in group_a)) - sum((values_by_label[label] for label in group_b)))
    if variant == 'section_total_except_named':
        included = [str(label) for label in trace['target_groups']['included']]
        return int(sum((values_by_label[label] for label in included)))
    if variant == 'section_icon_total_value':
        included = [str(label) for label in trace['target_groups']['filtered_icon_cards']]
        return int(sum((values_by_label[label] for label in included)))
    if variant == 'section_icon_total_difference_value':
        group_a = [str(label) for label in trace['target_groups']['section_a_filtered_icon_cards']]
        group_b = [str(label) for label in trace['target_groups']['section_b_filtered_icon_cards']]
        total_a = sum((values_by_label[label] for label in group_a))
        total_b = sum((values_by_label[label] for label in group_b))
        return int(abs(total_a - total_b))
    if variant == 'section_icon_extremum_label':
        direction = str(trace['rank_direction'])
        totals = {str(section): int(total) for section, total in trace['filtered_section_totals'].items()}
        target_total = max(totals.values()) if direction == 'highest' else min(totals.values())
        winners = [section for section, total in totals.items() if int(total) == int(target_total)]
        assert len(winners) == 1
        return str(winners[0])
    if variant in set(METRIC_RANKED_ITEM_QUERY_IDS):
        ranked = [dict(candidate) for candidate in trace['ranked_candidates']]
        return str(ranked[int(trace['rank_position']) - 1]['label'])
    raise AssertionError(f'unsupported variant: {variant}')

def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height

def _assert_bbox_min_side(bbox: list[float], *, min_side_px: float = 24.0) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert min(x1 - x0, y1 - y0) >= float(min_side_px)

def _annotation_bboxes(annotation_value) -> list[list[float]]:
    if isinstance(annotation_value, dict):
        return [list(value) for value in annotation_value.values()]
    if isinstance(annotation_value, list) and len(annotation_value) == 4 and all(isinstance(value, (int, float)) for value in annotation_value):
        return [list(annotation_value)]
    return [list(value) for value in annotation_value]

def _bbox_overlap_area(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_w * overlap_h)

def _bbox_area(bbox: list[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def _mixed_infographic_text_bboxes(trace: dict) -> list[tuple[str, list[float]]]:
    boxes: list[tuple[str, list[float]]] = []
    for entity in trace['scene_ir']['entities']:
        entity_id = str(entity.get('entity_id', '?'))
        if entity.get('title_bbox_px'):
            boxes.append((f'{entity_id}.module_title', entity['title_bbox_px']))
        if entity.get('text_bbox_px'):
            boxes.append((f'{entity_id}.text_block', entity['text_bbox_px']))
        for field in entity.get('fields', []) or []:
            if field.get('bbox_px'):
                boxes.append((f"{entity_id}.{field.get('field_id')}.field", field['bbox_px']))
        for item in entity.get('items', []) or []:
            item_id = str(item.get('item_id'))
            if item.get('label_bbox_px'):
                boxes.append((f'{entity_id}.{item_id}.label', item['label_bbox_px']))
            for value in item.get('values', []) or []:
                if value.get('bbox_px'):
                    boxes.append((f"{entity_id}.{item_id}.{value.get('field_id')}.value", value['bbox_px']))
    return boxes

def _assert_mixed_infographic_text_has_no_significant_overlap(trace: dict) -> None:
    boxes = _mixed_infographic_text_bboxes(trace)
    for index, (name_a, bbox_a) in enumerate(boxes):
        for name_b, bbox_b in boxes[index + 1:]:
            overlap = _bbox_overlap_area(bbox_a, bbox_b)
            if overlap == 0.0:
                continue
            smaller_area = max(1.0, min(_bbox_area(bbox_a), _bbox_area(bbox_b)))
            assert overlap / smaller_area <= 0.05, (name_a, bbox_a, name_b, bbox_b)

def _assert_radial_modules_have_separate_text_bands(trace: dict) -> None:
    render_map = trace['render_map']
    modules = trace['execution_trace']['modules']
    for module in modules:
        if str(module['module_kind']) not in {'radial_bubbles', 'ring_summary'}:
            continue
        module_id = str(module['module_id'])
        for item in module['items']:
            item_id = str(item['item_id'])
            label_bbox = render_map['item_label_bboxes_px'][module_id][item_id]
            icon_bbox = render_map['icon_bboxes_px'][module_id][item_id]
            assert _bbox_overlap_area(label_bbox, icon_bbox) == 0.0
            for value_bbox in render_map['value_cell_bboxes_px'][module_id][item_id].values():
                assert _bbox_overlap_area(label_bbox, value_bbox) == 0.0

def _visible_int(value: str) -> int:
    digits = ''.join((char for char in str(value) if char.isdigit()))
    assert digits
    return int(digits)

def test_pages_hero_callout_composite_metric_extremum_label_contract() -> None:
    task = PagesHeroCalloutCompositeMetricExtremumLabelTask()
    assert tuple(task.supported_query_ids) == (
        HERO_CALLOUT_HIGHEST_COMPOSITE_QUERY_ID,
        HERO_CALLOUT_LOWEST_COMPOSITE_QUERY_ID,
    )
    cases = (
        (HERO_CALLOUT_HIGHEST_COMPOSITE_QUERY_ID, 'highest', max),
        (HERO_CALLOUT_LOWEST_COMPOSITE_QUERY_ID, 'lowest', min),
    )
    for index, (query_id, rank_direction, extremum_fn) in enumerate(cases):
        out = task.generate(99101 + index, params={'query_id': query_id, 'scene_variant': 'radial_halo', 'callout_count': 6, 'pages_context_text_enabled': False}, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        target = execution['target']
        callout_id = str(target['callout_id'])
        candidates = [dict(candidate) for candidate in target['candidate_values']]
        expected = extremum_fn(candidates, key=lambda candidate: int(candidate['composite_value']))
        assert task.task_id == HERO_CALLOUT_COMPOSITE_EXTREMUM_TASK_ID
        assert resolve_task_taxonomy(task.task_id).scene_id == 'hero_callout_infographic'
        assert out.scene_id == 'hero_callout_infographic'
        assert out.query_id == query_id
        assert str(execution['prompt_query_key']) == 'callout_composite_metric_extremum_label'
        assert out.answer_gt.type == 'string'
        assert out.annotation_gt.type == 'bbox_map'
        assert str(target['rank_direction']) == rank_direction
        assert str(target['first_field_label']) == 'Score'
        assert str(target['second_field_label']) == 'Count'
        assert all(
            int(candidate['composite_value']) == int(candidate['first_numeric_value']) + int(candidate['second_numeric_value'])
            for candidate in candidates
        )
        assert len([candidate for candidate in candidates if int(candidate['composite_value']) == int(expected['composite_value'])]) == 1
        assert str(out.answer_gt.value) == str(expected['callout_title']) == str(target['answer_value'])
        assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
        assert out.annotation_gt.value['winning_callout_card'] == trace['render_map']['callout_bboxes_px'][callout_id]
        for candidate_index, candidate in enumerate(candidates, start=1):
            assert out.annotation_gt.value[f'candidate_{candidate_index}_first_field_row'] == trace['render_map']['field_row_bboxes_px'][str(candidate['callout_id'])][str(candidate['first_field_id'])]
            assert out.annotation_gt.value[f'candidate_{candidate_index}_second_field_row'] == trace['render_map']['field_row_bboxes_px'][str(candidate['callout_id'])][str(candidate['second_field_id'])]
        assert str(render['layout']['scene_variant']) == 'radial_halo'
        assert render['page_visual_assets']['asset_root'] == 'assets/pages/visual_assets'
        assert render['page_visual_assets']['semantic_policy'] == 'non_answer_visual_context'
        assert render.get('context_text_layer', {}).get('enabled') is False
        for bbox in out.annotation_gt.value.values():
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        example = _extract_prompt_json_example(out.prompt)
        assert list(example.keys()) == ['annotation', 'answer']
        assert isinstance(example['answer'], str)
        assert 'winning_callout_card' in example['annotation']
        assert 'candidate_1_first_field_row' in example['annotation']
        assert 'candidate_1_second_field_row' in example['annotation']
        assert 'hero visual' not in out.prompt.lower()

def test_pages_hero_callout_metric_extremum_label_contract() -> None:
    task = PagesHeroCalloutMetricExtremumLabelTask()
    assert tuple(task.supported_query_ids) == (
        HERO_CALLOUT_HIGHEST_FIELD_VALUE_QUERY_ID,
        HERO_CALLOUT_LOWEST_FIELD_VALUE_QUERY_ID,
    )
    cases = (
        (HERO_CALLOUT_HIGHEST_FIELD_VALUE_QUERY_ID, 'highest', max),
        (HERO_CALLOUT_LOWEST_FIELD_VALUE_QUERY_ID, 'lowest', min),
    )
    for index, (query_id, rank_direction, extremum_fn) in enumerate(cases):
        out = task.generate(99102 + index, params={'query_id': query_id, 'scene_variant': 'side_rail_poster', 'callout_count': 6, 'target_field_label': 'Score', 'rank_direction': 'lowest' if rank_direction == 'highest' else 'highest', 'pages_context_text_enabled': False}, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        target = execution['target']
        callout_id = str(target['callout_id'])
        field_id = str(target['field_id'])
        candidates = [dict(candidate) for candidate in target['candidate_values']]
        expected = extremum_fn(candidates, key=lambda candidate: int(candidate['numeric_value']))
        assert task.task_id == HERO_CALLOUT_METRIC_EXTREMUM_TASK_ID
        assert resolve_task_taxonomy(task.task_id).scene_id == 'hero_callout_infographic'
        assert out.query_id == query_id
        assert str(execution['query_id']) == query_id
        assert str(execution['prompt_query_key']) == 'callout_metric_extremum_label'
        assert out.answer_gt.type == 'string'
        assert out.annotation_gt.type == 'bbox_map'
        assert str(target['rank_direction']) == rank_direction
        assert str(target['field_label']) == 'Score'
        assert len([candidate for candidate in candidates if int(candidate['numeric_value']) == int(expected['numeric_value'])]) == 1
        assert str(out.answer_gt.value) == str(expected['callout_title']) == str(target['answer_value'])
        assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
        assert out.annotation_gt.value['winning_callout_card'] == trace['render_map']['callout_bboxes_px'][callout_id]
        for candidate_index, candidate in enumerate(candidates, start=1):
            assert out.annotation_gt.value[f'candidate_{candidate_index}_field_row'] == trace['render_map']['field_row_bboxes_px'][str(candidate['callout_id'])][str(candidate['field_id'])]
        for bbox in out.annotation_gt.value.values():
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        example = _extract_prompt_json_example(out.prompt)
        assert list(example.keys()) == ['annotation', 'answer']
        assert isinstance(example['answer'], str)
        assert 'winning_callout_card' in example['annotation']
        assert 'candidate_1_field_row' in example['annotation']
        assert 'hero visual' not in out.prompt.lower()
    with pytest.raises(ValueError):
        task.generate(99122, params={'query_id': SINGLE_QUERY_ID}, max_attempts=10)

def test_pages_hero_callout_condition_count_contract() -> None:
    task = PagesHeroCalloutConditionCountTask()
    assert tuple(task.supported_query_ids) == (
        HERO_CALLOUT_ABOVE_THRESHOLD_QUERY_ID,
        HERO_CALLOUT_BELOW_THRESHOLD_QUERY_ID,
    )
    cases = (
        (HERO_CALLOUT_ABOVE_THRESHOLD_QUERY_ID, 'above', lambda value, threshold: int(value) > int(threshold)),
        (HERO_CALLOUT_BELOW_THRESHOLD_QUERY_ID, 'below', lambda value, threshold: int(value) < int(threshold)),
    )
    for index, (query_id, operator, predicate) in enumerate(cases):
        out = task.generate(99103 + index, params={'query_id': query_id, 'scene_variant': 'split_poster', 'callout_count': 7, 'target_field_label': 'Score', 'pages_context_text_enabled': False}, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        target = execution['target']
        threshold = int(target['threshold_value'])
        expected_matches = [candidate for candidate in target['candidate_values'] if predicate(int(candidate['numeric_value']), int(threshold))]
        expected_bboxes = [trace['render_map']['field_row_bboxes_px'][str(match['callout_id'])][str(match['field_id'])] for match in expected_matches]
        assert task.task_id == HERO_CALLOUT_CONDITION_COUNT_TASK_ID
        assert resolve_task_taxonomy(task.task_id).scene_id == 'hero_callout_infographic'
        assert out.query_id == query_id
        assert str(execution['query_id']) == query_id
        assert str(execution['prompt_query_key']) == 'callout_condition_count'
        assert out.answer_gt.type == 'integer'
        assert out.annotation_gt.type == 'bbox_set'
        assert str(target['condition_operator']) == operator
        assert str(target['field_label']) == 'Score'
        assert int(out.answer_gt.value) == len(expected_matches) == int(target['answer_value'])
        assert 0 < int(out.answer_gt.value) < len(target['candidate_values'])
        assert out.annotation_gt.value == expected_bboxes
        assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        example = _extract_prompt_json_example(out.prompt)
        assert list(example.keys()) == ['annotation', 'answer']
        assert isinstance(example['answer'], int)
        assert isinstance(example['annotation'], list)
        assert 'hero visual' not in out.prompt.lower()

def test_pages_mixed_infographic_module_field_value_contract() -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    out = task.generate(98100, params={'query_id': MIXED_INFOGRAPHIC_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 9, 'native_layout_mode': 'top_right_callout', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    item_id = str(target['item_id'])
    selector_field_id = str(target['selector_field_id'])
    answer_field_id = str(target['answer_field_id'])
    assert task.task_id == MIXED_INFOGRAPHIC_TASK_ID
    assert not hasattr(task, 'scene_id')
    assert resolve_task_taxonomy(MIXED_INFOGRAPHIC_TASK_ID).scene_id == 'mixed_infographic_page'
    assert out.scene_id == 'mixed_infographic_page'
    assert out.query_id == MIXED_INFOGRAPHIC_QUERY_ID
    assert str(execution['prompt_query_key']) == 'module_field_value_label'
    assert 'module in reading order' in str(target['module_position_phrase'])
    assert int(target['module_position_index']) >= 1
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert str(out.answer_gt.value) == str(target['answer_value'])
    assert str(execution['answer_value']) == str(target['answer_value'])
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    assert int(execution['module_count']) == 9
    assert len(execution['modules']) == 9
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    assert out.annotation_gt.value == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][answer_field_id]
    _assert_bbox_min_side(out.annotation_gt.value)
    diagnostic = trace['projected_annotation']['bbox_map']
    assert diagnostic['module_title'] == trace['render_map']['module_title_bboxes_px'][module_id]
    assert diagnostic['ranked_item_label'] == trace['render_map']['item_label_bboxes_px'][module_id][item_id]
    assert diagnostic['ranked_item_container'] == trace['render_map']['item_container_bboxes_px'][module_id][item_id]
    assert diagnostic['selector_field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][selector_field_id]
    assert diagnostic['answer_field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][answer_field_id]
    assert diagnostic['ranked_selector_value'] == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][selector_field_id]
    assert diagnostic['answer_value_cell'] == out.annotation_gt.value
    assert str(render['background_style']['style_spec']['kind']) == 'information_scene_style'
    assert str(render['information_scene_style']['kind']) == 'information_scene_style'
    assert int(render['information_scene_style']['layout_style']['shadow_offset_px']) == 0
    assert str(render['information_scene_style']['pages_adapter']['information_scene_shadow_policy']) == 'none'
    assert str(render['layout']['scene_variant']) == 'dashboard_blocks'
    assert str(render['layout']['native_layout_mode']) == 'top_right_callout'
    assert str(render['native_layout_mode']) == 'top_right_callout'
    assert str(execution['native_layout_mode']) == 'top_right_callout'
    assert render['layout']['page_backdrops']
    for backdrop in render['layout']['page_backdrops']:
        assert float(backdrop['blend_scale']) == pytest.approx(0.35)
    assert len(render['module_kinds']) == 9
    visual_assets = render['page_visual_assets']
    assert visual_assets['asset_root'] == 'assets/pages/visual_assets'
    assert visual_assets['semantic_policy'] == 'non_answer_visual_context'
    assert visual_assets['roles']['hero_anchor']['role'] == 'hero_anchor'
    assert visual_assets['hero_anchor_drawn'] is True
    assert 'hero_anchor' in trace['render_map']['visual_asset_bboxes_px']['hero_anchor']
    assert sorted(visual_assets['roles']['module_section_assets']) == sorted(trace['render_map']['module_bboxes_px'])
    assert sorted(visual_assets['roles']['item_badge_assets']) == sorted(trace['render_map']['module_bboxes_px'])
    assert sorted(trace['render_map']['visual_asset_bboxes_px']['module_section_assets']) == sorted(trace['render_map']['module_bboxes_px'])
    assert sorted(trace['render_map']['visual_asset_bboxes_px']['item_badge_assets']) == sorted(trace['render_map']['module_bboxes_px'])
    assert len(render['infographic_text_blocks']) >= 2
    assert set(trace['render_map']['infographic_text_block_bboxes_px']) == {str(block['block_id']) for block in render['infographic_text_blocks']}
    assert 'mixed_infographic_font_profile' in render['font_assets']
    assert 'text_legibility' in render['information_scene_style']
    _assert_radial_modules_have_separate_text_bands(trace)
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    for bbox in trace['render_map']['infographic_text_block_bboxes_px'].values():
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        assert _bbox_overlap_area(bbox, out.annotation_gt.value) == 0.0
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert len(example['annotation']) == 4

@pytest.mark.parametrize('rank_position', [1, 2])
def test_pages_mixed_infographic_module_field_ranked_item_label_contract(rank_position: int) -> None:
    task = PagesMixedInfographicModuleFieldRankedItemLabelTask()
    out = task.generate(98103 + int(rank_position), params={'query_id': MIXED_INFOGRAPHIC_FIELD_RANKED_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 8, 'rank_direction': 'highest', 'rank_position': int(rank_position), 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    field_id = str(target['field_id'])
    item_id = str(target['item_id'])
    ranked = sorted([dict(candidate) for candidate in target['candidate_values']], key=lambda candidate: int(candidate['numeric_value']), reverse=True)
    expected = ranked[int(target['rank_position']) - 1]
    assert task.task_id == MIXED_INFOGRAPHIC_FIELD_RANKED_TASK_ID
    assert resolve_task_taxonomy(task.task_id).scene_id == 'mixed_infographic_page'
    assert out.query_id == MIXED_INFOGRAPHIC_FIELD_RANKED_QUERY_ID
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert int(target['rank_position']) == int(rank_position)
    assert str(target['rank_direction']) == 'highest'
    assert len({int(candidate['numeric_value']) for candidate in ranked}) == len(ranked)
    assert str(out.answer_gt.value) == str(expected['item_label']) == str(target['item_label'])
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    assert out.annotation_gt.value == trace['render_map']['item_container_bboxes_px'][module_id][item_id]
    _assert_bbox_min_side(out.annotation_gt.value)
    diagnostic = trace['projected_annotation']['bbox_map']
    assert diagnostic['module_title'] == trace['render_map']['module_title_bboxes_px'][module_id]
    assert diagnostic['field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][field_id]
    assert diagnostic['ranked_item'] == out.annotation_gt.value
    assert diagnostic['ranked_item_label'] == trace['render_map']['item_label_bboxes_px'][module_id][item_id]
    assert diagnostic['ranked_value'] == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][field_id]
    for index, candidate in enumerate(ranked, start=1):
        assert diagnostic[f'candidate_{index}'] == trace['render_map']['value_cell_bboxes_px'][module_id][str(candidate['item_id'])][field_id]
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert len(example['annotation']) == 4

def test_pages_mixed_infographic_page_field_extremum_module_label_contract() -> None:
    task = PagesMixedInfographicPageFieldExtremumModuleLabelTask()
    out = task.generate(98107, params={'query_id': MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 8, 'target_field_label': 'Score', 'rank_direction': 'highest', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    field_id = str(target['field_id'])
    item_id = str(target['item_id'])
    candidates = [dict(candidate) for candidate in target['candidate_values']]
    expected = max(candidates, key=lambda candidate: int(candidate['numeric_value']))
    assert task.task_id == MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_TASK_ID
    assert resolve_task_taxonomy(task.task_id).scene_id == 'mixed_infographic_page'
    assert out.query_id == MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_QUERY_ID
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert str(target['rank_direction']) == 'highest'
    assert str(target['field_label']) == 'Score'
    assert len({str(candidate['module_id']) for candidate in candidates}) >= 3
    assert len([candidate for candidate in candidates if int(candidate['numeric_value']) == int(expected['numeric_value'])]) == 1
    assert str(out.answer_gt.value) == str(expected['module_title']) == str(target['answer_value'])
    assert str(target['module_id']) == str(expected['module_id'])
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    assert out.annotation_gt.value == trace['render_map']['module_bboxes_px'][module_id]
    diagnostic = trace['projected_annotation']['bbox_map']
    assert diagnostic['winning_module_title'] == trace['render_map']['module_title_bboxes_px'][module_id]
    assert diagnostic['winning_field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][field_id]
    assert diagnostic['winning_item'] == trace['render_map']['item_label_bboxes_px'][module_id][item_id]
    assert diagnostic['winning_value'] == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][field_id]
    for index, candidate in enumerate(candidates, start=1):
        assert diagnostic[f'candidate_{index}'] == trace['render_map']['value_cell_bboxes_px'][str(candidate['module_id'])][str(candidate['item_id'])][str(candidate['field_id'])]
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert len(example['annotation']) == 4

def test_pages_mixed_infographic_module_two_field_condition_item_label_contract() -> None:
    task = PagesMixedInfographicModuleTwoFieldConditionItemLabelTask()
    out = task.generate(98106, params={'query_id': MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 8, 'condition_operator': 'above', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    item_id = str(target['item_id'])
    numeric_field_id = str(target['numeric_field_id'])
    category_field_id = str(target['category_field_id'])
    threshold = int(target['threshold_value'])
    category_value = str(target['category_value'])
    numeric_matches = {str(value['item_id']) for value in target['numeric_matches']}
    category_matches = {str(value['item_id']) for value in target['category_matches']}
    intersection = sorted(numeric_matches.intersection(category_matches))
    assert task.task_id == MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_TASK_ID
    assert resolve_task_taxonomy(task.task_id).scene_id == 'mixed_infographic_page'
    assert out.query_id == MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_QUERY_ID
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert str(target['condition_operator']) == 'above'
    assert len(numeric_matches) > 1
    assert len(category_matches) > 1
    assert intersection == [item_id]
    assert int(target['matching_numeric_value']['numeric_value']) > int(threshold)
    assert str(target['matching_category_value']['visible_value']) == str(category_value)
    assert str(out.answer_gt.value) == str(target['item_label']) == str(target['answer_value'])
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    assert out.annotation_gt.value == trace['render_map']['item_container_bboxes_px'][module_id][item_id]
    _assert_bbox_min_side(out.annotation_gt.value)
    diagnostic = trace['projected_annotation']['bbox_map']
    assert diagnostic['module_title'] == trace['render_map']['module_title_bboxes_px'][module_id]
    assert diagnostic['numeric_field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][numeric_field_id]
    assert diagnostic['category_field_label'] == trace['render_map']['field_label_bboxes_px'][module_id][category_field_id]
    assert diagnostic['matching_item'] == out.annotation_gt.value
    assert diagnostic['matching_item_label'] == trace['render_map']['item_label_bboxes_px'][module_id][item_id]
    assert diagnostic['numeric_value_cell'] == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][numeric_field_id]
    assert diagnostic['category_value_cell'] == trace['render_map']['value_cell_bboxes_px'][module_id][item_id][category_field_id]
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert len(example['annotation']) == 4

def test_pages_mixed_infographic_module_condition_item_count_contract() -> None:
    task = PagesMixedInfographicModuleConditionItemCountTask()
    out = task.generate(98102, params={'query_id': MIXED_INFOGRAPHIC_CONDITION_COUNT_QUERY_ID, 'scene_variant': 'poster_sections', 'module_count': 5, 'condition_operator': 'above', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    field_id = str(target['field_id'])
    threshold = int(target['threshold_value'])
    expected_matches = [candidate for candidate in target['candidate_values'] if int(candidate['numeric_value']) > int(threshold)]
    expected_bboxes = [trace['render_map']['value_cell_bboxes_px'][module_id][str(match['item_id'])][field_id] for match in expected_matches]
    assert task.task_id == MIXED_INFOGRAPHIC_CONDITION_COUNT_TASK_ID
    assert out.query_id == MIXED_INFOGRAPHIC_CONDITION_COUNT_QUERY_ID
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    assert int(out.answer_gt.value) == len(expected_matches) == int(target['answer_value'])
    assert 0 < int(out.answer_gt.value) < len(target['candidate_values'])
    assert out.annotation_gt.value == expected_bboxes
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        _assert_bbox_min_side(bbox)
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], int)
    assert isinstance(example['annotation'], list)

def test_pages_mixed_infographic_retries_infeasible_row_layout() -> None:
    task = PagesMixedInfographicModuleConditionItemCountTask()
    out_a = task.generate(500730, params={}, max_attempts=100)
    out_b = task.generate(500730, params={}, max_attempts=100)

    execution = out_a.trace_payload["execution_trace"]
    assert int(execution["generation_attempt_index"]) == 1
    assert int(execution["generation_attempt_seed"]) == hash64(
        500730,
        f"{task.task_id}.mixed_infographic_annotation_retry",
        1,
    )
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert execution == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize(
    "error",
    (
        ValueError("unrelated construction error"),
        TypeError("unrelated construction type error"),
        KeyError("unrelated construction key error"),
    ),
)
def test_pages_mixed_infographic_does_not_retry_unrelated_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    calls = 0

    def fail_scene_construction(**_kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(
        mixed_infographic_lifecycle,
        "build_scene_context",
        fail_scene_construction,
    )
    task = PagesMixedInfographicModuleConditionItemCountTask()
    with pytest.raises(type(error)) as caught:
        task.generate(500731, params={}, max_attempts=5)

    assert caught.value is error
    assert calls == 1

def test_pages_mixed_infographic_module_field_total_value_contract() -> None:
    task = PagesMixedInfographicModuleFieldTotalValueTask()
    out = task.generate(98101, params={'query_id': MIXED_INFOGRAPHIC_FIELD_TOTAL_QUERY_ID, 'scene_variant': 'compact_newsletter', 'module_count': 8, 'target_field_label': 'Count', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_id = str(target['module_id'])
    field_id = str(target['field_id'])
    expected_total = sum((_visible_int(value_payload['visible_value']) for value_payload in target['summed_values']))
    expected_bboxes = [trace['render_map']['value_cell_bboxes_px'][module_id][str(value_payload['item_id'])][field_id] for value_payload in target['summed_values']]
    assert task.task_id == MIXED_INFOGRAPHIC_FIELD_TOTAL_TASK_ID
    assert out.query_id == MIXED_INFOGRAPHIC_FIELD_TOTAL_QUERY_ID
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    assert int(out.answer_gt.value) == int(expected_total) == int(target['answer_value'])
    assert out.annotation_gt.value == expected_bboxes
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        _assert_bbox_min_side(bbox)
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], int)
    assert isinstance(example['annotation'], list)

@pytest.mark.parametrize(
    'task_cls',
    [
        PagesMixedInfographicModuleConditionItemCountTask,
        PagesMixedInfographicModuleFieldRankedItemLabelTask,
        PagesMixedInfographicModuleFieldTotalValueTask,
        PagesMixedInfographicModuleFieldValueLabelTask,
        PagesMixedInfographicModuleTwoFieldConditionItemLabelTask,
    ],
)
def test_pages_mixed_infographic_public_bbox_annotations_satisfy_min_side(task_cls) -> None:
    task = task_cls()
    for index in range(8):
        out = task.generate(
            hash64(20260701, str(task.task_id), index),
            params={'pages_context_text_enabled': False},
            max_attempts=80,
        )
        for bbox in _annotation_bboxes(out.annotation_gt.value):
            _assert_bbox_min_side(bbox)

def test_pages_mixed_infographic_two_module_field_total_comparison_module_label_contract() -> None:
    task = PagesMixedInfographicTwoModuleFieldTotalComparisonModuleLabelTask()
    out = task.generate(98108, params={'query_id': MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 8, 'target_field_label': 'Score', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target']
    module_a = dict(target['module_a'])
    module_b = dict(target['module_b'])
    module_a_id = str(module_a['module_id'])
    module_b_id = str(module_b['module_id'])
    field_a_id = str(module_a['field_id'])
    field_b_id = str(module_b['field_id'])
    module_a_total = sum((_visible_int(value_payload['visible_value']) for value_payload in module_a['summed_values']))
    module_b_total = sum((_visible_int(value_payload['visible_value']) for value_payload in module_b['summed_values']))
    expected_answer = str(module_a['module_title'] if module_a_total > module_b_total else module_b['module_title'])
    assert task.task_id == MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_TASK_ID
    assert resolve_task_taxonomy(task.task_id).scene_id == 'mixed_infographic_page'
    assert out.query_id == MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_QUERY_ID
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert str(target['field_label']) == 'Score'
    assert module_a_total == int(target['module_a_total'])
    assert module_b_total == int(target['module_b_total'])
    assert module_a_total != module_b_total
    assert str(out.answer_gt.value) == expected_answer == str(target['answer_value'])
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    winning_module_id = module_a_id if str(target['winning_side']) == 'module_a' else module_b_id
    assert out.annotation_gt.value == trace['render_map']['module_bboxes_px'][winning_module_id]
    diagnostic = trace['projected_annotation']['bbox_map']
    assert diagnostic['module_a_title'] == trace['render_map']['module_title_bboxes_px'][module_a_id]
    assert diagnostic['module_b_title'] == trace['render_map']['module_title_bboxes_px'][module_b_id]
    assert diagnostic['field_label_a'] == trace['render_map']['field_label_bboxes_px'][module_a_id][field_a_id]
    assert diagnostic['field_label_b'] == trace['render_map']['field_label_bboxes_px'][module_b_id][field_b_id]
    for index, value_payload in enumerate(module_a['summed_values'], start=1):
        assert diagnostic[f'module_a_value_{index}'] == trace['render_map']['value_cell_bboxes_px'][module_a_id][str(value_payload['item_id'])][field_a_id]
    for index, value_payload in enumerate(module_b['summed_values'], start=1):
        assert diagnostic[f'module_b_value_{index}'] == trace['render_map']['value_cell_bboxes_px'][module_b_id][str(value_payload['item_id'])][field_b_id]
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert len(example['annotation']) == 4

def test_pages_mixed_infographic_native_text_blocks_and_font_profile_default() -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    out = task.generate(98122, params={'query_id': MIXED_INFOGRAPHIC_QUERY_ID, 'scene_variant': 'collage_board', 'module_count': 9}, max_attempts=10)
    trace = out.trace_payload
    render = trace['render_spec']
    render_map = trace['render_map']
    context_layer = render.get('context_text_layer', {})
    assert context_layer.get('enabled') is False
    assert context_layer.get('layout_spec', {}).get('reason') == 'disabled'
    assert int(trace['execution_trace']['native_text_block_count']) in {4, 5}
    assert len(render['infographic_text_blocks']) == int(trace['execution_trace']['native_text_block_count'])
    assert len(render_map['infographic_text_block_bboxes_px']) == len(render['infographic_text_blocks'])
    assert any((entity['kind'] == 'mixed_infographic_text_block' for entity in trace['scene_ir']['entities']))
    paragraph_blocks = [block for block in render['infographic_text_blocks'] if block['kind'] == 'paragraph_note']
    assert len(paragraph_blocks) == 2
    for block in paragraph_blocks:
        bbox = render_map['infographic_text_block_bboxes_px'][str(block['block_id'])]
        assert float(bbox[2] - bbox[0]) >= 150.0
        assert float(bbox[3] - bbox[1]) >= 70.0
    font_profile = render['font_assets']['mixed_infographic_font_profile']
    families = {str(font_profile['readout_family']), str(font_profile['section_header_family']), str(font_profile['accent_context_family'])}
    assert len(families) >= 2
    assert set(font_profile['module_title_families_by_id']) == set(render_map['module_title_bboxes_px'])
    for block in render['infographic_text_blocks']:
        bbox = render_map['infographic_text_block_bboxes_px'][str(block['block_id'])]
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        for annotation_bbox in _annotation_bboxes(out.annotation_gt.value):
            assert _bbox_overlap_area(bbox, annotation_bbox) == 0.0

@pytest.mark.parametrize('native_layout_mode', MIXED_INFOGRAPHIC_NATIVE_LAYOUT_MODES)
def test_pages_mixed_infographic_native_layout_modes_render_inside_canvas(native_layout_mode: str) -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    out = task.generate(98270 + MIXED_INFOGRAPHIC_NATIVE_LAYOUT_MODES.index(native_layout_mode), params={'query_id': MIXED_INFOGRAPHIC_QUERY_ID, 'scene_variant': 'dashboard_blocks', 'module_count': 9, 'native_layout_mode': native_layout_mode, 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    render = trace['render_spec']
    render_map = trace['render_map']
    layout = render['layout']
    assert str(render['native_layout_mode']) == str(native_layout_mode)
    assert str(layout['native_layout_mode']) == str(native_layout_mode)
    assert str(trace['execution_trace']['native_layout_mode']) == str(native_layout_mode)
    assert len(layout['slot_bboxes_px']) == 9
    _assert_bbox_inside_canvas(layout['content_bbox_px'], width=int(render['canvas_width']), height=int(render['canvas_height']))
    _assert_bbox_inside_canvas(layout['native_text_footer_bbox_px'], width=int(render['canvas_width']), height=int(render['canvas_height']))
    hero_bboxes = render_map['visual_asset_bboxes_px']['hero_anchor']
    if str(native_layout_mode) == 'footer_only':
        assert render['page_visual_assets']['hero_anchor_drawn'] is False
        assert hero_bboxes == {}
    else:
        assert render['page_visual_assets']['hero_anchor_drawn'] is True
        assert 'hero_anchor' in hero_bboxes
        _assert_bbox_inside_canvas(hero_bboxes['hero_anchor'], width=int(render['canvas_width']), height=int(render['canvas_height']))
    if str(native_layout_mode) == 'left_side_rail':
        assert float(layout['content_bbox_px'][0]) > float(render['page_bbox_px'][0]) + 120.0
    if str(native_layout_mode) == 'right_side_rail':
        assert float(layout['content_bbox_px'][2]) < float(render['page_bbox_px'][2]) - 120.0
    if str(native_layout_mode) == 'poster_anchor_strip':
        assert float(layout['content_bbox_px'][1]) > float(render['page_bbox_px'][1]) + 190.0
    for bbox in render_map['infographic_text_block_bboxes_px'].values():
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        for annotation_bbox in _annotation_bboxes(out.annotation_gt.value):
            assert _bbox_overlap_area(bbox, annotation_bbox) == 0.0

@pytest.mark.parametrize('scene_variant', MIXED_INFOGRAPHIC_SCENE_VARIANTS)
def test_pages_mixed_infographic_scene_variants_render_inside_canvas(scene_variant: str) -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    out = task.generate(98140 + MIXED_INFOGRAPHIC_SCENE_VARIANTS.index(scene_variant), params={'query_id': MIXED_INFOGRAPHIC_QUERY_ID, 'scene_variant': scene_variant, 'module_count': 9, 'pages_context_text_enabled': False}, max_attempts=10)
    render = out.trace_payload['render_spec']
    assert str(render['layout']['scene_variant']) == str(scene_variant)
    assert len(render['layout']['slot_bboxes_px']) == 9
    assert int(out.trace_payload['execution_trace']['module_count']) == 9
    _assert_radial_modules_have_separate_text_bands(out.trace_payload)
    for bbox in _annotation_bboxes(out.annotation_gt.value):
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))

@pytest.mark.parametrize('instance_seed', [60023, 60043, 60058, 60059, 60062, 60098, 60126, 60151, 60157, 60170, 60194, 60241, 60260, 60294, 60314, 60335, 60483])
def test_pages_mixed_infographic_text_bboxes_do_not_occlude(instance_seed: int) -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    out = task.generate(int(instance_seed), params={'pages_context_text_enabled': False}, max_attempts=10)
    _assert_mixed_infographic_text_has_no_significant_overlap(out.trace_payload)

def test_pages_mixed_infographic_is_deterministic() -> None:
    task = PagesMixedInfographicModuleFieldValueLabelTask()
    params = {'query_id': MIXED_INFOGRAPHIC_QUERY_ID, 'scene_variant': 'compact_newsletter', 'module_count': 9, 'pages_context_text_enabled': False}
    out_a = task.generate(98190, params=dict(params), max_attempts=10)
    out_b = task.generate(98190, params=dict(params), max_attempts=10)
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()

def test_pages_sectioned_infographic_item_count_matches_contract() -> None:
    task = PagesSectionedInfographicSectionItemCountTask()
    out = task.generate(97010, params={'query_id': SINGLE_QUERY_ID, 'scene_variant': 'topic_cards', 'section_count': 4, 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target_section']
    section_id = str(target['section_id'])
    assert task.task_id == SECTIONED_INFOGRAPHIC_ITEM_COUNT_TASK_ID
    assert out.scene_id == 'sectioned_infographic'
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['prompt_query_key'] == SECTIONED_PROMPT_QUERY_KEY
    assert execution['source_query_id'] == SECTIONED_PROMPT_QUERY_KEY
    assert trace['query_spec']['prompt_variant']['prompt_schema_version'] == 'v1'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    assert int(execution['section_count']) == 4
    assert int(out.answer_gt.value) == int(target['item_count'])
    assert int(execution['answer_value']) == int(target['item_count'])
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert str(render['background_style']['style_spec']['kind']) == 'information_scene_style'
    assert str(render['information_scene_style']['kind']) == 'information_scene_style'
    expected_bboxes = [trace['render_map']['item_row_bboxes_px'][section_id][str(item_id)] for item_id in target['item_ids']]
    assert out.annotation_gt.value == expected_bboxes
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
    _assert_bbox_inside_canvas(trace['render_map']['section_title_bboxes_px'][section_id], width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], int)
    assert isinstance(example['annotation'], list)

def test_pages_sectioned_infographic_filtered_item_label_matches_contract() -> None:
    task = PagesSectionedInfographicSectionFilteredItemLabelTask()
    out = task.generate(97021, params={'query_id': SINGLE_QUERY_ID, 'scene_variant': 'checklist_bands', 'section_count': 4, 'item_count_support': [5], 'target_section_index': 1, 'target_marker': 'flag_marker', 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    target = execution['target_section']
    section_id = str(target['section_id'])
    item_id = str(target['item_id'])
    assert task.task_id == SECTIONED_INFOGRAPHIC_FILTERED_ITEM_TASK_ID
    assert out.scene_id == 'sectioned_infographic'
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['prompt_query_key'] == SECTIONED_FILTERED_PROMPT_QUERY_KEY
    assert execution['source_query_id'] == SECTIONED_FILTERED_PROMPT_QUERY_KEY
    assert trace['query_spec']['prompt_variant']['prompt_schema_version'] == 'v1'
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    assert str(out.answer_gt.value) == str(target['item_label'])
    assert str(target['marker']) == 'flag_marker'
    assert str(target['filter_marker_label']) == 'flag marker'
    assert str(execution['answer_value']) == str(target['item_label'])
    assert trace['projected_annotation']['type'] == 'bbox'
    assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox'] == out.annotation_gt.value
    assert out.annotation_gt.value == trace['render_map']['item_row_bboxes_px'][section_id][item_id]
    reasoning_bboxes = trace['render_map']['reasoning_bboxes_px']
    assert set(reasoning_bboxes) == {'section_title', 'filter_marker', 'target_item'}
    assert reasoning_bboxes['section_title'] == trace['render_map']['section_title_bboxes_px'][section_id]
    assert reasoning_bboxes['filter_marker'] == trace['render_map']['item_marker_bboxes_px'][section_id][item_id]
    assert reasoning_bboxes['target_item'] == out.annotation_gt.value
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    for bbox in reasoning_bboxes.values():
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
    target_section = next((section for section in execution['sections'] if str(section['section_id']) == section_id))
    marker_matches = [item for item in target_section['items'] if str(item['marker']) == str(target['marker'])]
    assert len(marker_matches) == 1
    assert str(marker_matches[0]['item_id']) == item_id
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert isinstance(example['annotation'], list)
    assert len(example['annotation']) == 4

@pytest.mark.parametrize('scene_variant', SECTIONED_SCENE_VARIANTS)
def test_pages_sectioned_infographic_scene_variants_render_inside_canvas(scene_variant: str) -> None:
    task = PagesSectionedInfographicSectionItemCountTask()
    out = task.generate(97040 + SECTIONED_SCENE_VARIANTS.index(scene_variant), params={'query_id': SINGLE_QUERY_ID, 'scene_variant': scene_variant, 'section_count': 5, 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    render = trace['render_spec']
    assert str(render['scene_variant']) == str(scene_variant)
    for section_id, section_bbox in trace['render_map']['section_bboxes_px'].items():
        _assert_bbox_inside_canvas(section_bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        _assert_bbox_inside_canvas(trace['render_map']['section_title_bboxes_px'][str(section_id)], width=int(render['canvas_width']), height=int(render['canvas_height']))
        for bbox in trace['render_map']['item_row_bboxes_px'][str(section_id)].values():
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
        for bbox in trace['render_map']['item_marker_bboxes_px'][str(section_id)].values():
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))

@pytest.mark.parametrize('scene_variant', SECTIONED_SCENE_VARIANTS)
def test_pages_sectioned_infographic_filtered_item_scene_variants_render_inside_canvas(scene_variant: str) -> None:
    task = PagesSectionedInfographicSectionFilteredItemLabelTask()
    out = task.generate(97060 + SECTIONED_SCENE_VARIANTS.index(scene_variant), params={'query_id': SINGLE_QUERY_ID, 'scene_variant': scene_variant, 'section_count': 5, 'item_count_support': [6], 'pages_context_text_enabled': False}, max_attempts=10)
    trace = out.trace_payload
    render = trace['render_spec']
    target = trace['execution_trace']['target_section']
    assert str(render['scene_variant']) == str(scene_variant)
    assert str(out.answer_gt.value) == str(target['item_label'])
    for bbox in trace['render_map']['section_title_bboxes_px'].values():
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
    for item_map in trace['render_map']['item_marker_bboxes_px'].values():
        for bbox in item_map.values():
            _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))

@pytest.mark.parametrize('query_id', METRIC_VALUE_QUERY_IDS)
def test_pages_infographic_metric_arithmetic_variants_match_contract(query_id: str) -> None:
    task = _metric_value_task_for_query(query_id)
    out = task.generate(97100 + METRIC_VALUE_QUERY_IDS.index(query_id), params={'card_count': 20, 'section_count': 4, 'operand_count': 4}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['source_query_id'] == query_id
    assert trace['query_spec']['params']['internal_query_id'] == query_id
    assert out.answer_gt.type == 'integer'
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    assert int(execution['card_count']) == 20
    assert int(execution['section_count']) == 4
    assert len(execution['cards']) == 20
    assert len(trace['scene_ir']['entities']) == 20
    assert set(trace['render_map']['label_bboxes_px'].keys()) == set(execution['labels'])
    assert set(trace['render_map']['value_bboxes_px'].keys()) == set(execution['labels'])
    expected = _expected_answer(execution)
    assert int(out.answer_gt.value) == int(expected)
    assert int(execution['answer_value']) == int(expected)
    assert int(trace['query_spec']['params']['target_answer']) == int(expected)
    projected = trace['projected_annotation']
    target_labels = [str(label) for label in execution['target_labels']]
    if query_id in METRIC_VALUE_BBOX_SET_QUERY_IDS:
        assert out.annotation_gt.type == 'bbox_set'
        expected_bboxes = [
            [float(value) for value in projected['card_bbox_map'][label]]
            for label in target_labels
        ]
        annotation_bboxes = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]
        assert annotation_bboxes == expected_bboxes
        assert projected['bbox_set'] == out.annotation_gt.value
        if query_id == 'section_total_except_named':
            assert set(target_labels) == set(execution['target_groups']['included'])
            assert not (set(target_labels) & set(execution['target_groups']['excluded']))
        example_annotation_type = list
    elif query_id in METRIC_VALUE_BBOX_SET_MAP_QUERY_IDS:
        assert out.annotation_gt.type == 'bbox_set_map'
        assert projected['bbox_set_map'] == out.annotation_gt.value
        expected_group_keys = METRIC_VALUE_BBOX_SET_MAP_GROUP_KEYS[query_id]
        assert tuple(out.annotation_gt.value.keys()) == expected_group_keys
        annotation_bboxes = []
        for group_key in expected_group_keys:
            group_labels = [str(label) for label in execution['target_groups'][group_key]]
            expected_group_bboxes = [
                [float(value) for value in projected['card_bbox_map'][label]]
                for label in group_labels
            ]
            actual_group_bboxes = [
                [float(value) for value in bbox]
                for bbox in out.annotation_gt.value[group_key]
            ]
            assert actual_group_bboxes == expected_group_bboxes
            annotation_bboxes.extend(actual_group_bboxes)
        assert len(annotation_bboxes) == len(target_labels)
        assert set(projected['bbox_map'].keys()) == set(target_labels)
        example_annotation_type = dict
    else:
        assert query_id == 'section_extrema_arithmetic'
        assert out.annotation_gt.type == 'bbox_map'
        assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
        assert tuple(out.annotation_gt.value.keys()) == ('section_a_extremum', 'section_b_extremum')
        annotation_bboxes = []
        for role_key in ('section_a_extremum', 'section_b_extremum'):
            label = str(execution['target_groups'][role_key][0])
            expected_bbox = [float(value) for value in projected['card_bbox_map'][label]]
            actual_bbox = [float(value) for value in out.annotation_gt.value[role_key]]
            assert actual_bbox == expected_bbox
            annotation_bboxes.append(actual_bbox)
        assert len(annotation_bboxes) == len(target_labels)
        example_annotation_type = dict
    for bbox in annotation_bboxes:
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
    for label in target_labels:
        assert label in trace['projected_annotation']['card_bbox_map']
    for value in execution['target_values']:
        assert isinstance(int(value), int)
    assert set(execution['target_groups'].keys()).issubset({'highest_total_section', 'included', 'lowest_total_section', 'excluded', 'section_a_extremum', 'section_b_extremum', 'filtered_icon_cards', 'section_a_filtered_icon_cards', 'section_b_filtered_icon_cards'})
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], int)
    assert isinstance(example['annotation'], example_annotation_type)

def test_pages_infographic_metric_arithmetic_is_deterministic() -> None:
    task = PagesInfographicSumNamedMetricsValueTask()
    sectioned_task = PagesSectionedInfographicSectionFilteredItemLabelTask()
    params = {'card_count': 22, 'section_count': 4, 'operand_count': 5}
    sectioned_params = {'scene_variant': 'topic_cards', 'section_count': 4, 'item_count_support': [4], 'target_section_index': 2, 'target_marker': 'star_marker', 'pages_context_text_enabled': False}
    out_a = task.generate(98231, params=params, max_attempts=10)
    out_b = task.generate(98231, params=params, max_attempts=10)
    sectioned_a = sectioned_task.generate(98232, params=sectioned_params, max_attempts=10)
    sectioned_b = sectioned_task.generate(98232, params=sectioned_params, max_attempts=10)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload['execution_trace'] == out_b.trace_payload['execution_trace']
    assert sectioned_a.prompt == sectioned_b.prompt
    assert sectioned_a.answer_gt.to_dict() == sectioned_b.answer_gt.to_dict()
    assert sectioned_a.annotation_gt.to_dict() == sectioned_b.annotation_gt.to_dict()
    assert sectioned_a.trace_payload['execution_trace'] == sectioned_b.trace_payload['execution_trace']

def test_pages_infographic_metric_arithmetic_supports_larger_variable_sections() -> None:
    task = PagesInfographicSectionTotalExceptNamedValueTask()
    out = task.generate(98531, params={'card_count': 30, 'section_count': 6}, max_attempts=10)
    trace = out.trace_payload['execution_trace']
    render = out.trace_payload['render_spec']
    assert int(trace['card_count']) == 30
    assert int(trace['section_count']) == 6
    assert trace['section_card_counts'] == [5, 5, 5, 5, 5, 5]
    assert int(trace['target_operand_count']) >= 3
    assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
    for card in trace['cards']:
        _assert_bbox_inside_canvas(card['card_bbox_px'], width=int(render['canvas_width']), height=int(render['canvas_height']))

def test_pages_infographic_section_ranked_total_label_matches_contract() -> None:
    task = PagesInfographicSectionRankedTotalLabelTask()
    out = task.generate(99421, params={'card_count': 24, 'section_count': 4, 'rank_direction': 'highest', 'rank_position': 2}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['source_query_id'] == 'section_ranked_total_label'
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox_set'
    assert execution['question_format'] == 'label_open'
    assert execution['answer_type'] == 'string'
    assert execution['rank_direction'] == 'highest'
    assert int(execution['rank_position']) == 2
    ranked = sorted(((str(section), int(total)) for section, total in execution['section_totals'].items()), key=lambda item: item[1], reverse=True)
    expected_section = ranked[1][0]
    assert str(out.answer_gt.value) == expected_section
    assert str(execution['answer_value']) == expected_section
    assert str(trace['query_spec']['params']['target_answer']) == expected_section
    target_labels = [str(label) for label in execution['target_groups']['answer_section']]
    annotation_bboxes = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]
    expected_bboxes = [
        [float(value) for value in trace['projected_annotation']['card_bbox_map'][label]]
        for label in target_labels
    ]
    assert len(target_labels) == execution['section_card_counts'][execution['section_titles'].index(expected_section)]
    assert len(annotation_bboxes) == len(target_labels)
    assert annotation_bboxes == expected_bboxes
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert set(trace['projected_annotation']['bbox_map'].keys()) == set(target_labels)
    for bbox in annotation_bboxes:
        _assert_bbox_inside_canvas(bbox, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert isinstance(example['annotation'], list)
    assert all(
        not str(card.get('caption_text', '')).startswith('Ref ')
        for card in execution['cards']
    )

@pytest.mark.parametrize('query_id', SECTION_RANK_QUERY_IDS)
def test_pages_infographic_section_rank_label_variants_match_contract(query_id: str) -> None:
    task = _section_rank_task_for_query(query_id)
    out = task.generate(99731 + SECTION_RANK_QUERY_IDS.index(query_id), params={'card_count': 24, 'section_count': 4}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    assert out.query_id == SINGLE_QUERY_ID
    assert execution['source_query_id'] == query_id
    assert out.scene_id == 'infographic'
    if query_id == 'section_icon_extremum_label':
        assert execution['answer_type'] == 'string'
        assert execution['question_format'] == 'label_open'
        expected = _expected_answer(execution)
        assert str(out.answer_gt.value) == str(expected)
        assert str(trace['query_spec']['params']['target_answer']) == str(out.answer_gt.value)
        assert execution['comparison_icon_kind']
        assert execution['rank_direction'] in {'highest', 'lowest'}
        assert out.annotation_gt.type == 'bbox'
        assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
        assert trace['projected_annotation']['annotation_targets'] == [
            {'key': 'answer_section', 'section': expected, 'bbox_kind': 'section'}
        ]
        _assert_bbox_inside_canvas(
            out.annotation_gt.value,
            width=int(trace['render_spec']['canvas_width']),
            height=int(trace['render_spec']['canvas_height']),
        )
    else:
        assert execution['answer_type'] == 'string'
        assert execution['question_format'] == 'label_open'
        ranked = sorted(((str(section), int(total)) for section, total in execution['section_totals'].items()), key=lambda item: item[1], reverse=str(execution['rank_direction']) == 'highest')
        expected = ranked[int(execution['rank_position']) - 1][0]
        assert str(out.answer_gt.value) == expected
        assert str(trace['query_spec']['params']['target_answer']) == expected
        assert out.annotation_gt.type == 'bbox_set'
        assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value

@pytest.mark.parametrize('query_id', METRIC_RANKED_ITEM_QUERY_IDS)
def test_pages_infographic_metric_ranked_item_label_variants_match_contract(query_id: str) -> None:
    task = _metric_ranked_item_task_for_query(query_id)
    out = task.generate(100231 + METRIC_RANKED_ITEM_QUERY_IDS.index(query_id), params={'query_id': query_id, 'card_count': 24, 'section_count': 4, 'rank_position': 2}, max_attempts=10)
    trace = out.trace_payload
    execution = trace['execution_trace']
    render = trace['render_spec']
    projected = trace['projected_annotation']
    source_query_id = str(execution['source_query_id'])
    assert out.query_id == query_id
    assert source_query_id in set(METRIC_RANKED_ITEM_QUERY_IDS)
    assert out.scene_id == 'infographic'
    assert out.answer_gt.type == 'string'
    assert execution['answer_type'] == 'string'
    assert execution['question_format'] == 'label_open'
    expected_scope = 'section' if source_query_id.endswith('_in_section_label') else 'global'
    expected_direction = 'highest' if 'highest' in source_query_id else 'lowest'
    assert execution['rank_scope'] == expected_scope
    assert execution['rank_direction'] == expected_direction
    assert int(execution['rank_position']) == 2
    assert execution['ranked_candidates']
    ranked = [dict(candidate) for candidate in execution['ranked_candidates']]
    values = [int(candidate['value']) for candidate in ranked]
    assert len(values) == len(set(values))
    assert values == sorted(values, reverse=expected_direction == 'highest')
    expected_label = str(ranked[int(execution['rank_position']) - 1]['label'])
    assert str(out.answer_gt.value) == expected_label
    assert str(execution['answer_value']) == expected_label
    assert str(trace['query_spec']['params']['target_answer']) == expected_label
    if expected_scope == 'section':
        assert out.annotation_gt.type == 'bbox'
        target_section = str(execution['target_sections'][0])
        assert projected['bbox'] == out.annotation_gt.value
        assert out.annotation_gt.value == projected['card_bbox_map'][expected_label]
        assert projected['bbox_map']['section'] == trace['render_spec']['section_bboxes_px'][target_section]
        assert projected['bbox_map']['target_card'] == projected['card_bbox_map'][expected_label]
        assert all((str(candidate['section']) == target_section for candidate in ranked))
        assert list(projected['bbox_map'].keys()) == ['section', 'target_card']
        _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    else:
        assert out.annotation_gt.type == 'bbox'
        assert execution['target_sections'] == []
        assert projected['bbox'] == out.annotation_gt.value
        assert out.annotation_gt.value == projected['card_bbox_map'][expected_label]
        _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render['canvas_width']), height=int(render['canvas_height']))
    example = _extract_prompt_json_example(out.prompt)
    assert list(example.keys()) == ['annotation', 'answer']
    assert isinstance(example['answer'], str)
    assert isinstance(example['annotation'], list)

@pytest.mark.parametrize('instance_seed', [44, 61, 87])
def test_pages_infographic_metric_ranked_item_label_tie_breaking_is_retry_free(instance_seed: int) -> None:
    task = PagesInfographicGlobalMetricRankedItemLabelTask()
    out = task.generate(instance_seed, params={}, max_attempts=1)
    assert out.scene_id == 'infographic'
    assert out.query_id in set(GLOBAL_METRIC_RANKED_ITEM_QUERY_IDS)
    assert out.trace_payload['execution_trace']['source_query_id'] in set(GLOBAL_METRIC_RANKED_ITEM_QUERY_IDS)
    ranked = [dict(candidate) for candidate in out.trace_payload['execution_trace']['ranked_candidates']]
    values = [int(candidate['value']) for candidate in ranked]
    assert len(values) == len(set(values))

def test_pages_infographic_retired_public_modules_are_removed() -> None:
    retired_modules = [
        'trace_tasks.tasks.pages.infographic.value_for_named_item',
        'trace_tasks.tasks.pages.infographic.item_for_named_value',
        'trace_tasks.tasks.pages.infographic.detail_for_named_item',
        'trace_tasks.tasks.pages.infographic.metric_ranked_item_label',
        'trace_tasks.tasks.pages.infographic.metric_card_field_lookup',
    ]
    for module_name in retired_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
