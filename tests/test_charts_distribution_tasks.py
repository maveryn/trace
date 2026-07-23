"""Behavior tests for chart distribution tasks."""
from __future__ import annotations
import json
from trace_tasks.tasks.charts.boxplot.iqr_extremum_label import ChartsDistributionBoxplotIqrExtremumLabelTask
from trace_tasks.tasks.charts.boxplot.median_rank_difference_value import ChartsDistributionBoxplotMedianRankDifferenceValueTask
from trace_tasks.tasks.charts.boxplot.paired_median_shift_label import ChartsDistributionBoxplotPairedMedianShiftLabelTask
from trace_tasks.tasks.charts.histogram.cumulative_rank_bin_label import ChartsDistributionHistogramCumulativeRankLabelTask
from trace_tasks.tasks.charts.histogram.interval_mass import ChartsDistributionHistogramIntervalMassTask
from trace_tasks.tasks.charts.violin.modality_label import ChartsDistributionViolinModalityLabelTask
from trace_tasks.tasks.charts.violin.mode_extremum_label import ChartsDistributionViolinModeExtremumLabelTask
from trace_tasks.tasks.charts.violin.support_width_extremum_label import ChartsDistributionViolinSupportWidthExtremumLabelTask

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _assert_value_axis_covers_values(render: dict, values: list[int]) -> None:
    assert int(render['value_axis_min']) <= min((int(value) for value in values))
    assert max((int(value) for value in values)) <= int(render['value_axis_max'])
    assert int(render['value_axis_span']) == int(render['value_axis_max']) - int(render['value_axis_min'])
    assert set((int(value) for value in render['y_ticks'])).issubset(set((int(value) for value in render['value_axis_minor_ticks'])))

def test_chart_distribution_histogram_variants_match_contract() -> None:
    cases = (
        (ChartsDistributionHistogramIntervalMassTask, 'inside_interval_mass'),
        (ChartsDistributionHistogramIntervalMassTask, 'outside_interval_mass'),
        (ChartsDistributionHistogramCumulativeRankLabelTask, 'single'),
    )
    for seed, (task_cls, query_id) in enumerate(cases, start=11010):
        out = task_cls().generate(seed, params={'query_id': query_id}, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        labels = [str(label) for label in execution['labels']]
        counts = [int(value) for value in execution['bin_counts']]
        counts_by_label = {str(label): int(value) for label, value in execution['counts_by_label'].items()}
        intervals_by_label = {str(entity['attrs']['label']): (int(entity['attrs']['interval_start']), int(entity['attrs']['interval_end'])) for entity in trace['scene_ir']['entities']}
        annotation_labels = [str(label) for label in execution['annotation_labels']]
        axis_values = [int(label) for label in labels]
        assert str(out.query_id) == str(query_id)
        assert out.answer_gt.type == 'integer'
        if task_cls is ChartsDistributionHistogramCumulativeRankLabelTask:
            assert out.annotation_gt.type == 'bbox'
            annotation_bboxes = [list(out.annotation_gt.value)]
        else:
            assert out.annotation_gt.type == 'bbox_set'
            annotation_bboxes = [list(bbox) for bbox in out.annotation_gt.value]
        assert str(execution['scene_variant']) == 'histogram'
        assert str(render['scene_variant']) == 'histogram'
        assert render['information_scene_style']['kind'] == 'information_scene_style'
        assert render['information_scene_style']['style_request']['style_family'] == 'information_scene'
        assert render['information_scene_style']['style_request']['domain'] == 'charts'
        if out.annotation_gt.type == 'bbox':
            assert trace['projected_annotation']['bbox'] == annotation_bboxes[0]
        else:
            assert trace['projected_annotation']['bbox_set'] == annotation_bboxes
        projected_items = annotation_bboxes
        assert len(projected_items) == len(annotation_labels)
        assert len(annotation_bboxes) == len(annotation_labels)
        assert len(trace['scene_ir']['entities']) == int(execution['bin_count'])
        assert set((str(entity['attrs']['label']) for entity in trace['scene_ir']['entities'])) == set(labels)
        assert set(trace['render_map']['label_centers_px'].keys()) == set(labels)
        assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
        assert max(axis_values) <= 99
        assert all((int(entity['attrs']['interval_start']) == int(entity['attrs']['interval_end']) for entity in trace['scene_ir']['entities']))
        _assert_value_axis_covers_values(render, counts)
        assert int(render['value_axis_min']) == 0
        assert render['guide_line_style'] in {'dashed', 'dotted', 'solid'}
        assert len(render['guide_lines']) == int(execution['bin_count'])
        if task_cls is ChartsDistributionHistogramIntervalMassTask:
            query_start = int(execution['query_interval_start_value'])
            query_end = int(execution['query_interval_end_value'])
            assert str(execution['query_interval_label']) == f'{query_start}-{query_end}'
            inside_interval_labels = [str(label) for label in labels if int(intervals_by_label[str(label)][0]) >= query_start and int(intervals_by_label[str(label)][1]) <= query_end]
            outside_interval_labels = [str(label) for label in labels if str(label) not in set(inside_interval_labels)]
        if str(query_id) == 'inside_interval_mass':
            assert 5 <= len(annotation_labels) <= int(execution['bin_count'])
            assert annotation_labels == inside_interval_labels
            assert int(out.answer_gt.value) == sum((int(counts_by_label[label]) for label in annotation_labels))
        elif task_cls is ChartsDistributionHistogramCumulativeRankLabelTask:
            answer_index = int(execution['answer_bin_index'])
            target_rank = int(execution['target_rank'])
            assert annotation_labels == [labels[answer_index]]
            assert str(execution['answer_bin_label']) == str(labels[answer_index])
            assert int(out.answer_gt.value) == int(labels[answer_index])
            assert int(execution['cumulative_count_before_answer_bin']) < target_rank
            assert target_rank <= int(execution['cumulative_count_through_answer_bin'])
        else:
            assert str(query_id) == 'outside_interval_mass'
            assert str(execution['interval_relation']) == 'outside'
            assert 2 <= len(annotation_labels) <= int(execution['bin_count'])
            assert annotation_labels == outside_interval_labels
            assert int(out.answer_gt.value) == sum((int(counts_by_label[label]) for label in annotation_labels))
            assert int(execution['outside_bin_count']) == len(annotation_labels)
            assert int(execution['outside_left_bin_count']) >= 1
            assert int(execution['outside_right_bin_count']) >= 1
            assert int(execution['outside_left_bin_count']) + int(execution['outside_right_bin_count']) == len(annotation_labels)
            assert int(execution['excluded_interval_bin_span']) == len(inside_interval_labels)
            assert int(execution['excluded_interval_bin_span']) >= 2

def test_histogram_bins_are_contiguous_numeric_intervals() -> None:
    task = ChartsDistributionHistogramIntervalMassTask()
    out = task.generate(11030, params={'query_id': 'inside_interval_mass'}, max_attempts=10)
    entities = out.trace_payload['scene_ir']['entities']
    intervals = [(int(entity['attrs']['interval_start']), int(entity['attrs']['interval_end'])) for entity in entities]
    for index in range(len(intervals) - 1):
        assert int(intervals[index][1]) + 1 == int(intervals[index + 1][0])
    assert all((int(start) == int(end) for start, end in intervals))
    assert max((int(end) for _, end in intervals)) <= 99

def test_chart_distribution_histogram_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsDistributionHistogramIntervalMassTask, 'inside_interval_mass', {'annotation': [[210, 320, 246, 520], [252, 280, 288, 520]], 'answer': 17}),
        (ChartsDistributionHistogramCumulativeRankLabelTask, 'single', {'annotation': [336, 240, 372, 520], 'answer': 18}),
    )
    for index, (task_cls, query_id, expected) in enumerate(cases, start=11040):
        out = task_cls().generate(index, params={'query_id': query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected
        assert answer_only == {'answer': expected['answer']}

def test_chart_distribution_histogram_task_is_deterministic() -> None:
    task = ChartsDistributionHistogramIntervalMassTask()
    params = {'query_id': 'inside_interval_mass'}
    out_a = task.generate(11060, params=params, max_attempts=10)
    out_b = task.generate(11060, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload['execution_trace'] == out_b.trace_payload['execution_trace']
    assert out_a.trace_payload['query_spec']['prompt_variant'] == out_b.trace_payload['query_spec']['prompt_variant']
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()

def test_chart_distribution_histogram_interval_mass_sampler_varies_answers() -> None:
    task = ChartsDistributionHistogramIntervalMassTask()
    interval_mass_answers = []
    for sampling_index in range(42):
        out = task.generate(11070 + sampling_index, params={'query_id': 'inside_interval_mass'}, max_attempts=10)
        assert str(out.query_id) == 'inside_interval_mass'
        interval_mass_answers.append(int(out.answer_gt.value))
    assert min(interval_mass_answers) >= 0
    assert len(set(interval_mass_answers)) >= 8

def test_chart_distribution_histogram_cumulative_rank_public_task_contract() -> None:
    task = ChartsDistributionHistogramCumulativeRankLabelTask()
    out = task.generate(11080, params={}, max_attempts=10)
    execution = out.trace_payload['execution_trace']
    query_spec = out.trace_payload['query_spec']
    assert str(out.query_id) == 'single'
    assert str(execution['query_id']) == 'single'
    assert str(query_spec['params']['query_id']) == 'single'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox'

def test_chart_distribution_boxplot_variants_match_contract() -> None:
    cases = (
        (ChartsDistributionBoxplotIqrExtremumLabelTask, {'query_id': 'largest_iqr_label'}),
        (ChartsDistributionBoxplotIqrExtremumLabelTask, {'query_id': 'smallest_iqr_label'}),
    )
    for seed, (task_cls, extra_params) in enumerate(cases, start=11110):
        out = task_cls().generate(seed, params=extra_params, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        quartiles_by_label = {str(label): dict(values) for label, values in execution['quartiles_by_label'].items()}
        assert str(out.query_id) == str(extra_params['query_id'])
        assert out.answer_gt.type == 'string'
        assert str(execution['scene_variant']) == 'boxplot'
        assert str(render['scene_variant']) == 'boxplot'
        assert task_cls is ChartsDistributionBoxplotIqrExtremumLabelTask
        assert out.annotation_gt.type == 'bbox'
        assert len(out.annotation_gt.value) == 4
        assert trace['projected_annotation']['type'] == 'bbox'
        assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
        assert trace['projected_annotation']['pixel_bbox'] == out.annotation_gt.value
        assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
        all_box_values = [int(value) for stats in quartiles_by_label.values() for value in (stats['whisker_min'], stats['q1'], stats['median'], stats['q3'], stats['whisker_max'])]
        _assert_value_axis_covers_values(render, all_box_values)
        for stats in quartiles_by_label.values():
            assert int(stats['whisker_min']) <= int(stats['q1']) < int(stats['median']) < int(stats['q3']) <= int(stats['whisker_max'])
        if str(execution['extremum_direction']) == 'largest':
            target_label = max(quartiles_by_label, key=lambda label: int(quartiles_by_label[label]['iqr']))
            assert str(out.answer_gt.value) == str(target_label)
            assert int(execution['annotation_value']) == int(quartiles_by_label[target_label]['iqr'])
            assert out.annotation_gt.value == trace['render_map']['box_bboxes_px'][str(target_label)]
        else:
            assert task_cls is ChartsDistributionBoxplotIqrExtremumLabelTask
            assert str(execution['extremum_direction']) == 'smallest'
            target_label = min(quartiles_by_label, key=lambda label: int(quartiles_by_label[label]['iqr']))
            assert str(out.answer_gt.value) == str(target_label)
            assert int(execution['annotation_value']) == int(quartiles_by_label[target_label]['iqr'])
            assert out.annotation_gt.value == trace['render_map']['box_bboxes_px'][str(target_label)]

def test_chart_distribution_boxplot_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsDistributionBoxplotIqrExtremumLabelTask, {'annotation': [394, 210, 466, 290], 'answer': 'Ivory'}),
    )
    for index, (task_cls, expected) in enumerate(cases, start=11140):
        out = task_cls().generate(index, params={}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected
        assert answer_only == {'answer': expected['answer']}

def test_chart_distribution_boxplot_task_is_deterministic() -> None:
    task = ChartsDistributionBoxplotIqrExtremumLabelTask()
    params = {'query_id': 'largest_iqr_label'}
    out_a = task.generate(11160, params=params, max_attempts=10)
    out_b = task.generate(11160, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload['execution_trace'] == out_b.trace_payload['execution_trace']
    assert out_a.trace_payload['query_spec']['prompt_variant'] == out_b.trace_payload['query_spec']['prompt_variant']
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()

def test_chart_distribution_boxplot_can_tighten_iqr_winner_gap() -> None:
    task = ChartsDistributionBoxplotIqrExtremumLabelTask()
    for seed, (query_id, extremum_direction) in enumerate((('largest_iqr_label', 'largest'), ('smallest_iqr_label', 'smallest')), start=11163):
        out = task.generate(seed, params={'query_id': query_id, 'category_count_min': 7, 'category_count_max': 7, 'iqr_winner_gap_min': 1, 'iqr_winner_gap_max': 1}, max_attempts=10)
        quartiles_by_label = out.trace_payload['execution_trace']['quartiles_by_label']
        iqrs = sorted((int(stats['iqr']) for stats in quartiles_by_label.values()))
        if extremum_direction == 'largest':
            assert iqrs[-1] - iqrs[-2] == 1
        else:
            assert iqrs[1] - iqrs[0] == 1

def test_chart_distribution_boxplot_uses_configured_iqr_winner_gap() -> None:
    task = ChartsDistributionBoxplotIqrExtremumLabelTask()
    for seed, (query_id, extremum_direction) in enumerate((('largest_iqr_label', 'largest'), ('smallest_iqr_label', 'smallest')), start=11167):
        out = task.generate(seed, params={'query_id': query_id, 'category_count_min': 7, 'category_count_max': 7}, max_attempts=10)
        quartiles_by_label = out.trace_payload['execution_trace']['quartiles_by_label']
        iqrs = sorted((int(stats['iqr']) for stats in quartiles_by_label.values()))
        if extremum_direction == 'largest':
            assert iqrs[-1] - iqrs[-2] == 1
        else:
            assert iqrs[1] - iqrs[0] == 1

def test_chart_distribution_boxplot_public_role_bound_tasks_use_keyed_annotation() -> None:
    iqr = ChartsDistributionBoxplotIqrExtremumLabelTask().generate(11179, params={'query_id': 'largest_iqr_label'}, max_attempts=10)
    assert iqr.answer_gt.type == 'string'
    assert iqr.annotation_gt.type == 'bbox'
    assert len(iqr.annotation_gt.value) == 4
    assert iqr.trace_payload['projected_annotation']['type'] == 'bbox'
    assert iqr.trace_payload['projected_annotation']['bbox'] == iqr.annotation_gt.value
    median_rank = ChartsDistributionBoxplotMedianRankDifferenceValueTask().generate(11180, params={'query_id': 'median_top_second_difference_value'}, max_attempts=10)
    assert median_rank.answer_gt.type == 'integer'
    assert median_rank.annotation_gt.type == 'point_map'
    assert set(median_rank.annotation_gt.value) == {'highest_median_boxplot', 'second_highest_median_boxplot'}
    assert median_rank.trace_payload['projected_annotation']['type'] == 'point_map'
    assert median_rank.trace_payload['projected_annotation']['point_map'] == median_rank.annotation_gt.value
    paired_shift = ChartsDistributionBoxplotPairedMedianShiftLabelTask().generate(11181, params={'query_id': 'paired_median_greatest_increase_label'}, max_attempts=10)
    assert paired_shift.answer_gt.type == 'string'
    assert paired_shift.annotation_gt.type == 'point_map'
    assert set(paired_shift.annotation_gt.value) == {'before_boxplot', 'after_boxplot'}
    assert paired_shift.trace_payload['projected_annotation']['type'] == 'point_map'
    assert paired_shift.trace_payload['projected_annotation']['point_map'] == paired_shift.annotation_gt.value

def test_chart_distribution_violin_variants_match_contract() -> None:
    cases = (
        (ChartsDistributionViolinModeExtremumLabelTask, 'highest_mode'),
        (ChartsDistributionViolinModeExtremumLabelTask, 'lowest_mode'),
        (ChartsDistributionViolinModalityLabelTask, 'single'),
        (ChartsDistributionViolinSupportWidthExtremumLabelTask, 'widest_support'),
        (ChartsDistributionViolinSupportWidthExtremumLabelTask, 'narrowest_support'),
    )
    for seed, (task_cls, query_id) in enumerate(cases, start=11210):
        out = task_cls().generate(seed, params={'query_id': query_id}, max_attempts=10)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render = trace['render_spec']
        support_by_label = {str(label): dict(values) for label, values in execution['support_by_label'].items()}
        assert str(out.query_id) == str(query_id)
        assert out.answer_gt.type == 'string'
        assert out.annotation_gt.type == 'bbox'
        assert len(out.annotation_gt.value) == 4
        assert str(execution['scene_variant']) == 'violin'
        assert str(render['scene_variant']) == 'violin'
        assert str(render['violin_style']['mode_line_style']) in {'full', 'short', 'dot', 'none'}
        assert str(render['violin_style']['fill_style']) in {'solid', 'light', 'outline', 'hatch'}
        assert str(render['violin_style']['palette_mode']) in {'single', 'per_violin_muted'}
        assert str(render['font_assets']['chart_font_family']).strip()
        assert str(trace['scene_ir']['scene_kind']) == 'chart_violin_distribution'
        assert trace['projected_annotation']['type'] == 'bbox'
        assert trace['projected_annotation']['bbox'] == out.annotation_gt.value
        assert trace['projected_annotation']['pixel_bbox'] == out.annotation_gt.value
        assert out.image.size == (int(render['canvas_width']), int(render['canvas_height']))
        assert len(trace['scene_ir']['entities']) == int(execution['category_count'])
        assert set(trace['render_map']['label_centers_px'].keys()) == set(execution['labels'])
        if str(query_id) == 'highest_mode':
            expected = max(support_by_label, key=lambda label: int(support_by_label[label]['mode_values'][0]))
        elif str(query_id) == 'lowest_mode':
            expected = min(support_by_label, key=lambda label: int(support_by_label[label]['mode_values'][0]))
        elif str(query_id) == 'single':
            assert str(execution['prompt_query_key']) == 'bimodal_label'
            expected = next((label for label, values in support_by_label.items() if bool(values['bimodal'])))
            assert len(execution['annotation_values']) == 2
        elif str(query_id) == 'widest_support':
            expected = max(support_by_label, key=lambda label: int(support_by_label[label]['support_span']))
        else:
            expected = min(support_by_label, key=lambda label: int(support_by_label[label]['support_span']))
        assert str(out.answer_gt.value) == str(expected)

def test_chart_distribution_violin_public_task_contract() -> None:
    cases = ((ChartsDistributionViolinModeExtremumLabelTask, {'highest_mode', 'lowest_mode'}), (ChartsDistributionViolinSupportWidthExtremumLabelTask, {'widest_support', 'narrowest_support'}), (ChartsDistributionViolinModalityLabelTask, {'single'}))
    for task_cls, allowed_query_ids in cases:
        out = task_cls().generate(11280 + len(task_cls.task_id), params={}, max_attempts=10)
        execution = out.trace_payload['execution_trace']
        query_spec = out.trace_payload['query_spec']
        assert str(out.query_id) in allowed_query_ids
        assert str(execution['query_id']) == str(out.query_id)
        assert str(query_spec['params']['query_id']) == str(out.query_id)
        assert out.answer_gt.type == 'string'
        assert out.annotation_gt.type == 'bbox'
        assert out.trace_payload['projected_annotation']['bbox'] == out.annotation_gt.value
