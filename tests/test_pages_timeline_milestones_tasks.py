"""Behavior tests for the milestone-timeline task."""
from __future__ import annotations
from collections import Counter, defaultdict
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.shared.text_rendering import resolve_text_stroke_fill
from trace_tasks.tasks.shared.time_artifact_style import SUPPORTED_TIME_ARTIFACT_COLOR_NAMES, SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS
from trace_tasks.tasks.pages.timeline.date_threshold_event_count import PagesTimelineDateThresholdEventCountTask
from trace_tasks.tasks.pages.timeline.interval_membership_count import PagesTimelineIntervalMembershipCountTask
from trace_tasks.tasks.pages.timeline.relative_position_event_label import PagesTimelineRelativePositionEventLabelTask
from tests.helpers import extract_prompt_json_example

def test_pages_timeline_milestones_contract_matches_trace() -> None:
    task = PagesTimelineIntervalMembershipCountTask()
    query_ids = ('between_reference_events_count', 'outside_reference_interval_count')
    scene_variants = ('classic', 'roadmap')
    style_variants = ('studio', 'marker')
    accent_colors = ('blue', 'orange')
    for index, query_id in enumerate(query_ids):
        out = task.generate(
            22624 + index,
            params={
                'query_id': query_id,
                'scene_variant': scene_variants[index],
                'style_variant': style_variants[index],
                'accent_color_name': accent_colors[index],
            },
            max_attempts=20,
        )
        trace = out.trace_payload
        execution = trace['execution_trace']
        render_map = trace['render_map']
        expected_relation = 'between' if query_id.startswith('between') else 'outside'
        assert out.query_id == query_id
        assert str(execution['query_id']) == query_id
        assert str(execution['source_query_id']) == query_id
        assert str(execution['interval_relation']) == expected_relation
        assert out.answer_gt.type == 'integer'
        assert out.annotation_gt.type == 'bbox_set'
        assert int(out.answer_gt.value) == len(execution['answer_event_ids'])
        expected_boxes = [
            render_map['event_bboxes_by_id'][str(event_id)]
            for event_id in execution['answer_event_ids']
        ]
        assert out.annotation_gt.value == expected_boxes
        assert trace['projected_annotation']['bbox_set'] == expected_boxes

def test_pages_timeline_date_threshold_event_count_contract_matches_trace() -> None:
    task = PagesTimelineDateThresholdEventCountTask()
    out = task.generate(
        22632,
        params={
            'query_id': 'after_threshold_date_count',
            'scene_variant': 'minimal',
            'style_variant': 'accented',
            'accent_color_name': 'green',
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace['execution_trace']
    render_map = trace['render_map']
    threshold_day = int(execution['threshold_day'])
    assert out.query_id == 'after_threshold_date_count'
    assert str(execution['query_id']) == 'after_threshold_date_count'
    assert str(execution['source_query_id']) == 'after_threshold_date_count'
    assert str(execution['interval_relation']) == 'after'
    assert execution['reference_event_ids'] == []
    assert execution['endpoint_event_ids'] == []
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'bbox_set'
    expected_event_ids = [
        str(event['event_id'])
        for event in execution['events']
        if int(event['day_of_month']) > threshold_day
    ]
    assert int(out.answer_gt.value) == len(expected_event_ids)
    assert expected_event_ids == execution['answer_event_ids']
    expected_boxes = [
        render_map['event_bboxes_by_id'][str(event_id)]
        for event_id in expected_event_ids
    ]
    assert out.annotation_gt.value == expected_boxes
    assert trace['projected_annotation']['bbox_set'] == expected_boxes

def test_pages_timeline_relative_position_event_label_contract_matches_trace() -> None:
    task = PagesTimelineRelativePositionEventLabelTask()
    out = task.generate(
        22636,
        params={
            'query_id': 'event_after_dated_event_label',
            'relative_offset': 4,
            'scene_variant': 'classic',
            'style_variant': 'studio',
            'accent_color_name': 'blue',
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace['execution_trace']
    render_map = trace['render_map']
    assert out.query_id == 'event_after_dated_event_label'
    assert str(execution['query_id']) == 'event_after_dated_event_label'
    assert str(execution['source_query_id']) == 'event_after_dated_event_label'
    assert str(execution['interval_relation']) == 'after'
    assert int(execution['relative_offset']) == 4
    assert len(execution['prompt_reference_event_ids']) == 1
    assert len(execution['answer_event_ids']) == 1
    reference_id = str(execution['prompt_reference_event_ids'][0])
    target_id = str(execution['answer_event_ids'][0])
    events_by_id = {str(event['event_id']): event for event in execution['events']}
    assert int(events_by_id[target_id]['order_index']) == int(events_by_id[reference_id]['order_index']) + 4
    assert str(events_by_id[reference_id]['date_text']) == str(execution['reference_date_text'])
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'bbox'
    assert str(out.answer_gt.value) == str(events_by_id[target_id]['label'])
    expected_box = render_map['event_bboxes_by_id'][target_id]
    assert out.annotation_gt.value == expected_box
    assert trace['projected_annotation']['bbox'] == expected_box

def test_pages_timeline_milestones_prompt_examples_match_variants() -> None:
    task = PagesTimelineIntervalMembershipCountTask()
    expected = {'between_reference_events_count': ({'annotation': [[412, 410, 516, 480], [538, 158, 642, 228]], 'answer': 2}, {'answer': 2}), 'outside_reference_interval_count': ({'annotation': [[160, 158, 264, 228], [790, 410, 894, 480]], 'answer': 2}, {'answer': 2})}
    for index, (query_id, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected.items(), start=22640):
        out = task.generate(index, params={'query_id': query_id}, max_attempts=20)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only

def test_pages_timeline_date_threshold_event_count_prompt_examples_match_variants() -> None:
    task = PagesTimelineDateThresholdEventCountTask()
    expected = {
        'before_threshold_date_count': {'annotation': [[160, 158, 264, 228], [286, 410, 390, 480]], 'answer': 2},
        'after_threshold_date_count': {'annotation': [[664, 410, 768, 480], [790, 158, 894, 228]], 'answer': 2},
    }
    for index, (query_id, expected_answer_and_annotation) in enumerate(expected.items(), start=22643):
        out = task.generate(index, params={'query_id': query_id}, max_attempts=20)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == {'answer': 2}

def test_pages_timeline_relative_position_event_label_prompt_examples_match_variants() -> None:
    task = PagesTimelineRelativePositionEventLabelTask()
    expected = {
        'event_before_dated_event_label': ({'annotation': [286, 410, 390, 480], 'answer': 'B'}, {'answer': 'B'}),
        'event_after_dated_event_label': ({'annotation': [664, 410, 768, 480], 'answer': 'E'}, {'answer': 'E'}),
    }
    for index, (query_id, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected.items(), start=22646):
        out = task.generate(index, params={'query_id': query_id}, max_attempts=20)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only

def test_pages_timeline_milestones_balanced_sampling_defaults_cover_axes() -> None:
    task = PagesTimelineIntervalMembershipCountTask()
    query_ids: Counter[str] = Counter()
    interval_relations: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    event_counts: Counter[int] = Counter()
    scenes_by_query_id: defaultdict[str, Counter[str]] = defaultdict(Counter)
    styles_by_query_id: defaultdict[str, Counter[str]] = defaultdict(Counter)
    answers_by_query_id: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for index in range(90):
        out = task.generate(hash64(22680, 'pages_timeline_interval_membership_count', index), params={}, max_attempts=20)
        execution = out.trace_payload['execution_trace']
        query_id = str(execution['source_query_id'])
        interval_relation = str(execution['interval_relation'])
        scene_variant = str(execution['scene_variant'])
        style_variant = str(execution['style_variant'])
        query_ids[query_id] += 1
        interval_relations[interval_relation] += 1
        scene_variants[scene_variant] += 1
        style_variants[style_variant] += 1
        accent_color_names[str(execution['accent_color_name'])] += 1
        event_counts[int(execution['event_count'])] += 1
        scenes_by_query_id[query_id][scene_variant] += 1
        styles_by_query_id[query_id][style_variant] += 1
        answers_by_query_id[query_id][int(out.answer_gt.value)] += 1
    assert set(query_ids.keys()) == {'between_reference_events_count', 'outside_reference_interval_count'}
    assert set(interval_relations.keys()) == {'between', 'outside'}
    assert set(scene_variants.keys()) == {'classic', 'roadmap', 'minimal'}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_COLOR_NAMES)
    assert set(event_counts.keys()).issubset({6, 7, 8, 9, 10, 11, 12})
    assert max(event_counts) >= 10
    for query_id in query_ids:
        assert set(scenes_by_query_id[query_id].keys()) == {'classic', 'roadmap', 'minimal'}
        assert set(styles_by_query_id[query_id].keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
        assert len(answers_by_query_id[query_id]) >= 3

def test_pages_timeline_date_threshold_event_count_balanced_sampling_covers_answers_and_queries() -> None:
    task = PagesTimelineDateThresholdEventCountTask()
    answers: Counter[int] = Counter()
    query_ids: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(hash64(22690, 'pages_timeline_date_threshold_event_count', index), params={}, max_attempts=20)
        execution = out.trace_payload['execution_trace']
        answers[int(out.answer_gt.value)] += 1
        query_ids[str(execution['source_query_id'])] += 1
        scene_variants[str(execution['scene_variant'])] += 1
        style_variants[str(execution['style_variant'])] += 1
        accent_color_names[str(execution['accent_color_name'])] += 1
    assert len(answers) >= 5
    assert set(query_ids.keys()) == {'before_threshold_date_count', 'after_threshold_date_count'}
    assert set(scene_variants.keys()) == {'classic', 'roadmap', 'minimal'}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_COLOR_NAMES)

def test_pages_timeline_relative_position_event_label_balanced_sampling_covers_offsets_and_queries() -> None:
    task = PagesTimelineRelativePositionEventLabelTask()
    offsets: Counter[int] = Counter()
    query_ids: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(hash64(22700, 'pages_timeline_relative_position_event_label', index), params={}, max_attempts=20)
        execution = out.trace_payload['execution_trace']
        offsets[int(execution['relative_offset'])] += 1
        query_ids[str(execution['source_query_id'])] += 1
        answers[str(out.answer_gt.value)] += 1
        scene_variants[str(execution['scene_variant'])] += 1
        style_variants[str(execution['style_variant'])] += 1
        accent_color_names[str(execution['accent_color_name'])] += 1
    assert set(offsets.keys()) == {1, 2, 3, 4}
    assert set(query_ids.keys()) == {'event_before_dated_event_label', 'event_after_dated_event_label'}
    assert len(answers) >= 6
    assert set(scene_variants.keys()) == {'classic', 'roadmap', 'minimal'}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_COLOR_NAMES)

def test_resolve_text_stroke_fill_tracks_text_luminance() -> None:
    assert resolve_text_stroke_fill((255, 255, 255)) == (36, 42, 52)
    assert resolve_text_stroke_fill((44, 52, 64)) == (255, 255, 255)
