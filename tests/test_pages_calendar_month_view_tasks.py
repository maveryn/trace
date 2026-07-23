"""Behavior tests for the month-view calendar task."""
from __future__ import annotations
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.calendar.date_range_day_class_count import PagesCalendarDateRangeDayClassCountTask
from trace_tasks.tasks.pages.calendar.date_weekday_label import PagesCalendarDateWeekdayLabelTask
from trace_tasks.tasks.pages.calendar.marked_day_class_count import PagesCalendarMarkedDayClassCountTask
from trace_tasks.tasks.pages.calendar.weekday_occurrence_date import PagesCalendarWeekdayOccurrenceDateTask
from trace_tasks.tasks.pages.calendar.workday_offset_date import PagesCalendarWorkdayOffsetDateTask
from trace_tasks.tasks.pages.calendar.shared.defaults import SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES, SUPPORTED_PAGE_CALENDAR_TITLE_MODES, SUPPORTED_PAGE_CALENDAR_WEEK_STARTS
from trace_tasks.tasks.pages.calendar.shared.rendering import calendar_theme_from_information_style
from trace_tasks.tasks.pages.shared.information_style import resolve_pages_information_style
from trace_tasks.tasks.shared.time_format import weekday_abbreviation
from trace_tasks.tasks.shared.text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO, contrast_ratio
from trace_tasks.tasks.shared.visual_style.information_scene import INFORMATION_SCENE_TREATMENT_IDS
from tests.helpers import extract_prompt_json_example

def _weekday_target_date(*, reference_date: int, offset: int, direction: str, days_in_month: int, start_weekday_index: int, weekend_weekday_indices: list[int]) -> int:
    weekend_indices = {int(value) for value in weekend_weekday_indices}
    workdays = [int(day) for day in range(1, int(days_in_month) + 1) if (int(start_weekday_index) + int(day) - 1) % 7 not in weekend_indices]
    reference_index = workdays.index(int(reference_date))
    target_index = int(reference_index + int(offset)) if str(direction) == 'after' else int(reference_index - int(offset))
    return int(workdays[int(target_index)])

def _range_day_class_dates(*, start_date: int, end_date: int, day_class: str, start_weekday_index: int, weekend_weekday_indices: list[int]) -> list[int]:
    weekend_indices = {int(value) for value in weekend_weekday_indices}
    target_weekend = str(day_class) == 'weekend'
    return [
        int(day)
        for day in range(int(start_date), int(end_date) + 1)
        if (((int(start_weekday_index) + int(day) - 1) % 7) in weekend_indices) == bool(target_weekend)
    ]

def test_pages_calendar_month_view_contract_matches_trace() -> None:
    weekday_out = PagesCalendarWeekdayOccurrenceDateTask().generate(21820, params={'scene_variant': 'classic'}, max_attempts=20)
    weekday_execution = weekday_out.trace_payload['execution_trace']
    weekday_render_map = weekday_out.trace_payload['render_map']
    assert weekday_out.query_id == 'single'
    assert str(weekday_execution['query_id']) == 'single'
    assert str(weekday_execution['prompt_query_key']) == 'weekday_occurrence_date'
    assert weekday_out.annotation_gt.type == 'bbox'
    assert weekday_out.annotation_gt.value == weekday_render_map['date_cells_by_day'][str(weekday_execution['answer_value'])]

    for index, marked_day_class in enumerate(('weekend', 'weekday'), start=1):
        count_out = PagesCalendarMarkedDayClassCountTask().generate(21820 + index, params={'marked_day_class': marked_day_class, 'scene_variant': 'outline'}, max_attempts=20)
        count_execution = count_out.trace_payload['execution_trace']
        count_render_map = count_out.trace_payload['render_map']
        expected_boxes = [count_render_map['date_cells_by_day'][str(day)] for day in count_execution['annotation_dates']]
        assert count_out.query_id == 'single'
        assert str(count_execution['query_id']) == 'single'
        assert str(count_execution['prompt_query_key']) == 'marked_day_class_count'
        assert str(count_execution['marked_day_class']) == marked_day_class
        assert count_out.answer_gt.type == 'integer'
        assert count_out.annotation_gt.type == 'bbox_set'
        assert int(count_out.answer_gt.value) == len(expected_boxes)
        assert count_out.annotation_gt.value == expected_boxes

def test_pages_calendar_week_start_changes_display_order_without_changing_semantics() -> None:
    task = PagesCalendarWeekdayOccurrenceDateTask()
    common_params = {'year': 2023, 'month': 1, 'scene_variant': 'classic', 'layout_mode': 'center_clean', 'title_mode': 'none'}
    monday_out = task.generate(21827, params={**common_params, 'week_start': 'monday'}, max_attempts=20)
    sunday_out = task.generate(21827, params={**common_params, 'week_start': 'sunday'}, max_attempts=20)

    monday_execution = monday_out.trace_payload['execution_trace']
    sunday_execution = sunday_out.trace_payload['execution_trace']
    assert str(monday_execution['week_start']) == 'monday'
    assert int(monday_execution['first_weekday_index']) == 0
    assert str(sunday_execution['week_start']) == 'sunday'
    assert int(sunday_execution['first_weekday_index']) == 6
    assert int(monday_execution['start_weekday_index']) == int(sunday_execution['start_weekday_index']) == 6
    assert int(monday_execution['days_in_month']) == int(sunday_execution['days_in_month']) == 31

    monday_date_one = next(entity for entity in monday_out.trace_payload['scene_ir']['entities'] if str(entity['entity_id']) == 'date_1')
    sunday_date_one = next(entity for entity in sunday_out.trace_payload['scene_ir']['entities'] if str(entity['entity_id']) == 'date_1')
    assert int(monday_date_one['attrs']['weekday_index']) == 6
    assert int(monday_date_one['attrs']['display_weekday_index']) == 6
    assert int(sunday_date_one['attrs']['weekday_index']) == 6
    assert int(sunday_date_one['attrs']['display_weekday_index']) == 0

def test_pages_calendar_date_weekday_label_contract_matches_visible_header_format() -> None:
    task = PagesCalendarDateWeekdayLabelTask()
    for index, week_start in enumerate(('monday', 'sunday')):
        out = task.generate(
            21830 + index,
            params={
                'year': 2024,
                'month': 2,
                'target_date': 17,
                'week_start': week_start,
                'scene_variant': 'classic',
                'layout_mode': 'center_clean',
                'title_mode': 'full_month_year',
            },
            max_attempts=20,
        )
        execution = out.trace_payload['execution_trace']
        render_map = out.trace_payload['render_map']
        expected_label = weekday_abbreviation(int(execution['answer_weekday_index']))
        assert out.query_id == 'single'
        assert str(execution['prompt_query_key']) == 'date_weekday_label'
        assert out.answer_gt.type == 'string'
        assert out.answer_gt.value == expected_label
        assert execution['answer_value'] == expected_label
        assert execution['answer_weekday_header_label'] == expected_label
        assert execution['valid_answer_labels'] == ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        assert '"answer"' in out.prompt
        assert 'Mon' in out.prompt and 'Sun' in out.prompt
        assert out.annotation_gt.type == 'bbox'
        assert out.annotation_gt.value == render_map['date_cells_by_day'][str(execution['target_date'])]
        assert out.trace_payload['projected_annotation']['bbox'] == out.annotation_gt.value

def test_pages_calendar_workday_offset_date_contract_matches_trace() -> None:
    task = PagesCalendarWorkdayOffsetDateTask()
    for index, query_id in enumerate(('workday_after_offset_date', 'workday_before_offset_date')):
        out = task.generate(21832 + index, params={'query_id': query_id, 'scene_variant': 'outline', 'layout_mode': 'center_clean', 'title_mode': 'full_month_year'}, max_attempts=20)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render_map = trace['render_map']
        reference_date = int(execution['reference_date'])
        target_date = int(execution['target_date'])
        expected_target = _weekday_target_date(reference_date=reference_date, offset=int(execution['workday_offset']), direction=str(execution['workday_direction']), days_in_month=int(execution['days_in_month']), start_weekday_index=int(execution['start_weekday_index']), weekend_weekday_indices=[int(value) for value in execution['weekend_weekday_indices']])
        assert out.query_id == str(query_id)
        assert str(execution['query_id']) == str(query_id)
        assert str(execution['prompt_query_key']) == str(query_id)
        assert out.answer_gt.type == 'integer'
        assert out.annotation_gt.type == 'bbox_map'
        assert int(out.answer_gt.value) == int(target_date) == int(expected_target)
        assert execution['marked_dates'] == [reference_date]
        assert execution['annotation_dates'] == [reference_date, target_date]
        assert set(out.annotation_gt.value.keys()) == {'reference_date', 'target_date'}
        assert out.annotation_gt.value['reference_date'] == render_map['date_cells_by_day'][str(reference_date)]
        assert out.annotation_gt.value['target_date'] == render_map['date_cells_by_day'][str(target_date)]
        assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value

def test_pages_calendar_date_range_day_class_count_contract_matches_trace() -> None:
    task = PagesCalendarDateRangeDayClassCountTask()
    for index, query_id in enumerate(('weekday_range_count', 'weekend_range_count')):
        out = task.generate(21836 + index, params={'query_id': query_id, 'scene_variant': 'classic', 'layout_mode': 'center_clean', 'title_mode': 'full_month_year'}, max_attempts=20)
        trace = out.trace_payload
        execution = trace['execution_trace']
        render_map = trace['render_map']
        range_start = int(execution['range_start_date'])
        range_end = int(execution['range_end_date'])
        day_class = str(execution['range_day_class'])
        expected_dates = _range_day_class_dates(
            start_date=range_start,
            end_date=range_end,
            day_class=day_class,
            start_weekday_index=int(execution['start_weekday_index']),
            weekend_weekday_indices=[int(value) for value in execution['weekend_weekday_indices']],
        )
        expected_boxes = [render_map['date_cells_by_day'][str(day)] for day in expected_dates]
        assert out.query_id == str(query_id)
        assert str(execution['query_id']) == str(query_id)
        assert str(execution['prompt_query_key']) == str(query_id)
        assert out.answer_gt.type == 'integer'
        assert out.annotation_gt.type == 'bbox_set'
        assert int(out.answer_gt.value) == len(expected_dates)
        assert execution['marked_dates'] == [range_start, range_end]
        assert execution['annotation_dates'] == expected_dates
        assert execution['range_counted_dates'] == expected_dates
        assert out.annotation_gt.value == expected_boxes
        assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value

def test_pages_calendar_month_view_prompt_examples_match_variants() -> None:
    expected = ((PagesCalendarWeekdayOccurrenceDateTask(), {}, ({'annotation': [354, 256, 456, 352], 'answer': 18}, {'answer': 18})), (PagesCalendarDateWeekdayLabelTask(), {}, ({'annotation': [354, 256, 456, 352], 'answer': 'Wed'}, {'answer': 'Wed'})), (PagesCalendarMarkedDayClassCountTask(), {'marked_day_class': 'weekend'}, ({'annotation': [[148, 352, 250, 448], [560, 352, 662, 448]], 'answer': 2}, {'answer': 2})), (PagesCalendarMarkedDayClassCountTask(), {'marked_day_class': 'weekday'}, ({'annotation': [[148, 352, 250, 448], [560, 352, 662, 448]], 'answer': 2}, {'answer': 2})), (PagesCalendarWorkdayOffsetDateTask(), {'query_id': 'workday_after_offset_date'}, ({'annotation': {'reference_date': [354, 256, 456, 352], 'target_date': [560, 256, 662, 352]}, 'answer': 20}, {'answer': 20})), (PagesCalendarWorkdayOffsetDateTask(), {'query_id': 'workday_before_offset_date'}, ({'annotation': {'reference_date': [560, 256, 662, 352], 'target_date': [354, 256, 456, 352]}, 'answer': 18}, {'answer': 18})), (PagesCalendarDateRangeDayClassCountTask(), {'query_id': 'weekday_range_count'}, ({'annotation': [[148, 256, 250, 352], [251, 256, 353, 352], [354, 256, 456, 352]], 'answer': 3}, {'answer': 3})), (PagesCalendarDateRangeDayClassCountTask(), {'query_id': 'weekend_range_count'}, ({'annotation': [[560, 256, 662, 352], [663, 256, 765, 352]], 'answer': 2}, {'answer': 2})))
    for index, (task, params, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected, start=21840):
        out = task.generate(index, params=params, max_attempts=20)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only

def test_pages_calendar_workday_offset_date_balanced_sampling_defaults_cover_query_axes() -> None:
    task = PagesCalendarWorkdayOffsetDateTask()
    query_ids: Counter[str] = Counter()
    offsets: Counter[int] = Counter()
    directions: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    layout_modes: Counter[str] = Counter()
    week_starts: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(hash64(21870, task.task_id, index), params={}, max_attempts=20)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        offsets[int(execution['workday_offset'])] += 1
        directions[str(execution['workday_direction'])] += 1
        scene_variants[str(execution['scene_variant'])] += 1
        layout_modes[str(execution['layout_mode'])] += 1
        week_starts[str(execution['week_start'])] += 1
    assert set(query_ids.keys()) == {'workday_after_offset_date', 'workday_before_offset_date'}
    assert set(directions.keys()) == {'after', 'before'}
    assert set(offsets.keys()) == {2, 3, 4, 5, 6, 7}
    assert set(scene_variants.keys()) == {'classic', 'minimal', 'outline'}
    assert set(layout_modes.keys()) == set(SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES)
    assert set(week_starts.keys()) == set(SUPPORTED_PAGE_CALENDAR_WEEK_STARTS)

def test_pages_calendar_date_range_day_class_count_sampling_covers_query_axes() -> None:
    task = PagesCalendarDateRangeDayClassCountTask()
    query_ids: Counter[str] = Counter()
    range_day_classes: Counter[str] = Counter()
    answer_values: Counter[int] = Counter()
    for index in range(120):
        out = task.generate(hash64(21876, task.task_id, index), params={}, max_attempts=20)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        range_day_classes[str(execution['range_day_class'])] += 1
        answer_values[int(execution['answer_value'])] += 1
        assert int(execution['range_start_date']) < int(execution['range_end_date'])
        assert execution['marked_dates'] == [int(execution['range_start_date']), int(execution['range_end_date'])]
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert set(query_ids.keys()) == {'weekday_range_count', 'weekend_range_count'}
    assert set(range_day_classes.keys()) == {'weekday', 'weekend'}
    assert set(answer_values.keys()).issuperset({1, 2, 3, 4})

def test_pages_calendar_month_view_balanced_sampling_defaults_cover_axes() -> None:
    tasks = (PagesCalendarWeekdayOccurrenceDateTask(), PagesCalendarDateWeekdayLabelTask(), PagesCalendarMarkedDayClassCountTask(), PagesCalendarDateRangeDayClassCountTask())
    query_ids: Counter[str] = Counter()
    marked_day_classes: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    information_scene_treatments: Counter[str] = Counter()
    layout_modes: Counter[str] = Counter()
    title_modes: Counter[str] = Counter()
    week_starts: Counter[str] = Counter()
    row_counts: Counter[int] = Counter()
    for task in tasks:
        for index in range(90):
            out = task.generate(hash64(21880, task.task_id, index), params={}, max_attempts=20)
            execution = out.trace_payload['execution_trace']
            query_ids[str(execution['query_id'])] += 1
            if execution['marked_day_class'] is not None:
                marked_day_classes[str(execution['marked_day_class'])] += 1
            scene_variants[str(execution['scene_variant'])] += 1
            information_scene_treatments[str(execution['information_scene_treatment'])] += 1
            layout_modes[str(execution['layout_mode'])] += 1
            title_modes[str(execution['title_mode'])] += 1
            week_starts[str(execution['week_start'])] += 1
            row_counts[int(execution['row_count'])] += 1
    assert set(query_ids.keys()) == {'single', 'weekday_range_count', 'weekend_range_count'}
    assert set(marked_day_classes.keys()) == {'weekend', 'weekday'}
    assert set(scene_variants.keys()) == {'classic', 'minimal', 'outline'}
    assert set(information_scene_treatments.keys()).issubset(set(INFORMATION_SCENE_TREATMENT_IDS))
    assert any(str(value).startswith('dark_') for value in information_scene_treatments)
    assert any(not str(value).startswith('dark_') for value in information_scene_treatments)
    assert set(layout_modes.keys()) == set(SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES)
    assert set(title_modes.keys()) == set(SUPPORTED_PAGE_CALENDAR_TITLE_MODES)
    assert set(week_starts.keys()) == set(SUPPORTED_PAGE_CALENDAR_WEEK_STARTS)
    assert set(row_counts.keys()).issubset({4, 5, 6})

def test_pages_calendar_information_scene_treatment_pool_matches_pages_baseline() -> None:
    observed: set[str] = set()
    for index in range(500):
        style, metadata = resolve_pages_information_style(
            instance_seed=hash64(21910, "calendar-information-style", index),
            params={},
            scene_id="calendar",
        )
        observed.add(str(style.treatment))
        assert set(metadata["selection"]["requested_treatments"]) == set(INFORMATION_SCENE_TREATMENT_IDS)
    assert observed == set(INFORMATION_SCENE_TREATMENT_IDS)


def test_pages_calendar_information_scene_theme_keeps_required_text_readable() -> None:
    for index, treatment in enumerate(INFORMATION_SCENE_TREATMENT_IDS):
        style, _metadata = resolve_pages_information_style(
            instance_seed=hash64(21911, "calendar-information-style-readability", index),
            params={"information_scene_treatments": [str(treatment)]},
            scene_id="calendar",
        )
        theme = calendar_theme_from_information_style(style)
        assert str(theme.style_variant) == str(treatment)
        assert contrast_ratio(theme.title_text_rgb, theme.panel_fill_rgb) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
        assert contrast_ratio(theme.date_text_rgb, theme.panel_fill_rgb) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
        assert contrast_ratio(theme.weekday_text_rgb, theme.weekday_fill_rgb) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
        marker_surface = theme.marker_fill_rgb if str(theme.marker_kind) == 'fill' else theme.panel_fill_rgb
        assert contrast_ratio(theme.marker_text_rgb, marker_surface) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO


def test_pages_calendar_dark_information_style_drives_context_text_colors() -> None:
    out = PagesCalendarWeekdayOccurrenceDateTask().generate(
        12345,
        params={
            "information_scene_treatments": ["dark_report_card"],
            "pages_context_mode": "paragraph_box",
            "pages_context_simple_count": 0,
            "pages_context_paragraph_box_count": 1,
            "pages_context_text_max_elements": 2,
        },
        max_attempts=20,
    )
    render_spec = out.trace_payload["render_spec"]
    context_layer = render_spec["context_text_layer"]

    assert render_spec["calendar_style"]["information_scene_style"]["treatment"] == "dark_report_card"
    assert context_layer["enabled"] is True
    assert context_layer["element_count"] > 0
    assert context_layer["layout_spec"]["context_profile"] == "report_paragraph"
    assert context_layer["layout_spec"]["mode"] == "paragraph_box"
    assert context_layer["layout_spec"]["paragraph_box_count"] == 1
    assert context_layer["layout_spec"]["context_color_source"] == "information_scene_style"
