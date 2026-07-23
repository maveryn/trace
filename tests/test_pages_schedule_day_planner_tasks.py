"""Behavior tests for the day-planner schedule task."""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.schedule.longer_than_reference_count import PagesScheduleLongerThanReferenceCountTask
from trace_tasks.tasks.pages.schedule.maximum_non_overlapping_count import PagesScheduleMaximumNonOverlappingCountTask
from trace_tasks.tasks.pages.schedule.overlap_count import PagesScheduleOverlapCountTask
from trace_tasks.tasks.shared.time_artifact_style import SUPPORTED_TIME_ARTIFACT_COLOR_NAMES, SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS
from tests.helpers import extract_prompt_json_example

def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open slot intervals overlap."""
    return bool(int(left[0]) < int(right[1]) and int(right[0]) < int(left[1]))

def _maximum_non_overlapping_subsets(events: list[dict[str, object]]) -> tuple[int, list[tuple[str, ...]]]:
    """Return the maximum compatible subset size and every optimum subset."""
    event_ids = [str(event['event_id']) for event in events]
    intervals = {str(event['event_id']): (int(event['start_slot']), int(event['end_slot'])) for event in events}
    best_size = 0
    best_subsets: list[tuple[str, ...]] = []
    for subset_size in range(1, len(event_ids) + 1):
        for subset in combinations(event_ids, subset_size):
            if any((_intervals_overlap(intervals[left], intervals[right]) for left, right in combinations(subset, 2))):
                continue
            if int(subset_size) > int(best_size):
                best_size = int(subset_size)
                best_subsets = [tuple(sorted(subset))]
            elif int(subset_size) == int(best_size):
                best_subsets.append(tuple(sorted(subset)))
    return (int(best_size), sorted(set((tuple(subset) for subset in best_subsets))))

def test_pages_schedule_day_planner_contract_matches_trace() -> None:
    task_cases = ((PagesScheduleOverlapCountTask(), 'overlap_count'), (PagesScheduleLongerThanReferenceCountTask(), 'longer_than_reference_count'), (PagesScheduleMaximumNonOverlappingCountTask(), 'maximum_non_overlapping_count'))
    scene_variants = ('classic', 'outline')
    style_variants = ('studio', 'marker')
    accent_colors = ('blue', 'orange')
    for task, source_query_id in task_cases:
        for index, scene_variant in enumerate(scene_variants):
            out = task.generate(hash64(22100, task.task_id, index), params={'scene_variant': scene_variant, 'style_variant': style_variants[index], 'accent_color_name': accent_colors[index]}, max_attempts=20)
            trace = out.trace_payload
            execution = trace['execution_trace']
            assert out.query_id == 'single'
            assert execution['source_query_id'] == source_query_id
            assert out.answer_gt.type == 'integer'
            assert out.annotation_gt.type == 'bbox_set'
            assert int(out.answer_gt.value) == len(execution['answer_event_ids'])
            expected_boxes = [trace['render_map']['event_bboxes_by_id'][str(event_id)] for event_id in execution['answer_event_ids']]
            assert out.annotation_gt.value == expected_boxes
            assert trace['projected_annotation']['bbox_set'] == expected_boxes
            if source_query_id == 'overlap_count':
                assert all(entity.get('type') != 'reference_time_band' for entity in trace['scene_ir']['entities'])

def test_pages_schedule_maximum_non_overlapping_count_is_activity_selection() -> None:
    task = PagesScheduleMaximumNonOverlappingCountTask()
    for index in range(50):
        out = task.generate(hash64(22125, task.task_id, index), params={}, max_attempts=20)
        trace = out.trace_payload
        execution = trace['execution_trace']
        answer_event_ids = tuple(str(event_id) for event_id in execution['answer_event_ids'])
        answer_set = set(answer_event_ids)
        best_size, best_subsets = _maximum_non_overlapping_subsets(list(execution['events']))
        assert int(out.answer_gt.value) in {2, 3, 4, 5}
        assert int(best_size) == int(out.answer_gt.value)
        assert tuple(best_subsets) == (tuple(sorted(answer_event_ids)),)
        expected_boxes = [trace['render_map']['event_bboxes_by_id'][str(event_id)] for event_id in answer_event_ids]
        assert out.annotation_gt.value == expected_boxes
        lane_by_event_id = {
            str(event['event_id']): int(event['lane_index'])
            for event in execution['events']
        }
        answer_lanes = {lane_by_event_id[str(event_id)] for event_id in answer_event_ids}
        assert len(answer_lanes) > 1
        lane_sets: defaultdict[int, set[str]] = defaultdict(set)
        for event_id, lane_index in lane_by_event_id.items():
            lane_sets[int(lane_index)].add(str(event_id))
        assert all(event_ids != answer_set for event_ids in lane_sets.values())

def test_pages_schedule_day_planner_prompt_examples_match_variants() -> None:
    expected = ((PagesScheduleOverlapCountTask(), 'overlap_count', ({'annotation': [[250, 276, 396, 366], [404, 318, 550, 438]], 'answer': 2}, {'answer': 2})), (PagesScheduleLongerThanReferenceCountTask(), 'longer_than_reference_count', ({'annotation': [[250, 240, 396, 408], [404, 430, 550, 634], [558, 352, 704, 568]], 'answer': 3}, {'answer': 3})), (PagesScheduleMaximumNonOverlappingCountTask(), 'maximum_non_overlapping_count', ({'annotation': [[250, 220, 396, 316], [404, 316, 550, 412], [558, 412, 704, 508], [250, 508, 396, 604]], 'answer': 4}, {'answer': 4})))
    for index, (task, query_id, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected, start=22110):
        out = task.generate(index, params={}, max_attempts=20)
        assert out.query_id == 'single'
        assert out.trace_payload['execution_trace']['source_query_id'] == query_id
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
        answer_only = extract_prompt_json_example(out.prompt_variants['answer_only'])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only

def test_pages_schedule_day_planner_balanced_sampling_defaults_cover_axes() -> None:
    tasks = (PagesScheduleOverlapCountTask(), PagesScheduleLongerThanReferenceCountTask(), PagesScheduleMaximumNonOverlappingCountTask())
    query_ids: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    lane_counts: Counter[int] = Counter()
    scenes_by_query_id: defaultdict[str, Counter[str]] = defaultdict(Counter)
    styles_by_query_id: defaultdict[str, Counter[str]] = defaultdict(Counter)
    answers_by_query_id: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for task in tasks:
        for index in range(90):
            out = task.generate(hash64(22140, task.task_id, index), params={}, max_attempts=20)
            execution = out.trace_payload['execution_trace']
            query_id = str(execution['source_query_id'])
            scene_variant = str(execution['scene_variant'])
            style_variant = str(execution['style_variant'])
            query_ids[query_id] += 1
            scene_variants[str(execution['scene_variant'])] += 1
            style_variants[style_variant] += 1
            accent_color_names[str(execution['accent_color_name'])] += 1
            lane_counts[int(execution['lane_count'])] += 1
            scenes_by_query_id[query_id][scene_variant] += 1
            styles_by_query_id[query_id][style_variant] += 1
            answers_by_query_id[query_id][int(out.answer_gt.value)] += 1
    assert set(query_ids.keys()) == {'overlap_count', 'longer_than_reference_count', 'maximum_non_overlapping_count'}
    assert set(scene_variants.keys()) == {'classic', 'minimal', 'outline'}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_COLOR_NAMES)
    assert set(lane_counts.keys()).issubset({1, 2, 3, 4, 5})
    for query_id in query_ids:
        assert set(scenes_by_query_id[query_id].keys()) == {'classic', 'minimal', 'outline'}
        assert set(styles_by_query_id[query_id].keys()) == set(SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS)
        expected_min_answers = 4 if query_id == 'maximum_non_overlapping_count' else 5
        assert len(answers_by_query_id[query_id]) >= expected_min_answers
