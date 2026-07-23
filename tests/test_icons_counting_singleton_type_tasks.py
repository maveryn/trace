"""Behavior tests for icon singleton-type counting task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.icon_field.frequency_extreme_type_label import IconsIconFieldFrequencyExtremeTypeLabelTask
from trace_tasks.tasks.icons.icon_field.most_frequent_type_count import IconsIconFieldMostFrequentTypeCountTask
from trace_tasks.tasks.icons.icon_field.singleton_type_count import IconsIconFieldSingletonTypeCountTask
from trace_tasks.tasks.icons.icon_grid.distinct_color_count import IconsIconGridDistinctColorCountTask
from trace_tasks.tasks.icons.icon_grid.distinct_type_count import IconsIconGridDistinctTypeCountTask

def _overlap_fraction_smaller(left: list[int], right: list[int]) -> float:
    ix0 = max(int(left[0]), int(right[0]))
    iy0 = max(int(left[1]), int(right[1]))
    ix1 = min(int(left[2]), int(right[2]))
    iy1 = min(int(left[3]), int(right[3]))
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    left_area = max(1, int(left[2]) - int(left[0])) * max(1, int(left[3]) - int(left[1]))
    right_area = max(1, int(right[2]) - int(right[0])) * max(1, int(right[3]) - int(right[1]))
    return float(inter) / float(min(left_area, right_area))

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _representative_bboxes_by_field(entities: list[dict], field_name: str) -> list[list[int]]:
    selected: dict[str, dict] = {}
    sorted_entities = sorted(
        entities,
        key=lambda entity: (
            int(entity.get("row_index", 0)),
            int(entity.get("col_index", 0)),
            int(entity["index"]),
        ),
    )
    for entity in sorted_entities:
        key = str(entity[field_name])
        if key not in selected:
            selected[key] = entity
    representatives = sorted(
        selected.values(),
        key=lambda entity: (
            int(entity.get("row_index", 0)),
            int(entity.get("col_index", 0)),
            int(entity["index"]),
        ),
    )
    return [[int(value) for value in entity["cell_bbox_xyxy"]] for entity in representatives]

def test_icons_counting_singleton_type_contract_matches_scene() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    out = task.generate(18310, params={'object_count': 9, 'target_count': 3}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert out.annotation_gt.value == sorted(out.annotation_gt.value, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    assert trace['scene_ir']['scene_kind'] == 'icons_singleton_type_counting'
    assert out.query_id == 'single'
    assert execution['question_format'] == 'count_singleton_type_icons'
    assert execution['query_id'] == 'single'
    assert execution['prompt_query_key'] == 'singleton_type_count'
    assert execution['scene_variant'] == 'single_panel_scene'
    assert int(execution['object_count']) == 9
    assert int(execution['target_count']) == 3
    assert int(execution['repeated_type_count']) >= 1
    assert int(execution['unique_type_total']) >= 4
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 1
    assert len(scene_entities) == 9
    assert 'reference_panel_xyxy' not in trace['render_spec']['panel_geometry']
    sampled_palette = [tuple((int(channel) for channel in color)) for color in trace['render_spec']['style']['sampled_palette_rgb']]
    assert 8 <= len(sampled_palette) <= 12
    assert list(trace['render_spec']['style']['icon_noise_edit_count_range']) == [0, 2]
    assert float(trace['render_spec']['style']['scene_max_overlap_fraction']) == 0.1
    singleton_indices = {int(value) for value in execution['singleton_indices']}
    type_frequencies = {str(key): int(value) for key, value in execution['type_frequencies'].items()}
    assert len(singleton_indices) == 3
    assert sum((1 for value in execution['scene_icon_ids'] if type_frequencies[str(value)] == 1)) == 3
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == out.annotation_gt.value
    assert trace['projected_annotation']['pixel_bbox_set'] == out.annotation_gt.value
    assert len(trace['projected_annotation']['pixel_point_set']) == len(out.annotation_gt.value)
    annotation_bboxes = {tuple(bbox) for bbox in out.annotation_gt.value}
    for index, entity in enumerate(scene_entities):
        icon_id = str(entity['icon_id'])
        assert int(entity['type_frequency']) == int(type_frequencies[icon_id])
        assert bool(entity['is_singleton_type']) == (int(index) in singleton_indices)
        assert tuple((int(channel) for channel in entity['tint_rgb'])) in sampled_palette
        assert int(entity['rotation_degrees']) in {0, 90, 180, 270}
        assert isinstance(entity['noise_edits'], list)
        if bool(entity['is_singleton_type']):
            assert tuple(entity['bbox_xyxy']) in annotation_bboxes
            assert int(type_frequencies[icon_id]) == 1
        else:
            assert int(type_frequencies[icon_id]) >= 2
    for left_index, left in enumerate(scene_entities):
        for right in scene_entities[left_index + 1:]:
            assert _overlap_fraction_smaller(left['bbox_xyxy'], right['bbox_xyxy']) <= 0.1 + 1e-06

def test_icons_counting_singleton_type_supports_zero_singletons() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    out = task.generate(18311, params={'object_count': 8, 'target_count': 0}, max_attempts=200)
    execution = out.trace_payload['execution_trace']
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert execution['singleton_indices'] == []
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 1

def test_icons_counting_singleton_type_prompt_example_matches_contract() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    out = task.generate(18312, params={'object_count': 9, 'target_count': 2}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 2
    assert all((len(bbox) == 4 for bbox in answer_and_annotation['annotation']))
    assert answer_and_annotation['answer'] == 2

def test_icons_counting_singleton_type_repeated_distractors_share_visual_style() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    out = task.generate(18314, params={'object_count': 8, 'target_count': 2}, max_attempts=200)
    trace = out.trace_payload
    execution = trace['execution_trace']
    assert int(out.answer_gt.value) == 2
    assert len(out.annotation_gt.value) == 2
    assert execution['question_format'] == 'count_singleton_type_icons'
    assert int(execution['object_count']) == 8
    assert int(execution['target_count']) == 2
    assert int(execution['singleton_count']) == 2
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 1
    assert int(execution['repeated_icon_count']) == 6
    assert len(execution['repeated_indices']) == 6
    assert len(execution['annotation_indices']) == 2
    assert trace['scene_ir']['relations']['counting_rule'] == 'singleton_icon_type_frequency'
    annotation_bboxes = {tuple(bbox) for bbox in out.annotation_gt.value}
    repeated_styles: dict[str, set[tuple[object, ...]]] = {}
    for entity in trace['scene_ir']['entities']:
        if bool(entity['is_repeated_type']):
            assert tuple(entity['bbox_xyxy']) not in annotation_bboxes
            assert int(entity['type_frequency']) >= 2
            repeated_styles.setdefault(str(entity['icon_id']), set()).add((int(entity['rotation_degrees']), int(entity['nominal_size_px']), tuple((int(channel) for channel in entity['tint_rgb'])), json.dumps(entity['noise_edits'], sort_keys=True), entity['noise_seed']))
        else:
            assert tuple(entity['bbox_xyxy']) in annotation_bboxes
            assert int(entity['type_frequency']) == 1
    assert repeated_styles
    assert all((len(styles) == 1 for styles in repeated_styles.values()))

def test_icons_counting_singleton_type_sampling_defaults() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    object_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(18313, 'icons_counting_singleton_type', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        object_count = int(execution['object_count'])
        target_count = int(execution['target_count'])
        object_counts[object_count] += 1
        target_counts[target_count] += 1
        assert execution['query_id'] == 'single'
        assert execution['prompt_query_key'] == 'singleton_type_count'
        assert 5 <= object_count <= 10
        assert 0 <= target_count <= 4
        assert int(execution["unique_color_total"]) == 1
        assert len(set(execution["scene_color_keys"])) == 1
    assert set(target_counts.keys()) == set(range(0, 5))
    assert min(object_counts.keys()) >= 5
    assert max(object_counts.keys()) <= 10


def test_icons_counting_most_frequent_type_sampling_defaults() -> None:
    task = IconsIconFieldMostFrequentTypeCountTask()
    object_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(18315, 'icons_counting_most_frequent_type', index), params={}, max_attempts=200)
        execution = out.trace_payload['execution_trace']
        object_count = int(execution['object_count'])
        target_count = int(execution['target_count'])
        object_counts[object_count] += 1
        target_counts[target_count] += 1
        assert execution['query_id'] == 'single'
        assert execution['prompt_query_key'] == 'most_frequent_type_count'
        assert 7 <= object_count <= 12
        assert 2 <= target_count <= 6
        assert int(execution["unique_color_total"]) == 1
        assert len(set(execution["scene_color_keys"])) == 1
    assert set(target_counts.keys()) == set(range(2, 7))
    assert min(object_counts.keys()) >= 7
    assert max(object_counts.keys()) <= 12


def test_icons_frequency_extreme_type_label_most_contract_matches_scene() -> None:
    task = IconsIconFieldFrequencyExtremeTypeLabelTask()
    out = task.generate(
        18324,
        params={"query_id": "most_frequent_type_label", "option_count": 4},
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    frequencies = {str(key): int(value) for key, value in execution["type_frequencies"].items()}
    winner_icon_id = str(execution["winner_icon_id"])
    assert out.scene_id == "icon_field"
    assert out.query_id == "most_frequent_type_label"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_set"
    assert int(execution["option_count"]) == 4
    assert int(execution["unique_type_total"]) == 4
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 1
    assert frequencies[winner_icon_id] == max(frequencies.values())
    assert sum(1 for value in frequencies.values() if int(value) == max(frequencies.values())) == 1
    assert str(out.answer_gt.value) == str(execution["winner_label"])
    assert len(out.annotation_gt.value) == int(execution["winner_frequency"])
    assert len(out.annotation_gt.value) == frequencies[winner_icon_id]
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert sorted(record["label"] for record in execution["candidate_markers"]) == ["A", "B", "C", "D"]
    assert {str(record["icon_id"]) for record in execution["candidate_markers"]} == set(frequencies)
    assert any(str(record["label"]) == str(out.answer_gt.value) for record in execution["candidate_markers"])
    assert "marked icon type" in out.prompt


def test_icons_frequency_extreme_type_label_least_contract_matches_scene() -> None:
    task = IconsIconFieldFrequencyExtremeTypeLabelTask()
    out = task.generate(
        18325,
        params={"query_id": "least_frequent_type_label", "option_count": 6},
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    frequencies = {str(key): int(value) for key, value in execution["type_frequencies"].items()}
    winner_icon_id = str(execution["winner_icon_id"])
    assert out.scene_id == "icon_field"
    assert out.query_id == "least_frequent_type_label"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_set"
    assert int(execution["option_count"]) == 6
    assert int(execution["unique_type_total"]) == 6
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 1
    assert frequencies[winner_icon_id] == min(frequencies.values())
    assert sum(1 for value in frequencies.values() if int(value) == min(frequencies.values())) == 1
    assert str(out.answer_gt.value) == str(execution["winner_label"])
    assert len(out.annotation_gt.value) == int(execution["winner_frequency"])
    assert len(out.annotation_gt.value) == frequencies[winner_icon_id]
    assert sorted(record["label"] for record in execution["candidate_markers"]) == ["A", "B", "C", "D", "E", "F"]
    assert {str(record["icon_id"]) for record in execution["candidate_markers"]} == set(frequencies)
    assert "marked icon type" in out.prompt


def test_icons_frequency_extreme_type_label_sampling_defaults() -> None:
    task = IconsIconFieldFrequencyExtremeTypeLabelTask()
    seen_queries: Counter[str] = Counter()
    seen_option_counts: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(18326, "icons_frequency_extreme_type_label", index), params={}, max_attempts=200)
        execution = out.trace_payload["execution_trace"]
        frequencies = {str(key): int(value) for key, value in execution["type_frequencies"].items()}
        seen_queries[str(out.query_id)] += 1
        seen_option_counts[int(execution["option_count"])] += 1
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox_set"
        assert str(out.query_id) in {"most_frequent_type_label", "least_frequent_type_label"}
        assert int(execution["option_count"]) in {4, 6}
        assert int(execution["unique_type_total"]) == int(execution["option_count"])
        assert int(execution["unique_color_total"]) == 1
        assert len(out.annotation_gt.value) == int(execution["winner_frequency"])
        if str(out.query_id) == "most_frequent_type_label":
            assert int(execution["winner_frequency"]) == max(frequencies.values())
        else:
            assert int(execution["winner_frequency"]) == min(frequencies.values())
    assert set(seen_queries) == {"most_frequent_type_label", "least_frequent_type_label"}
    assert set(seen_option_counts) == {4, 6}


def test_icons_counting_distinct_type_contract_matches_representatives() -> None:
    task = IconsIconGridDistinctTypeCountTask()
    out = task.generate(18316, params={"target_count": 4, "object_count": 9}, max_attempts=200)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    entities = trace["scene_ir"]["entities"]
    assert out.scene_id == "icon_grid"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 4
    assert execution["prompt_query_key"] == "distinct_type_count"
    assert execution["question_format"] == "count_distinct_icon_types_in_grid"
    assert execution["representative_rule"] == "topmost_row_then_left_to_right_cell_per_category"
    assert int(execution["unique_type_total"]) == 4
    assert int(execution["unique_color_total"]) == 1
    assert len(set(execution["scene_icon_ids"])) == 4
    assert len(set(execution["scene_color_keys"])) == 1
    assert out.annotation_gt.value == _representative_bboxes_by_field(entities, "icon_id")
    assert "for each different icon type" in out.prompt
    assert "topmost row" in out.prompt


def test_icons_counting_distinct_color_contract_matches_representatives() -> None:
    task = IconsIconGridDistinctColorCountTask()
    out = task.generate(18317, params={"target_count": 3, "object_count": 9}, max_attempts=200)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    entities = trace["scene_ir"]["entities"]
    assert out.scene_id == "icon_grid"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 3
    assert execution["prompt_query_key"] == "distinct_color_count"
    assert execution["question_format"] == "count_distinct_icon_colors_in_grid"
    assert execution["representative_rule"] == "topmost_row_then_left_to_right_cell_per_category"
    assert int(execution["unique_color_total"]) == 3
    assert int(execution["unique_type_total"]) == 1
    assert len(set(execution["scene_color_keys"])) == 3
    assert len(set(execution["scene_icon_ids"])) == 1
    assert out.annotation_gt.value == _representative_bboxes_by_field(entities, "color_key")
    assert "for each different icon color" in out.prompt
    assert "topmost row" in out.prompt


def test_icons_counting_distinct_category_sampling_defaults() -> None:
    type_task = IconsIconGridDistinctTypeCountTask()
    color_task = IconsIconGridDistinctColorCountTask()
    type_targets: Counter[int] = Counter()
    color_targets: Counter[int] = Counter()
    for index in range(80):
        type_out = type_task.generate(hash64(18318, "distinct_type", index), params={}, max_attempts=200)
        color_out = color_task.generate(hash64(18319, "distinct_color", index), params={}, max_attempts=200)
        type_execution = type_out.trace_payload["execution_trace"]
        color_execution = color_out.trace_payload["execution_trace"]
        type_targets[int(type_out.answer_gt.value)] += 1
        color_targets[int(color_out.answer_gt.value)] += 1
        assert 1 <= int(type_out.answer_gt.value) <= 5
        assert 1 <= int(color_out.answer_gt.value) <= 5
        assert len(type_out.annotation_gt.value) == int(type_out.answer_gt.value)
        assert len(color_out.annotation_gt.value) == int(color_out.answer_gt.value)
        assert 5 <= int(type_execution["object_count"]) <= 12
        assert 5 <= int(color_execution["object_count"]) <= 12
        assert int(type_execution["unique_color_total"]) == 1
        assert len(set(type_execution["scene_color_keys"])) == 1
        assert int(type_execution["unique_type_total"]) == int(type_out.answer_gt.value)
        assert int(color_execution["unique_type_total"]) == 1
        assert len(set(color_execution["scene_icon_ids"])) == 1
        assert int(color_execution["unique_color_total"]) == int(color_out.answer_gt.value)
    assert set(type_targets.keys()) == set(range(1, 6))
    assert set(color_targets.keys()) == set(range(1, 6))
