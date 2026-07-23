"""Behavior tests for named-icon strip run-length task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.named_strip.shape_run_length import IconsNamedStripShapeRunLengthTask
from trace_tasks.tasks.icons.named_strip.shape_run_count import IconsNamedStripShapeRunCountTask
QUERY_IDS = ('longest_shape_run_length', 'shortest_shape_run_length')

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def _target_runs(shape_ids: list[str], target_shape_id: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, shape_id in enumerate(shape_ids):
        if str(shape_id) == str(target_shape_id):
            if start is None:
                start = int(index)
        elif start is not None:
            runs.append((int(start), int(index) - 1))
            start = None
    if start is not None:
        runs.append((int(start), len(shape_ids) - 1))
    return runs

def _target_run_lengths(shape_ids: list[str], target_shape_id: str) -> list[int]:
    return [int(end) - int(start) + 1 for start, end in _target_runs(shape_ids, target_shape_id)]

def test_icons_sequence_named_shape_run_length_longest_contract_matches_scene() -> None:
    task = IconsNamedStripShapeRunLengthTask()
    out = task.generate(2026052810, params={'query_id': 'longest_shape_run_length', 'target_shape_id': 'star', 'target_run_length': 5, 'strip_length': 15}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    icon_entities = [entity for entity in entities if str(entity['entity_kind']) == 'named_icon']
    cell_entities = [entity for entity in entities if str(entity['entity_kind']) == 'strip_cell']
    assert out.scene_id == 'named_strip'
    assert out.query_id == 'longest_shape_run_length'
    assert out.answer_gt.type == 'integer'
    assert out.answer_gt.value == 5
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 5
    assert trace['scene_ir']['scene_kind'] == 'icons_named_strip_run_length'
    assert execution['question_format'] == 'named_shape_run_length'
    assert int(execution['strip_length']) == 15
    assert str(execution['target_shape_id']) == 'star'
    assert len(icon_entities) == 15
    assert len(cell_entities) == 15
    shape_ids = [str(value) for value in execution['shape_ids_by_cell']]
    lengths = _target_run_lengths(shape_ids, 'star')
    assert max(lengths) == 5
    assert lengths.count(5) == 1
    selected_indices = [int(value) for value in execution['selected_run_indices']]
    assert len(selected_indices) == 5
    assert selected_indices == list(range(min(selected_indices), max(selected_indices) + 1))
    assert all((shape_ids[index] == 'star' for index in selected_indices))
    selected_boxes = sorted([entity['bbox_xyxy'] for entity in icon_entities if bool(entity['is_selected_run_member'])], key=lambda box: (int(box[1]), int(box[0]), int(box[3]), int(box[2])))
    assert out.annotation_gt.value == selected_boxes
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == selected_boxes
    assert trace['projected_annotation']['pixel_bbox_set'] == selected_boxes
    style = trace['render_spec']['style']
    assert style['text_legibility']['failure_count'] == 0

def test_icons_sequence_named_shape_run_length_shortest_contract_matches_scene() -> None:
    task = IconsNamedStripShapeRunLengthTask()
    out = task.generate(2026052811, params={'query_id': 'shortest_shape_run_length', 'target_shape_id': 'bell', 'target_run_length': 2, 'strip_length': 14}, max_attempts=100)
    execution = out.trace_payload['execution_trace']
    shape_ids = [str(value) for value in execution['shape_ids_by_cell']]
    lengths = _target_run_lengths(shape_ids, 'bell')
    assert out.query_id == 'shortest_shape_run_length'
    assert out.answer_gt.value == 2
    assert min(lengths) == 2
    assert lengths.count(2) == 1
    assert len(out.annotation_gt.value) == 2
    assert all((shape_ids[int(index)] == 'bell' for index in execution['selected_run_indices']))

def test_icons_sequence_named_shape_run_count_contract_matches_scene() -> None:
    task = IconsNamedStripShapeRunCountTask()
    out = task.generate(2026070101, params={'target_shape_id': 'star', 'run_count': 3, 'strip_length': 14}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    entities = trace['scene_ir']['entities']
    icon_entities = [entity for entity in entities if str(entity['entity_kind']) == 'named_icon']
    assert out.scene_id == 'named_strip'
    assert out.query_id == 'single'
    assert out.answer_gt.type == 'integer'
    assert out.answer_gt.value == 3
    assert out.annotation_gt.type == 'bbox_set'
    assert len(out.annotation_gt.value) == 3
    assert trace['scene_ir']['scene_kind'] == 'icons_named_strip_run_count'
    assert execution['question_format'] == 'named_shape_run_count'
    assert int(execution['strip_length']) == 14
    assert str(execution['target_shape_id']) == 'star'
    shape_ids = [str(value) for value in execution['shape_ids_by_cell']]
    runs = _target_runs(shape_ids, 'star')
    starts = [int(start) for start, _ in runs]
    assert len(runs) == 3
    assert [int(value) for value in execution['selected_run_start_indices']] == starts
    assert all((shape_ids[index] == 'star' for index in starts))
    selected_boxes = sorted([entity['bbox_xyxy'] for entity in icon_entities if bool(entity['is_selected_run_member'])], key=lambda box: (int(box[1]), int(box[0]), int(box[3]), int(box[2])))
    start_boxes = sorted([entity['bbox_xyxy'] for entity in icon_entities if int(entity['cell_index']) in set(starts)], key=lambda box: (int(box[1]), int(box[0]), int(box[3]), int(box[2])))
    assert selected_boxes == start_boxes
    assert out.annotation_gt.value == start_boxes
    assert trace['projected_annotation']['type'] == 'bbox_set'
    assert trace['projected_annotation']['bbox_set'] == start_boxes
    assert trace['render_map']['selected_run_start_instance_ids'] == execution['selected_run_start_instance_ids']
    assert trace['query_spec']['prompt_variant']['query_key'] == 'shape_run_count'

def test_icons_sequence_named_shape_run_length_prompt_example_matches_contract() -> None:
    task = IconsNamedStripShapeRunLengthTask()
    out = task.generate(2026052812, params={'query_id': 'longest_shape_run_length', 'target_shape_id': 'guitar', 'target_run_length': 3}, max_attempts=100)
    assert '"guitar"' in out.prompt
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 3}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 3
    assert answer_and_annotation['answer'] == 3

def test_icons_sequence_named_shape_run_count_prompt_example_matches_contract() -> None:
    task = IconsNamedStripShapeRunCountTask()
    out = task.generate(2026070102, params={'target_shape_id': 'guitar', 'run_count': 3}, max_attempts=100)
    assert '"guitar"' in out.prompt
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 3}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert isinstance(answer_and_annotation['annotation'], list)
    assert len(answer_and_annotation['annotation']) == 3
    assert answer_and_annotation['answer'] == 3
    assert 'first icon in each counted run' in out.prompt_variants['answer_and_annotation']

def test_icons_sequence_named_shape_run_length_sampling_smoke() -> None:
    task = IconsNamedStripShapeRunLengthTask()
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    strip_lengths: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(2026052813, 'icons_named_strip_shape_run_length', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_id = str(out.query_id)
        answer = int(out.answer_gt.value)
        target_shape_id = str(execution['target_shape_id'])
        shape_ids = [str(value) for value in execution['shape_ids_by_cell']]
        lengths = _target_run_lengths(shape_ids, target_shape_id)
        if query_id == 'longest_shape_run_length':
            assert answer == max(lengths)
            assert 2 <= answer <= 6
        else:
            assert answer == min(lengths)
            assert 1 <= answer <= 5
        assert lengths.count(answer) == 1
        assert len(out.annotation_gt.value) == answer
        assert 12 <= int(execution['strip_length']) <= 16
        query_counts[query_id] += 1
        answer_counts[answer] += 1
        strip_lengths[int(execution['strip_length'])] += 1
    assert set(query_counts) == set(QUERY_IDS)
    assert len(answer_counts) >= 5
    assert len(strip_lengths) >= 4

def test_icons_sequence_named_shape_run_count_sampling_smoke() -> None:
    task = IconsNamedStripShapeRunCountTask()
    answer_counts: Counter[int] = Counter()
    strip_lengths: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(2026070103, 'icons_named_strip_shape_run_count', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        answer = int(out.answer_gt.value)
        target_shape_id = str(execution['target_shape_id'])
        shape_ids = [str(value) for value in execution['shape_ids_by_cell']]
        runs = _target_runs(shape_ids, target_shape_id)
        starts = [int(start) for start, _ in runs]
        assert 1 <= answer <= 4
        assert len(runs) == answer
        assert [int(value) for value in execution['selected_run_start_indices']] == starts
        assert len(out.annotation_gt.value) == answer
        assert 12 <= int(execution['strip_length']) <= 16
        answer_counts[answer] += 1
        strip_lengths[int(execution['strip_length'])] += 1
    assert set(answer_counts) == {1, 2, 3, 4}
    assert len(strip_lengths) >= 4
