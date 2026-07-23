"""Behavior tests for the single-transform curated-icon option task."""
from __future__ import annotations
import json
from pathlib import Path
from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.shared.icon_assets import icon_transform_signature, resolve_icon_pool
from trace_tasks.tasks.icons.shared.icon_transform import IDENTITY_TRANSFORM_ID
from trace_tasks.tasks.icons.single_transform_options.geometric_transform_result_label import IconsSingleTransformOptionsGeometricTransformResultLabelTask, SUPPORTED_QUERY_IDS, TASK_ID
from trace_tasks.tasks.icons.single_transform_options.inverse_geometric_transform_source_label import IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask, SUPPORTED_QUERY_IDS as INVERSE_SUPPORTED_QUERY_IDS, TASK_ID as INVERSE_TASK_ID
from trace_tasks.tasks.icons.single_transform_options.shared.sampling import compose_transform_ids
from tests.helpers import read_jsonl
QUERY_TO_TRANSFORM = {'rotate_90_clockwise_result_label': 'rot270', 'rotate_90_counterclockwise_result_label': 'rot90', 'rotate_180_result_label': 'rot180', 'flip_horizontal_result_label': 'flip_h', 'flip_vertical_result_label': 'flip_v'}
INVERSE_QUERY_TO_TRANSFORM = {'rotate_90_clockwise_source_label': 'rot270', 'rotate_90_counterclockwise_source_label': 'rot90', 'rotate_180_source_label': 'rot180', 'flip_horizontal_source_label': 'flip_h', 'flip_vertical_source_label': 'flip_v'}

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())

def test_icons_single_transform_options_contract_matches_scene() -> None:
    task = IconsSingleTransformOptionsGeometricTransformResultLabelTask()
    out = task.generate(2026060801, params={'query_id': 'rotate_90_clockwise_result_label', 'answer_label': 'D'}, max_attempts=300)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'scene']
    reference_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'reference']
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'D'
    assert out.annotation_gt.type == 'bbox_map'
    assert sorted(out.annotation_gt.value.keys()) == ['reference_icon', 'selected_option']
    assert out.scene_id == 'single_transform_options'
    assert out.query_id == 'rotate_90_clockwise_result_label'
    assert tuple(task.supported_query_ids) == SUPPORTED_QUERY_IDS
    assert trace['scene_ir']['scene_kind'] == 'icons_single_transform_options_result_label'
    assert execution['question_format'] == 'select_transformed_reference_icon_option'
    assert int(execution['object_count']) == 6
    assert len(reference_entities) == 1
    assert len(scene_entities) == 6
    assert [str(entity['label']) for entity in scene_entities] == list('ABCDEF')
    reference = reference_entities[0]
    reference_box = list(reference['icon_bbox_xyxy'])
    reference_dims = (int(reference_box[2] - reference_box[0]), int(reference_box[3] - reference_box[1]))
    assert min(reference_dims) >= 100
    option_dims = {
        (
            int(entity['icon_bbox_xyxy'][2] - entity['icon_bbox_xyxy'][0]),
            int(entity['icon_bbox_xyxy'][3] - entity['icon_bbox_xyxy'][1]),
        )
        for entity in scene_entities
    }
    assert all(abs(width - reference_dims[0]) <= 1 and abs(height - reference_dims[1]) <= 1 for width, height in option_dims)
    assert str(reference['transform_id']) == IDENTITY_TRANSFORM_ID
    assert str(reference['target_transform_id']) == 'rot270'
    assert str(reference['operation_cue']) == 'Rotate 90 CW'
    assert str(execution['target_transform_id']) == 'rot270'
    assert str(execution['operation_cue']) == 'Rotate 90 CW'
    assert str(execution['icon_id']) in set(resolve_icon_pool('non_symmetry.txt'))
    answer_label = str(out.answer_gt.value)
    matching = [entity for entity in scene_entities if bool(entity['is_match'])]
    assert len(matching) == 1
    assert str(matching[0]['label']) == answer_label
    assert str(matching[0]['transform_id']) == 'rot270'
    assert str(execution['option_transform_ids_by_label'][answer_label]) == 'rot270'
    assert sorted((str(entity['transform_id']) for entity in scene_entities)) == sorted([IDENTITY_TRANSFORM_ID, 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v'])
    signatures = {icon_transform_signature(str(execution['icon_id']), 96, str(entity['transform_id'])) for entity in scene_entities}
    signatures.add(icon_transform_signature(str(execution['icon_id']), 96, IDENTITY_TRANSFORM_ID))
    assert len(signatures) == 6
    assert out.annotation_gt.value == {'reference_icon': list(reference['icon_bbox_xyxy']), 'selected_option': list(matching[0]['cell_bbox_xyxy'])}
    assert trace['projected_annotation']['type'] == 'bbox_map'
    assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
    assert trace['witness_symbolic']['selected_option_label'] == answer_label
    assert trace['query_spec']['template_id'] == 'icons_single_transform_options_v1'
    assert trace['query_spec']['prompt_variant']['prompt_schema_version'] == 'v1'
    assert trace['query_spec']['prompt_variant']['query_key'] == 'rotate_90_clockwise_result_label'
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert '90' in out.prompt
    assert 'clockwise' in out.prompt

def test_icons_single_transform_options_all_queries_map_to_expected_transform() -> None:
    task = IconsSingleTransformOptionsGeometricTransformResultLabelTask()
    for index, (query_id, transform_id) in enumerate(QUERY_TO_TRANSFORM.items()):
        out = task.generate(hash64(2026060802, query_id, index), params={'query_id': query_id}, max_attempts=300)
        execution = out.trace_payload['execution_trace']
        selected_label = str(out.answer_gt.value)
        assert str(out.query_id) == query_id
        assert str(execution['target_transform_id']) == transform_id
        assert str(execution['option_transform_ids_by_label'][selected_label]) == transform_id
        assert int(execution['object_count']) == 6

def test_icons_single_transform_options_inverse_contract_matches_scene() -> None:
    task = IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask()
    out = task.generate(2026070101, params={'query_id': 'rotate_90_clockwise_source_label', 'answer_label': 'B'}, max_attempts=300)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'scene']
    reference_entities = [entity for entity in trace['scene_ir']['entities'] if str(entity.get('panel')) == 'reference']
    assert out.answer_gt.type == 'option_letter'
    assert out.answer_gt.value == 'B'
    assert out.annotation_gt.type == 'bbox_map'
    assert sorted(out.annotation_gt.value.keys()) == ['reference_icon', 'selected_option']
    assert out.scene_id == 'single_transform_options'
    assert out.query_id == 'rotate_90_clockwise_source_label'
    assert tuple(task.supported_query_ids) == INVERSE_SUPPORTED_QUERY_IDS
    assert trace['scene_ir']['scene_kind'] == 'icons_single_transform_options_inverse_source_label'
    assert execution['question_format'] == 'select_source_option_that_becomes_reference_after_operation'
    assert int(execution['object_count']) == 4
    assert len(reference_entities) == 1
    assert len(scene_entities) == 4
    assert [str(entity['label']) for entity in scene_entities] == list('ABCD')
    reference = reference_entities[0]
    selected = [entity for entity in scene_entities if bool(entity['is_match'])]
    assert len(selected) == 1
    selected = selected[0]
    assert str(selected['label']) == 'B'
    assert str(execution['target_transform_id']) == 'rot270'
    assert str(reference['target_transform_id']) == 'rot270'
    assert str(reference['transform_id']) == str(execution['reference_transform_id'])
    assert str(selected['transform_id']) == str(execution['correct_source_transform_id'])
    assert str(selected['transform_id']) != str(reference['transform_id'])
    assert compose_transform_ids(
        after_transform_id=str(execution['target_transform_id']),
        before_transform_id=str(selected['transform_id']),
    ) == str(reference['transform_id'])
    for entity in scene_entities:
        produced = compose_transform_ids(
            after_transform_id=str(execution['target_transform_id']),
            before_transform_id=str(entity['transform_id']),
        )
        if str(entity['label']) == str(out.answer_gt.value):
            assert produced == str(reference['transform_id'])
        else:
            assert produced != str(reference['transform_id'])
            assert str(entity['transform_id']) != str(reference['transform_id'])
    assert out.annotation_gt.value == {'reference_icon': list(reference['icon_bbox_xyxy']), 'selected_option': list(selected['cell_bbox_xyxy'])}
    assert trace['projected_annotation']['type'] == 'bbox_map'
    assert trace['projected_annotation']['bbox_map'] == out.annotation_gt.value
    assert trace['witness_symbolic']['selected_option_label'] == 'B'
    assert trace['witness_symbolic']['selected_option_after_operation_transform_id'] == str(reference['transform_id'])
    assert trace['query_spec']['template_id'] == 'icons_single_transform_options_v1'
    assert trace['query_spec']['prompt_variant']['query_key'] == 'rotate_90_clockwise_source_label'
    assert 'four labeled source-option icons' in out.prompt
    assert 'Reference icon' in out.prompt

def test_icons_single_transform_options_inverse_all_queries_map_to_expected_transform() -> None:
    task = IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask()
    for index, (query_id, transform_id) in enumerate(INVERSE_QUERY_TO_TRANSFORM.items()):
        out = task.generate(hash64(2026070102, query_id, index), params={'query_id': query_id}, max_attempts=300)
        execution = out.trace_payload['execution_trace']
        selected_label = str(out.answer_gt.value)
        selected_source = str(execution['option_source_transform_ids_by_label'][selected_label])
        assert str(out.query_id) == query_id
        assert str(execution['target_transform_id']) == transform_id
        assert int(execution['object_count']) == 4
        assert compose_transform_ids(
            after_transform_id=str(execution['target_transform_id']),
            before_transform_id=selected_source,
        ) == str(execution['reference_transform_id'])

def test_icons_single_transform_options_prompt_example_matches_contract() -> None:
    task = IconsSingleTransformOptionsGeometricTransformResultLabelTask()
    out = task.generate(2026060803, params={'query_id': 'flip_vertical_result_label', 'answer_label': 'C'}, max_attempts=300)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 'C'}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert sorted(answer_and_annotation['annotation'].keys()) == ['reference_icon', 'selected_option']
    assert answer_and_annotation['answer'] == 'C'

def test_icons_single_transform_options_inverse_prompt_example_matches_contract() -> None:
    task = IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask()
    out = task.generate(2026070103, params={'query_id': 'flip_vertical_source_label', 'answer_label': 'C'}, max_attempts=300)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 'C'}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert sorted(answer_and_annotation['annotation'].keys()) == ['reference_icon', 'selected_option']
    assert answer_and_annotation['answer'] == 'C'

def test_icons_single_transform_options_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / TASK_ID
    config = BuildConfig(output_root=str(output_root), dataset_name=f'build_smoke_{TASK_ID}', instance_version='v0', image_format='png', tasks=[BuildTaskConfig(task_id=TASK_ID, count=3, params={'query_id': 'rotate_180_result_label'})], strict_repro=False, max_attempts_per_instance=300, sampling_seed=37)
    final_path = build_dataset(config, code_hash='icons-single-transform-options-smoke')
    assert final_path.exists()
    train_records = read_jsonl(final_path / 'train_instances.jsonl')
    assert len(train_records) == 3
    assert all((record['domain'] == 'icons' for record in train_records))
    assert all((record['task'] == TASK_ID for record in train_records))
    build_report = json.loads((final_path / 'build_report.json').read_text(encoding='utf-8'))
    assert int(build_report['accepted_counts_by_task'][TASK_ID]) == 3
    validation = json.loads((final_path / 'validation_report.json').read_text(encoding='utf-8'))
    assert validation['total_errors'] == 0

def test_icons_single_transform_options_inverse_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / INVERSE_TASK_ID
    config = BuildConfig(output_root=str(output_root), dataset_name=f'build_smoke_{INVERSE_TASK_ID}', instance_version='v0', image_format='png', tasks=[BuildTaskConfig(task_id=INVERSE_TASK_ID, count=3, params={'query_id': 'rotate_180_source_label'})], strict_repro=False, max_attempts_per_instance=300, sampling_seed=41)
    final_path = build_dataset(config, code_hash='icons-single-transform-options-inverse-smoke')
    assert final_path.exists()
    train_records = read_jsonl(final_path / 'train_instances.jsonl')
    assert len(train_records) == 3
    assert all((record['domain'] == 'icons' for record in train_records))
    assert all((record['task'] == INVERSE_TASK_ID for record in train_records))
    build_report = json.loads((final_path / 'build_report.json').read_text(encoding='utf-8'))
    assert int(build_report['accepted_counts_by_task'][INVERSE_TASK_ID]) == 3
    validation = json.loads((final_path / 'validation_report.json').read_text(encoding='utf-8'))
    assert validation['total_errors'] == 0
