"""Behavior tests for graph same-component relation task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.same_component_count import GraphRelationSameComponentCountTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_relation_same_component_count_contract_matches_trace() -> None:
    task = GraphRelationSameComponentCountTask()
    out = task.generate(19201, params={'node_count': 8, 'component_count': 3, 'target_component_size': 3, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_same_component_relation'
    assert execution['question_format'] == 'count_nodes_in_same_component_including_query'
    assert execution['graph_directionality'] == 'undirected'
    assert 2 <= int(execution['component_count']) <= 4
    assert str(execution['query_label']) in execution['matching_labels']
    assert len(node_entities) == 8
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant']['query_key'] == 'same_component_count'
    assert 'component' in str(out.prompt_variants['answer_only']).lower()
    adjacency_by_label = {str(key): [str(value) for value in values] for key, values in execution['adjacency_by_label'].items()}
    components = [tuple(component) for component in execution['components_by_label']]
    query_label = str(execution['query_label'])
    matching_labels = [str(value) for value in execution['matching_labels']]
    annotation_points = list(out.annotation_gt.value)
    matching_component = next((component for component in components if query_label in component))
    assert tuple(matching_labels) == matching_component
    assert int(out.answer_gt.value) == len(matching_labels) == len(annotation_points)
    assert int(execution['target_component_size']) == len(matching_labels)
    assert trace['witness_symbolic']['labels'] == matching_labels
    assert 'label_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    assert len(trace['projected_annotation']['pixel_bbox_set']) == len(annotation_points)
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))
    for label in matching_labels:
        assert str(label) in adjacency_by_label

def test_graph_relation_same_component_prompt_examples_follow_label_variant() -> None:
    task = GraphRelationSameComponentCountTask()
    letters = task.generate(19202, params={'label_variant': 'letters', 'node_count': 8, 'component_count': 3, 'target_component_size': 3}, max_attempts=80)
    numbers = task.generate(19203, params={'label_variant': 'numbers', 'node_count': 8, 'component_count': 3, 'target_component_size': 3}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[180, 220], [310, 180], [430, 260]], 'answer': 3}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_relation_same_component_supports_numeric_labels_and_named_colors() -> None:
    task = GraphRelationSameComponentCountTask()
    out = task.generate(19204, params={'node_count': 10, 'component_count': 4, 'target_component_size': 4, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    assert all((str(label).isdigit() for label in labels))
    assert trace['witness_symbolic']['labels'] == sorted(trace['witness_symbolic']['labels'], key=lambda value: int(str(value)))
    assert execution['label_variant'] == 'numbers'
    assert execution['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert execution['node_color_name'] == 'orange'
    assert tuple(trace['render_spec']['style']['node_fill_rgb']) == tuple(named_color('orange'))

def test_graph_relation_same_component_balanced_sampling_defaults() -> None:
    task = GraphRelationSameComponentCountTask()
    component_counts: Counter[int] = Counter()
    target_sizes: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    node_counts_by_answer: dict[int, Counter[int]] = {}
    topology_profiles_by_answer: dict[int, Counter[str]] = {}
    layout_variants_by_answer: dict[int, Counter[str]] = {}
    label_variants_by_answer: dict[int, Counter[str]] = {}
    for index in range(120):
        out = task.generate(hash64(19205, 'graph_relation_same_component_count', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        component_counts[int(execution['component_count'])] += 1
        target_sizes[int(execution['target_component_size'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        node_counts_by_answer.setdefault(int(execution['target_component_size']), Counter())[int(execution['node_count'])] += 1
        topology_profiles_by_answer.setdefault(int(execution['target_component_size']), Counter())[str(execution['topology_profile'])] += 1
        layout_variants_by_answer.setdefault(int(execution['target_component_size']), Counter())[str(execution['layout_variant_requested'])] += 1
        label_variants_by_answer.setdefault(int(execution['target_component_size']), Counter())[str(execution['label_variant'])] += 1
        assert 6 <= int(execution['node_count']) <= 15
        assert 2 <= int(execution['component_count']) <= 4
        assert 2 <= int(execution['target_component_size']) <= 7
    assert set(component_counts.keys()) == {2, 3, 4}
    assert set(target_sizes.keys()) == {2, 3, 4, 5, 6, 7}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
    for answer in range(2, 8):
        assert len(node_counts_by_answer[answer]) >= 2
        assert len(topology_profiles_by_answer[answer]) >= 2
        assert len(layout_variants_by_answer[answer]) >= 2
        assert len(label_variants_by_answer[answer]) == 3
