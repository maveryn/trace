"""Behavior tests for graph largest-component comparison task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.largest_component_size import GraphComparisonLargestComponentSizeTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_comparison_largest_component_size_contract_matches_trace() -> None:
    task = GraphComparisonLargestComponentSizeTask()
    out = task.generate(19301, params={'node_count': 9, 'component_count': 3, 'target_largest_component_size': 4, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_largest_component_comparison'
    assert execution['question_format'] == 'count_nodes_in_unique_largest_component'
    assert execution['graph_directionality'] == 'undirected'
    assert 2 <= int(execution['component_count']) <= 4
    assert len(node_entities) == 9
    assert len(edge_entities) == int(execution['edge_count'])
    assert int(execution['largest_component_count']) == 1
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    components = [tuple(component) for component in execution['components_by_label']]
    component_sizes = [int(size) for size in execution['component_sizes']]
    matching_labels = [str(value) for value in execution['matching_labels']]
    annotation_points = list(out.annotation_gt.value)
    largest_component = next((component for component in components if len(component) == len(matching_labels)))
    assert tuple(matching_labels) == largest_component
    assert int(out.answer_gt.value) == len(matching_labels) == len(annotation_points)
    assert int(execution['target_largest_component_size']) == len(matching_labels)
    assert component_sizes.count(int(out.answer_gt.value)) == 1
    assert trace['witness_symbolic']['labels'] == matching_labels
    assert 'label_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    assert len(trace['projected_annotation']['pixel_bbox_set']) == len(annotation_points)
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))

def test_graph_comparison_largest_component_size_prompt_examples_follow_label_variant() -> None:
    task = GraphComparisonLargestComponentSizeTask()
    letters = task.generate(19302, params={'label_variant': 'letters', 'node_count': 9, 'component_count': 3, 'target_largest_component_size': 4}, max_attempts=80)
    numbers = task.generate(19303, params={'label_variant': 'numbers', 'node_count': 9, 'component_count': 3, 'target_largest_component_size': 4}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[180, 220], [310, 180], [430, 260], [520, 340]], 'answer': 4}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_comparison_largest_component_size_supports_numeric_labels_and_named_colors() -> None:
    task = GraphComparisonLargestComponentSizeTask()
    out = task.generate(19304, params={'node_count': 10, 'component_count': 4, 'target_largest_component_size': 5, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
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

def test_graph_comparison_largest_component_size_balanced_sampling_defaults() -> None:
    task = GraphComparisonLargestComponentSizeTask()
    component_counts: Counter[int] = Counter()
    target_sizes: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    for index in range(48):
        out = task.generate(hash64(19305, 'graph_comparison_largest_component_size', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        component_counts[int(execution['component_count'])] += 1
        target_sizes[int(execution['target_largest_component_size'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        assert 6 <= int(execution['node_count']) <= 15
        assert 2 <= int(execution['component_count']) <= 4
        assert 3 <= int(execution['target_largest_component_size']) <= 9
    assert set(component_counts.keys()) == {2, 3, 4}
    assert set(target_sizes.keys()) == {3, 4, 5, 6, 7, 8, 9}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
