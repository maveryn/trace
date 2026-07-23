"""Behavior tests for graph articulation-point counting task."""
from __future__ import annotations
import json
from collections import Counter
import networkx as nx
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.articulation_point_count import GraphCountingArticulationPointCountTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _graph_from_trace_adjacency(adjacency_by_label: dict[str, list[str]]) -> nx.Graph:
    graph = nx.Graph()
    for label, neighbors in adjacency_by_label.items():
        graph.add_node(str(label))
        for neighbor in neighbors:
            graph.add_edge(str(label), str(neighbor))
    return graph

def test_graph_counting_articulation_point_count_contract_matches_trace() -> None:
    task = GraphCountingArticulationPointCountTask()
    out = task.generate(19401, params={'node_count': 9, 'target_count': 3, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_articulation_point_counting'
    assert execution['question_format'] == 'count_articulation_points'
    assert execution['graph_directionality'] == 'undirected'
    assert len(node_entities) == 9
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    adjacency_by_label = {str(key): [str(value) for value in values] for key, values in execution['adjacency_by_label'].items()}
    graph = _graph_from_trace_adjacency(adjacency_by_label)
    articulation_labels = list(nx.articulation_points(graph))
    articulation_labels = sorted((str(label) for label in articulation_labels), key=lambda value: int(value) if str(value).isdigit() else str(value))
    annotation_points = list(out.annotation_gt.value)
    assert int(out.answer_gt.value) == len(articulation_labels) == len(annotation_points)
    assert int(execution['target_count']) == len(articulation_labels)
    assert articulation_labels == [str(value) for value in execution['matching_labels']]
    assert trace['witness_symbolic']['labels'] == articulation_labels
    assert 'label_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    assert len(trace['projected_annotation']['pixel_bbox_set']) == len(annotation_points)
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))

def test_graph_counting_articulation_point_count_prompt_examples_follow_label_variant() -> None:
    task = GraphCountingArticulationPointCountTask()
    letters = task.generate(19402, params={'label_variant': 'letters', 'node_count': 9, 'target_count': 3}, max_attempts=80)
    numbers = task.generate(19403, params={'label_variant': 'numbers', 'node_count': 9, 'target_count': 3}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[180, 220], [310, 180], [430, 260]], 'answer': 3}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_counting_articulation_point_count_supports_numeric_labels_and_named_colors() -> None:
    task = GraphCountingArticulationPointCountTask()
    out = task.generate(19404, params={'node_count': 10, 'target_count': 5, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
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

def test_graph_counting_articulation_point_count_balanced_sampling_defaults() -> None:
    task = GraphCountingArticulationPointCountTask()
    target_counts: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    for index in range(54):
        out = task.generate(hash64(19405, 'graph_counting_articulation_point_count', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        target_counts[int(execution['target_count'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        assert 5 <= int(execution['node_count']) <= 10
        assert 0 <= int(execution['target_count']) <= 5
    assert set(target_counts.keys()) == set(range(0, 6))
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
