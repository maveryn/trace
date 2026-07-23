"""Behavior tests for graph bridge-edge counting task."""
from __future__ import annotations
import json
from collections import Counter
import networkx as nx
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.bridge_count import GraphCountingBridgeCountTask
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

def test_graph_counting_bridge_count_contract_matches_trace() -> None:
    task = GraphCountingBridgeCountTask()
    out = task.generate(19601, params={'node_count': 9, 'target_count': 3, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'segment_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_bridge_counting'
    assert execution['question_format'] == 'count_bridge_edges'
    assert execution['graph_directionality'] == 'undirected'
    assert len(node_entities) == 9
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    adjacency_by_label = {str(key): [str(value) for value in values] for key, values in execution['adjacency_by_label'].items()}
    graph = _graph_from_trace_adjacency(adjacency_by_label)
    bridge_edges = sorted((tuple(sorted((str(left), str(right)), key=lambda value: int(value) if str(value).isdigit() else str(value))) for left, right in nx.bridges(graph)), key=lambda pair: (int(pair[0]) if str(pair[0]).isdigit() else str(pair[0]), int(pair[1]) if str(pair[1]).isdigit() else str(pair[1])))
    annotation_point_pairs = list(out.annotation_gt.value)
    assert int(out.answer_gt.value) == len(bridge_edges) == len(annotation_point_pairs)
    assert int(execution['target_count']) == len(bridge_edges)
    assert bridge_edges == [tuple((str(value) for value in edge)) for edge in execution['matching_edges']]
    assert trace['witness_symbolic']['type'] == 'edge_pair_set'
    assert trace['witness_symbolic']['edges'] == [list(edge) for edge in bridge_edges]
    assert 'edge_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'segment_set'
    assert trace['projected_annotation']['segment_set'] == annotation_point_pairs
    assert sum((1 for edge in edge_entities if bool(edge['is_bridge']))) == len(annotation_point_pairs)
    width, height = trace['render_spec']['canvas_size']
    for pair in annotation_point_pairs:
        assert len(pair) == 2
        for point in pair:
            assert 0 <= float(point[0]) <= float(width)
            assert 0 <= float(point[1]) <= float(height)

def test_graph_counting_bridge_count_prompt_examples_follow_label_variant() -> None:
    task = GraphCountingBridgeCountTask()
    letters = task.generate(19602, params={'label_variant': 'letters', 'node_count': 9, 'target_count': 2}, max_attempts=80)
    numbers = task.generate(19603, params={'label_variant': 'numbers', 'node_count': 9, 'target_count': 2}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[[180, 220], [310, 180]], [[310, 180], [430, 260]]], 'answer': 2}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_counting_bridge_count_supports_numeric_labels_and_named_colors() -> None:
    task = GraphCountingBridgeCountTask()
    out = task.generate(19604, params={'node_count': 10, 'target_count': 5, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    assert all((str(label).isdigit() for label in labels))
    assert execution['label_variant'] == 'numbers'
    assert execution['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert execution['node_color_name'] == 'orange'
    assert tuple(trace['render_spec']['style']['node_fill_rgb']) == tuple(named_color('orange'))

def test_graph_counting_bridge_count_balanced_sampling_defaults() -> None:
    task = GraphCountingBridgeCountTask()
    target_counts: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    for index in range(54):
        out = task.generate(hash64(19605, 'graph_counting_bridge_count', index), params={}, max_attempts=80)
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
