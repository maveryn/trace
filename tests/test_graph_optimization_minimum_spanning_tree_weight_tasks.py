"""Behavior tests for graph minimum-spanning-tree-weight task."""
from __future__ import annotations
import json
from collections import Counter
import networkx as nx
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.mst_weight import GraphOptimizationMinimumSpanningTreeWeightTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.graph.shared.graph_scene import _segment_intersects_bbox
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _weighted_graph_from_trace(trace: dict) -> nx.Graph:
    graph = nx.Graph()
    for entity in trace['scene_ir']['entities']:
        if entity['entity_kind'] == 'graph_node':
            graph.add_node(str(entity['label']))
    for edge_info in trace['execution_trace']['edge_weights_by_label']:
        left, right = [str(value) for value in edge_info['endpoints']]
        graph.add_edge(left, right, weight=int(edge_info['weight']))
    return graph

def test_graph_optimization_minimum_spanning_tree_weight_contract_matches_trace() -> None:
    task = GraphOptimizationMinimumSpanningTreeWeightTask()
    out = task.generate(19601, params={'node_count': 7, 'extra_edge_count': 2, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'segment_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_minimum_spanning_tree_weight'
    assert execution['question_format'] == 'sum_unique_mst_weights'
    assert execution['graph_directionality'] == 'undirected'
    assert int(execution['node_count']) == 7
    assert int(execution['extra_edge_count']) == 2
    assert len(node_entities) == 7
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    weighted_graph = _weighted_graph_from_trace(trace)
    mst_graph = nx.minimum_spanning_tree(weighted_graph, weight='weight', algorithm='kruskal')
    mst_edges = sorted((tuple(sorted((str(left), str(right)), key=lambda value: int(value) if str(value).isdigit() else str(value))) for left, right in mst_graph.edges()), key=lambda pair: (int(pair[0]) if str(pair[0]).isdigit() else str(pair[0]), int(pair[1]) if str(pair[1]).isdigit() else str(pair[1])))
    annotation_point_pairs = list(out.annotation_gt.value)
    assert mst_edges == [tuple((str(value) for value in edge)) for edge in execution['minimum_spanning_tree_edges']]
    assert trace['witness_symbolic']['type'] == 'edge_pair_set'
    assert trace['witness_symbolic']['edges'] == [list(edge) for edge in mst_edges]
    assert len(annotation_point_pairs) == len(mst_edges)
    assert int(out.answer_gt.value) == int(sum((int(data['weight']) for _, _, data in mst_graph.edges(data=True))))
    assert int(out.answer_gt.value) == int(execution['minimum_spanning_tree_total_weight'])
    assert 'edge_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'segment_set'
    assert trace['projected_annotation']['segment_set'] == annotation_point_pairs
    assert sum((1 for edge in edge_entities if bool(edge['is_in_minimum_spanning_tree']))) == len(annotation_point_pairs)
    width, height = trace['render_spec']['canvas_size']
    for pair in annotation_point_pairs:
        assert len(pair) == 2
        for point in pair:
            assert 0 <= float(point[0]) <= float(width)
            assert 0 <= float(point[1]) <= float(height)
    all_weights = [int(edge['weight']) for edge in edge_entities]
    assert len(all_weights) == len(set(all_weights))
    assert all((1 <= int(weight) <= 9 for weight in all_weights))
    assert all((edge['weight_label_bbox_xyxy'] is not None for edge in edge_entities))

def test_graph_optimization_minimum_spanning_tree_weight_prompt_examples_follow_label_variant() -> None:
    task = GraphOptimizationMinimumSpanningTreeWeightTask()
    letters = task.generate(19602, params={'label_variant': 'letters', 'node_count': 7, 'extra_edge_count': 2}, max_attempts=80)
    numbers = task.generate(19603, params={'label_variant': 'numbers', 'node_count': 7, 'extra_edge_count': 2}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[[180, 220], [310, 180]], [[310, 180], [430, 260]], [[430, 260], [520, 340]]], 'answer': 12}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_optimization_minimum_spanning_tree_weight_supports_numeric_labels_and_named_colors() -> None:
    task = GraphOptimizationMinimumSpanningTreeWeightTask()
    out = task.generate(19604, params={'node_count': 7, 'extra_edge_count': 1, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    assert all((str(label).isdigit() for label in labels))
    assert execution['label_variant'] == 'numbers'
    assert execution['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert execution['node_color_name'] == 'orange'
    assert tuple(trace['render_spec']['style']['node_fill_rgb']) == tuple(named_color('orange'))

def test_graph_optimization_minimum_spanning_tree_weight_balanced_sampling_defaults() -> None:
    task = GraphOptimizationMinimumSpanningTreeWeightTask()
    node_counts: Counter[int] = Counter()
    extra_edge_counts: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    node_colors: Counter[str] = Counter()
    for index in range(60):
        out = task.generate(hash64(19605, 'graph_optimization_minimum_spanning_tree_weight', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        node_counts[int(execution['node_count'])] += 1
        extra_edge_counts[int(execution['extra_edge_count'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        node_colors[str(execution['node_color_name'])] += 1
        assert 4 <= int(execution['node_count']) <= 7
        assert 1 <= int(execution['extra_edge_count']) <= 2
        assert 1 <= int(execution['edge_weight_min']) <= int(execution['edge_weight_max']) <= 9
    assert set(node_counts.keys()) == {4, 5, 6, 7}
    assert set(extra_edge_counts.keys()) == {1, 2}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
    assert set(node_colors.keys()) == {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'maroon'}

def test_graph_optimization_minimum_spanning_tree_weight_weight_labels_avoid_edge_crossings() -> None:
    task = GraphOptimizationMinimumSpanningTreeWeightTask()
    out = task.generate(5710676742424900, params={'node_count': 7, 'extra_edge_count': 2, 'layout_variant': 'shell', 'topology_profile': 'hub_heavy', 'label_variant': 'letters', 'node_shape_variant': 'circle', 'layout_transform_variant': 'rotate_180', 'node_color_name': 'green'}, max_attempts=80)
    edge_entities = [entity for entity in out.trace_payload['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge']
    assert edge_entities
    for edge_entity in edge_entities:
        label_box = edge_entity['weight_label_bbox_xyxy']
        assert label_box is not None
        for other_edge_entity in edge_entities:
            other_segment = (tuple((int(value) for value in other_edge_entity['segment_px'][0])), tuple((int(value) for value in other_edge_entity['segment_px'][1])))
            assert not _segment_intersects_bbox(other_segment, tuple((int(value) for value in label_box)))
