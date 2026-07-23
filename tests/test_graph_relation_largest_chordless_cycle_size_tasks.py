"""Behavior tests for graph largest-chordless-cycle relation task."""
from __future__ import annotations
from collections import Counter
import networkx as nx
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.largest_chordless_cycle_size import GraphRelationLargestChordlessCycleSizeTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.named_colors import named_color

def _graph_from_scene_entities(node_entities: list[dict[str, object]], edge_entities: list[dict[str, object]]) -> nx.Graph:
    graph = nx.Graph()
    for entity in node_entities:
        graph.add_node(str(entity["label"]))
    for entity in edge_entities:
        graph.add_edge(str(entity["node_u_label"]), str(entity["node_v_label"]))
    return graph

def _is_chordless_cycle(graph: nx.Graph, labels: list[str]) -> bool:
    if len(labels) < 3 or len(set(labels)) != len(labels):
        return False
    cycle_edges = {frozenset((labels[index], labels[(index + 1) % len(labels)])) for index in range(len(labels))}
    for edge in cycle_edges:
        left, right = tuple(edge)
        if not graph.has_edge(str(left), str(right)):
            return False
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            edge = frozenset((labels[left_index], labels[right_index]))
            if edge in cycle_edges:
                continue
            if graph.has_edge(str(labels[left_index]), str(labels[right_index])):
                return False
    return True

def test_graph_relation_largest_chordless_cycle_size_contract_matches_trace() -> None:
    task = GraphRelationLargestChordlessCycleSizeTask()
    out = task.generate(19601, params={'node_count': 9, 'target_cycle_size': 5, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    node_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_largest_chordless_cycle'
    assert execution['question_format'] == 'largest_chordless_cycle_size'
    assert execution['graph_directionality'] == 'undirected'
    assert len(node_entities) == 9
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant']['query_key'] == 'largest_chordless_cycle_size'
    assert 'chordless' in str(out.prompt_variants['answer_only']).lower()
    graph = _graph_from_scene_entities(node_entities, edge_entities)
    matching_labels = [str(label) for label in execution['matching_labels']]
    annotation_points = list(out.annotation_gt.value)
    assert int(out.answer_gt.value) == len(matching_labels) == len(annotation_points)
    assert int(execution['target_cycle_size']) == int(out.answer_gt.value)
    assert max((int(size) for size in execution['chordless_cycle_sizes'])) == int(out.answer_gt.value)
    assert _is_chordless_cycle(graph, matching_labels)
    assert trace['witness_symbolic']['labels'] == matching_labels
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))

def test_graph_relation_largest_chordless_cycle_size_supports_numeric_labels_and_named_colors() -> None:
    task = GraphRelationLargestChordlessCycleSizeTask()
    out = task.generate(19602, params={'node_count': 10, 'target_cycle_size': 6, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    style = trace['render_spec']['style']
    assert all((str(label).isdigit() for label in labels))
    assert int(out.answer_gt.value) == 6
    assert execution['label_variant'] == 'numbers'
    assert style['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert style['node_color_name'] == 'orange'
    assert {tuple(entity['fill_rgb']) for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node'} == {tuple(named_color('orange'))}

def test_graph_relation_largest_chordless_cycle_size_balanced_sampling_defaults() -> None:
    task = GraphRelationLargestChordlessCycleSizeTask()
    target_sizes: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    for index in range(48):
        out = task.generate(hash64(19603, 'graph_relation_largest_chordless_cycle_size', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        target_sizes[int(execution['target_cycle_size'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(out.trace_payload['render_spec']['style']['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_used'])] += 1
        assert 8 <= int(execution['node_count']) <= 10
        assert 3 <= int(execution['target_cycle_size']) <= 7
        assert max((int(size) for size in execution['chordless_cycle_sizes'])) == int(execution['target_cycle_size'])
    assert set(target_sizes.keys()).issuperset({3, 4, 5, 6, 7})
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()).issubset(set(SUPPORTED_LAYOUT_VARIANTS))
    assert len(layout_variants) >= 3
