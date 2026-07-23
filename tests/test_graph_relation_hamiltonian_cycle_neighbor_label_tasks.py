"""Behavior tests for graph Hamiltonian-cycle neighbor relation task."""
from __future__ import annotations
from collections import Counter
import itertools
import networkx as nx
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.hamiltonian_cycle_neighbor_label import GraphRelationHamiltonianCycleNeighborLabelTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.shared.named_colors import named_color

def _graph_from_scene_entities(node_entities: list[dict[str, object]], edge_entities: list[dict[str, object]]) -> nx.Graph:
    graph = nx.Graph()
    for entity in node_entities:
        graph.add_node(str(entity["label"]))
    for entity in edge_entities:
        graph.add_edge(str(entity["node_u_label"]), str(entity["node_v_label"]))
    return graph

def _canonical_cycle_order(cycle: tuple[str, ...]) -> tuple[str, ...]:
    rotations: list[tuple[str, ...]] = []
    ordered_cycle = tuple((str(label) for label in cycle))
    for ordered in (ordered_cycle, tuple(reversed(ordered_cycle))):
        for index in range(len(ordered)):
            rotations.append(tuple(ordered[index:] + ordered[:index]))
    return min(rotations)

def _hamiltonian_cycles(graph: nx.Graph) -> tuple[tuple[str, ...], ...]:
    labels = tuple(sorted((str(node) for node in graph.nodes())))
    if len(labels) < 3:
        return ()
    anchor = labels[0]
    seen: set[tuple[str, ...]] = set()
    cycles: list[tuple[str, ...]] = []
    for suffix in itertools.permutations(labels[1:]):
        candidate = (anchor, *tuple((str(label) for label in suffix)))
        if all((graph.has_edge(candidate[index], candidate[(index + 1) % len(candidate)]) for index in range(len(candidate)))):
            canonical = _canonical_cycle_order(candidate)
            if canonical not in seen:
                seen.add(canonical)
                cycles.append(canonical)
    return tuple(sorted(cycles))

def _assert_task_contract(seed: int, *, query_id: str) -> None:
    task = GraphRelationHamiltonianCycleNeighborLabelTask()
    out = task.generate(seed, params={'node_count': 6, 'query_id': query_id, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    node_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge']
    assert 'task_graph__node_link__hamiltonian_cycle_neighbor_label' in TASK_REGISTRY
    assert out.scene_id == 'node_link'
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_hamiltonian_cycle_neighbor'
    assert execution['question_format'] == query_id
    assert execution['graph_directionality'] == 'undirected'
    assert execution['query_id'] == query_id
    assert len(node_entities) == 6
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant']['query_key'] == query_id
    assert 'Hamiltonian cycle' in str(out.prompt_variants['answer_only'])
    graph = _graph_from_scene_entities(node_entities, edge_entities)
    cycles = _hamiltonian_cycles(graph)
    matching_labels = [str(label) for label in execution['matching_labels']]
    annotation_points = list(out.annotation_gt.value)
    assert len(cycles) == 1
    assert len(matching_labels) == 6
    assert len(set(matching_labels)) == 6
    assert len(annotation_points) == 1
    assert all((graph.has_edge(matching_labels[index], matching_labels[(index + 1) % len(matching_labels)]) for index in range(len(matching_labels))))
    assert int(execution["edge_count"]) <= int(execution["node_count"]) + 2
    query_label = str(execution['query_label'])
    if str(execution['relation_mode']) == 'next':
        assert query_label == matching_labels[0] == str(execution['orientation_start_label'])
        assert str(execution['orientation_final_label']) == matching_labels[-1]
        expected_answer = matching_labels[1]
    else:
        assert query_label == matching_labels[-1] == str(execution['orientation_final_label'])
        assert str(execution['orientation_start_label']) == matching_labels[0]
        expected_answer = matching_labels[-2]
    assert str(out.answer_gt.value) == expected_answer == str(execution['answer_label'])
    assert trace['witness_symbolic']['labels'] == [expected_answer]
    answer_entity = [entity for entity in node_entities if str(entity["label"]) == expected_answer][0]
    assert annotation_points == [[float(answer_entity["center_px"][0]), float(answer_entity["center_px"][1])]]
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))

def test_graph_relation_hamiltonian_cycle_next_contract_matches_trace() -> None:
    _assert_task_contract(19701, query_id='next_in_hamiltonian_cycle_label')

def test_graph_relation_hamiltonian_cycle_previous_contract_matches_trace() -> None:
    _assert_task_contract(19702, query_id='previous_in_hamiltonian_cycle_label')

def test_graph_relation_hamiltonian_cycle_supports_numeric_labels_and_named_colors() -> None:
    task = GraphRelationHamiltonianCycleNeighborLabelTask()
    out = task.generate(19703, params={'node_count': 5, 'query_id': 'next_in_hamiltonian_cycle_label', 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    node_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    labels = [entity['label'] for entity in node_entities]
    style = trace["render_spec"]["style"]
    assert all((str(label).isdigit() for label in labels))
    assert 4 <= int(execution['node_count']) <= 6
    assert execution['label_variant'] == 'numbers'
    assert style['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert style['node_color_name'] == 'orange'
    assert {tuple(entity['fill_rgb']) for entity in node_entities} == {tuple(named_color('orange'))}

def test_graph_relation_hamiltonian_cycle_balanced_sampling_defaults() -> None:
    task = GraphRelationHamiltonianCycleNeighborLabelTask()
    node_counts: Counter[int] = Counter()
    query_ids: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    for index in range(48):
        out = task.generate(hash64(19704, 'graph_relation_hamiltonian_cycle_neighbor_label', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        style = out.trace_payload['render_spec']['style']
        node_counts[int(execution['node_count'])] += 1
        query_ids[str(execution['query_id'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(style['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_used'])] += 1
        assert 4 <= int(execution['node_count']) <= 6
        assert int(execution['edge_count']) <= int(execution['node_count']) + 2
    assert set(node_counts.keys()) == {4, 5, 6}
    assert set(query_ids.keys()) == {'next_in_hamiltonian_cycle_label', 'previous_in_hamiltonian_cycle_label'}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
