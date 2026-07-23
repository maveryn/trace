"""Behavior tests for graph named-node degree-value counting task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.named_node_degree_value import GraphCountingNamedNodeDegreeValueTask

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_counting_named_node_degree_value_undirected_contract_matches_trace() -> None:
    task = GraphCountingNamedNodeDegreeValueTask()
    out = task.generate(20301, params={'graph_directionality': 'undirected', 'degree_mode': 'degree', 'node_count': 7, 'target_degree': 2, 'label_variant': 'named', 'edge_routing_variant': 'mixed_arc', 'layout_variant': 'shell'}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert 'task_graph__node_link__named_node_degree_value' in TASK_REGISTRY
    assert out.scene_id == 'node_link'
    assert out.query_id == 'undirected_named_node_degree_value'
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 2
    assert out.annotation_gt.type == 'segment_set'
    assert len(out.annotation_gt.value) == 2
    assert trace['scene_ir']['scene_kind'] == 'graph_named_node_degree_value'
    assert execution['query_id'] == 'undirected_named_node_degree_value'
    assert execution['graph_directionality'] == 'undirected'
    assert execution['degree_mode'] == 'degree'
    assert execution['question_format'] == 'named_node_degree_value'
    assert execution['layout_variant_requested'] == 'shell'
    assert execution['edge_routing_variant'] == 'mixed_arc'
    assert execution['label_variant'] == 'named'
    assert 'node "' in str(out.prompt)
    query_label = str(execution['query_label'])
    counted_edges = [list(edge) for edge in execution['counted_edges']]
    adjacency_by_label = {str(label): {str(value) for value in neighbors} for label, neighbors in execution['adjacency_by_label'].items()}
    queried_degrees = {str(label): int(value) for label, value in execution['queried_degrees_by_label'].items()}
    assert queried_degrees[query_label] == int(out.answer_gt.value)
    assert trace['witness_symbolic']['query_label'] == query_label
    assert trace['witness_symbolic']['edges'] == counted_edges
    assert trace['projected_annotation']['type'] == 'segment_set'
    assert trace['projected_annotation']['segment_set'] == out.annotation_gt.value
    for left, right in counted_edges:
        assert query_label in {str(left), str(right)}
        other = str(right) if str(left) == query_label else str(left)
        assert other in adjacency_by_label[query_label]
        assert query_label in adjacency_by_label[other]
    assert sum((1 for node in node_entities if bool(node['is_query_node']))) == 1
    query_node = next((node for node in node_entities if bool(node['is_query_node'])))
    assert query_node['label'] == query_label
    assert int(query_node['queried_degree']) == 2
    assert int(query_node['incident_annotation_edge_count']) == 2
    assert sum((1 for edge in edge_entities if bool(edge['is_counted']))) == 2
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']

def test_graph_counting_named_node_degree_value_directed_modes_use_edge_direction() -> None:
    task = GraphCountingNamedNodeDegreeValueTask()
    expected_query_ids = {'in_degree': 'directed_named_node_in_degree_value', 'out_degree': 'directed_named_node_out_degree_value', 'total_degree': 'directed_named_node_total_degree_value'}
    for index, degree_mode in enumerate(('in_degree', 'out_degree', 'total_degree')):
        out = task.generate(20310 + index, params={'graph_directionality': 'directed', 'degree_mode': degree_mode, 'node_count': 8, 'target_degree': 2, 'label_variant': 'letters'}, max_attempts=100)
        trace = out.trace_payload
        execution = trace['execution_trace']
        query_label = str(execution['query_label'])
        counted_edges = [list(edge) for edge in execution['counted_edges']]
        queried_degrees = {str(label): int(value) for label, value in execution['queried_degrees_by_label'].items()}
        edge_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge']
        assert out.query_id == expected_query_ids[degree_mode]
        assert execution['query_id'] == expected_query_ids[degree_mode]
        assert execution['graph_directionality'] == 'directed'
        assert execution['degree_mode'] == degree_mode
        assert int(out.answer_gt.value) == queried_degrees[query_label] == 2
        assert len(counted_edges) == len(out.annotation_gt.value) == 2
        assert all((bool(edge['directed']) for edge in edge_entities))
        assert sum((1 for edge in edge_entities if bool(edge['is_counted']))) == 2
        if degree_mode == 'in_degree':
            assert all((str(right) == query_label for _, right in counted_edges))
        elif degree_mode == 'out_degree':
            assert all((str(left) == query_label for left, _ in counted_edges))
        else:
            assert all((query_label in {str(left), str(right)} for left, right in counted_edges))

def test_graph_counting_named_node_degree_value_prompt_examples_match_contract() -> None:
    task = GraphCountingNamedNodeDegreeValueTask()
    out = task.generate(20320, params={'graph_directionality': 'undirected', 'node_count': 8, 'target_degree': 1, 'label_variant': 'letters'}, max_attempts=100)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == [[[180, 220], [310, 180]], [[180, 220], [430, 260]]]
    assert answer_and_annotation['answer'] == 2

def test_graph_counting_named_node_degree_value_balanced_sampling_defaults() -> None:
    task = GraphCountingNamedNodeDegreeValueTask()
    query_ids: Counter[str] = Counter()
    directionality: Counter[str] = Counter()
    degree_modes: Counter[str] = Counter()
    target_degrees: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(96):
        out = task.generate(hash64(20330, 'graph_counting_named_node_degree_value', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        directionality[str(execution['graph_directionality'])] += 1
        degree_modes[str(execution['degree_mode'])] += 1
        target_degrees[int(execution['target_degree'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        edge_routing_variants[str(execution['edge_routing_variant'])] += 1
        assert 5 <= int(execution['node_count']) <= 10
        assert 0 <= int(execution['target_degree']) <= 4
        assert int(out.answer_gt.value) == int(execution['target_degree'])
    assert set(query_ids) == {'undirected_named_node_degree_value', 'directed_named_node_in_degree_value', 'directed_named_node_out_degree_value', 'directed_named_node_total_degree_value'}
    assert set(directionality) == {'undirected', 'directed'}
    assert set(degree_modes) == {'degree', 'in_degree', 'out_degree', 'total_degree'}
    assert set(target_degrees) == set(range(0, 5))
    assert set(label_variants) == {'letters', 'numbers', 'named'}
    assert set(edge_routing_variants) == {'straight', 'mixed_arc'}
