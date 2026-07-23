"""Behavior tests for graph topological-order endpoint task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.topological_endpoint_node_label import GraphOrderTopologicalEndpointNodeLabelTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.graph_algorithms import unique_topological_order_by_adjacency

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_order_topological_endpoint_contract_matches_trace() -> None:
    task = GraphOrderTopologicalEndpointNodeLabelTask()
    out = task.generate(19640, params={'node_count': 7, 'query_id': 'last_in_topological_order_label', 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    relations = trace['scene_ir']['relations']
    node_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'string'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_topological_endpoint_node_label'
    assert execution['question_format'] == 'last_in_topological_order_label'
    assert execution['graph_directionality'] == 'directed'
    assert int(execution['node_count']) == 7
    assert len(node_entities) == 7
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    successors_by_label = {str(key): tuple(str(value) for value in values) for key, values in relations['successors_by_label'].items()}
    order = tuple(str(label) for label in execution['topological_order_labels'])
    verified_order = unique_topological_order_by_adjacency(successors_by_label, node_order=order)
    assert verified_order is not None
    assert tuple(str(label) for label in verified_order) == order
    assert str(out.answer_gt.value) == str(order[-1])
    assert str(execution['answer_label']) == str(out.answer_gt.value)
    assert execution['matching_labels'] == [str(out.answer_gt.value)]
    assert trace['witness_symbolic']['type'] == 'object_set'
    assert trace['witness_symbolic']['labels'] == [str(out.answer_gt.value)]
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == out.annotation_gt.value
    width, height = trace['render_spec']['canvas_size']
    assert all(0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in out.annotation_gt.value)

def test_graph_order_topological_endpoint_prompt_examples_follow_contract() -> None:
    task = GraphOrderTopologicalEndpointNodeLabelTask()
    out = task.generate(19641, params={'label_variant': 'letters', 'node_count': 7, 'query_id': 'first_in_topological_order_label'}, max_attempts=80)
    example = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert example == {'annotation': [[180, 160]], 'answer': 'A'}

def test_graph_order_topological_endpoint_balanced_sampling_defaults() -> None:
    task = GraphOrderTopologicalEndpointNodeLabelTask()
    query_ids: Counter[str] = Counter()
    node_counts: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    for index in range(54):
        out = task.generate(hash64(19644, 'graph_order_topological_endpoint', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        query_ids[str(out.query_id)] += 1
        node_counts[int(execution['node_count'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        layout_variants[str(execution['layout_variant_used'])] += 1
        assert 3 <= int(execution['node_count']) <= 7
        order = list(execution['topological_order_labels'])
        if out.query_id == 'first_in_topological_order_label':
            assert str(out.answer_gt.value) == str(order[0])
        else:
            assert str(out.answer_gt.value) == str(order[-1])
    assert set(query_ids.keys()) == {'first_in_topological_order_label', 'last_in_topological_order_label'}
    assert set(node_counts.keys()) == {3, 4, 5, 6, 7}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(layout_variants.keys()).issubset(set(SUPPORTED_LAYOUT_VARIANTS))
