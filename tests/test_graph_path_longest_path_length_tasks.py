"""Behavior tests for graph longest-path-length task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.longest_path_length import GraphPathLongestPathLengthTask
from trace_tasks.tasks.graph.shared.graph_sample_types import graph_label_sort_key

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def _unique_longest_path(successors_by_label: dict[str, list[str]]) -> list[str] | None:
    labels = sorted(successors_by_label.keys(), key=graph_label_sort_key)
    indegree = {str(label): 0 for label in labels}
    for values in successors_by_label.values():
        for value in values:
            indegree[str(value)] = int(indegree.get(str(value), 0)) + 1
    queue = sorted([label for label, degree in indegree.items() if int(degree) == 0], key=graph_label_sort_key)
    topo: list[str] = []
    while queue:
        label = queue.pop(0)
        topo.append(str(label))
        for successor in sorted(successors_by_label.get(str(label), []), key=graph_label_sort_key):
            indegree[str(successor)] -= 1
            if int(indegree[str(successor)]) == 0:
                queue.append(str(successor))
                queue.sort(key=graph_label_sort_key)
    if len(topo) != len(indegree):
        return None
    best_length = {label: 0 for label in topo}
    best_count = {label: 1 for label in topo}
    best_path = {label: [label] for label in topo}
    for source in topo:
        for target in sorted(successors_by_label.get(str(source), []), key=graph_label_sort_key):
            candidate = int(best_length[str(source)]) + 1
            if int(candidate) > int(best_length[str(target)]):
                best_length[str(target)] = int(candidate)
                best_count[str(target)] = int(best_count[str(source)])
                best_path[str(target)] = [*best_path[str(source)], str(target)]
            elif int(candidate) == int(best_length[str(target)]):
                best_count[str(target)] += int(best_count[str(source)])
    max_length = max(best_length.values())
    endpoints = [label for label in topo if int(best_length[str(label)]) == int(max_length)]
    if sum((best_count[str(label)] for label in endpoints)) != 1:
        return None
    endpoint = next((label for label in endpoints if int(best_count[str(label)]) == 1))
    return [str(label) for label in best_path[str(endpoint)]]

def test_graph_path_longest_path_length_contract_matches_trace() -> None:
    task = GraphPathLongestPathLengthTask()
    out = task.generate(20701, params={'target_longest_path_length': 4, 'node_count': 8, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters', 'edge_routing_variant': 'mixed_arc'}, max_attempts=100)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert 'task_graph__node_link__longest_path_length' in TASK_REGISTRY
    assert out.scene_id == 'node_link'
    assert out.query_id == 'single'
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_sequence'
    assert trace['scene_ir']['scene_kind'] == 'graph_longest_path_length'
    assert execution['question_format'] == 'directed_longest_path_length'
    assert execution['query_id'] == 'single'
    assert execution['graph_directionality'] == 'directed'
    assert int(out.answer_gt.value) == 4
    assert int(execution['target_longest_path_length']) == 4
    assert len(node_entities) == 8
    assert len(edge_entities) == int(execution['edge_count'])
    assert all((bool(edge['directed']) for edge in edge_entities))
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    annotation_labels = [str(label) for label in execution['longest_path_labels']]
    annotation_path = list(out.annotation_gt.value)
    successors = {str(key): [str(value) for value in values] for key, values in execution['successors_by_label'].items()}
    assert _unique_longest_path(successors) == annotation_labels
    assert int(out.answer_gt.value) == len(annotation_path) - 1 == len(annotation_labels) - 1
    assert str(execution['source_label']) == str(annotation_labels[0])
    assert str(execution['goal_label']) == str(annotation_labels[-1])
    assert sum((1 for node in node_entities if bool(node['is_on_longest_path']))) == len(annotation_labels)
    assert sum((1 for edge in edge_entities if bool(edge['is_on_longest_path']))) == len(annotation_labels) - 1
    assert trace['witness_symbolic']['type'] == 'node_path'
    assert trace['witness_symbolic']['nodes'] == annotation_labels
    assert trace['projected_annotation']['type'] == 'point_sequence'
    assert trace['projected_annotation']['point_sequence'] == annotation_path
    assert trace['projected_annotation']['pixel_point_sequence'] == annotation_path

def test_graph_path_longest_path_prompt_examples_match_contract() -> None:
    task = GraphPathLongestPathLengthTask()
    out = task.generate(20702, params={'target_longest_path_length': 2, 'label_variant': 'numbers'}, max_attempts=100)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == [[180, 220], [310, 180], [430, 260]]
    assert answer_and_annotation['answer'] == 2

def test_graph_path_longest_path_balanced_sampling_defaults() -> None:
    task = GraphPathLongestPathLengthTask()
    query_ids: Counter[str] = Counter()
    target_lengths: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    for index in range(100):
        out = task.generate(hash64(20710, 'graph_path_longest_path_length', index), params={}, max_attempts=100)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        target_lengths[int(execution['target_longest_path_length'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        edge_routing_variants[str(execution['edge_routing_variant'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        assert execution['graph_directionality'] == 'directed'
        assert 5 <= int(execution['node_count']) <= 10
        assert 2 <= int(execution['target_longest_path_length']) <= 6
        assert int(execution['attachment_count']) >= 1
        assert int(out.answer_gt.value) == int(execution['target_longest_path_length'])
    assert set(query_ids) == {'single'}
    assert set(target_lengths) == {2, 3, 4, 5, 6}
    assert all((count > 0 for count in target_lengths.values()))
    assert set(label_variants) == {'letters', 'numbers', 'named'}
    assert set(edge_routing_variants) == {'straight', 'mixed_arc'}
    assert set(topology_profiles) == {'balanced', 'hub_heavy', 'low_degree'}
