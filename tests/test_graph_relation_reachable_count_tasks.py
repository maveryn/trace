"""Behavior tests for graph reachable-count relation task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.reachable_count import GraphRelationReachableCountTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.graph_algorithms import bfs_dist_count_by_adjacency
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_relation_reachable_count_contract_matches_trace() -> None:
    task = GraphRelationReachableCountTask()
    out = task.generate(19701, params={'node_count': 8, 'target_reachable_count': 4, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_set'
    assert trace['scene_ir']['scene_kind'] == 'graph_reachable_count'
    assert execution['question_format'] == 'reachable_count'
    assert execution['graph_directionality'] == 'directed'
    assert 5 <= int(execution['node_count']) <= 15
    assert 1 <= int(execution['target_reachable_count']) <= 6
    assert len(node_entities) == 8
    assert len(edge_entities) == int(execution['edge_count'])
    assert all((bool(edge['directed']) for edge in edge_entities))
    assert str(execution['query_label']) not in set(execution['matching_labels'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant']['query_key'] == 'reachable_count'
    assert 'reachable' in str(out.prompt).lower()
    assert 'arrow' in str(out.prompt).lower() or 'directed' in str(out.prompt).lower()
    successors = {str(key): [str(value) for value in values] for key, values in execution['successors_by_label'].items()}
    dist_start, _ = bfs_dist_count_by_adjacency(successors, start=str(execution['query_label']))
    reachable_labels = sorted((str(label) for label in dist_start.keys() if str(label) != str(execution['query_label'])), key=lambda value: int(value) if str(value).isdigit() else str(value))
    annotation_points = list(out.annotation_gt.value)
    assert int(out.answer_gt.value) == len(reachable_labels) == len(annotation_points)
    assert reachable_labels == [str(value) for value in execution['matching_labels']]
    assert trace['witness_symbolic']['labels'] == reachable_labels
    assert 'label_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    assert len(trace['projected_annotation']['pixel_bbox_set']) == len(annotation_points)
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))

def test_graph_relation_reachable_count_prompt_examples_follow_label_variant() -> None:
    task = GraphRelationReachableCountTask()
    letters = task.generate(19702, params={'label_variant': 'letters', 'node_count': 8, 'target_reachable_count': 4}, max_attempts=80)
    numbers = task.generate(19703, params={'label_variant': 'numbers', 'node_count': 8, 'target_reachable_count': 4}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[180, 220], [310, 180]], 'answer': 2}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_relation_reachable_count_supports_numeric_labels_and_named_colors() -> None:
    task = GraphRelationReachableCountTask()
    out = task.generate(19704, params={'node_count': 9, 'target_reachable_count': 5, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
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

def test_graph_relation_reachable_count_balanced_sampling_defaults() -> None:
    task = GraphRelationReachableCountTask()
    target_counts: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    for index in range(48):
        out = task.generate(hash64(19705, 'graph_relation_reachable_count', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        target_counts[int(execution['target_reachable_count'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        assert 5 <= int(execution['node_count']) <= 15
        assert 1 <= int(execution['target_reachable_count']) <= 6
        assert int(execution['target_reachable_count']) < int(execution['node_count'])
        assert str(execution['query_label']) not in set(execution['matching_labels'])
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
    assert min(target_counts.keys()) >= 1
    assert max(target_counts.keys()) <= 6
