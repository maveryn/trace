"""Behavior tests for graph shortest-path-length task."""
from __future__ import annotations
import json
from collections import Counter
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.shortest_path_length import GraphPathShortestPathLengthTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.graph_algorithms import bfs_dist_count_by_adjacency, reconstruct_unique_shortest_path_by_adjacency
from trace_tasks.tasks.shared.named_colors import named_color
FULL_NODE_LINK_LAYOUT_VARIANTS = set(SUPPORTED_LAYOUT_VARIANTS)

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_path_shortest_path_length_contract_matches_trace() -> None:
    task = GraphPathShortestPathLengthTask()
    out = task.generate(19601, params={'query_id': 'undirected_shortest_path_length', 'node_count': 8, 'target_shortest_path_length': 4, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert out.annotation_gt.type == 'point_sequence'
    assert trace['scene_ir']['scene_kind'] == 'graph_shortest_path_length'
    assert execution['question_format'] == 'count_edges_in_unique_shortest_path'
    assert out.query_id == 'undirected_shortest_path_length'
    assert execution['query_id'] == 'undirected_shortest_path_length'
    assert execution['internal_query_id'] == 'shortest_path_length'
    assert execution['graph_directionality'] == 'undirected'
    assert 5 <= int(execution['node_count']) <= 15
    assert 2 <= int(execution['target_shortest_path_length']) <= 5
    annotation_labels = list((str(label) for label in execution['shortest_path_labels']))
    annotation_path = list(out.annotation_gt.value)
    assert int(out.answer_gt.value) == len(annotation_path) - 1
    assert str(execution['source_label']) == str(annotation_labels[0])
    assert str(execution['goal_label']) == str(annotation_labels[-1])
    assert len(node_entities) == 8
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    adjacency_by_label = {str(key): [str(value) for value in values] for key, values in execution['adjacency_by_label'].items()}
    dist_start, count_start = bfs_dist_count_by_adjacency(adjacency_by_label, start=str(execution['source_label']))
    dist_goal, _ = bfs_dist_count_by_adjacency(adjacency_by_label, start=str(execution['goal_label']))
    reconstructed = reconstruct_unique_shortest_path_by_adjacency(adjacency_by_label, start=str(execution['source_label']), goal=str(execution['goal_label']), dist_start=dist_start, dist_goal=dist_goal)
    assert reconstructed == annotation_labels
    assert int(dist_start[str(execution['goal_label'])]) == int(out.answer_gt.value)
    assert int(count_start[str(execution['goal_label'])]) == 1
    assert trace['witness_symbolic']['type'] == 'node_path'
    assert trace['witness_symbolic']['nodes'] == annotation_labels
    assert 'label_path' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_sequence'
    assert trace['projected_annotation']['point_sequence'] == annotation_path
    assert trace['projected_annotation']['pixel_point_sequence'] == annotation_path
    assert len(trace['projected_annotation']['pixel_bbox_set']) == len(annotation_path)
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_path))

def test_graph_path_directed_shortest_path_contract_matches_trace() -> None:
    task = GraphPathShortestPathLengthTask()
    out = task.generate(19611, params={'query_id': 'directed_shortest_path_length', 'node_count': 9, 'target_shortest_path_length': 4, 'layout_variant': 'shell', 'topology_profile': 'balanced', 'label_variant': 'letters'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.query_id == 'directed_shortest_path_length'
    assert execution['query_id'] == 'directed_shortest_path_length'
    assert execution['internal_query_id'] == 'directed_shortest_path_length'
    assert execution['graph_directionality'] == 'directed'
    assert trace['query_spec']['params']['graph_directionality'] == 'directed'
    assert trace['scene_ir']['relations']['graph_directionality'] == 'directed'
    assert all((bool(edge['directed']) for edge in edge_entities))
    assert 5 <= int(execution['node_count']) <= 15
    assert 2 <= int(execution['target_shortest_path_length']) <= 5
    assert int(out.answer_gt.value) == 4
    successors = {str(key): [str(value) for value in values] for key, values in trace['scene_ir']['relations']['adjacency_by_label'].items()}
    predecessors = {str(entity['label']): [str(value) for value in entity['predecessors']] for entity in scene_entities if entity['entity_kind'] == 'graph_node'}
    dist_start, count_start = bfs_dist_count_by_adjacency(successors, start=str(execution['source_label']))
    dist_goal, _ = bfs_dist_count_by_adjacency(predecessors, start=str(execution['goal_label']))
    reconstructed = reconstruct_unique_shortest_path_by_adjacency(successors, start=str(execution['source_label']), goal=str(execution['goal_label']), dist_start=dist_start, dist_goal=dist_goal)
    assert reconstructed == list((str(label) for label in execution['shortest_path_labels']))
    assert int(dist_start[str(execution['goal_label'])]) == int(out.answer_gt.value)
    assert int(count_start[str(execution['goal_label'])]) == 1
    assert trace['query_spec']['prompt_variant']['query_key'] == 'directed_shortest_path_length'
    assert 'arrow' in str(out.prompt).lower() or 'directed' in str(out.prompt).lower()

def test_graph_path_shortest_path_prompt_examples_follow_label_variant() -> None:
    task = GraphPathShortestPathLengthTask()
    letters = task.generate(19602, params={'query_id': 'undirected_shortest_path_length', 'label_variant': 'letters', 'node_count': 8, 'target_shortest_path_length': 3}, max_attempts=80)
    numbers = task.generate(19603, params={'query_id': 'undirected_shortest_path_length', 'label_variant': 'numbers', 'node_count': 8, 'target_shortest_path_length': 3}, max_attempts=80)
    letters_example = _extract_prompt_json_example(letters.prompt_variants['answer_and_annotation'])
    numbers_example = _extract_prompt_json_example(numbers.prompt_variants['answer_and_annotation'])
    expected_example = {'annotation': [[180, 220], [310, 180], [430, 260]], 'answer': 2}
    assert letters_example == expected_example
    assert numbers_example == expected_example

def test_graph_path_shortest_path_supports_numeric_labels_and_named_colors() -> None:
    task = GraphPathShortestPathLengthTask()
    out = task.generate(19604, params={'query_id': 'undirected_shortest_path_length', 'node_count': 10, 'target_shortest_path_length': 5, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    assert all((str(label).isdigit() for label in labels))
    assert execution['label_variant'] == 'numbers'
    assert execution['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert execution['node_color_name'] == 'orange'
    assert tuple(trace['render_spec']['style']['node_fill_rgb']) == tuple(named_color('orange'))

def test_graph_path_shortest_path_balanced_sampling_defaults() -> None:
    task = GraphPathShortestPathLengthTask()
    query_ids: Counter[str] = Counter()
    target_lengths: Counter[int] = Counter()
    target_lengths_by_variant: dict[str, Counter[int]] = {}
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    node_shapes: Counter[str] = Counter()
    layout_transforms: Counter[str] = Counter()
    node_colors: Counter[str] = Counter()
    for index in range(100):
        out = task.generate(hash64(19605, 'graph_path_shortest_path_length', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        target_lengths[int(execution['target_shortest_path_length'])] += 1
        target_lengths_by_variant.setdefault(str(execution['query_id']), Counter())[int(execution['target_shortest_path_length'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shapes[str(execution['node_shape_variant'])] += 1
        layout_transforms[str(execution['layout_transform_variant'])] += 1
        node_colors[str(execution['node_color_name'])] += 1
        assert 5 <= int(execution['node_count']) <= 15
        assert 2 <= int(execution['target_shortest_path_length']) <= 5
        assert int(execution['attachment_count']) >= 1
        assert int(out.answer_gt.value) == int(execution['target_shortest_path_length'])
    assert set(query_ids.keys()) == {'undirected_shortest_path_length', 'directed_shortest_path_length'}
    assert set(target_lengths.keys()) == {2, 3, 4, 5}
    assert {variant: set(lengths.keys()) for variant, lengths in target_lengths_by_variant.items()} == {'undirected_shortest_path_length': {2, 3, 4, 5}, 'directed_shortest_path_length': {2, 3, 4, 5}}
    assert set(layout_variants.keys()) == FULL_NODE_LINK_LAYOUT_VARIANTS
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shapes.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_transforms.keys()) == {'identity', 'rotate_90', 'rotate_180', 'rotate_270', 'mirror_left_right', 'mirror_up_down'}
    assert set(node_colors.keys()) == {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'maroon'}
