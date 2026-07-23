"""Behavior tests for graph degree-count counting task."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.degree_value_filter_count import GraphCountingDegreeValueFilterCountTask
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS
from trace_tasks.tasks.shared.named_colors import named_color

def _extract_prompt_json_example(prompt: str) -> dict:
    marker = 'Example JSON:\n'
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)

def test_graph_counting_degree_count_contract_matches_trace() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    out = task.generate(19101, params={'query_id': 'undirected_degree_count', 'node_count': 7, 'query_degree': 2, 'target_count': 2, 'layout_variant': 'shell', 'topology_profile': 'balanced'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    scene_entities = trace['scene_ir']['entities']
    node_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_node']
    edge_entities = [entity for entity in scene_entities if entity['entity_kind'] == 'graph_edge']
    assert out.answer_gt.type == 'integer'
    assert int(out.answer_gt.value) == 2
    assert out.annotation_gt.type == 'point_set'
    assert len(out.annotation_gt.value) == 2
    assert trace['scene_ir']['scene_kind'] == 'graph_degree_counting'
    assert execution['question_format'] == 'count_nodes_with_degree'
    assert out.query_id == 'undirected_degree_count'
    assert execution['query_id'] == 'undirected_degree_count'
    assert execution['internal_query_id'] == 'degree_count'
    assert execution['graph_directionality'] == 'undirected'
    assert execution['degree_mode'] == 'degree'
    assert int(execution['node_count']) == 7
    assert int(execution['query_degree']) == 2
    assert int(execution['target_count']) == 2
    assert execution['layout_variant_requested'] == 'shell'
    assert execution['topology_profile'] == 'balanced'
    assert execution['label_variant'] in {'letters', 'numbers', 'named'}
    assert execution['node_shape_variant'] in {'circle', 'rounded_square', 'hexagon'}
    assert execution['layout_transform_variant'] in {'identity', 'rotate_90', 'rotate_180', 'rotate_270', 'mirror_left_right', 'mirror_up_down'}
    assert execution['node_color_name']
    assert trace['query_spec']['params']['query_id_probabilities']
    assert len(node_entities) == 7
    assert len(edge_entities) == int(execution['edge_count'])
    assert sorted(out.prompt_variants.keys()) == ['answer_and_annotation', 'answer_only']
    assert trace['query_spec']['prompt_variant_active_key'] == 'answer_and_annotation'
    degrees_by_label = {str(key): int(value) for key, value in execution['degrees_by_label'].items()}
    adjacency_by_label = {str(key): [str(value) for value in values] for key, values in execution['adjacency_by_label'].items()}
    assert execution['in_degrees_by_label'] == execution['degrees_by_label']
    assert execution['out_degrees_by_label'] == execution['degrees_by_label']
    assert sorted(degrees_by_label.keys()) == sorted((entity['label'] for entity in node_entities))
    for label, degree in degrees_by_label.items():
        assert int(degree) == len(adjacency_by_label[str(label)])
        for neighbor in adjacency_by_label[str(label)]:
            assert str(label) in adjacency_by_label[str(neighbor)]
    matching_labels = [str(value) for value in execution['matching_labels']]
    annotation_points = list(out.annotation_gt.value)
    assert matching_labels == trace['witness_symbolic']['labels']
    assert int(out.answer_gt.value) == len(matching_labels) == len(annotation_points)
    assert all((int(degrees_by_label[str(label)]) == 2 for label in matching_labels))
    assert 'label_set' not in trace['projected_annotation']
    assert trace['projected_annotation']['type'] == 'point_set'
    assert trace['projected_annotation']['point_set'] == annotation_points
    assert trace['projected_annotation']['pixel_point_set'] == annotation_points
    assert len(trace['projected_annotation']['pixel_bbox_set']) == 2
    width, height = trace['render_spec']['canvas_size']
    assert all((0 <= float(point[0]) <= float(width) and 0 <= float(point[1]) <= float(height) for point in annotation_points))
    assert trace['render_spec']['style']['node_shape_variant'] == execution['node_shape_variant']
    assert trace['render_spec']['style']['node_color_name'] == execution['node_color_name']
    assert int(trace['render_spec']['style']['resolved_label_font_size_px']) > 0
    assert int(trace['render_spec']['style']['label_stroke_width_px']) >= 1
    assert int(trace['render_spec']['style']['resolved_label_font_size_px']) <= int(trace['render_spec']['style']['label_font_size_px'])

def test_graph_counting_degree_count_prompt_example_matches_contract() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    out = task.generate(19102, params={'query_id': 'undirected_degree_count', 'node_count': 8, 'query_degree': 1, 'target_count': 2, 'label_variant': 'letters'}, max_attempts=80)
    answer_only = _extract_prompt_json_example(out.prompt_variants['answer_only'])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert answer_only == {'answer': 2}
    assert list(answer_and_annotation.keys()) == ['annotation', 'answer']
    assert answer_and_annotation['annotation'] == [[180, 220], [310, 180]]
    assert answer_and_annotation['answer'] == 2

def test_graph_counting_degree_count_supports_numeric_labels_and_named_colors() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    out = task.generate(19104, params={'query_id': 'undirected_degree_count', 'node_count': 10, 'query_degree': 1, 'target_count': 3, 'label_variant': 'numbers', 'node_shape_variant': 'hexagon', 'layout_transform_variant': 'rotate_90', 'node_color_name': 'orange'}, max_attempts=80)
    trace = out.trace_payload
    execution = trace['execution_trace']
    labels = [entity['label'] for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_node']
    assert all((str(label).isdigit() for label in labels))
    assert trace['witness_symbolic']['labels'] == sorted(trace['witness_symbolic']['labels'], key=lambda value: int(str(value)))
    assert execution['label_variant'] == 'numbers'
    assert execution['node_shape_variant'] == 'hexagon'
    assert execution['layout_transform_variant'] == 'rotate_90'
    assert execution['node_color_name'] == 'orange'
    assert trace['render_spec']['style']['node_shape_variant'] == 'hexagon'
    assert tuple(trace['render_spec']['style']['node_fill_rgb']) == tuple(named_color('orange'))
    prompt_example = _extract_prompt_json_example(out.prompt_variants['answer_and_annotation'])
    assert prompt_example['annotation'] == [[180, 220], [310, 180]]
    assert prompt_example['answer'] == 2

def test_graph_counting_degree_count_fits_numeric_labels_to_node_glyphs() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    common_params = {'query_id': 'undirected_degree_count', 'node_count': 10, 'query_degree': 1, 'target_count': 3, 'node_shape_variant': 'circle', 'node_radius_px': 18}
    letters = task.generate(19105, params={**common_params, 'label_variant': 'letters'}, max_attempts=80)
    numbers = task.generate(19106, params={**common_params, 'label_variant': 'numbers'}, max_attempts=80)
    letters_style = letters.trace_payload['render_spec']['style']
    numbers_style = numbers.trace_payload['render_spec']['style']
    assert int(numbers_style['resolved_label_font_size_px']) <= int(letters_style['resolved_label_font_size_px'])
    assert int(numbers_style['resolved_label_font_size_px']) >= 10
    assert int(numbers_style['label_stroke_width_px']) >= 1

def test_graph_counting_degree_count_directed_variants_use_in_out_degree_semantics() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    for degree_mode in ('in_degree', 'out_degree'):
        query_id = 'directed_in_degree_count' if degree_mode == 'in_degree' else 'directed_out_degree_count'
        out = task.generate(19107 if degree_mode == 'in_degree' else 19108, params={'query_id': query_id, 'degree_mode': degree_mode, 'node_count': 8, 'query_degree': 1, 'target_count': 2}, max_attempts=120)
        trace = out.trace_payload
        execution = trace['execution_trace']
        assert out.query_id == f'directed_{degree_mode}_count'
        assert execution['query_id'] == f'directed_{degree_mode}_count'
        assert execution['internal_query_id'] == 'directed_degree_count'
        assert execution['graph_directionality'] == 'directed'
        assert execution['degree_mode'] in {'in_degree', 'out_degree'}
        assert trace['query_spec']['params']['graph_directionality'] == 'directed'
        assert trace['scene_ir']['relations']['graph_directionality'] == 'directed'
        assert any((bool(entity['directed']) for entity in trace['scene_ir']['entities'] if entity['entity_kind'] == 'graph_edge'))
        if degree_mode == 'in_degree':
            assert execution['degree_mode'] == 'in_degree'
            queried = {str(key): int(value) for key, value in execution['in_degrees_by_label'].items()}
        else:
            assert execution['degree_mode'] == 'out_degree'
            queried = {str(key): int(value) for key, value in execution['out_degrees_by_label'].items()}
        matching_labels = [str(value) for value in execution['matching_labels']]
        assert trace['witness_symbolic']['labels'] == sorted(trace['witness_symbolic']['labels'], key=lambda value: int(str(value)) if str(value).isdigit() else str(value))
        assert len(out.annotation_gt.value) == len(matching_labels)
        assert all((int(queried[str(label)]) == 1 for label in matching_labels))

def test_graph_counting_degree_count_directed_variants_use_full_node_range() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    node_counts: Counter[int] = Counter()
    for degree_mode in ('in_degree', 'out_degree'):
        for index in range(80):
            query_id = 'directed_in_degree_count' if degree_mode == 'in_degree' else 'directed_out_degree_count'
            out = task.generate(hash64(19109 if degree_mode == 'in_degree' else 19110, degree_mode, index), params={'query_id': query_id, 'degree_mode': degree_mode}, max_attempts=120)
            node_count = int(out.trace_payload['execution_trace']['node_count'])
            assert 5 <= node_count <= 10
            node_counts[node_count] += 1
    assert 10 in node_counts

def test_graph_counting_degree_count_balanced_sampling_defaults() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    query_ids: Counter[str] = Counter()
    degree_modes: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    targets_by_mode: defaultdict[str, Counter[int]] = defaultdict(Counter)
    query_degrees: Counter[int] = Counter()
    layout_variants: Counter[str] = Counter()
    topology_profiles: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    node_shape_variants: Counter[str] = Counter()
    layout_transform_variants: Counter[str] = Counter()
    node_colors: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(hash64(19103, 'graph_counting_degree_count', index), params={}, max_attempts=80)
        execution = out.trace_payload['execution_trace']
        query_ids[str(execution['query_id'])] += 1
        degree_modes[str(execution['degree_mode'])] += 1
        target_counts[int(execution['target_count'])] += 1
        targets_by_mode[str(execution['degree_mode'])][int(execution['target_count'])] += 1
        if 'query_degree' in execution:
            query_degrees[int(execution['query_degree'])] += 1
        layout_variants[str(execution['layout_variant_requested'])] += 1
        topology_profiles[str(execution['topology_profile'])] += 1
        label_variants[str(execution['label_variant'])] += 1
        node_shape_variants[str(execution['node_shape_variant'])] += 1
        layout_transform_variants[str(execution['layout_transform_variant'])] += 1
        node_colors[str(execution['node_color_name'])] += 1
        assert 5 <= int(execution['node_count']) <= 10
        if 'query_degree' in execution:
            assert 0 <= int(execution['query_degree']) <= 4
        assert 0 <= int(execution['target_count']) <= 5
    assert set(query_ids.keys()) == {'undirected_degree_count', 'directed_in_degree_count', 'directed_out_degree_count'}
    assert set(degree_modes.keys()) == {'degree', 'in_degree', 'out_degree'}
    assert min(target_counts.keys()) == 0
    assert max(target_counts.keys()) <= 5
    for degree_mode in ('degree', 'in_degree', 'out_degree'):
        assert len(targets_by_mode[degree_mode]) >= 4
    assert set(query_degrees.keys()).issubset(set(range(0, 5)))
    assert {0, 1}.issubset(set(query_degrees.keys()))
    assert set(layout_variants.keys()) == set(SUPPORTED_LAYOUT_VARIANTS)
    assert set(topology_profiles.keys()) == {'balanced', 'hub_heavy', 'low_degree'}
    assert set(label_variants.keys()) == {'letters', 'numbers', 'named'}
    assert set(node_shape_variants.keys()) == {'circle', 'rounded_square', 'hexagon'}
    assert set(layout_transform_variants.keys()) == {'identity', 'rotate_90', 'rotate_180', 'rotate_270', 'mirror_left_right', 'mirror_up_down'}
    assert set(node_colors.keys()) == {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'maroon'}

def test_graph_counting_degree_count_node_support_is_not_locked_to_smallest_feasible_count() -> None:
    task = GraphCountingDegreeValueFilterCountTask()
    node_counts: Counter[int] = Counter()
    for index in range(120):
        out = task.generate(hash64(19111, 'graph_counting_degree_count_node_balance', index), params={'query_id': 'undirected_degree_count'}, max_attempts=120)
        node_counts[int(out.trace_payload['execution_trace']['node_count'])] += 1
    assert set(node_counts) >= {5, 6, 7, 8, 9, 10}
    assert node_counts[5] < 90
