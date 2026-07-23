from ...registry import register_task
from ._lifecycle import NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_degree_count_graph
TASK_ID = 'task_graph__node_link__degree_value_filter_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('undirected_degree_count', 'directed_in_degree_count', 'directed_out_degree_count')

def _mode(query_id):
    return 'in_degree' if 'in_degree' in str(query_id) else 'out_degree' if 'out_degree' in str(query_id) else None

def _prompt_key(axes):
    mode = _mode(axes.query_id)
    return 'in_degree_count' if mode == 'in_degree' else 'out_degree_count' if mode == 'out_degree' else 'degree_count'

def _hint_key(axes):
    return 'annotation_hint_' + _prompt_key(axes)

def _sample_graph(rng, axes, attempts):
    directed = str(axes.query_id).startswith('directed')
    target_count = int(axes.values['target_count'])
    return sample_degree_count_graph(rng, query_id='directed_degree_count' if directed else 'degree_count', degree_mode=_mode(axes.query_id), node_count=int(axes.node_count), query_degree=int(axes.values['query_degree']), target_count=int(target_count), max_degree=4, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), search_attempts=int(attempts))

def _build_objective_plan():
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingDegreeValueFilterCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key=_prompt_key, annotation_hint_key=_hint_key, graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_degree_value_filter_counting', question_format=lambda axes: str(axes.query_id), value_ranges={'target_count': (1, 4), 'query_degree': (1, 4)}, prompt_bundle_id='graph_node_link_counting_v1', prompt_scene_key='single_graph_counting', prompt_task_key='degree_count_query')

@register_task
class GraphCountingDegreeValueFilterCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _build_objective_plan(self):
        return _build_objective_plan()

    def generate(self, instance_seed, *, params, max_attempts):
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
