from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from ._lifecycle import build_integer_angle_relation_trace, render_angle_relation_runtime, select_angle_relation_case
from .shared.construction import make_triangle_exterior_case
from .shared.sampling import triangle_exterior_parameters_for_answer
from .shared.state import DOMAIN, SCENE_ID
TASK_ID = 'task_geometry__angle_relations__triangle_exterior_angle'
TRIANGLE_EXTERIOR_QUERY_ID = 'triangle_exterior_angle'
SUPPORTED_QUERY_IDS = (TRIANGLE_EXTERIOR_QUERY_ID,)
TRIANGLE_EXTERIOR_CASE_SUPPORT = tuple((triangle_exterior_parameters_for_answer(answer_value, variant_index=11) for answer_value in range(35, 108)))
TRIANGLE_EXTERIOR_CASES = tuple((make_triangle_exterior_case(*values) for values in TRIANGLE_EXTERIOR_CASE_SUPPORT))
_RENDER_DEFAULTS = load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)[1]

@register_task
class GeometryAngleRelationsTriangleExteriorAngleTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts) -> TaskOutput:
        triangle_branch, branch_weights, resolved_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=TRIANGLE_EXTERIOR_QUERY_ID, task_id=TASK_ID)
        branch_name = str(triangle_branch)
        case, case_index = select_angle_relation_case(cases=TRIANGLE_EXTERIOR_CASES, params=resolved_params, instance_seed=int(instance_seed), namespace=f'{TASK_ID}.case')
        runtime = render_angle_relation_runtime(case=case, case_index=int(case_index), prompt_query_key=branch_name, instance_seed=int(instance_seed), params=resolved_params, render_defaults=_RENDER_DEFAULTS, max_attempts=int(max_attempts))
        answer_gt = TypedValue(type='integer', value=int(runtime.rendered_context.rendered_scene.answer))
        annotation_gt = TypedValue(type=str(runtime.annotation_artifacts.annotation_type), value=runtime.annotation_artifacts.value)
        trace_payload = build_integer_angle_relation_trace(runtime=runtime, branch_name=branch_name, branch_probabilities=branch_weights, answer_value=int(answer_gt.value))
        return TaskOutput(str(runtime.prompt_artifacts.prompt), answer_gt, annotation_gt, runtime.rendered_context.image, 'img0', trace_payload, default_task_versions(), SCENE_ID, branch_name, dict(runtime.prompt_artifacts.prompt_variants))
__all__ = ['GeometryAngleRelationsTriangleExteriorAngleTask']
