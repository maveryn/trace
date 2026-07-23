"""Compute a shared radius from compound cylinder-cone volume data."""

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import build_solid_formula_plan, prepare_solid_formula_task_parts
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import answer_support_probability_map, decimal_support, round1
from .shared.rendering import render_cylinder_cone_radius
from .shared.sampling import select_case_option, select_support_value
from .shared.state import SolidFormulaProblem

TASK_ID = "task_geometry__solid_formula__cylinder_cone_radius_from_volume_heights"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = "cylinder_cone_radius_from_volume_heights"
ANSWER_SUPPORT = decimal_support(2, 61, step=1)
CONSTRUCTION_OPTIONS = (
    (6.0, 3.0),
    (8.0, 6.0),
    (10.0, 9.0),
    (12.0, 6.0),
    (14.0, 12.0),
    (16.0, 9.0),
)


@register_task
class GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate one cylinder-cone radius task with task-owned output binding."""

        query_id, query_distribution, resolved_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        radius_value = select_support_value(
            instance_seed=int(instance_seed),
            params=resolved_params,
            namespace=f"{TASK_ID}.{query_id}.answer",
            support=ANSWER_SUPPORT,
        )
        (cylinder_height, cone_height), case_count = select_case_option(
            instance_seed=int(instance_seed),
            params=resolved_params,
            namespace=f"{TASK_ID}.{query_id}.construction",
            options=CONSTRUCTION_OPTIONS,
        )
        total_height = cylinder_height + cone_height
        volume_pi_multiple = radius_value**2 * (cylinder_height + (cone_height / 3.0))
        answer_distribution = answer_support_probability_map(ANSWER_SUPPORT, radius_value)
        problem = SolidFormulaProblem(
            solid_kind="cylinder_cone",
            answer=round1(radius_value),
            unknown_dimension="radius",
            formula_family="cylinder_cone_radius_from_volume_heights",
            formula="V = pi r^2(H-c) + (1/3)pi r^2c, solve r from V, total height H, and cone height c",
            radius=round1(radius_value),
            total_height=round1(total_height),
            cylinder_height=round1(cylinder_height),
            cone_height=round1(cone_height),
            volume_pi_multiple=round1(volume_pi_multiple),
            answer_support_probabilities=answer_distribution,
            construction_case_count_for_answer=case_count,
        )
        plan = build_solid_formula_plan(
            prompt_key=PROMPT_KEY,
            problem=problem,
            render_scene=render_cylinder_cone_radius,
            branch_probabilities=query_distribution,
            support_probabilities=answer_distribution,
        )
        parts = prepare_solid_formula_task_parts(
            task_id=TASK_ID,
            selected_query=str(query_id),
            branch_probabilities=query_distribution,
            params=resolved_params,
            plan=plan,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        output_fields = {
            "prompt": parts.prompt,
            "answer_gt": TypedValue(type="number", value=float(plan.answer_value)),
            "annotation_gt": TypedValue(type="bbox", value=list(parts.annotation_value)),
            "image": parts.image,
            "image_id": "img0",
            "trace_payload": parts.trace_payload,
            "task_versions": parts.task_versions,
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "prompt_variants": dict(parts.prompt_variants),
        }
        return TaskOutput(**output_fields)
