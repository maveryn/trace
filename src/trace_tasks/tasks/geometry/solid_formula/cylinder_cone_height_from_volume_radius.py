"""Compute cylinder height from compound cylinder-cone volume data."""

from trace_tasks.core import types as core_types
from trace_tasks.tasks import base as task_base
from trace_tasks.tasks import registry as task_registry
from trace_tasks.tasks.shared import fixed_query

from . import _lifecycle as lifecycle
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import answer_support_probability_map, decimal_support, round1
from .shared.rendering import render_cylinder_cone_height
from .shared import sampling as formula_sampling
from .shared.state import SolidFormulaProblem

HEIGHT_TASK_ID = "task_geometry__solid_formula__cylinder_cone_height_from_volume_radius"
HEIGHT_QUERY_ID = "single"
HEIGHT_QUERY_IDS = (HEIGHT_QUERY_ID,)
HEIGHT_DEFAULT_QUERY_ID = HEIGHT_QUERY_ID
HEIGHT_PROMPT_KEY = "cylinder_cone_height_from_volume_radius"
HEIGHT_SUPPORT_VALUES = decimal_support(2, 61, step=1)
HEIGHT_CONSTRUCTION_CHOICES = (
    (3.0, 3.0),
    (4.0, 6.0),
    (5.0, 9.0),
    (6.0, 12.0),
    (7.0, 6.0),
    (8.0, 9.0),
)


def _prepare_height_objective(
    *,
    instance_seed,
    params,
    selected_query,
    branch_probabilities,
):
    # This task binds cylinder height as the answer before rendering.
    cylinder_height = formula_sampling.select_support_value(
        instance_seed=instance_seed,
        params=params,
        namespace=f"{HEIGHT_TASK_ID}.{selected_query}.answer",
        support=HEIGHT_SUPPORT_VALUES,
    )
    (radius, cone_height), construction_count = formula_sampling.select_case_option(
        instance_seed=instance_seed,
        params=params,
        namespace=f"{HEIGHT_TASK_ID}.{selected_query}.construction",
        options=HEIGHT_CONSTRUCTION_CHOICES,
    )
    volume_pi_multiple = radius**2 * (cylinder_height + (cone_height / 3.0))
    support_probabilities = answer_support_probability_map(HEIGHT_SUPPORT_VALUES, cylinder_height)
    problem = SolidFormulaProblem(
        solid_kind="cylinder_cone",
        answer=round1(cylinder_height),
        unknown_dimension="cylinder_height",
        formula_family="cylinder_cone_height_from_volume_radius",
        formula="V = pi r^2x + (1/3)pi r^2c, solve cylinder height x from V, r, and cone height c",
        radius=round1(radius),
        cylinder_height=round1(cylinder_height),
        cone_height=round1(cone_height),
        volume_pi_multiple=round1(volume_pi_multiple),
        answer_support_probabilities=support_probabilities,
        construction_case_count_for_answer=construction_count,
    )
    return lifecycle.build_solid_formula_plan(
        prompt_key=HEIGHT_PROMPT_KEY,
        problem=problem,
        render_scene=render_cylinder_cone_height,
        branch_probabilities=branch_probabilities,
        support_probabilities=support_probabilities,
    )


@task_registry.register_task
class GeometrySolidFormulaCylinderConeHeightFromVolumeRadiusTask:
    task_id = HEIGHT_TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = HEIGHT_QUERY_IDS
    default_query_id = HEIGHT_DEFAULT_QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> task_base.TaskOutput:
        """Generate one cylinder-cone height task with task-owned output binding."""

        selected_query, query_probabilities, task_params = fixed_query.select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=HEIGHT_QUERY_IDS,
            default_query_id=HEIGHT_DEFAULT_QUERY_ID,
            task_id=HEIGHT_TASK_ID,
            namespace=f"{HEIGHT_TASK_ID}.query",
        )
        plan = _prepare_height_objective(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_query=str(selected_query),
            branch_probabilities=query_probabilities,
        )
        parts = lifecycle.prepare_solid_formula_task_parts(
            task_id=HEIGHT_TASK_ID,
            selected_query=str(selected_query),
            branch_probabilities=query_probabilities,
            params=task_params,
            plan=plan,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        return task_base.TaskOutput(
            prompt=parts.prompt,
            answer_gt=core_types.TypedValue(type="number", value=float(plan.answer_value)),
            annotation_gt=core_types.TypedValue(type="bbox", value=list(parts.annotation_value)),
            image=parts.image,
            image_id="img0",
            trace_payload=parts.trace_payload,
            task_versions=parts.task_versions,
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(parts.prompt_variants),
        )
