"""Compute length from a house-prism volume."""

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import build_solid_formula_plan, prepare_solid_formula_task_parts
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import answer_support_probability_map, decimal_support, round1
from .shared.rendering import render_house_prism
from .shared.sampling import select_case_option, select_support_value
from .shared.state import SolidFormulaProblem

TASK_ID = "task_geometry__solid_formula__house_prism_length_from_volume"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = "house_prism_length_from_volume"
ANSWER_SUPPORT = decimal_support(2, 61, step=1)
CONSTRUCTION_OPTIONS = (
    (6.0, 4.0, 3.0),
    (8.0, 5.0, 4.0),
    (10.0, 4.0, 6.0),
    (12.0, 6.0, 3.0),
    (14.0, 5.0, 5.0),
    (16.0, 4.0, 6.0),
)


def _prepare_house_length_objective(
    *,
    instance_seed,
    params,
    selected_query,
    branch_probabilities,
):
    # This task binds prism length as the answer before rendering.
    prism_length = select_support_value(
        instance_seed=instance_seed,
        params=params,
        namespace=f"{TASK_ID}.{selected_query}.answer",
        support=ANSWER_SUPPORT,
    )
    (triangle_base, wall_height, roof_height), construction_count = select_case_option(
        instance_seed=instance_seed,
        params=params,
        namespace=f"{TASK_ID}.{selected_query}.construction",
        options=CONSTRUCTION_OPTIONS,
    )
    cross_section_area = (triangle_base * wall_height) + (0.5 * triangle_base * roof_height)
    volume = cross_section_area * prism_length
    support_probabilities = answer_support_probability_map(ANSWER_SUPPORT, prism_length)
    problem = SolidFormulaProblem(
        solid_kind="house_prism",
        answer=round1(prism_length),
        unknown_dimension="length",
        formula_family="house_prism_length_from_volume",
        formula="V = (bh + (1/2)bt)L, solve length L from rectangular wall and triangular roof cross-section",
        triangle_base=round1(triangle_base),
        prism_length=round1(prism_length),
        wall_height=round1(wall_height),
        roof_height=round1(roof_height),
        volume=round1(volume),
        answer_support_probabilities=support_probabilities,
        construction_case_count_for_answer=construction_count,
    )
    return build_solid_formula_plan(
        prompt_key=PROMPT_KEY,
        problem=problem,
        render_scene=render_house_prism,
        branch_probabilities=branch_probabilities,
        support_probabilities=support_probabilities,
    )


@register_task
class GeometrySolidFormulaHousePrismLengthFromVolumeTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID

    @staticmethod
    def _bind_house_plan(instance_seed: int, params: dict, query_id: str, query_probabilities: dict) -> object:
        """Bind the house-prism formula target and visual witnesses for this objective."""

        plan = _prepare_house_length_objective(
            instance_seed=int(instance_seed),
            params=params,
            selected_query=str(query_id),
            branch_probabilities=query_probabilities,
        )
        return plan

    @staticmethod
    def _emit_task_output(*, parts, plan, query_id: str) -> TaskOutput:
        """Build the verifier payload for the house-prism length public task."""

        prompt_variants = dict(parts.prompt_variants)
        return TaskOutput(
            prompt=parts.prompt,
            answer_gt=TypedValue(type="number", value=float(plan.answer_value)),
            annotation_gt=TypedValue(type="bbox", value=list(parts.annotation_value)),
            image=parts.image,
            image_id="img0",
            trace_payload=parts.trace_payload,
            task_versions=parts.task_versions,
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants=prompt_variants,
        )

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate one house-prism length task with task-owned output binding."""

        selected_query, probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        plan = self._bind_house_plan(
            int(instance_seed),
            task_params,
            str(selected_query),
            dict(probabilities),
        )
        parts = prepare_solid_formula_task_parts(
            task_id=TASK_ID,
            selected_query=str(selected_query),
            branch_probabilities=probabilities,
            params=task_params,
            plan=plan,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        return self._emit_task_output(parts=parts, plan=plan, query_id=str(selected_query))
