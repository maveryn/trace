"""Compute prism height from a prism-plus-pyramid volume."""

from dataclasses import dataclass

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import build_solid_formula_plan, prepare_solid_formula_task_parts
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import answer_support_probability_map, decimal_support, round1
from .shared.rendering import render_prism_pyramid
from .shared.sampling import select_case_option, select_support_value
from .shared.state import SolidFormulaProblem

TASK_ID = "task_geometry__solid_formula__prism_pyramid_height_from_volume"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = "prism_pyramid_height_from_volume"
ANSWER_SUPPORT = decimal_support(2, 61, step=1)
CONSTRUCTION_OPTIONS = (
    (5.0, 4.0, 3.0),
    (6.0, 5.0, 6.0),
    (7.0, 4.0, 9.0),
    (8.0, 6.0, 3.0),
    (9.0, 5.0, 6.0),
    (10.0, 7.0, 9.0),
)


@dataclass(frozen=True)
class PrismPyramidBinding:
    """Task-local formula binding for a prism topped by a pyramid."""

    problem: SolidFormulaProblem
    support_probabilities: dict[float, float]


def _select_prism_pyramid_binding(*, instance_seed: int, params: dict, query_id: str) -> PrismPyramidBinding:
    """Choose the answer first, then choose one compatible prism-pyramid construction."""

    prism_height = select_support_value(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{query_id}.answer",
        support=ANSWER_SUPPORT,
    )
    (side_a, side_b, pyramid_height), construction_count = select_case_option(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{query_id}.construction",
        options=CONSTRUCTION_OPTIONS,
    )
    base_area = side_a * side_b
    support_probabilities = answer_support_probability_map(ANSWER_SUPPORT, prism_height)
    problem = SolidFormulaProblem(
        solid_kind="prism_pyramid",
        answer=round1(prism_height),
        unknown_dimension="prism_height",
        formula_family="prism_pyramid_height_from_volume",
        formula="V = lwx + (1/3)lwp, solve prism height x from V, l, w, and pyramid height p",
        side_a=round1(side_a),
        side_b=round1(side_b),
        prism_height=round1(prism_height),
        pyramid_height=round1(pyramid_height),
        volume=round1(base_area * (prism_height + (pyramid_height / 3.0))),
        answer_support_probabilities=support_probabilities,
        construction_case_count_for_answer=construction_count,
    )
    return PrismPyramidBinding(
        problem=problem,
        support_probabilities=support_probabilities,
    )


@register_task
class GeometrySolidFormulaPrismPyramidHeightFromVolumeTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate one prism-pyramid height task with task-owned output binding."""

        query_id, probabilities_by_query, resolved_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        binding = _select_prism_pyramid_binding(
            instance_seed=int(instance_seed),
            params=resolved_params,
            query_id=str(query_id),
        )
        plan = build_solid_formula_plan(
            prompt_key=PROMPT_KEY,
            problem=binding.problem,
            render_scene=render_prism_pyramid,
            branch_probabilities=probabilities_by_query,
            support_probabilities=binding.support_probabilities,
        )
        parts = prepare_solid_formula_task_parts(
            task_id=TASK_ID,
            selected_query=str(query_id),
            branch_probabilities=probabilities_by_query,
            params=resolved_params,
            plan=plan,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        answer = TypedValue(type="number", value=float(plan.answer_value))
        annotation = TypedValue(type="bbox", value=list(parts.annotation_value))
        return TaskOutput(
            parts.prompt,
            answer,
            annotation,
            parts.image,
            "img0",
            parts.trace_payload,
            parts.task_versions,
            SCENE_ID,
            str(query_id),
            dict(parts.prompt_variants),
        )
