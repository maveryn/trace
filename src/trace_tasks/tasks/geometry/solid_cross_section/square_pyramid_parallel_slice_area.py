from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import SolidCrossSectionObjectivePlan, prepare_solid_cross_section_task_parts
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import pyramid_problem_from_case
from .shared.rendering import render_square_pyramid_cross_section
from .shared.sampling import pyramid_answer_support_size, resolve_pyramid_slice_case

TASK_ID = "task_geometry__solid_cross_section__square_pyramid_parallel_slice_area"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = QUERY_ID


def _prepare_pyramid_objective(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
) -> SolidCrossSectionObjectivePlan:
    """Bind the square-pyramid slice-area objective for one generated instance."""

    case, support_probabilities, construction_case_count = resolve_pyramid_slice_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{selected_query}",
    )
    problem = pyramid_problem_from_case(
        case,
        formula_family="square_pyramid_parallel_slice_area",
        formula="parallel square-pyramid slice: s_slice/s = d/H, area = s_slice^2",
        answer_support_probabilities=support_probabilities,
        construction_case_count_for_answer=int(construction_case_count),
    )
    trace_values = {
        "solid_kind": "square_pyramid",
        "base_side": float(problem.base_side or 0.0),
        "solid_height": float(problem.solid_height),
        "slice_distance_from_apex": float(problem.slice_distance_from_apex),
        "similarity_scale": float(problem.slice_distance_from_apex) / float(problem.solid_height),
        "slice_side": float(problem.slice_side or 0.0),
        "answer_support_size": int(pyramid_answer_support_size()),
        "construction_case_count_for_answer": int(construction_case_count),
    }
    return SolidCrossSectionObjectivePlan(
        prompt_key=PROMPT_KEY,
        object_description="a square pyramid cut by a plane parallel to its base, with the cross-section marked and dimensions labeled",
        problem=problem,
        render_scene=render_square_pyramid_cross_section,
        answer_value=float(problem.answer),
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "answer_support_probabilities": dict(support_probabilities),
            **dict(trace_values),
        },
        trace_values=trace_values,
    )


@register_task
class GeometrySquarePyramidParallelSliceAreaTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate one square-pyramid parallel-slice area task with task-owned output binding."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        plan = _prepare_pyramid_objective(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
        )
        parts = prepare_solid_cross_section_task_parts(
            task_id=TASK_ID,
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            params=task_params,
            plan=plan,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        return TaskOutput(
            prompt=parts.prompt,
            answer_gt=TypedValue(type="number", value=float(plan.answer_value)),
            annotation_gt=TypedValue(type="bbox", value=list(parts.annotation_value)),
            image=parts.image,
            image_id="img0",
            trace_payload=parts.trace_payload,
            task_versions=parts.task_versions,
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(parts.prompt_variants),
        )
