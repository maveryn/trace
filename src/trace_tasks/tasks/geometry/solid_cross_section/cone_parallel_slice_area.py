from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import SolidCrossSectionObjectivePlan, prepare_solid_cross_section_task_parts
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.measurements import cone_problem_from_case
from .shared.rendering import render_cone_cross_section
from .shared.sampling import cone_answer_support_size, resolve_cone_slice_case

TASK_ID = "task_geometry__solid_cross_section__cone_parallel_slice_area"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = QUERY_ID


@register_task
class GeometryConeParallelSliceAreaTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate one cone parallel-slice area task with task-owned output binding."""

        query_id, probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        case, answer_probabilities, case_count = resolve_cone_slice_case(
            instance_seed=int(instance_seed),
            params=task_params,
            namespace=f"{TASK_ID}.{query_id}",
        )
        problem = cone_problem_from_case(
            case,
            formula_family="cone_parallel_slice_area",
            formula="parallel cone slice: r_slice/R = d/H, area = pi * r_slice^2",
            answer_support_probabilities=answer_probabilities,
            construction_case_count_for_answer=int(case_count),
        )
        if problem.base_radius is None or problem.slice_radius is None:
            raise RuntimeError("cone slice area task requires base and slice radii")
        cone_trace = {
            "solid_kind": "cone",
            "base_radius": float(problem.base_radius or 0.0),
            "solid_height": float(problem.solid_height),
            "slice_distance_from_apex": float(problem.slice_distance_from_apex),
            "similarity_scale": float(problem.slice_distance_from_apex) / float(problem.solid_height),
            "slice_radius": float(problem.slice_radius or 0.0),
            "answer_support_size": int(cone_answer_support_size()),
            "construction_case_count_for_answer": int(case_count),
        }
        plan = SolidCrossSectionObjectivePlan(
            prompt_key=PROMPT_KEY,
            object_description="a cone cut by a plane parallel to its base, with the cross-section marked and dimensions labeled",
            problem=problem,
            render_scene=render_cone_cross_section,
            answer_value=float(problem.answer),
            query_params={
                "query_id_probabilities": dict(probabilities),
                "answer_support_probabilities": dict(answer_probabilities),
                **cone_trace,
            },
            trace_values=cone_trace,
        )
        parts = prepare_solid_cross_section_task_parts(
            task_id=TASK_ID,
            selected_query=str(query_id),
            query_probabilities=probabilities,
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
            query_id=str(query_id),
            prompt_variants=dict(parts.prompt_variants),
        )
