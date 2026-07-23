"""Find a regular-polygon side length."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import RegularPolygonObjectivePlan, run_regular_polygon_public_entry
from .shared.defaults import DOMAIN
from .shared.sampling import side_from_area_apothem, side_from_perimeter, side_from_piece_area_apothem

TASK_ID = "task_geometry__regular_polygon_decomposition__side_length_value"
QUERY_ID_PERIMETER = "side_length_from_perimeter"
QUERY_ID_AREA_APOTHEM = "side_length_from_total_area_and_apothem"
QUERY_ID_PIECE_APOTHEM = "side_length_from_wedge_area_and_apothem"
SUPPORTED_QUERY_IDS = (QUERY_ID_PERIMETER, QUERY_ID_AREA_APOTHEM, QUERY_ID_PIECE_APOTHEM)
DEFAULT_QUERY_ID = QUERY_ID_PERIMETER
PROMPT_TASK_KEY = "side_length_value_query"


def _prepare_side_length(instance_seed, task_params, selected_branch, branch_probabilities):
    if selected_branch == QUERY_ID_PERIMETER:
        problem = side_from_perimeter(int(instance_seed), task_params, seed_namespace=f"{TASK_ID}.{selected_branch}")
        annotation_roles = ("O", "A", "B")
    elif selected_branch == QUERY_ID_AREA_APOTHEM:
        problem = side_from_area_apothem(int(instance_seed), task_params, seed_namespace=f"{TASK_ID}.{selected_branch}")
        annotation_roles = ("O", "A", "B", "M")
    elif selected_branch == QUERY_ID_PIECE_APOTHEM:
        problem = side_from_piece_area_apothem(int(instance_seed), task_params, seed_namespace=f"{TASK_ID}.{selected_branch}")
        annotation_roles = ("O", "A", "B", "M", "W")
    else:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    return RegularPolygonObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=str(selected_branch),
        problem=problem,
        answer_gt=TypedValue(type="integer", value=int(round(float(problem.answer)))),
        annotation_roles=tuple(annotation_roles),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "case_index": int(problem.case_index),
            "n_sides": int(problem.n_sides),
            "wedge_count": int(problem.wedge_count),
        },
        trace_values={"answer_family": "length", "target_role": "side_length"},
    )


@register_task
class GeometryRegularPolygonDecompositionSideLengthTask:
    """Find a regular-polygon side length."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_side_length)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_regular_polygon_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
