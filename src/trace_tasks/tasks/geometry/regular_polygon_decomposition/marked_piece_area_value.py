"""Find the area of marked equal wedges in a regular polygon."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import RegularPolygonObjectivePlan, run_regular_polygon_public_entry
from .shared.defaults import DOMAIN
from .shared.sampling import area_from_marked_equal_pieces

TASK_ID = "task_geometry__regular_polygon_decomposition__marked_piece_area_value"
QUERY_ID_MARKED_WEDGES = "marked_wedges_area_from_total"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
PROMPT_TASK_KEY = "marked_piece_area_value_query"
ANNOTATION_ROLES = ("O", "A", "B")


def _prepare_marked_piece_area(instance_seed, task_params, selected_branch, branch_probabilities):
    if selected_branch != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    problem = area_from_marked_equal_pieces(
        int(instance_seed),
        task_params,
        seed_namespace=f"{TASK_ID}.{QUERY_ID_MARKED_WEDGES}",
    )
    answer_value = round(float(problem.answer), 1)
    return RegularPolygonObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=QUERY_ID_MARKED_WEDGES,
        problem=problem,
        answer_gt=TypedValue(type="number", value=float(answer_value)),
        annotation_roles=ANNOTATION_ROLES,
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "case_index": int(problem.case_index),
            "n_sides": int(problem.n_sides),
            "wedge_count": int(problem.wedge_count),
        },
        trace_values={
            "answer_family": "area",
            "target_role": "marked_wedge_group_area",
            "answer_rounding": "one_decimal",
        },
    )


@register_task
class GeometryRegularPolygonDecompositionMarkedPieceAreaTask:
    """Find the area of marked equal wedges in a regular polygon."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_marked_piece_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_regular_polygon_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
