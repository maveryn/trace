"""Compute a central angle from a visible sector measure and radius."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import SectorObjectivePlan, run_sector_public_entry
from .shared.defaults import DOMAIN
from .shared.sampling import sample_sector_angle_from_arc_length, sample_sector_angle_from_area
from .shared.state import SectorProblem

TASK_ID = "task_geometry__sector__central_angle_from_sector_measure_value"
FROM_ARC_LENGTH = "from_arc_length"
FROM_SECTOR_AREA = "from_sector_area"
SUPPORTED_QUERY_IDS = (FROM_ARC_LENGTH, FROM_SECTOR_AREA)
PROMPT_TASK_KEY = "central_angle_from_sector_measure_value_query"
PROGRAM_SCOPE = "central_angle_from_sector_measure_value"

def _sample_from_arc_length(instance_seed: int, seed_namespace: str, params: Mapping[str, Any]) -> SectorProblem:
    return sample_sector_angle_from_arc_length(instance_seed, seed_namespace=seed_namespace, params=params)


def _sample_from_sector_area(instance_seed: int, seed_namespace: str, params: Mapping[str, Any]) -> SectorProblem:
    return sample_sector_angle_from_area(instance_seed, seed_namespace=seed_namespace, params=params)


_BRANCH_SAMPLERS = {
    FROM_ARC_LENGTH: _sample_from_arc_length,
    FROM_SECTOR_AREA: _sample_from_sector_area,
}


def _prepare_objective(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: dict[str, float],
    task_params: Mapping[str, Any],
) -> SectorObjectivePlan:
    """Bind the central-angle objective for one visible measure branch."""

    branch = str(selected_branch)
    if branch not in _BRANCH_SAMPLERS:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    problem = _BRANCH_SAMPLERS[branch](int(instance_seed), f"{TASK_ID}.{branch}", task_params)
    return SectorObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=branch,
        problem=problem,
        answer_type="number",
        annotation_roles=("target_sector_angle_region",),
        replay_params={
            "query_id_probabilities": dict(branch_probabilities),
            "case_index": int(problem.case_index),
            "formula_family": str(problem.formula_family),
        },
        trace_values={
            "program_scope": PROGRAM_SCOPE,
            "measure_branch": branch,
        },
    )


@register_task
class GeometrySectorCentralAngleFromSectorMeasureValueTask:
    """Compute a central angle from a visible sector measure and radius."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = FROM_ARC_LENGTH
    prepare_objective = staticmethod(_prepare_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_sector_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))


__all__ = ["GeometrySectorCentralAngleFromSectorMeasureValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
