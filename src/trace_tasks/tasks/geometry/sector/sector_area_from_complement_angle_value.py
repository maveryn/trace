"""Compute sector area after deriving a complementary sector angle."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import SectorObjectivePlan, run_sector_public_entry
from .shared.defaults import DOMAIN
from .shared.sampling import sample_sector_area_from_complement_angle

TASK_ID = "task_geometry__sector__sector_area_from_complement_angle_value"
SINGLE_QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "sector_area_from_complement_angle_value_query"
PROGRAM_SCOPE = "sector_area_from_complement_angle_value"


def _sample(instance_seed: int, seed_namespace: str, params: Mapping[str, Any]):
    return sample_sector_area_from_complement_angle(instance_seed, seed_namespace=seed_namespace, params=params)


def _prepare_objective(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: dict[str, float],
    task_params: Mapping[str, Any],
) -> SectorObjectivePlan:
    """Bind the area-from-complement objective before neutral rendering."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    problem = _sample(int(instance_seed), f"{TASK_ID}.{SINGLE_QUERY_ID}", task_params)
    return SectorObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=SINGLE_QUERY_ID,
        problem=problem,
        answer_type="number",
        annotation_roles=("target_sector_region",),
        replay_params={
            "query_id_probabilities": dict(branch_probabilities),
            "case_index": int(problem.case_index),
            "formula_family": str(problem.formula_family),
        },
        trace_values={"program_scope": PROGRAM_SCOPE},
    )


@register_task
class GeometrySectorAreaFromComplementAngleValueTask:
    """Compute sector area after deriving a complementary sector angle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    prepare_objective = staticmethod(_prepare_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_sector_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))


__all__ = ["GeometrySectorAreaFromComplementAngleValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
