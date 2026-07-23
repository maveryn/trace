"""Find a corresponding side length between similar figures."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SimilarFigureObjectivePlan, forced_or_sampled_family, run_similar_figure_public_entry
from .shared.measurements import corresponding_side_case


TASK_ID = "task_geometry__similar_figure_measure_transfer__corresponding_side_value"
SUPPORTED_QUERY_IDS = ("single",)
DEFAULT_QUERY_ID = "single"
CONFIG_GROUP_KEY = "corresponding_side_value"
PROMPT_BRANCH_KEY = "single"
CONSTRUCTION_FAMILIES = ("direct_side_transfer", "two_pair_side_transfer", "nested_side_transfer")


def _prepare_corresponding_side_objective(instance_seed: int, selected_branch: str, branch_probabilities: dict[str, float], task_params: dict) -> SimilarFigureObjectivePlan:
    """Bind one public side-transfer task to a concrete construction family."""

    family = forced_or_sampled_family(
        context_label=TASK_ID,
        options=CONSTRUCTION_FAMILIES,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=f"{TASK_ID}.construction_family",
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{family}.case")
    shape_kind = str(rng.choice(("triangle", "quadrilateral", "pentagon")))
    scale_factor = int(rng.randint(2, 7))
    source_side = int(rng.randint(4, 15))
    support_source = int(rng.randint(3, 12))
    if support_source == source_side:
        support_source += 1
    case = corresponding_side_case(
        construction_family=family,
        shape_kind=shape_kind,
        scale_factor=scale_factor,
        source_side=source_side,
        support_source=support_source,
    )
    return SimilarFigureObjectivePlan(
        case=case,
        config_group_key=CONFIG_GROUP_KEY,
        prompt_branch_key=PROMPT_BRANCH_KEY,
        answer_type="integer",
        answer_hint_key="answer_hint_integer_length",
        program_scope="corresponding_side_value",
        public_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
    )


@register_task
class GeometrySimilarFigureMeasureTransferCorrespondingSideValueTask:
    """Find a corresponding side length between similar figures."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_corresponding_side_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_similar_figure_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))


__all__ = ["GeometrySimilarFigureMeasureTransferCorrespondingSideValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
