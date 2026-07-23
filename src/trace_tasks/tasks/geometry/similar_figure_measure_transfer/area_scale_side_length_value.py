"""Use an area scale relation to find a corresponding side length."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SimilarFigureObjectivePlan, forced_or_sampled_family, run_similar_figure_public_entry
from .shared.measurements import side_length_from_area_case


TASK_ID = "task_geometry__similar_figure_measure_transfer__area_scale_side_length_value"
SUPPORTED_QUERY_IDS = ("side_length_from_area_pair", "side_length_from_area_ratio")
DEFAULT_QUERY_ID = SUPPORTED_QUERY_IDS[0]
CONFIG_GROUP_KEY = "area_scale_side_length_value"
AREA_PAIR_FAMILIES = ("area_pair_labels", "area_known_side_nested")


def _prepare_area_scale_side_length_objective(instance_seed: int, selected_branch: str, branch_probabilities: dict[str, float], task_params: dict) -> SimilarFigureObjectivePlan:
    """Bind an area-scale branch to one side-transfer construction."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{selected_branch}.case")
    shape_kind = str(rng.choice(("triangle", "quadrilateral", "pentagon")))
    scale_factor = int(rng.randint(2, 8))
    source_side = int(rng.randint(4, 15))
    source_area = int(rng.randint(3, 22))
    if selected_branch == "side_length_from_area_ratio":
        family = "area_ratio_label"
        use_ratio = True
    elif selected_branch == "side_length_from_area_pair":
        family = forced_or_sampled_family(
            context_label=TASK_ID,
            options=AREA_PAIR_FAMILIES,
            instance_seed=int(instance_seed),
            params=task_params,
            namespace=f"{TASK_ID}.area_pair_family",
        )
        use_ratio = False
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    case = side_length_from_area_case(
        construction_family=family,
        shape_kind=shape_kind,
        scale_factor=scale_factor,
        source_side=source_side,
        source_area=source_area,
        use_ratio_label=use_ratio,
    )
    return SimilarFigureObjectivePlan(
        case=case,
        config_group_key=CONFIG_GROUP_KEY,
        prompt_branch_key=str(selected_branch),
        answer_type="integer",
        answer_hint_key="answer_hint_integer_length",
        program_scope="area_scale_side_length_value",
        public_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
    )


@register_task
class GeometrySimilarFigureMeasureTransferAreaScaleSideLengthValueTask:
    """Use an area scale relation to find a corresponding side length."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_area_scale_side_length_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_similar_figure_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))


__all__ = ["GeometrySimilarFigureMeasureTransferAreaScaleSideLengthValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
