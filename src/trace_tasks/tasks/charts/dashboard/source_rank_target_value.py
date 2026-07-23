"""Public task for `task_charts__dashboard__source_rank_target_value`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.metrics import category_by_id, panel_by_id, rank_phrase, ranked_category_id
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_SOURCE_QUERY_ID = "largest_source_rank_target_value"
SMALLEST_SOURCE_QUERY_ID = "smallest_source_rank_target_value"
RANK_DIRECTION_BY_QUERY_ID = {
    LARGEST_SOURCE_QUERY_ID: "largest",
    SMALLEST_SOURCE_QUERY_ID: "smallest",
}


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


def _build_rank_transfer_trace(*, source_id: str, target_id: str, category_id: str) -> dict[str, Any]:
    """Return target-readout trace fields for the transfer-style objective."""

    transfer_steps = [
        "find_ranked_category_in_source_panel",
        "carry_shared_category_label",
        "read_same_label_in_target_panel",
    ]
    readout_path_descriptor = {
        "initial_operand": "source_rank_ordering",
        "bridge_operand": "shared_category_identity",
        "terminal_operand": "target_panel_readout",
    }
    return {
        "source_rank_panel": str(source_id),
        "target_readout_panel": str(target_id),
        "carried_category": str(category_id),
        "readout_path": list(transfer_steps),
        "readout_path_descriptor": dict(readout_path_descriptor),
        "readout_terminal": "target_panel_numeric_value",
    }


def _build_source_target_transfer_plan(instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
    """Bind the rank-transfer objective: choose by source rank, answer from target value."""
    rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.source_rank_target.transfer")
    base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
    source_id, target_id = rng.sample([str(panel.panel_id) for panel in base_sample.panels], 2)
    panels = {
        "source": panel_by_id(base_sample.panels, source_id),
        "target": panel_by_id(base_sample.panels, target_id),
    }
    direction = RANK_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    rank_n = 1
    chosen_category_id = ranked_category_id(categories=base_sample.categories, panel=panels["source"], direction=direction, rank_n=rank_n)
    chosen_category = category_by_id(base_sample.categories, chosen_category_id)
    mark_values = {role: int(panel.values_by_category_id[str(chosen_category_id)]) for role, panel in panels.items()}
    mark_refs = {role: (str(panel.panel_id), str(chosen_category_id)) for role, panel in panels.items()}
    transfer_value = int(mark_values["target"])
    transfer_trace = _build_rank_transfer_trace(source_id=str(source_id), target_id=str(target_id), category_id=str(chosen_category_id))
    relations = dict(base_sample.common_params)
    relations.update(
        {
            "source_panel_id": str(transfer_trace["source_rank_panel"]),
            "source_panel_name": str(panels["source"].name),
            "target_panel_id": str(transfer_trace["target_readout_panel"]),
            "target_panel_name": str(panels["target"].name),
            "rank_direction": str(direction),
            "rank_n": int(rank_n),
            "rank_phrase": rank_phrase(str(direction), int(rank_n)),
            "selected_category_id": str(transfer_trace["carried_category"]),
            "selected_category_label": str(chosen_category.label),
            "source_value": int(mark_values["source"]),
            "target_value": int(transfer_value),
            "transfer_readout_roles": ["source_rank_mark", "target_value_mark"],
            "transfer_operation": "rank_selected_value_lookup",
            "transfer_trace": dict(transfer_trace),
        }
    )
    refs = (mark_refs["source"], mark_refs["target"])
    dataset = DashboardDataset(
        scene_variant=SCENE_VARIANT,
        categories=base_sample.categories,
        panels=base_sample.panels,
        query=DashboardQuery(answer=int(transfer_value), answer_type="integer", annotation_refs=refs, params=dict(relations)),
    )
    prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
    return DashboardTaskPlan(
        dataset=dataset,
        prompt_artifacts=prompt_artifacts,
        relations=relations,
        answer_gt=TypedValue(type="integer", value=int(transfer_value)),
        annotation_refs=refs,
        annotation_roles={"source_panel": refs[0], "target_panel": refs[1]},
    )


@register_task
class ChartsDashboardSourceRankTargetValueTask:
    """Use a ranked category in one panel to read the same category in another panel."""

    task_id = "task_charts__dashboard__source_rank_target_value"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "source_rank_target_value"
    supported_query_ids = (LARGEST_SOURCE_QUERY_ID, SMALLEST_SOURCE_QUERY_ID)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=LARGEST_SOURCE_QUERY_ID, task_id=self.task_id)
        return run_dashboard_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=_build_source_target_transfer_plan, build_output=_build_task_output)


__all__ = ["ChartsDashboardSourceRankTargetValueTask"]
