"""Public task for `task_charts__dashboard__panel_value_range_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import category_by_id, panel_by_id, panel_value_range_support
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample, make_panel_with_controlled_range, replace_panels_by_id
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_RANGE_QUERY_ID = "largest_panel_value_range_label"
SMALLEST_RANGE_QUERY_ID = "smallest_panel_value_range_label"
RANGE_DIRECTION_BY_QUERY_ID = {
    LARGEST_RANGE_QUERY_ID: "largest",
    SMALLEST_RANGE_QUERY_ID: "smallest",
}


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


def _range_assignment(*, rng, panels: tuple[Any, ...], support: tuple[int, ...], answer_panel_id: str, direction: str) -> dict[str, int]:
    sampled_ranges = sorted(int(value) for value in rng.sample(list(support), len(panels)))
    answer_range = int(sampled_ranges[-1] if str(direction) == "largest" else sampled_ranges[0])
    other_ranges = list(sampled_ranges[:-1] if str(direction) == "largest" else sampled_ranges[1:])
    rng.shuffle(other_ranges)
    assigned: dict[str, int] = {str(answer_panel_id): int(answer_range)}
    other_index = 0
    for panel in panels:
        panel_id = str(panel.panel_id)
        if panel_id == str(answer_panel_id):
            continue
        assigned[panel_id] = int(other_ranges[other_index])
        other_index += 1
    return assigned


def _range_extremum_relations(
    *,
    common_params: dict[str, Any],
    direction: str,
    answer_panel: Any,
    answer_extrema: dict[str, Any],
    largest_category: Any,
    smallest_category: Any,
    extrema_by_panel_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        **dict(common_params),
        "range_extremum_direction": str(direction),
        "answer_panel_id": str(answer_panel.panel_id),
        "answer_panel_name": str(answer_panel.name),
        "answer_range_value": int(answer_extrema["range_value"]),
        "largest_category_id": str(largest_category.category_id),
        "largest_category_label": str(largest_category.label),
        "largest_value": int(answer_extrema["largest_value"]),
        "smallest_category_id": str(smallest_category.category_id),
        "smallest_category_label": str(smallest_category.label),
        "smallest_value": int(answer_extrema["smallest_value"]),
        "panel_range_by_panel_id": {str(panel_id): int(record["range_value"]) for panel_id, record in extrema_by_panel_id.items()},
        "selection_operation": "panel_value_range_extremum",
        "range_operation": "panel_max_minus_min",
    }


@register_task
class ChartsDashboardPanelValueRangeExtremumLabelTask:
    """Find which dashboard panel has the largest or smallest category-value range."""

    task_id = "task_charts__dashboard__panel_value_range_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "panel_value_range_extremum_label"
    supported_query_ids = (LARGEST_RANGE_QUERY_ID, SMALLEST_RANGE_QUERY_ID)
    default_dataset_enabled = True

    def _construct_panel_value_range_extremum_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Choose the answer panel by a unique max/min within-panel value range."""

        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        direction = RANGE_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        support = panel_value_range_support(
            params,
            value_min=int(value_min),
            value_max=int(value_max),
            explicit_keys=(),
        )
        if len(support) < len(base_sample.panels):
            raise ValueError("panel_value_range_support must contain enough unique values for all panels")

        answer_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
        assigned_ranges = _range_assignment(
            rng=rng,
            panels=base_sample.panels,
            support=support,
            answer_panel_id=str(answer_panel.panel_id),
            direction=str(direction),
        )

        updated_by_panel_id = {}
        extrema_by_panel_id: dict[str, dict[str, Any]] = {}
        for panel in base_sample.panels:
            panel_id = str(panel.panel_id)
            updated_panel, extrema = make_panel_with_controlled_range(
                rng,
                panel=panel,
                categories=base_sample.categories,
                value_min=int(value_min),
                value_max=int(value_max),
                target_range=int(assigned_ranges[panel_id]),
            )
            updated_by_panel_id[panel_id] = updated_panel
            extrema_by_panel_id[panel_id] = dict(extrema)

        panels = replace_panels_by_id(base_sample.panels, updated_by_panel_id)
        answer_panel_record = panel_by_id(panels, str(answer_panel.panel_id))
        answer_extrema = extrema_by_panel_id[str(answer_panel_record.panel_id)]
        largest_category = category_by_id(base_sample.categories, str(answer_extrema["largest_category_id"]))
        smallest_category = category_by_id(base_sample.categories, str(answer_extrema["smallest_category_id"]))
        largest_ref = (str(answer_panel_record.panel_id), str(largest_category.category_id))
        smallest_ref = (str(answer_panel_record.panel_id), str(smallest_category.category_id))
        relations = _range_extremum_relations(
            common_params=dict(base_sample.common_params),
            direction=str(direction),
            answer_panel=answer_panel_record,
            answer_extrema=answer_extrema,
            largest_category=largest_category,
            smallest_category=smallest_category,
            extrema_by_panel_id=extrema_by_panel_id,
        )
        refs = (largest_ref, smallest_ref)
        dataset = DashboardDataset(
            scene_variant=SCENE_VARIANT,
            categories=base_sample.categories,
            panels=panels,
            query=DashboardQuery(answer=str(answer_panel_record.name), answer_type="string", annotation_refs=refs, params=dict(relations)),
        )
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(
            dataset=dataset,
            prompt_artifacts=prompt_artifacts,
            relations=relations,
            answer_gt=TypedValue(type="string", value=str(answer_panel_record.name)),
            annotation_refs=refs,
            annotation_roles={"largest_value": largest_ref, "smallest_value": smallest_ref},
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_RANGE_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_panel_value_range_extremum_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardPanelValueRangeExtremumLabelTask"]
