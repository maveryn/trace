"""Public task for `task_charts__dashboard__category_total_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import (
    DashboardTaskPlan,
    MaterializedDashboardTask,
    dashboard_task_output_fields,
    dashboard_total_extremum_plan_from_sample,
    run_dashboard_public_task,
)
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import DashboardTotalExtremumSample, build_dashboard_base_sample, make_category_total_extremum_sample
from trace_tasks.tasks.charts.dashboard.shared.state import Category, DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery, Panel
from trace_tasks.tasks.shared.unanswerable import (
    UNANSWERABLE_ANSWER,
    absence_proof,
    should_use_unanswerable_branch,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_CATEGORY_TOTAL_QUERY_ID = "largest_category_total_label"
SMALLEST_CATEGORY_TOTAL_QUERY_ID = "smallest_category_total_label"
TOTAL_DIRECTION_BY_QUERY_ID = {
    LARGEST_CATEGORY_TOTAL_QUERY_ID: "largest",
    SMALLEST_CATEGORY_TOTAL_QUERY_ID: "smallest",
}
UNANSWERABLE_PROMPT_INSTRUCTION = (
    'If any category is not shown in every dashboard panel, treat the question as unanswerable '
    'and set the answer to exactly "unanswerable".'
)


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


def _category_total_map(*, categories: tuple[Category, ...], panels: tuple[Panel, ...]) -> dict[str, int]:
    return {
        str(category.category_id): sum(
            int(panel.values_by_category_id[str(category.category_id)])
            for panel in panels
        )
        for category in categories
    }


def _validate_category_total_sample(
    *,
    categories: tuple[Category, ...],
    panels: tuple[Panel, ...],
    sample: DashboardTotalExtremumSample,
    direction: str,
) -> None:
    """Check that the selected category is the unique requested total extremum."""

    totals = _category_total_map(categories=categories, panels=panels)
    if totals != {str(key): int(value) for key, value in sample.totals_by_id.items()}:
        raise ValueError("category total sample metadata does not match generated values")
    ranked = sorted(totals.items(), key=lambda item: int(item[1]), reverse=str(direction) == "largest")
    if str(ranked[0][0]) != str(sample.answer_id):
        raise ValueError("selected category is not the requested total extremum")
    if len(ranked) > 1 and int(ranked[0][1]) == int(ranked[1][1]):
        raise ValueError("selected category total is tied")


def _category_total_rank_records(*, categories: tuple[Category, ...], totals_by_category_id: dict[str, int]) -> list[dict[str, Any]]:
    """Return category-total ranks in both directions for trace inspection."""

    descending_ids = [
        str(category_id)
        for category_id, _total in sorted(
            totals_by_category_id.items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    ]
    ascending_ids = list(reversed(descending_ids))
    largest_rank = {str(category_id): int(index + 1) for index, category_id in enumerate(descending_ids)}
    smallest_rank = {str(category_id): int(index + 1) for index, category_id in enumerate(ascending_ids)}
    labels_by_id = {str(category.category_id): str(category.label) for category in categories}
    return [
        {
            "category_id": str(category_id),
            "category_label": str(labels_by_id[str(category_id)]),
            "category_total": int(totals_by_category_id[str(category_id)]),
            "largest_rank": int(largest_rank[str(category_id)]),
            "smallest_rank": int(smallest_rank[str(category_id)]),
        }
        for category_id in descending_ids
    ]


def _omit_category_from_panel(
    *,
    panels: tuple[Panel, ...],
    panel_id: str,
    category_id: str,
) -> tuple[Panel, ...]:
    """Return panels where one category mark is removed from one panel."""

    updated: list[Panel] = []
    for panel in panels:
        values = dict(panel.values_by_category_id)
        if str(panel.panel_id) == str(panel_id):
            values.pop(str(category_id), None)
        updated.append(
            Panel(
                panel_id=str(panel.panel_id),
                kind=str(panel.kind),
                name=str(panel.name),
                values_by_category_id=values,
            )
        )
    return tuple(updated)


@register_task
class ChartsDashboardCategoryTotalExtremumLabelTask:
    """Find which category has the largest or smallest total across panels."""

    task_id = "task_charts__dashboard__category_total_extremum_label"
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = "category_total_extremum_label"
    supported_query_ids = (LARGEST_CATEGORY_TOTAL_QUERY_ID, SMALLEST_CATEGORY_TOTAL_QUERY_ID)
    default_dataset_enabled = True
    supports_unanswerable = True

    def _construct_category_total_extremum_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Choose the answer category by a unique sum across all panels."""

        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        direction = TOTAL_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        if should_use_unanswerable_branch(
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_ID}.{self.objective_contract}",
            enabled=bool(self.supports_unanswerable),
        ):
            missing_category = base_sample.categories[
                int(rng.randrange(len(base_sample.categories)))
            ]
            missing_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
            panels = _omit_category_from_panel(
                panels=base_sample.panels,
                panel_id=str(missing_panel.panel_id),
                category_id=str(missing_category.category_id),
            )
            present_panels = [
                panel for panel in panels if str(panel.panel_id) != str(missing_panel.panel_id)
            ]
            category_presence_by_panel_id = {
                str(panel.panel_id): str(missing_category.category_id) in panel.values_by_category_id
                for panel in panels
            }
            relations = {
                **dict(base_sample.common_params),
                "category_total_extremum_direction": str(direction),
                "answerability": "unanswerable",
                "answer_category_id": "",
                "answer_category_label": UNANSWERABLE_ANSWER,
                "missing_category_id": str(missing_category.category_id),
                "missing_category_label": str(missing_category.label),
                "missing_category_panel_id": str(missing_panel.panel_id),
                "missing_category_panel_name": str(missing_panel.name),
                "category_presence_by_panel_id": dict(category_presence_by_panel_id),
                "category_total_by_category_id": {},
                "category_total_rank_records": [],
                "selection_operation": "category_total_extremum",
                "sum_operation": "sum_panels_for_category",
                "unanswerable_instruction": UNANSWERABLE_PROMPT_INSTRUCTION,
                "absence_proof": absence_proof(
                    requested_item=f"{str(missing_category.label)} in every dashboard panel",
                    visible_candidates=[
                        f"{str(missing_category.label)} in {str(panel.name)}"
                        for panel in present_panels
                    ],
                    checked_scope="dashboard category presence by panel",
                    absence_reason="a category needed for a cross-panel total is not shown in every panel",
                ),
            }
            dataset = DashboardDataset(
                scene_variant=SCENE_VARIANT,
                categories=base_sample.categories,
                panels=panels,
                query=DashboardQuery(
                    answer=UNANSWERABLE_ANSWER,
                    answer_type="string",
                    annotation_refs=(),
                    params=dict(relations),
                ),
            )
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(selected_query_id),
                dynamic_slots=build_prompt_slots(dataset=dataset),
                instance_seed=int(instance_seed),
            )
            return DashboardTaskPlan(
                dataset=dataset,
                prompt_artifacts=prompt_artifacts,
                relations=relations,
                answer_gt=TypedValue(type="string", value=UNANSWERABLE_ANSWER),
                annotation_refs=(),
                annotation_type="point_set",
            )
        answer_category = base_sample.categories[int(rng.randrange(len(base_sample.categories)))]
        total_sample = make_category_total_extremum_sample(
            rng,
            panels=base_sample.panels,
            categories=base_sample.categories,
            answer_category_id=str(answer_category.category_id),
            direction=str(direction),
            value_min=int(value_min),
            value_max=int(value_max),
        )
        _validate_category_total_sample(
            categories=base_sample.categories,
            panels=total_sample.panels,
            sample=total_sample,
            direction=str(direction),
        )
        relations = {
            **dict(base_sample.common_params),
            "category_total_extremum_direction": str(direction),
            "answer_category_id": str(total_sample.answer_id),
            "answer_category_label": str(total_sample.answer_label),
            "answer_category_total": int(total_sample.answer_total),
            "category_total_by_category_id": dict(total_sample.totals_by_id),
            "category_total_rank_records": _category_total_rank_records(
                categories=base_sample.categories,
                totals_by_category_id=dict(total_sample.totals_by_id),
            ),
            "selection_operation": "category_total_extremum",
            "sum_operation": "sum_panels_for_category",
            "unanswerable_instruction": UNANSWERABLE_PROMPT_INSTRUCTION,
        }
        return dashboard_total_extremum_plan_from_sample(
            categories=base_sample.categories,
            total_sample=total_sample,
            relations=relations,
            prompt_query_key=str(selected_query_id),
            instance_seed=int(instance_seed),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_CATEGORY_TOTAL_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_category_total_extremum_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardCategoryTotalExtremumLabelTask"]
