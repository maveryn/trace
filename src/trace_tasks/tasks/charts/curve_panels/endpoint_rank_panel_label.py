"""Public task for `task_charts__curve_panels__endpoint_rank_panel_label`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.curve_panels._lifecycle import (
    CurvePanelTaskPlan,
    build_curve_panel_plan_from_query,
    build_curve_panel_query_record,
    run_curve_panel_task_lifecycle,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import SCENE_NAMESPACE
from trace_tasks.tasks.charts.curve_panels.shared.sampling import (
    base_curve_panel_sample,
    choose_method_label,
    point_id,
)
from trace_tasks.tasks.registry import register_task

START_HIGHEST_QUERY_ID = "start_highest_panel_label"
START_LOWEST_QUERY_ID = "start_lowest_panel_label"
END_HIGHEST_QUERY_ID = "end_highest_panel_label"
END_LOWEST_QUERY_ID = "end_lowest_panel_label"
QUERY_PARAMS = {
    START_HIGHEST_QUERY_ID: ("start", "highest"),
    START_LOWEST_QUERY_ID: ("start", "lowest"),
    END_HIGHEST_QUERY_ID: ("end", "highest"),
    END_LOWEST_QUERY_ID: ("end", "lowest"),
}
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 3,
    "method_count_max": 5,
    "x_tick_count_min": 5,
    "x_tick_count_max": 8,
}


def _endpoint_index_for_query(endpoint: str, x_values: tuple[int, ...]) -> int:
    """Return the x-index implied by a start/end endpoint query."""

    if str(endpoint) == "start":
        return 0
    if str(endpoint) == "end":
        return len(x_values) - 1
    raise ValueError(f"unsupported endpoint position: {endpoint}")


def _non_answer_endpoint_value(
    *, rng: Any, rank_direction: str, panel_index: int
) -> int:
    """Sample a distractor endpoint value on the opposite side of the target."""

    if str(rank_direction) == "highest":
        return int(rng.randint(18, 70 - (int(panel_index) % 7)))
    if str(rank_direction) == "lowest":
        return int(rng.randint(30 + (int(panel_index) % 7), 86))
    raise ValueError(f"unsupported endpoint rank direction: {rank_direction}")


def _ranked_endpoint_panel(
    endpoint_values: Mapping[str, int], *, rank_direction: str
) -> str:
    """Select the panel with the requested endpoint rank, tie-broken by label."""

    if str(rank_direction) == "highest":
        return max(endpoint_values, key=lambda label: (endpoint_values[label], label))
    if str(rank_direction) == "lowest":
        return min(endpoint_values, key=lambda label: (endpoint_values[label], label))
    raise ValueError(f"unsupported endpoint rank direction: {rank_direction}")


@register_task
class ChartsScientificEndpointRankPanelLabelTask:
    """Select the subplot where one method's start/end marker ranks highest/lowest."""

    task_id = "task_charts__curve_panels__endpoint_rank_panel_label"
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "endpoint_rank_panel_label"
    supported_query_ids = (
        START_HIGHEST_QUERY_ID,
        START_LOWEST_QUERY_ID,
        END_HIGHEST_QUERY_ID,
        END_LOWEST_QUERY_ID,
    )
    default_dataset_enabled = True

    def _build_endpoint_rank_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        try:
            endpoint, rank_direction = QUERY_PARAMS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported endpoint rank query: {selected_query_id}"
            ) from exc

        sample = base_curve_panel_sample(
            params=effective_params,
            instance_seed=int(instance_seed),
            min_x_tick_count=5,
            min_panel_count=4,
            min_method_count=3,
            namespace=f"{SCENE_NAMESPACE}.endpoint_rank",
        )
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.endpoint_rank")
        method_label = choose_method_label(
            method_labels=sample.method_labels,
            params=sample.non_answer_params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.endpoint_rank.method",
        )
        endpoint_index = _endpoint_index_for_query(str(endpoint), sample.x_values)
        endpoint_x = int(sample.x_values[int(endpoint_index)])
        answer_value = (
            int(rng.randint(86, 94))
            if str(rank_direction) == "highest"
            else int(rng.randint(6, 14))
        )
        if str(rank_direction) == "highest":
            for panel_index, panel in enumerate(sample.panel_labels):
                value = (
                    int(answer_value)
                    if str(panel) == str(sample.answer_panel)
                    else _non_answer_endpoint_value(
                        rng=rng,
                        rank_direction=str(rank_direction),
                        panel_index=int(panel_index),
                    )
                )
                sample.values[str(panel)][method_label][int(endpoint_index)] = int(
                    value
                )
        else:
            for panel_index, panel in enumerate(sample.panel_labels):
                value = (
                    int(answer_value)
                    if str(panel) == str(sample.answer_panel)
                    else _non_answer_endpoint_value(
                        rng=rng,
                        rank_direction=str(rank_direction),
                        panel_index=int(panel_index),
                    )
                )
                sample.values[str(panel)][method_label][int(endpoint_index)] = int(
                    value
                )

        endpoint_values = {
            str(panel): int(
                sample.values[str(panel)][method_label][int(endpoint_index)]
            )
            for panel in sample.panel_labels
        }
        selected_panel = _ranked_endpoint_panel(
            endpoint_values, rank_direction=str(rank_direction)
        )
        if str(selected_panel) != str(sample.answer_panel):
            raise RuntimeError("endpoint rank construction lost unique target")

        annotation_id = point_id(sample.answer_panel, method_label, int(endpoint_x))
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=sample.answer_panel,
            answer_type="string",
            panel_label=sample.answer_panel,
            method_label=method_label,
            x_value=endpoint_x,
            annotation_panel_labels=(sample.answer_panel,),
            annotation_point_ids=(annotation_id,),
            trace={
                "method_label": str(method_label),
                "endpoint_position": str(endpoint),
                "endpoint_rank_direction": str(rank_direction),
                "endpoint_x_value": int(endpoint_x),
                "endpoint_values_by_panel": dict(endpoint_values),
                "winning_panel_label": str(sample.answer_panel),
                "winning_point_id": str(annotation_id),
                **dict(sample.panel_label_meta),
            },
        )
        return build_curve_panel_plan_from_query(
            x_values=sample.x_values,
            y_min=sample.y_min,
            y_max=sample.y_max,
            panel_labels=sample.panel_labels,
            method_labels=sample.method_labels,
            colors=sample.colors,
            values_by_panel_method=sample.values,
            query=query,
            dynamic_slots={"method_label": f'"{query.method_label}"'},
            instance_seed=int(instance_seed),
            annotation_type="point",
        )

    def generate(
        self, instance_seed: int, *, params: dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Select the local query, then run neutral curve-panel lifecycle."""

        return run_curve_panel_task_lifecycle(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            supported_query_ids=self.supported_query_ids,
            default_query_id=START_HIGHEST_QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_endpoint_rank_plan,
        )


__all__ = ["ChartsScientificEndpointRankPanelLabelTask"]
