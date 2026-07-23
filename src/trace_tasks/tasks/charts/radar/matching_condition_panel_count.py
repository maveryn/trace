"""Count radar panels satisfying a multi-metric threshold condition."""

from __future__ import annotations

from typing import Any

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import build_radar_dataset_from_components, build_radar_plan, run_radar_task
from .shared.defaults import GEN_DEFAULTS, resolve_gen_int
from .shared.sampling import (
    balanced_choice,
    make_single_profile_panels,
    metric_count,
    panel_count,
    sample_small_multiple_frame,
    target_count_support,
    threshold_value,
    without_sample_cursor,
)
from .shared.state import DOMAIN, SMALL_MULTIPLE_SCENE_VARIANT


TASK_ID = "task_charts__radar__matching_condition_panel_count"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "matching_condition_panel_count"
TASK_PANEL_COUNT_MIN = 4
TASK_PANEL_COUNT_MAX = 6


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Construct panels so exactly the target panels meet the multi-metric condition."""

    task_params = dict(params)
    task_params["panel_count_min"] = TASK_PANEL_COUNT_MIN
    task_params["panel_count_max"] = TASK_PANEL_COUNT_MAX
    non_answer_params = without_sample_cursor(task_params)
    threshold = threshold_value(non_answer_params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.threshold")
    target_count = balanced_choice(
        target_count_support(task_params, upper=min(6, resolve_gen_int(task_params, "panel_count_max", 8) - 1)),
        task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_count",
    )
    resolved_panel_count = panel_count(
        non_answer_params,
        min_required=int(target_count) + 1,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.panel_count",
    )
    resolved_metric_count = metric_count(
        non_answer_params,
        min_required=5,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.metric_count",
    )
    min_condition_low = resolve_gen_int(params, "min_condition_metric_count_min", 2)
    min_condition_high = min(resolve_gen_int(params, "min_condition_metric_count_max", 4), int(resolved_metric_count))
    minimum_metric_count = balanced_choice(
        list(range(int(min_condition_low), int(min_condition_high) + 1)),
        non_answer_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.minimum_metric_count",
    )
    frame = sample_small_multiple_frame(
        params=task_params,
        metric_count=int(resolved_metric_count),
        panel_count=int(resolved_panel_count),
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
    )
    metrics = frame.metrics
    panel_labels = frame.panel_labels
    values = frame.values
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.values")
    panel_indices = list(range(len(panel_labels)))
    rng.shuffle(panel_indices)
    matching_panel_indices = set(int(index) for index in panel_indices[: int(target_count)])
    matching_labels: list[str] = []
    for panel_index, panel in enumerate(panel_labels):
        metric_indices = list(range(len(metrics)))
        rng.shuffle(metric_indices)
        if int(panel_index) in matching_panel_indices:
            above_count = int(rng.randint(int(minimum_metric_count), int(resolved_metric_count)))
            matching_labels.append(str(panel))
        else:
            above_count = int(rng.randint(0, int(minimum_metric_count) - 1))
        above_indices = set(int(index) for index in metric_indices[: int(above_count)])
        for metric_index, metric in enumerate(metrics):
            if int(metric_index) in above_indices:
                values[str(panel)][str(metric)] = int(rng.randint(int(threshold) + 1, int(frame.value_max)))
            else:
                values[str(panel)][str(metric)] = int(rng.randint(int(frame.value_min), int(threshold)))

    annotation_panel_labels = tuple(str(label) for label in matching_labels)
    annotation_point_ids = tuple(
        f"{str(panel)}|Profile|{str(metric)}"
        for panel in panel_labels
        if str(panel) in set(annotation_panel_labels)
        for metric in metrics
        if int(values[str(panel)][str(metric)]) > int(threshold)
    )
    dataset = build_radar_dataset_from_components(
        metrics=tuple(metrics),
        panels=make_single_profile_panels(panel_labels=panel_labels, values_by_panel=values, params=task_params),
        scene_variant=SMALL_MULTIPLE_SCENE_VARIANT,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        highlight_metric_label="",
        answer=int(target_count),
        answer_type="integer",
        annotation_type="bbox_set",
        metric_label="",
        panel_label="",
        profile_a_label="",
        profile_b_label="",
        threshold_value=int(threshold),
        minimum_metric_count=int(minimum_metric_count),
        annotation_point_ids=tuple(annotation_point_ids),
        annotation_panel_labels=tuple(annotation_panel_labels),
        annotation_point_id_pairs=tuple(),
        params={
            "program_code": "count(filter(radar_panels, count(filter(metrics, value(panel, metric) > threshold)) >= minimum_metric_count))",
            "threshold_value": int(threshold),
            "minimum_metric_count": int(minimum_metric_count),
            "matching_panel_labels": list(annotation_panel_labels),
            "values_by_panel": values,
            "minimum_metric_count_bounds": [
                int(group_default(GEN_DEFAULTS, "min_condition_metric_count_min", 2)),
                int(group_default(GEN_DEFAULTS, "min_condition_metric_count_max", 4)),
            ],
            **dict(frame.panel_label_meta),
        },
    )
    return build_radar_plan(dataset=dataset, prompt_query_key=PROMPT_QUERY_KEY)


@register_task
class ChartsRadarMatchingConditionPanelCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "matching_condition_panel_count"
    supported_query_ids = (QUERY_ID,)
    default_query_id = QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radar_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadarMatchingConditionPanelCountTask"]
