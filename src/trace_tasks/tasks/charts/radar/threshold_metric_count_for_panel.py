"""Count metrics above a threshold within one radar panel."""

from __future__ import annotations

from typing import Any

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ._lifecycle import build_radar_dataset_from_components, build_radar_plan, run_radar_task
from .shared.defaults import resolve_gen_int
from .shared.sampling import (
    balanced_choice,
    choose_index,
    force_metric_threshold_by_panel,
    make_single_profile_panels,
    metric_count,
    panel_count,
    sample_small_multiple_frame,
    shuffled_subset,
    target_count_support,
    threshold_value,
    without_sample_cursor,
)
from .shared.state import DOMAIN, SMALL_MULTIPLE_SCENE_VARIANT


TASK_ID = "task_charts__radar__threshold_metric_count_for_panel"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "threshold_metric_count_for_panel"


def _selected_panel_label(panel_labels: tuple[str, ...], params: dict[str, Any], instance_seed: int) -> str:
    panel_index = choose_index(
        len(panel_labels),
        without_sample_cursor(params),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.panel",
    )
    return str(panel_labels[int(panel_index)])


def _annotation_point_ids(panel_label: str, metric_labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{str(panel_label)}|Profile|{str(metric)}" for metric in metric_labels)


def _build_threshold_dataset(
    *,
    selected: str,
    probabilities: dict[str, float],
    frame,
    panel_label: str,
    threshold: int,
    target_count: int,
    matching_metric_labels: tuple[str, ...],
    panels,
):
    """Package the selected-panel point-set contract after task-specific sampling."""

    return build_radar_dataset_from_components(
        metrics=tuple(frame.metrics),
        panels=tuple(panels),
        scene_variant=SMALL_MULTIPLE_SCENE_VARIANT,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        highlight_metric_label="",
        answer=int(target_count),
        answer_type="integer",
        annotation_type="point_set",
        metric_label="",
        panel_label=str(panel_label),
        profile_a_label="",
        profile_b_label="",
        threshold_value=int(threshold),
        minimum_metric_count=0,
        annotation_point_ids=_annotation_point_ids(str(panel_label), tuple(matching_metric_labels)),
        annotation_panel_labels=tuple(),
        annotation_point_id_pairs=tuple(),
        params={
            "program_code": "count(filter(metrics_in_panel, value(selected_panel, metric) > threshold))",
            "query_panel_label": str(panel_label),
            "threshold_value": int(threshold),
            "matching_metric_labels": list(matching_metric_labels),
            "values_by_panel": frame.values,
            **dict(frame.panel_label_meta),
        },
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Construct one queried panel with a controlled count of high metric vertices."""

    non_answer_params = without_sample_cursor(params)
    threshold = threshold_value(non_answer_params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.threshold")
    target_count = balanced_choice(
        target_count_support(params, upper=min(6, resolve_gen_int(params, "metric_count_max", 7) - 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_count",
    )
    resolved_metric_count = metric_count(
        non_answer_params,
        min_required=int(target_count) + 1,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.metric_count",
    )
    resolved_panel_count = panel_count(
        non_answer_params,
        min_required=5,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.panel_count",
    )
    frame = sample_small_multiple_frame(
        params=params,
        metric_count=int(resolved_metric_count),
        panel_count=int(resolved_panel_count),
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
    )
    metrics = frame.metrics
    panel_labels = frame.panel_labels
    values = frame.values
    query_panel_label = _selected_panel_label(tuple(panel_labels), params, int(instance_seed))
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.values")
    matching_metric_labels = shuffled_subset(metrics, int(target_count), rng)
    force_metric_threshold_by_panel(
        values=values,
        metrics=metrics,
        panel_label=str(query_panel_label),
        matching_metric_labels=matching_metric_labels,
        threshold=int(threshold),
        value_min=int(frame.value_min),
        value_max=int(frame.value_max),
        rng=rng,
    )
    dataset = _build_threshold_dataset(
        selected=str(selected),
        probabilities=dict(probabilities),
        frame=frame,
        panel_label=str(query_panel_label),
        threshold=int(threshold),
        target_count=int(target_count),
        matching_metric_labels=tuple(matching_metric_labels),
        panels=make_single_profile_panels(panel_labels=panel_labels, values_by_panel=values, params=params),
    )
    return build_radar_plan(dataset=dataset, prompt_query_key=PROMPT_QUERY_KEY)


@register_task
class ChartsRadarThresholdMetricCountForPanelTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "threshold_metric_count_for_panel"
    supported_query_ids = (QUERY_ID,)
    default_query_id = QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radar_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadarThresholdMetricCountForPanelTask"]
