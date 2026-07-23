"""Count radar panels whose highlighted metric exceeds a threshold."""

from __future__ import annotations

from typing import Any

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ._lifecycle import build_radar_dataset_from_components, build_radar_plan, run_radar_task
from .shared.sampling import (
    axis_metric_count,
    axis_panel_count,
    balanced_choice,
    choose_index,
    force_panel_threshold_by_metric,
    make_single_profile_panels,
    sample_small_multiple_frame,
    shuffled_subset,
    target_count_support,
    threshold_value,
    without_sample_cursor,
)
from .shared.state import DOMAIN, SMALL_MULTIPLE_SCENE_VARIANT


TASK_ID = "task_charts__radar__highlighted_metric_threshold_panel_count"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "highlighted_metric_threshold_panel_count"


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Construct panels so exactly the target count exceeds the highlighted spoke threshold."""

    non_answer_params = without_sample_cursor(params)
    panel_count = axis_panel_count(
        non_answer_params,
        min_required=5,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.panel_count",
    )
    target_count = balanced_choice(
        target_count_support(params, upper=min(6, int(panel_count) - 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_count",
    )
    threshold = threshold_value(non_answer_params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.threshold")
    metric_count = axis_metric_count(
        non_answer_params,
        min_required=5,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.metric_count",
    )
    frame = sample_small_multiple_frame(
        params=params,
        metric_count=int(metric_count),
        panel_count=int(panel_count),
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        metric_seed_offset=101,
    )
    metrics = frame.metrics
    panel_labels = frame.panel_labels
    values = frame.values
    metric_index = choose_index(
        len(metrics),
        non_answer_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.highlight_metric",
    )
    metric_label = str(metrics[int(metric_index)])
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.values")
    annotation_panel_labels = shuffled_subset(panel_labels, int(target_count), rng)
    force_panel_threshold_by_metric(
        values=values,
        panel_labels=panel_labels,
        metric_label=str(metric_label),
        matching_panel_labels=annotation_panel_labels,
        threshold=int(threshold),
        value_min=int(frame.value_min),
        value_max=int(frame.value_max),
        rng=rng,
    )

    layout_panel_labels = list(panel_labels)
    layout_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.panel_layout")
    layout_rng.shuffle(layout_panel_labels)
    panels = make_single_profile_panels(panel_labels=layout_panel_labels, values_by_panel=values, params=params)
    annotation_point_ids = tuple(f"{str(panel)}|Profile|{str(metric_label)}" for panel in annotation_panel_labels)
    dataset = build_radar_dataset_from_components(
        metrics=tuple(metrics),
        panels=tuple(panels),
        scene_variant=SMALL_MULTIPLE_SCENE_VARIANT,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        highlight_metric_label=str(metric_label),
        answer=int(target_count),
        answer_type="integer",
        annotation_type="bbox_set",
        metric_label=str(metric_label),
        panel_label="",
        profile_a_label="",
        profile_b_label="",
        threshold_value=int(threshold),
        minimum_metric_count=0,
        annotation_point_ids=tuple(annotation_point_ids),
        annotation_panel_labels=tuple(annotation_panel_labels),
        annotation_point_id_pairs=tuple(),
        params={
            "program_code": "count(filter(radar_panels, value(panel, highlighted_metric) > threshold))",
            "query_metric_label": str(metric_label),
            "threshold_value": int(threshold),
            "matching_panel_labels": list(annotation_panel_labels),
            "panel_layout_labels": [str(panel) for panel in layout_panel_labels],
            "values_by_panel": values,
            **dict(frame.panel_label_meta),
        },
    )
    return build_radar_plan(dataset=dataset, prompt_query_key=PROMPT_QUERY_KEY)


@register_task
class ChartsRadarHighlightedMetricThresholdPanelCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "highlighted_metric_threshold_panel_count"
    supported_query_ids = (QUERY_ID,)
    default_query_id = QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radar_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadarHighlightedMetricThresholdPanelCountTask"]
