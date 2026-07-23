"""Count metrics where one radar profile exceeds another."""

from __future__ import annotations

from typing import Any

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ._lifecycle import build_radar_dataset_from_components, build_radar_plan, run_radar_task
from ..shared.label_assets import resolve_chart_entity_labels
from .shared.defaults import profile_palette, resolve_gen_int
from .shared.sampling import balanced_choice, metric_count, sample_metrics, target_count_support, value_bounds, without_sample_cursor
from .shared.state import DOMAIN, RadarPanel, RadarProfile, SINGLE_PROFILE_SCENE_VARIANT


TASK_ID = "task_charts__radar__profile_advantage_count"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "profile_advantage_count"


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Construct two profiles so the first profile leads on the target metric count."""

    value_min, value_max = value_bounds(params)
    non_answer_params = without_sample_cursor(params)
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
    metrics = sample_metrics(int(resolved_metric_count), instance_seed=int(instance_seed), namespace=f"{TASK_ID}.metric_labels")
    profile_labels = resolve_chart_entity_labels(
        spawn_rng(int(instance_seed), f"{TASK_ID}.profile_labels"),
        count=2,
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.values")
    metric_indices = list(range(len(metrics)))
    rng.shuffle(metric_indices)
    advantage_indices = set(int(index) for index in metric_indices[: int(target_count)])
    values_a: dict[str, int] = {}
    values_b: dict[str, int] = {}
    for index, metric in enumerate(metrics):
        if int(index) in advantage_indices:
            high = int(rng.randint(max(int(value_min) + 1, 4), int(value_max)))
            low = int(rng.randint(int(value_min), int(high) - 1))
            values_a[str(metric)] = int(high)
            values_b[str(metric)] = int(low)
        else:
            high = int(rng.randint(max(int(value_min) + 1, 4), int(value_max)))
            low = int(rng.randint(int(value_min), int(high)))
            values_a[str(metric)] = int(low)
            values_b[str(metric)] = int(high)

    colors = profile_palette(params)
    panel = RadarPanel(
        panel_label="",
        profiles=(
            RadarProfile(profile_label=str(profile_labels[0]), values=dict(values_a), color_rgb=tuple(colors[0])),
            RadarProfile(profile_label=str(profile_labels[1]), values=dict(values_b), color_rgb=tuple(colors[1])),
        ),
    )
    advantage_metric_labels = tuple(
        str(metric)
        for metric in metrics
        if int(values_a[str(metric)]) > int(values_b[str(metric)])
    )
    annotation_point_id_pairs = tuple(
        (
            f"|{str(profile_labels[0])}|{str(metric)}",
            f"|{str(profile_labels[1])}|{str(metric)}",
        )
        for metric in advantage_metric_labels
    )
    dataset = build_radar_dataset_from_components(
        metrics=tuple(metrics),
        panels=(panel,),
        scene_variant=SINGLE_PROFILE_SCENE_VARIANT,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        highlight_metric_label="",
        answer=int(target_count),
        answer_type="integer",
        annotation_type="segment_set",
        metric_label="",
        panel_label="",
        profile_a_label=str(profile_labels[0]),
        profile_b_label=str(profile_labels[1]),
        threshold_value=0,
        minimum_metric_count=0,
        annotation_point_ids=tuple(point_id for pair in annotation_point_id_pairs for point_id in pair),
        annotation_panel_labels=tuple(),
        annotation_point_id_pairs=tuple(annotation_point_id_pairs),
        params={
            "program_code": "count(filter(metrics, value(profile_a, metric) > value(profile_b, metric)))",
            "profile_a_label": str(profile_labels[0]),
            "profile_b_label": str(profile_labels[1]),
            "advantage_metric_labels": list(advantage_metric_labels),
            "values_by_profile": {
                str(profile_labels[0]): dict(values_a),
                str(profile_labels[1]): dict(values_b),
            },
        },
    )
    return build_radar_plan(dataset=dataset, prompt_query_key=PROMPT_QUERY_KEY)


@register_task
class ChartsRadarProfileAdvantageCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    objective_contract = "profile_advantage_count"
    supported_query_ids = (QUERY_ID,)
    default_query_id = QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radar_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadarProfileAdvantageCountTask"]
