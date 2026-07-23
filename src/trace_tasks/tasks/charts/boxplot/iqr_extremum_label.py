"""Public task for `task_charts__boxplot__iqr_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.boxplot._lifecycle import (
    SingleBoxplotTaskPlan,
    boxplot_attempt_seed,
    materialize_single_boxplot_plan,
)
from trace_tasks.tasks.charts.boxplot.shared.defaults import DOMAIN, SCENE_ID, SCENE_NAMESPACE, merge_task_defaults
from trace_tasks.tasks.charts.boxplot.shared.prompts import SINGLE_SCENE_PROMPT_KEY, build_prompt_artifacts
from trace_tasks.tasks.charts.boxplot.shared.rendering import resolve_mark_style
from trace_tasks.tasks.charts.boxplot.shared.sampling import (
    build_boxplot_for_median,
    choose_category_count,
    quartiles_by_label,
    resolve_value_bounds,
    sample_clustered_unique,
    sample_clustered_unique_low,
    sample_labels,
)
from trace_tasks.tasks.charts.shared.chart_scene_types import BoxPlotSpec
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions


TASK_ID = "task_charts__boxplot__iqr_extremum_label"
OBJECTIVE_CONTRACT = "iqr_extremum_label"
EXTREMUM_BRANCHES = ("largest", "smallest")
LARGEST_IQR_QUERY_ID = "largest_iqr_label"
SMALLEST_IQR_QUERY_ID = "smallest_iqr_label"
EXTREMUM_BY_QUERY_ID = {
    LARGEST_IQR_QUERY_ID: "largest",
    SMALLEST_IQR_QUERY_ID: "smallest",
}
TASK_PARAM_DEFAULTS = {
    "iqr_winner_gap_min": 1,
    "iqr_winner_gap_max": 1,
}


def _prompt_slots(direction: str) -> dict[str, str]:
    if str(direction) == "largest":
        return {"extremum_direction": "largest", "iqr_width_phrase": "widest"}
    if str(direction) == "smallest":
        return {"extremum_direction": "smallest", "iqr_width_phrase": "narrowest"}
    raise ValueError(f"unsupported IQR extremum direction: {direction}")


def _sample_iqr_values(
    *,
    direction: str,
    category_count: int,
    value_min: int,
    value_max: int,
    params: dict[str, Any],
    rng: Any,
) -> list[int]:
    """Sample IQR values with a unique largest or smallest winner."""

    support_min = 2
    support_max = int(value_max) - int(value_min) - 3
    feasible_iqrs = list(range(int(support_min), int(support_max) + 1))
    if len(feasible_iqrs) < int(category_count):
        raise ValueError("boxplot IQR support is too small for requested category count")
    gap_min = int(params.get("iqr_winner_gap_min", 0))
    gap_max = int(params.get("iqr_winner_gap_max", gap_min))
    if int(gap_max) <= 0:
        rng.shuffle(feasible_iqrs)
        values = feasible_iqrs[: int(category_count)]
        rng.shuffle(values)
        return [int(value) for value in values]

    gap_min = max(1, int(gap_min))
    gap_max = max(int(gap_min), int(gap_max))
    if str(direction) == "largest":
        answer_min = int(support_min) + int(gap_min) + int(category_count) - 2
        if int(answer_min) > int(support_max):
            raise ValueError("boxplot largest-IQR gap support is too small for requested category count")
        winner_iqr = int(rng.randint(int(answer_min), int(support_max)))
        gap_cap = min(int(gap_max), int(winner_iqr) - int(support_min) - int(category_count) + 2)
        runner_up = int(winner_iqr) - int(rng.randint(int(gap_min), int(gap_cap)))
        other_iqrs = sample_clustered_unique(rng, int(support_min), int(runner_up) - 1, int(category_count) - 2)
        values = [int(winner_iqr), int(runner_up), *other_iqrs]
    elif str(direction) == "smallest":
        answer_max = int(support_max) - int(gap_min) - int(category_count) + 2
        if int(answer_max) < int(support_min):
            raise ValueError("boxplot smallest-IQR gap support is too small for requested category count")
        winner_iqr = int(rng.randint(int(support_min), int(answer_max)))
        gap_cap = min(int(gap_max), int(support_max) - int(winner_iqr) - int(category_count) + 2)
        runner_up = int(winner_iqr) + int(rng.randint(int(gap_min), int(gap_cap)))
        other_iqrs = sample_clustered_unique_low(rng, int(runner_up) + 1, int(support_max), int(category_count) - 2)
        values = [int(winner_iqr), int(runner_up), *other_iqrs]
    else:
        raise ValueError(f"unsupported IQR extremum direction: {direction}")
    rng.shuffle(values)
    return [int(value) for value in values]


def _boxplot_for_iqr(
    *,
    label: str,
    iqr: int,
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
) -> BoxPlotSpec:
    q1_min = int(value_min) + 1
    q1_max = int(value_max) - int(iqr) - 2
    if int(q1_min) > int(q1_max):
        raise ValueError("no feasible q1 support for boxplot IQR construction")
    q1 = int(rng.randint(int(q1_min), int(q1_max)))
    q3 = int(q1) + int(iqr)
    median = int(rng.randint(int(q1) + 1, int(q3) - 1))
    whisker_min = max(int(value_min), int(q1) - int(rng.randint(0, min(2, int(q1) - int(value_min)))))
    whisker_max = min(int(value_max), int(q3) + int(rng.randint(0, min(2, int(value_max) - int(q3)))))
    return BoxPlotSpec(
        label=str(label),
        whisker_min=int(whisker_min),
        q1=int(q1),
        median=int(median),
        q3=int(q3),
        whisker_max=int(whisker_max),
        fill_rgb=fill_rgb,
        outline_rgb=outline_rgb,
    )


def _build_iqr_plan(params: dict[str, Any], instance_seed: int, selected_query_id: str) -> SingleBoxplotTaskPlan:
    """Bind the IQR-extremum objective before neutral scene rendering."""

    direction = EXTREMUM_BY_QUERY_ID[str(selected_query_id)]
    direction_probabilities = {value: 1.0 if value == direction else 0.0 for value in EXTREMUM_BRANCHES}
    effective_params = merge_task_defaults(params, TASK_PARAM_DEFAULTS)
    mark_style = resolve_mark_style(effective_params, instance_seed=int(instance_seed), mark_count=1)
    category_count, category_range = choose_category_count(
        params=effective_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.iqr.category_count.{direction}",
    )
    value_min, value_max = resolve_value_bounds(effective_params, instance_seed=int(instance_seed))
    labels = sample_labels(
        count=int(category_count),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.iqr.labels.{direction}.{int(category_count)}",
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.iqr.{direction}")
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    iqrs = _sample_iqr_values(
        direction=str(direction),
        category_count=int(category_count),
        value_min=int(value_min),
        value_max=int(value_max),
        params=effective_params,
        rng=rng,
    )
    boxplots = tuple(
        _boxplot_for_iqr(
            label=str(label),
            iqr=int(iqr),
            value_min=int(value_min),
            value_max=int(value_max),
            rng=rng,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
        )
        for label, iqr in zip(labels, iqrs)
    )
    key_func = max if str(direction) == "largest" else min
    answer_box = key_func(boxplots, key=lambda spec: int(spec.q3) - int(spec.q1))
    answer_label = str(answer_box.label)
    answer_value = int(answer_box.q3) - int(answer_box.q1)
    prompt_artifacts = build_prompt_artifacts(
        scene_key=SINGLE_SCENE_PROMPT_KEY,
        prompt_query_key=str(selected_query_id),
        dynamic_slots=_prompt_slots(str(direction)),
        instance_seed=int(instance_seed),
    )
    relations = {
        "query_id": str(selected_query_id),
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_range[0]), int(category_range[1])],
        "value_range": [int(value_min), int(value_max)],
        "labels": [str(spec.label) for spec in boxplots],
        "answer_label": str(answer_label),
        "annotation_value": int(answer_value),
        "extremum_direction": str(direction),
        "extremum_direction_probabilities": dict(direction_probabilities),
        "quartiles_by_label": quartiles_by_label(boxplots),
    }
    return SingleBoxplotTaskPlan(
        boxplots=boxplots,
        params=effective_params,
        mark_style=mark_style,
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        answer_value=str(answer_label),
        question_format="label_open",
        role_to_label={"answer_boxplot": str(answer_label)},
        relations=relations,
        prompt_artifacts=prompt_artifacts,
        annotation_kind="bbox",
    )


@register_task
class ChartsDistributionBoxplotIqrExtremumLabelTask:
    """Select the boxplot with the largest or smallest interquartile range."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (LARGEST_IQR_QUERY_ID, SMALLEST_IQR_QUERY_ID)
    default_dataset_enabled = True

    def _prepare_iqr_plan(
        self,
        params: dict[str, Any],
        instance_seed: int,
        selected_query_id: str,
    ) -> SingleBoxplotTaskPlan:
        """Build and validate the one-winner IQR annotation contract."""

        plan = _build_iqr_plan(dict(params), int(instance_seed), str(selected_query_id))
        if set(plan.role_to_label) != {"answer_boxplot"}:
            raise RuntimeError("IQR extremum annotation must contain only answer_boxplot")
        if plan.answer_gt.type != "string":
            raise RuntimeError("IQR extremum answer must be a string label")
        return plan

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_IQR_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_seed = boxplot_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = self._prepare_iqr_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_single_boxplot_plan(
                    instance_seed=int(attempt_seed),
                    selected_query_id=str(selected_query_id),
                    plan=plan,
                )
                return TaskOutput(
                    prompt=materialized.prompt,
                    answer_gt=materialized.answer_gt,
                    annotation_gt=materialized.annotation_gt,
                    image=materialized.image,
                    image_id="img0",
                    trace_payload=materialized.trace_payload,
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=materialized.query_id,
                    prompt_variants=materialized.prompt_variants,
                )
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"failed to generate {self.task_id}: {last_error}")


__all__ = ["ChartsDistributionBoxplotIqrExtremumLabelTask"]
