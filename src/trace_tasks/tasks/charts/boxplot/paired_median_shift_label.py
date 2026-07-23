"""Public task for `task_charts__boxplot__paired_median_shift_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.boxplot._lifecycle import (
    PairedBoxplotTaskPlan,
    boxplot_attempt_seed,
    materialize_paired_boxplot_plan,
)
from trace_tasks.tasks.charts.boxplot.shared.defaults import DOMAIN, SCENE_ID, SCENE_NAMESPACE, GENERATION_DEFAULTS, merge_task_defaults
from trace_tasks.tasks.charts.boxplot.shared.prompts import PAIRED_SCENE_PROMPT_KEY, build_prompt_artifacts
from trace_tasks.tasks.charts.boxplot.shared.rendering import resolve_mark_style
from trace_tasks.tasks.charts.boxplot.shared.sampling import (
    build_boxplot_for_median,
    choose_category_count,
    boxplot_stats,
    resolve_value_bounds,
    sample_labels,
)
from trace_tasks.tasks.charts.shared.chart_scene_types import BoxPlotSpec
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions


TASK_ID = "task_charts__boxplot__paired_median_shift_label"
OBJECTIVE_CONTRACT = "paired_median_shift_label"
INCREASE_QUERY_ID = "paired_median_greatest_increase_label"
DECREASE_QUERY_ID = "paired_median_greatest_decrease_label"
ABSOLUTE_QUERY_ID = "paired_median_greatest_absolute_change_label"
TASK_PARAM_DEFAULTS = {
    "category_count_min": 4,
    "category_count_max": 6,
    "paired_median_shift_min": 2,
    "paired_median_shift_max": 10,
}


def _signed_shifts_for_query(
    *,
    selected_query_id: str,
    shift_magnitudes: list[int],
    rng: Any,
) -> tuple[list[int], int, str]:
    """Resolve the median-shift direction program for the selected query."""

    if str(selected_query_id) == INCREASE_QUERY_ID:
        signed_shifts = [int(value) for value in shift_magnitudes]
        answer_index = max(range(len(signed_shifts)), key=lambda idx: int(signed_shifts[idx]))
        return signed_shifts, int(answer_index), "increase"
    if str(selected_query_id) == DECREASE_QUERY_ID:
        signed_shifts = [-int(value) for value in shift_magnitudes]
        answer_index = max(range(len(signed_shifts)), key=lambda idx: -int(signed_shifts[idx]))
        return signed_shifts, int(answer_index), "decrease"
    if str(selected_query_id) == ABSOLUTE_QUERY_ID:
        signed_shifts = [int(value) if int(rng.randint(0, 1)) == 1 else -int(value) for value in shift_magnitudes]
        answer_index = max(range(len(signed_shifts)), key=lambda idx: abs(int(signed_shifts[idx])))
        return signed_shifts, int(answer_index), "absolute_change"
    raise ValueError(f"unsupported paired boxplot query id: {selected_query_id}")


def _build_paired_specs(
    *,
    labels: tuple[str, ...],
    signed_shifts: list[int],
    value_min: int,
    value_max: int,
    rng: Any,
    before_fill: tuple[int, int, int],
    after_fill: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
) -> tuple[tuple[BoxPlotSpec, ...], tuple[BoxPlotSpec, ...], dict[str, dict[str, Any]]]:
    """Build matched before/after boxplots from task-selected signed shifts."""

    median_min = int(value_min) + 2
    median_max = int(value_max) - 2
    before_specs: list[BoxPlotSpec] = []
    after_specs: list[BoxPlotSpec] = []
    pairs: dict[str, dict[str, Any]] = {}
    for base_label, shift in zip(labels, signed_shifts):
        if int(shift) >= 0:
            before_min = int(median_min)
            before_max = int(median_max) - int(shift)
        else:
            before_min = int(median_min) - int(shift)
            before_max = int(median_max)
        if int(before_min) > int(before_max):
            raise ValueError("no feasible paired median support for requested shift")
        before_median = int(rng.randint(int(before_min), int(before_max)))
        after_median = int(before_median) + int(shift)
        before_specs.append(
            build_boxplot_for_median(
                label=str(base_label),
                median=int(before_median),
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=before_fill,
                outline_rgb=outline_rgb,
            )
        )
        after_specs.append(
            build_boxplot_for_median(
                label=str(base_label),
                median=int(after_median),
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=after_fill,
                outline_rgb=outline_rgb,
            )
        )
        pairs[str(base_label)] = {
            "before_label": f"{str(base_label)}__before",
            "after_label": f"{str(base_label)}__after",
            "display_label": str(base_label),
            "before_panel": "before",
            "after_panel": "after",
            "before_median": int(before_median),
            "after_median": int(after_median),
            "signed_shift": int(shift),
            "absolute_shift": abs(int(shift)),
        }
    return tuple(before_specs), tuple(after_specs), pairs


def _paired_quartiles_by_label(
    before_boxplots: tuple[BoxPlotSpec, ...],
    after_boxplots: tuple[BoxPlotSpec, ...],
) -> dict[str, dict[str, Any]]:
    return {
        **{
            f"{str(spec.label)}__before": {
                "display_label": str(spec.label),
                "panel": "before",
                **boxplot_stats(spec),
            }
            for spec in before_boxplots
        },
        **{
            f"{str(spec.label)}__after": {
                "display_label": str(spec.label),
                "panel": "after",
                **boxplot_stats(spec),
            }
            for spec in after_boxplots
        },
    }


def _quartiles_by_base_label(
    *,
    labels: tuple[str, ...],
    before_boxplots: tuple[BoxPlotSpec, ...],
    after_boxplots: tuple[BoxPlotSpec, ...],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(base_label): {
            "before": boxplot_stats(before_spec),
            "after": boxplot_stats(after_spec),
        }
        for base_label, before_spec, after_spec in zip(labels, before_boxplots, after_boxplots)
    }


def _build_shift_plan(params: dict[str, Any], instance_seed: int, selected_query_id: str) -> PairedBoxplotTaskPlan:
    """Bind the paired median-shift objective before neutral rendering."""

    effective_params = merge_task_defaults(params, TASK_PARAM_DEFAULTS)
    mark_style = resolve_mark_style(effective_params, instance_seed=int(instance_seed), mark_count=1)
    category_count, category_range = choose_category_count(
        params=effective_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.paired.category_count.{selected_query_id}",
    )
    value_min, value_max = resolve_value_bounds(effective_params, instance_seed=int(instance_seed))
    median_min = int(value_min) + 2
    median_max = int(value_max) - 2
    max_shift_support = max(1, (int(median_max) - int(median_min)) // 2)
    shift_min = max(1, int(effective_params.get("paired_median_shift_min", GENERATION_DEFAULTS.get("paired_median_shift_min", 2))))
    shift_max = min(
        int(max_shift_support),
        max(int(shift_min), int(effective_params.get("paired_median_shift_max", GENERATION_DEFAULTS.get("paired_median_shift_max", 12)))),
    )
    shift_support = list(range(int(shift_min), int(shift_max) + 1))
    if len(shift_support) < int(category_count):
        raise ValueError("paired boxplot shift support is too small for requested category count")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.paired.{selected_query_id}")
    labels = sample_labels(
        count=int(category_count),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.paired.labels.{selected_query_id}.{int(category_count)}",
    )
    shift_magnitudes = [int(value) for value in rng.sample(shift_support, int(category_count))]
    rng.shuffle(shift_magnitudes)
    signed_shifts, answer_index, shift_direction = _signed_shifts_for_query(
        selected_query_id=str(selected_query_id),
        shift_magnitudes=shift_magnitudes,
        rng=rng,
    )
    before_fill = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    after_fill = tuple(int(max(20, min(235, round(0.55 * float(channel) + 70.0)))) for channel in reversed(before_fill))
    before_boxplots, after_boxplots, pairs = _build_paired_specs(
        labels=labels,
        signed_shifts=signed_shifts,
        value_min=int(value_min),
        value_max=int(value_max),
        rng=rng,
        before_fill=before_fill,
        after_fill=after_fill,
        outline_rgb=outline_rgb,
    )
    answer_label = str(labels[int(answer_index)])
    annotation_labels = [str(pairs[str(answer_label)]["before_label"]), str(pairs[str(answer_label)]["after_label"])]
    prompt_artifacts = build_prompt_artifacts(
        scene_key=PAIRED_SCENE_PROMPT_KEY,
        prompt_query_key=str(selected_query_id),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    relations = {
        "query_id": str(selected_query_id),
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_range[0]), int(category_range[1])],
        "rendered_boxplot_count": int(len(before_boxplots) + len(after_boxplots)),
        "rendered_boxplot_count_range": [int(category_range[0]) * 2, int(category_range[1]) * 2],
        "value_range": [int(value_min), int(value_max)],
        "base_labels": [str(label) for label in labels],
        "labels": [str(label) for label in labels],
        "rendered_labels": [*(f"{str(label)}__before" for label in labels), *(f"{str(label)}__after" for label in labels)],
        "answer_label": str(answer_label),
        "answer_shift": int(pairs[str(answer_label)]["signed_shift"]),
        "answer_absolute_shift": int(pairs[str(answer_label)]["absolute_shift"]),
        "shift_direction": str(shift_direction),
        "annotation_labels": list(annotation_labels),
        "paired_panels": {"before": "Before", "after": "After"},
        "pairs_by_base_label": dict(pairs),
        "quartiles_by_label": _paired_quartiles_by_label(before_boxplots, after_boxplots),
        "quartiles_by_base_label": _quartiles_by_base_label(
            labels=labels,
            before_boxplots=before_boxplots,
            after_boxplots=after_boxplots,
        ),
        "before_medians_by_label": {str(spec.label): int(spec.median) for spec in before_boxplots},
        "after_medians_by_label": {str(spec.label): int(spec.median) for spec in after_boxplots},
    }
    return PairedBoxplotTaskPlan(
        before_boxplots=before_boxplots,
        after_boxplots=after_boxplots,
        params=effective_params,
        mark_style=mark_style,
        before_title="Before",
        after_title="After",
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        answer_value=str(answer_label),
        question_format="label_open",
        role_to_label={"before_boxplot": str(annotation_labels[0]), "after_boxplot": str(annotation_labels[1])},
        relations=relations,
        prompt_artifacts=prompt_artifacts,
    )


@register_task
class ChartsDistributionBoxplotPairedMedianShiftLabelTask:
    """Select the matched label with the requested before/after median shift."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (INCREASE_QUERY_ID, DECREASE_QUERY_ID, ABSOLUTE_QUERY_ID)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=INCREASE_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_seed = boxplot_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = _build_shift_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_paired_boxplot_plan(
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


__all__ = ["ChartsDistributionBoxplotPairedMedianShiftLabelTask"]
