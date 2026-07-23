"""Public task for `task_charts__multiseries__ranked_change_extremum_label`."""
from __future__ import annotations

from typing import Any, Mapping

from ._lifecycle import (
    MultiseriesTaskPlan,
    build_label_selection_plan,
    run_configured_multiseries_task,
    selected_trace_fields,
)
from .shared.defaults import DEFAULTS, DOMAIN, GEN_DEFAULTS
from ...registry import register_task
from .shared.data import build_delta_extremum_label_dataset
from .shared.prompts import (
    build_prompt_artifacts,
    change_measure_prompt_slots,
    change_prompt_slots,
    extremum_prompt_slots,
    object_description,
    ranked_phrase,
)
from .shared.sampling import (
    balance_answer_label_for_indexed_probe,
    internal_change_variant,
    params_for_variant_family,
    resolve_scene_variant,
)


TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "value_window_span_min": 24,
    "value_window_span_max": 25,
    "delta_category_count_min": 7,
    "delta_category_count_max": 10,
    "delta_series_count_min": 3,
    "delta_series_count_max": 4,
    "delta_value_min": 1,
    "delta_value_max": 30,
    "rank_min": 1,
    "rank_max": 1,
    "derived_score_min": 4,
    "derived_score_max": 24,
    "score_spread_extra_min": 0,
    "score_spread_extra_max": 4,
}
LARGEST_INCREASE_QUERY_ID = "largest_increase_label"
LARGEST_DECREASE_QUERY_ID = "largest_decrease_label"
LARGEST_ABSOLUTE_GAP_QUERY_ID = "largest_absolute_gap_label"
SMALLEST_ABSOLUTE_GAP_QUERY_ID = "smallest_absolute_gap_label"
RANKED_CHANGE_QUERY_IDS = (
    LARGEST_INCREASE_QUERY_ID,
    LARGEST_DECREASE_QUERY_ID,
    LARGEST_ABSOLUTE_GAP_QUERY_ID,
    SMALLEST_ABSOLUTE_GAP_QUERY_ID,
)
BRANCH_BY_QUERY_ID: dict[str, dict[str, str | None]] = {
    LARGEST_INCREASE_QUERY_ID: {
        "change_measure": "directional_change",
        "change_direction": "increase",
        "extremum_direction": None,
        "prompt_query_key": "ranked_directional_change",
    },
    LARGEST_DECREASE_QUERY_ID: {
        "change_measure": "directional_change",
        "change_direction": "decrease",
        "extremum_direction": None,
        "prompt_query_key": "ranked_directional_change",
    },
    LARGEST_ABSOLUTE_GAP_QUERY_ID: {
        "change_measure": "absolute_gap",
        "change_direction": None,
        "extremum_direction": "largest",
        "prompt_query_key": "ranked_absolute_gap",
    },
    SMALLEST_ABSOLUTE_GAP_QUERY_ID: {
        "change_measure": "absolute_gap",
        "change_direction": None,
        "extremum_direction": "smallest",
        "prompt_query_key": "ranked_absolute_gap",
    },
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the ranked change/gap objective before neutral rendering."""

    branch = BRANCH_BY_QUERY_ID[str(selected_query_id)]
    change_measure = str(branch["change_measure"])
    change_direction = branch["change_direction"]
    extremum_direction = branch["extremum_direction"]
    prompt_query_key = str(branch["prompt_query_key"])
    dataset_params = {
        **dict(params),
        "change_measure": str(change_measure),
        "rank_min": 1,
        "rank_max": 1,
    }
    if change_direction is not None:
        dataset_params["change_direction"] = str(change_direction)
    if extremum_direction is not None:
        dataset_params["extremum_direction"] = str(extremum_direction)
    internal_query_id = internal_change_variant(
        change_measure=str(change_measure),
        change_direction=change_direction,
        extremum_direction=extremum_direction,
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    values_by_category, answer_label, annotation_values, trace_extras = build_delta_extremum_label_dataset(
        variant_key=str(internal_query_id),
        params=params_for_variant_family(dataset_params, family="delta"),
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        namespace="ranked_change_extremum_label",
    )
    values_by_category, answer_label, trace_extras = balance_answer_label_for_indexed_probe(
        namespace="ranked_change_extremum_label",
        params=dataset_params,
        instance_seed=int(instance_seed),
        values_by_category=values_by_category,
        answer_label=str(answer_label),
        trace_extras=trace_extras,
    )
    answer_rank = int(trace_extras["answer_rank"])
    dynamic_slots = {
        "object_description": object_description(str(scene_variant)),
        "left_series": str(trace_extras.get("left_series_label", "")),
        "right_series": str(trace_extras.get("right_series_label", "")),
        "rank": answer_rank,
        "ranked_largest": ranked_phrase(answer_rank, "largest"),
        "ranked_greatest": ranked_phrase(answer_rank, "greatest"),
        **change_prompt_slots(change_direction),
        **change_measure_prompt_slots(change_measure, change_direction),
        **extremum_prompt_slots(extremum_direction, answer_rank=answer_rank),
    }
    prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(prompt_query_key), dynamic_slots=dynamic_slots, instance_seed=int(instance_seed))
    optional_trace = selected_trace_fields(
        trace_extras,
        (
            "left_series_label",
            "right_series_label",
            "derived_metric",
            "answer_rank",
            "answer_score",
            "rank_order",
            "ranked_category_labels",
            "derived_values_by_category",
        ),
        extra={
            "change_measure": str(change_measure),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
    )
    if change_direction is not None:
        optional_trace["change_direction"] = str(change_direction)
    if extremum_direction is not None:
        optional_trace["extremum_direction"] = str(extremum_direction)
    return build_label_selection_plan(
        values_by_category=values_by_category,
        trace_extras=trace_extras,
        scene_variant=str(scene_variant),
        prompt_artifacts=prompt_artifacts,
        answer_label=str(answer_label),
        annotation_values=annotation_values,
        annotation_category_labels=[str(answer_label)],
        annotation_series_labels=trace_extras["queried_series_labels"],
        variant_family="delta",
        internal_query_id=str(internal_query_id),
        optional_trace=optional_trace,
    )


@register_task
class ChartsMultiseriesRankedChangeExtremumTask:
    """Return the category label at a ranked change or absolute-gap extremum."""

    task_id = "task_charts__multiseries__ranked_change_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "ranked_change_extremum_label"
    supported_query_ids = RANKED_CHANGE_QUERY_IDS
    default_dataset_enabled = True

    default_query_id = LARGEST_INCREASE_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "LARGEST_ABSOLUTE_GAP_QUERY_ID",
    "LARGEST_DECREASE_QUERY_ID",
    "LARGEST_INCREASE_QUERY_ID",
    "RANKED_CHANGE_QUERY_IDS",
    "SMALLEST_ABSOLUTE_GAP_QUERY_ID",
    "ChartsMultiseriesRankedChangeExtremumTask",
]
