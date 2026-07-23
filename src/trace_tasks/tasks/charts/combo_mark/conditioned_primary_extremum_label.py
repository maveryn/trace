"""Public task for `task_charts__combo_mark__conditioned_primary_extremum_label`."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    combo_task_output_fields,
    make_combo_label_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN
from .shared.prompts import build_prompt_artifacts, combo_series_slots
from .shared.sampling import sample_base_dataset, select_threshold, unique_extremum
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


MAX_QUERY_ID = "max_primary_where_line_below_threshold"
MIN_QUERY_ID = "min_primary_where_line_below_threshold"
PRIMARY_EXTREMUM_BY_QUERY = {MAX_QUERY_ID: "max", MIN_QUERY_ID: "min"}


def _choose_primary_after_line_filter(dataset, *, selected_query_id, rng):
    line_threshold = select_threshold(dataset.line_values, rng=rng, above=False)
    candidate_indices = [
        idx
        for idx, line_value in enumerate(dataset.line_values)
        if int(line_value) < int(line_threshold)
    ]
    answer_label, _answer_value = unique_extremum(
        [dataset.labels[idx] for idx in candidate_indices],
        [dataset.primary_values[idx] for idx in candidate_indices],
        mode=PRIMARY_EXTREMUM_BY_QUERY[str(selected_query_id)],
    )
    return int(tuple(str(label) for label in dataset.labels).index(str(answer_label))), int(line_threshold), candidate_indices


def _build_task_output(materialized):
    return TaskOutput(**combo_task_output_fields(materialized))


@register_task
class ChartsComboConditionedPrimaryExtremumLabelTask:
    """Filter by the line series and find an extremum in the primary mark series."""

    task_id = "task_charts__combo_mark__conditioned_primary_extremum_label"
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = DOMAIN
    objective_contract = "conditioned_primary_extremum_label"
    supported_query_ids = (MAX_QUERY_ID, MIN_QUERY_ID)
    default_dataset_enabled = True
    default_primary_filter_dataset_enabled = True

    def _build_primary_after_line_filter_plan(self, instance_seed, params, selected_query_id):
        """Bind a primary-series extremum after filtering by line values."""

        primary_filter_dataset, primary_filter_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        primary_answer_index, line_cutoff, primary_candidate_indices = _choose_primary_after_line_filter(
            primary_filter_dataset,
            selected_query_id=str(selected_query_id),
            rng=spawn_rng(int(instance_seed), f"{self.task_id}.selection"),
        )
        answer = str(primary_filter_dataset.labels[int(primary_answer_index)])
        primary_filter_prompt = build_prompt_artifacts(
            scene_variant=primary_filter_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots=combo_series_slots(primary_filter_dataset, threshold_value=str(line_cutoff)),
            instance_seed=int(instance_seed),
        )
        return make_combo_label_plan(
            dataset=primary_filter_dataset,
            dataset_trace=primary_filter_trace,
            answer_label=str(answer),
            question_format="conditioned_extremum_query",
            annotation_index=int(primary_answer_index),
            relations={
                "line_cutoff_value": int(line_cutoff),
                "line_cutoff_relation": "below",
                "target_index": int(primary_answer_index),
                "primary_candidate_indices": [int(idx) for idx in primary_candidate_indices],
            },
            prompt_artifacts=primary_filter_prompt,
        )

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, primary_filter_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=MAX_QUERY_ID,
            task_id=self.task_id,
        )
        return run_combo_public_task(
            instance_seed=int(instance_seed),
            params=primary_filter_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_primary_after_line_filter_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsComboConditionedPrimaryExtremumLabelTask"]
