from trace_tasks.core.seed import spawn_rng
from ._lifecycle import (
    build_combo_task_output,
    make_combo_label_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN
from .shared.prompts import build_prompt_artifacts, combo_series_slots
from .shared.sampling import conditioned_extremum_index, sample_base_dataset
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


MAX_QUERY_ID = "max_line_where_primary_above_threshold"
MIN_QUERY_ID = "min_line_where_primary_above_threshold"
LINE_EXTREMUM_BY_QUERY = {MAX_QUERY_ID: "max", MIN_QUERY_ID: "min"}


@register_task
class ChartsComboConditionedLineExtremumLabelTask:
    task_id = "task_charts__combo_mark__conditioned_line_extremum_label"
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = DOMAIN
    objective_contract = "conditioned_line_extremum_label"
    supported_query_ids = (MAX_QUERY_ID, MIN_QUERY_ID)
    default_dataset_enabled = True

    def _build_line_after_primary_filter_plan(self, instance_seed, params, selected_query_id):
        line_filter_dataset, line_filter_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        line_answer_index, primary_cutoff, line_candidate_indices = conditioned_extremum_index(
            labels=line_filter_dataset.labels,
            primary_values=line_filter_dataset.primary_values,
            line_values=line_filter_dataset.line_values,
            condition_role="primary",
            condition_relation="above",
            target_role="line",
            extremum=LINE_EXTREMUM_BY_QUERY[str(selected_query_id)],
            rng=spawn_rng(int(instance_seed), f"{self.task_id}.selection"),
        )
        answer = str(line_filter_dataset.labels[int(line_answer_index)])
        line_filter_prompt = build_prompt_artifacts(
            scene_variant=line_filter_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots=combo_series_slots(line_filter_dataset, threshold_value=str(primary_cutoff)),
            instance_seed=int(instance_seed),
        )
        return make_combo_label_plan(
            dataset=line_filter_dataset,
            dataset_trace=line_filter_trace,
            answer_label=str(answer),
            question_format="conditioned_extremum_query",
            annotation_index=int(line_answer_index),
            relations={
                "primary_cutoff_value": int(primary_cutoff),
                "primary_cutoff_relation": "above",
                "target_index": int(line_answer_index),
                "line_candidate_indices": [int(idx) for idx in line_candidate_indices],
            },
            prompt_artifacts=line_filter_prompt,
        )

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, line_filter_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=MAX_QUERY_ID,
            task_id=self.task_id,
        )
        return run_combo_public_task(
            instance_seed=int(instance_seed),
            params=line_filter_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_line_after_primary_filter_plan,
            build_output=build_combo_task_output,
        )


__all__ = ["ChartsComboConditionedLineExtremumLabelTask"]
