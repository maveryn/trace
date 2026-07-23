"""Public task for `task_charts__combo_mark__directional_gap_extremum_label`."""

from __future__ import annotations

from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    combo_task_output_fields,
    make_combo_label_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN
from .shared.prompts import build_prompt_artifacts, combo_series_slots
from .shared.sampling import sample_base_dataset
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


PRIMARY_OVER_LINE_QUERY_ID = "largest_primary_over_line_gap_label"
LINE_OVER_PRIMARY_QUERY_ID = "largest_line_over_primary_gap_label"


def _select_directional_gap(primary_values, line_values, selected_query_id):
    sign = 1 if str(selected_query_id) == PRIMARY_OVER_LINE_QUERY_ID else -1
    candidates = [
        (idx, int(sign) * (int(primary_value) - int(line_value)))
        for idx, (primary_value, line_value) in enumerate(zip(primary_values, line_values))
        if int(sign) * (int(primary_value) - int(line_value)) > 0
    ]
    if not candidates:
        raise ValueError("no directional-gap candidates")
    best_gap = max(gap for _idx, gap in candidates)
    winners = [idx for idx, gap in candidates if int(gap) == int(best_gap)]
    if len(winners) != 1:
        raise ValueError("directional-gap tie")
    return int(winners[0]), int(best_gap), [int(idx) for idx, _gap in candidates]


def _build_task_output(materialized):
    return TaskOutput(**combo_task_output_fields(materialized))


@register_task
class ChartsComboDirectionalGapExtremumLabelTask:
    """Find the category with the largest directional gap between the two encodings."""

    task_id = "task_charts__combo_mark__directional_gap_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "directional_gap_extremum_label"
    supported_query_ids = (PRIMARY_OVER_LINE_QUERY_ID, LINE_OVER_PRIMARY_QUERY_ID)
    default_dataset_enabled = True
    default_direction_dataset_enabled = True

    def _build_directional_gap_plan(self, instance_seed, params, selected_query_id):
        """Bind directional-gap ranking before rendering projects mark centers."""

        direction_dataset, direction_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        direction_index, direction_gap_value, direction_candidates = _select_directional_gap(
            direction_dataset.primary_values,
            direction_dataset.line_values,
            str(selected_query_id),
        )
        answer = str(direction_dataset.labels[int(direction_index)])
        direction_prompt = build_prompt_artifacts(
            scene_variant=direction_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots=combo_series_slots(direction_dataset),
            instance_seed=int(instance_seed),
        )
        direction_signed_gaps = {
            str(direction_dataset.labels[idx]): int(direction_dataset.primary_values[idx]) - int(direction_dataset.line_values[idx])
            for idx in range(len(direction_dataset.labels))
        }
        return make_combo_label_plan(
            dataset=direction_dataset,
            dataset_trace=direction_trace,
            answer_label=str(answer),
            question_format="gap_extremum_label_query",
            annotation_index=int(direction_index),
            relations={
                "target_index": int(direction_index),
                "target_label": str(answer),
                "target_gap_value": int(direction_gap_value),
                "direction_candidates": [int(idx) for idx in direction_candidates],
                "gap_values": {label: int(abs(value)) for label, value in direction_signed_gaps.items()},
                "signed_gap_values": direction_signed_gaps,
            },
            prompt_artifacts=direction_prompt,
        )

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, direction_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=PRIMARY_OVER_LINE_QUERY_ID,
            task_id=self.task_id,
        )
        return run_combo_public_task(
            instance_seed=int(instance_seed),
            params=direction_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_directional_gap_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsComboDirectionalGapExtremumLabelTask"]
