"""Public task for `task_charts__combo_mark__absolute_gap_extremum_label`."""

from __future__ import annotations

from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    combo_task_output_fields,
    make_combo_label_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN
from .shared.prompts import build_prompt_artifacts, combo_series_slots
from .shared.sampling import rank_gap_index, sample_base_dataset
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_ABSOLUTE_QUERY_ID = "largest_absolute_gap_label"
SMALLEST_NONZERO_QUERY_ID = "smallest_nonzero_absolute_gap_label"


def _gap_mode(selected_query_id: str) -> str:
    if str(selected_query_id) == LARGEST_ABSOLUTE_QUERY_ID:
        return "largest_absolute"
    if str(selected_query_id) == SMALLEST_NONZERO_QUERY_ID:
        return "smallest_nonzero_absolute"
    raise ValueError(f"unsupported absolute-gap query: {selected_query_id}")


def _build_task_output(materialized):
    return TaskOutput(**combo_task_output_fields(materialized))


@register_task
class ChartsComboAbsoluteGapExtremumLabelTask:
    """Find the category with an extremal absolute gap between the two encodings."""

    task_id = "task_charts__combo_mark__absolute_gap_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "absolute_gap_extremum_label"
    supported_query_ids = (LARGEST_ABSOLUTE_QUERY_ID, SMALLEST_NONZERO_QUERY_ID)
    default_dataset_enabled = True
    default_absolute_dataset_enabled = True

    def _build_absolute_gap_plan(self, instance_seed, params, selected_query_id):
        """Bind absolute-gap ranking before rendering projects mark centers."""

        absolute_dataset, absolute_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        absolute_index, absolute_gap_value, absolute_candidates = rank_gap_index(
            primary_values=absolute_dataset.primary_values,
            line_values=absolute_dataset.line_values,
            gap_mode=_gap_mode(str(selected_query_id)),
        )
        answer = str(absolute_dataset.labels[int(absolute_index)])
        absolute_prompt = build_prompt_artifacts(
            scene_variant=absolute_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots=combo_series_slots(absolute_dataset),
            instance_seed=int(instance_seed),
        )
        absolute_relations = {
            "target_index": int(absolute_index),
            "target_label": str(answer),
            "target_absolute_gap_value": int(absolute_gap_value),
            "absolute_candidates": [int(idx) for idx in absolute_candidates],
            "absolute_gap_values": {
                str(absolute_dataset.labels[idx]): int(abs(int(absolute_dataset.primary_values[idx]) - int(absolute_dataset.line_values[idx])))
                for idx in range(len(absolute_dataset.labels))
            },
            "signed_absolute_gap_values": {
                str(absolute_dataset.labels[idx]): int(absolute_dataset.primary_values[idx]) - int(absolute_dataset.line_values[idx])
                for idx in range(len(absolute_dataset.labels))
            },
        }
        return make_combo_label_plan(
            dataset=absolute_dataset,
            dataset_trace=absolute_trace,
            answer_label=str(answer),
            question_format="gap_extremum_label_query",
            annotation_index=int(absolute_index),
            relations=absolute_relations,
            prompt_artifacts=absolute_prompt,
        )

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, absolute_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_ABSOLUTE_QUERY_ID,
            task_id=self.task_id,
        )
        return run_combo_public_task(
            instance_seed=int(instance_seed),
            params=absolute_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_absolute_gap_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsComboAbsoluteGapExtremumLabelTask"]
