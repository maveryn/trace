"""Public task for `task_charts__waterfall__running_total_extremum_value`."""

from __future__ import annotations

from dataclasses import dataclass

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.waterfall._lifecycle import (
    WaterfallTaskPlan,
    run_waterfall_lifecycle,
)
from trace_tasks.tasks.charts.waterfall.shared.annotations import bbox_artifacts
from trace_tasks.tasks.charts.waterfall.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.waterfall.shared.sampling import sample_waterfall_dataset
from trace_tasks.tasks.charts.waterfall.shared.state import WaterfallDataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__waterfall__running_total_extremum_value"
OBJECTIVE_CONTRACT = "running_total_extremum_value"
PROMPT_QUERY_KEY = "running_total_extremum_value"
MAXIMUM_QUERY_ID = "maximum_running_total"
MINIMUM_QUERY_ID = "minimum_running_total"
SUPPORTED_QUERY_IDS = (MAXIMUM_QUERY_ID, MINIMUM_QUERY_ID)
DEFAULT_QUERY_ID = MAXIMUM_QUERY_ID


@dataclass(frozen=True)
class RunningTotalCandidate:
    """One start/contribution bar candidate for running-total extrema."""

    bar_id: str
    label: str
    running_total: int
    ordinal: int


def _running_total_candidates(dataset: WaterfallDataset) -> tuple[RunningTotalCandidate, ...]:
    candidates = [
        RunningTotalCandidate(
            bar_id="start",
            label="Start",
            running_total=int(dataset.start_value),
            ordinal=0,
        )
    ]
    candidates.extend(
        RunningTotalCandidate(
            bar_id=str(step.step_id),
            label=str(step.label),
            running_total=int(step.running_after),
            ordinal=index + 1,
        )
        for index, step in enumerate(dataset.steps)
    )
    return tuple(candidates)


def _select_unique_extremum(
    candidates: tuple[RunningTotalCandidate, ...],
    *,
    query_id: str,
) -> tuple[RunningTotalCandidate, int]:
    if len(candidates) < 2:
        raise ValueError("waterfall running-total extremum requires multiple candidates")
    select_maximum = str(query_id) == MAXIMUM_QUERY_ID
    ranked = tuple(
        sorted(
            candidates,
            key=lambda candidate: int(candidate.running_total),
            reverse=bool(select_maximum),
        )
    )
    answer = ranked[0]
    nearest = ranked[1]
    if int(answer.running_total) == int(nearest.running_total):
        raise ValueError("waterfall running-total extremum is tied")
    return answer, abs(int(answer.running_total) - int(nearest.running_total))


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind the running-total extremum objective before neutral rendering."""

    if str(selected_branch) not in SUPPORTED_QUERY_IDS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    dataset = sample_waterfall_dataset(
        params,
        instance_seed=int(instance_seed),
        step_count_min=8,
        step_count_max=10,
    )
    candidates = _running_total_candidates(dataset)
    answer, nearest_margin = _select_unique_extremum(
        candidates,
        query_id=str(selected_branch),
    )
    extremum_word = "highest" if str(selected_branch) == MAXIMUM_QUERY_ID else "lowest"
    candidate_trace = [
        {
            "bar_id": str(candidate.bar_id),
            "label": str(candidate.label),
            "running_total": int(candidate.running_total),
            "ordinal": int(candidate.ordinal),
        }
        for candidate in candidates
    ]

    def _bind_annotation(rendered):
        return bbox_artifacts(rendered.bar_bboxes_px[str(answer.bar_id)])

    return WaterfallTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(answer.running_total)),
        annotation_builder=_bind_annotation,
        prompt_query_key=PROMPT_QUERY_KEY,
        dynamic_slots={"extremum_word": str(extremum_word)},
        query_params={
            "extremum_direction": str(selected_branch),
            "extremum_word": str(extremum_word),
            "answer_bar_id": str(answer.bar_id),
            "answer_bar_label": str(answer.label),
            "answer_bar_ordinal": int(answer.ordinal),
            "answer_running_total": int(answer.running_total),
            "nearest_extremum_margin": int(nearest_margin),
            "candidate_running_totals": candidate_trace,
            "step_count": int(len(dataset.steps)),
            "start_value": int(dataset.start_value),
            "final_value": int(dataset.final_value),
            "annotation_role": "answer_bar",
        },
        relations={
            "extremum_direction": str(selected_branch),
            "extremum_word": str(extremum_word),
            "answer_bar_id": str(answer.bar_id),
            "answer_bar_label": str(answer.label),
            "answer_bar_ordinal": int(answer.ordinal),
            "answer_running_total": int(answer.running_total),
            "nearest_extremum_margin": int(nearest_margin),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "waterfall_running_total_extremum_witness",
            "candidate_bar_ids": [str(candidate.bar_id) for candidate in candidates],
            "candidate_running_totals": candidate_trace,
            "direction": str(selected_branch),
            "answer_bar_id": str(answer.bar_id),
            "answer": int(answer.running_total),
        },
        question_format="waterfall_running_total_extremum_value",
    )


class ChartsWaterfallRunningTotalExtremumValueTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_waterfall_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_plan,
        )


register_task(ChartsWaterfallRunningTotalExtremumValueTask)


__all__ = [
    "ChartsWaterfallRunningTotalExtremumValueTask",
    "MAXIMUM_QUERY_ID",
    "MINIMUM_QUERY_ID",
]
