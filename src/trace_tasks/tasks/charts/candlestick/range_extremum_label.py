"""Public task for `task_charts__candlestick__range_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.candlestick.shared.annotations import annotation_segments
from trace_tasks.tasks.charts.candlestick.shared.defaults import (
    DOMAIN,
    SCENE_ID,
)
from trace_tasks.tasks.charts.candlestick.shared.output import build_trace_scaffold
from trace_tasks.tasks.charts.candlestick.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.candlestick.shared.rendering import render_dataset
from trace_tasks.tasks.charts.candlestick.shared.sampling import sample_candles
from trace_tasks.tasks.charts.candlestick.shared.state import Dataset, Selection
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


LARGEST_WICK_QUERY_ID = "largest_wick_range_label"
SMALLEST_WICK_QUERY_ID = "smallest_wick_range_label"
LARGEST_BODY_QUERY_ID = "largest_body_range_label"
SMALLEST_BODY_QUERY_ID = "smallest_body_range_label"
RANGE_QUERY_PARAMS = {
    LARGEST_WICK_QUERY_ID: ("wick", "largest"),
    SMALLEST_WICK_QUERY_ID: ("wick", "smallest"),
    LARGEST_BODY_QUERY_ID: ("body", "largest"),
    SMALLEST_BODY_QUERY_ID: ("body", "smallest"),
}


def _range_kind(selected_query_id: str) -> str:
    return RANGE_QUERY_PARAMS[str(selected_query_id)][0]


def _extremum(selected_query_id: str) -> str:
    return RANGE_QUERY_PARAMS[str(selected_query_id)][1]


def _build_range_extremum_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
    range_kind: str,
    extremum: str,
) -> Dataset:
    """Select the answer candle for the wick/body range objective."""

    candles = sample_candles(params, instance_seed=int(instance_seed))
    if str(range_kind) not in {"wick", "body"}:
        raise ValueError(f"unsupported candlestick range kind: {range_kind}")
    if str(extremum) not in {"largest", "smallest"}:
        raise ValueError(f"unsupported candlestick extremum: {extremum}")
    scored = [
        (
            int(candle.wick_range if str(range_kind) == "wick" else candle.body_size),
            candle,
        )
        for candle in candles
    ]
    scored.sort(key=lambda item: (int(item[0]), str(item[1].label)))
    target = scored[-1][1] if str(extremum) == "largest" else scored[0][1]
    if str(range_kind) == "wick":
        value_ids = (f"{target.candle_id}:high", f"{target.candle_id}:low")
        range_phrase = "high-low wick range"
        annotation_roles = ("answer_wick",)
        answer_range = int(target.wick_range)
    else:
        value_ids = (f"{target.candle_id}:open", f"{target.candle_id}:close")
        range_phrase = "open-close body size"
        annotation_roles = ("answer_body",)
        answer_range = int(target.body_size)
    selection = Selection(
        answer=str(target.label),
        answer_type="string",
        annotation_candle_ids=(str(target.candle_id),),
        annotation_label_ids=(f"x_label:{target.candle_id}", *value_ids),
        annotation_roles=tuple(annotation_roles),
        trace={
            "range_kind": str(range_kind),
            "range_kind_phrase": str(range_phrase),
            "extremum": str(extremum),
            "extremum_phrase": str(extremum),
            "answer_candle_id": str(target.candle_id),
            "answer_label": str(target.label),
            "answer_range_value": int(answer_range),
            "candle_count": int(len(candles)),
        },
    )
    return Dataset(candles=tuple(candles), selection=selection)


@register_task
class ChartsCandlestickRangeExtremumLabelTask:
    """Return the period label with an extremal wick or body range."""

    task_id = "task_charts__candlestick__range_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "range_extremum_label"
    supported_query_ids = (
        LARGEST_WICK_QUERY_ID,
        SMALLEST_WICK_QUERY_ID,
        LARGEST_BODY_QUERY_ID,
        SMALLEST_BODY_QUERY_ID,
    )
    default_dataset_enabled = True

    def _generate_once(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> TaskOutput:
        """Generate one range-extremum instance with task-owned answer binding."""

        extremum = _extremum(str(selected_query_id))
        extremum_probabilities = {value: 1.0 if value == extremum else 0.0 for value in ("largest", "smallest")}
        branch_params = dict(params)
        dataset = _build_range_extremum_dataset(
            params=branch_params,
            instance_seed=int(instance_seed),
            range_kind=_range_kind(str(selected_query_id)),
            extremum=str(extremum),
        )
        artifacts = render_dataset(dataset=dataset, params=branch_params, instance_seed=int(instance_seed))
        annotation_segments_px = annotation_segments(
            rendered=artifacts.rendered,
            selection=dataset.selection,
        )
        if len(annotation_segments_px) != 1:
            raise RuntimeError("candlestick range annotation must contain exactly one segment")
        annotation = segment_annotation_artifacts(annotation_segments_px[0])
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "range_kind_phrase": str(dataset.selection.trace["range_kind_phrase"]),
                "extremum_phrase": str(dataset.selection.trace["extremum_phrase"]),
            },
            instance_seed=int(instance_seed),
        )
        answer_value = str(dataset.selection.answer)
        relations: dict[str, Any] = {
            **dict(dataset.selection.trace),
            "answer": answer_value,
            "annotation_candle_ids": list(dataset.selection.annotation_candle_ids),
            "support_label_ids": list(dataset.selection.annotation_label_ids),
            "annotation_roles": list(dataset.selection.annotation_roles),
            "extremum_probabilities": dict(extremum_probabilities),
        }
        trace_payload = build_trace_scaffold(
            dataset=dataset,
            artifacts=artifacts,
            relations=relations,
            witness_symbolic={
                "type": "candlestick_ohlc_witness",
                "candle_ids": list(dataset.selection.annotation_candle_ids),
                "roles": list(dataset.selection.annotation_roles),
                "support_label_ids": list(dataset.selection.annotation_label_ids),
                "answer": answer_value,
            },
            projected_annotation={
                **annotation.projected_annotation,
                "candle_ids": list(dataset.selection.annotation_candle_ids),
                "roles": list(dataset.selection.annotation_roles),
            },
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
        trace_payload["query_spec"] = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params={"query_id": str(selected_query_id), **relations},
        )
        trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="string", value=answer_value),
            annotation_gt=annotation.annotation_gt,
            image=artifacts.rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_WICK_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt == 0
                else int(hash64(int(instance_seed), "charts.candlestick.retry", int(attempt)))
            )
            try:
                return self._generate_once(
                    int(attempt_seed),
                    params=task_params,
                    selected_query_id=str(selected_query_id),
                )
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"failed to generate {self.task_id}: {last_error}")


__all__ = ["ChartsCandlestickRangeExtremumLabelTask"]
