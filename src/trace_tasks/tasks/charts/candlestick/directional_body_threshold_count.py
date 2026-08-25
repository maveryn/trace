"""Public task for `task_charts__candlestick__directional_body_threshold_count`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.candlestick.shared.defaults import (
    DOMAIN,
    GENERATION_DEFAULTS,
    SCENE_ID,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.candlestick.shared.output import build_trace_scaffold
from trace_tasks.tasks.charts.candlestick.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.candlestick.shared.rendering import render_dataset
from trace_tasks.tasks.charts.candlestick.shared.state import Candle, Dataset, Selection
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_set_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice


TASK_ID = "task_charts__candlestick__directional_body_threshold_count"
UP_QUERY_ID = "up_body_threshold_count"
DOWN_QUERY_ID = "down_body_threshold_count"
SUPPORTED_QUERY_IDS = (UP_QUERY_ID, DOWN_QUERY_ID)
DIRECTION_BY_QUERY_ID = {UP_QUERY_ID: "up", DOWN_QUERY_ID: "down"}
DIRECTION_PHRASES = {"up": "closed above its open", "down": "closed below its open"}
# Sampled before the candle count so every answer stays equally likely; the
# candle count is then widened to fit the drawn target.
DIRECTIONAL_BODY_THRESHOLD_COUNT_SUPPORT = (1, 2, 3, 4, 5)
THRESHOLD_SUPPORT = (8, 10, 12)


def _support_sampling_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Decouple the sampling cursor from the public query-id round robin.

    Finite-support callers cycle `_sample_cursor` while the query id rotates in
    lockstep, so an undivided cursor would only ever reach the support entries
    whose index shares the query's residue.
    """

    support_params = dict(params)
    cursor = support_params.get("_sample_cursor")
    if cursor is None:
        return support_params
    support_params["_sample_cursor"] = abs(int(cursor)) // max(1, len(SUPPORTED_QUERY_IDS))
    return support_params


def _resolve_target_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    """Resolve how many same-direction candles must clear the body threshold."""

    target, _probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=_support_sampling_params(params),
        gen_defaults=GENERATION_DEFAULTS,
        support_key="directional_body_threshold_count_support",
        explicit_key="target_answer",
        fallback_support=DIRECTIONAL_BODY_THRESHOLD_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(target)


def _resolve_threshold(params: Mapping[str, Any], *, instance_seed: int) -> int:
    """Resolve the prompt-bound body-size threshold."""

    support = tuple(int(v) for v in params.get("body_threshold_support", THRESHOLD_SUPPORT))
    requested = params.get("body_threshold")
    if requested is not None:
        if int(requested) not in support:
            raise ValueError(f"unsupported body_threshold: {requested}")
        return int(requested)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.body_threshold")
    return int(support[rng.randrange(len(support))])


def _build_candles(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    direction: str,
    matching_count: int,
    decoy_count: int,
    opposite_count: int,
    threshold: int,
) -> tuple[Candle, ...]:
    """Construct an OHLC series with an exact above-threshold same-direction count.

    Every instance also carries at least one same-direction candle whose body is
    at or below the threshold, so candle colour alone never answers the question.
    """

    value_min, value_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="value_min", max_key="value_max",
        fallback_min=8, fallback_max=96, context="candlestick value range",
    )
    body_min, body_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="body_size_min", max_key="body_size_max",
        fallback_min=4, fallback_max=20, context="candlestick body size",
    )
    wick_min, wick_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="wick_size_min", max_key="wick_size_max",
        fallback_min=2, fallback_max=10, context="candlestick wick size",
    )
    above = [b for b in range(int(body_min), int(body_max) + 1) if b > int(threshold)]
    below = [b for b in range(int(body_min), int(body_max) + 1) if b <= int(threshold)]
    if len(above) < int(matching_count) or len(below) < int(decoy_count):
        raise ValueError("body-size support cannot satisfy the requested threshold plan")

    opposite = "down" if str(direction) == "up" else "up"
    for attempt in range(250):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.body_threshold.retry", int(attempt))
        match_bodies = rng.sample(above, int(matching_count))
        decoy_bodies = rng.sample(below, int(decoy_count))
        used = set(match_bodies) | set(decoy_bodies)
        remaining = [b for b in range(int(body_min), int(body_max) + 1) if b not in used]
        if len(remaining) < int(opposite_count):
            continue
        opposite_bodies = rng.sample(remaining, int(opposite_count))
        plan = (
            [(str(direction), int(b)) for b in match_bodies]
            + [(str(direction), int(b)) for b in decoy_bodies]
            + [(str(opposite), int(b)) for b in opposite_bodies]
        )
        rng.shuffle(plan)
        labels = sample_chart_labels(
            count=len(plan),
            instance_seed=int(hash64(int(instance_seed), "candlestick.body_threshold.labels", int(attempt))),
            namespace=f"{SCENE_NAMESPACE}.labels:{len(plan)}",
        )
        candles: list[Candle] = []
        for index, (candle_direction, body) in enumerate(plan):
            if candle_direction == "up":
                low_bound = int(value_min) + int(wick_max)
                high_bound = int(value_max) - int(body) - int(wick_max)
            else:
                low_bound = int(value_min) + int(body) + int(wick_max)
                high_bound = int(value_max) - int(wick_max)
            if low_bound > high_bound:
                break
            open_value = int(rng.randint(low_bound, high_bound))
            close_value = int(open_value) + (int(body) if candle_direction == "up" else -int(body))
            upper_wick = int(rng.randint(int(wick_min), int(wick_max)))
            lower_wick = int(rng.randint(int(wick_min), int(wick_max)))
            high_value = max(int(open_value), int(close_value)) + int(upper_wick)
            low_value = min(int(open_value), int(close_value)) - int(lower_wick)
            if not (int(value_min) <= int(low_value) < int(high_value) <= int(value_max)):
                break
            candles.append(
                Candle(
                    candle_id=f"candle_{index}",
                    label=str(labels[index]),
                    open_value=int(open_value),
                    high_value=int(high_value),
                    low_value=int(low_value),
                    close_value=int(close_value),
                )
            )
        if len(candles) != len(plan):
            continue
        if len({int(c.body_size) for c in candles}) != len(candles):
            continue
        if len({int(c.wick_range) for c in candles}) != len(candles):
            continue
        directions = [c.direction for c in candles]
        if directions.count("up") < 2 or directions.count("down") < 2:
            continue
        return tuple(candles)
    raise ValueError("could not construct a candlestick series for the requested threshold plan")


def _build_dataset(*, params: dict[str, Any], instance_seed: int, selected_query_id: str) -> Dataset:
    """Bind the directional body-threshold count answer to a constructed series."""

    direction = DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    candle_min, candle_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="candle_count_min", max_key="candle_count_max",
        fallback_min=7, fallback_max=10, context="candlestick candle count",
    )
    target_count = _resolve_target_count(params, instance_seed=int(instance_seed))
    threshold = _resolve_threshold(params, instance_seed=int(instance_seed))

    # One same-direction decoy plus the scene's two-per-direction invariant.
    lowest_total = max(int(candle_min), int(target_count) + 3)
    if lowest_total > int(candle_max):
        raise ValueError(f"target_answer {target_count} exceeds the candlestick count bounds")
    layout_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.candle_layout")
    candle_count = int(lowest_total) + int(layout_rng.randrange(int(candle_max) - int(lowest_total) + 1))
    remaining = int(candle_count) - int(target_count)
    decoy_count = 1 + int(layout_rng.randrange(int(remaining) - 2))
    opposite_count = int(remaining) - int(decoy_count)

    candles = _build_candles(
        params,
        instance_seed=int(instance_seed),
        direction=str(direction),
        matching_count=int(target_count),
        decoy_count=int(decoy_count),
        opposite_count=int(opposite_count),
        threshold=int(threshold),
    )
    same_direction = [c for c in candles if c.direction == str(direction)]
    matching = tuple(c for c in same_direction if int(c.body_size) > int(threshold))
    if len(matching) != int(target_count):
        raise ValueError("candlestick threshold construction lost its body plan")
    if len(same_direction) <= len(matching):
        raise ValueError("candlestick threshold instance has no same-direction decoy candle")
    selection = Selection(
        answer=int(target_count),
        answer_type="integer",
        annotation_candle_ids=tuple(str(c.candle_id) for c in matching),
        annotation_label_ids=tuple(f"x_label:{c.candle_id}" for c in matching),
        annotation_roles=tuple("matching_body" for _c in matching),
        trace={
            "direction": str(direction),
            "direction_phrase": str(DIRECTION_PHRASES[str(direction)]),
            "body_threshold": int(threshold),
            "matching_candle_ids": [str(c.candle_id) for c in matching],
            "matching_labels": [str(c.label) for c in matching],
            "same_direction_count": int(len(same_direction)),
            "same_direction_decoy_count": int(len(same_direction) - len(matching)),
            "target_answer": int(target_count),
            "candle_count": int(len(candles)),
        },
    )
    return Dataset(candles=tuple(candles), selection=selection)


@register_task
class ChartsCandlestickDirectionalBodyThresholdCountTask:
    """Count same-direction candles whose open-close body clears a threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "directional_body_threshold_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _generate_once(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> TaskOutput:
        dataset = _build_dataset(
            params=dict(params), instance_seed=int(instance_seed), selected_query_id=str(selected_query_id)
        )
        artifacts = render_dataset(dataset=dataset, params=dict(params), instance_seed=int(instance_seed))
        boxes = [
            artifacts.rendered.body_bboxes_px[str(candle_id)]
            for candle_id in dataset.selection.annotation_candle_ids
        ]
        annotation = bbox_set_annotation_artifacts(boxes)
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "direction_phrase": str(dataset.selection.trace["direction_phrase"]),
                "body_threshold": int(dataset.selection.trace["body_threshold"]),
            },
            instance_seed=int(instance_seed),
        )
        answer_value = int(dataset.selection.answer)
        relations: dict[str, Any] = {
            **dict(dataset.selection.trace),
            "answer": answer_value,
            "annotation_candle_ids": list(dataset.selection.annotation_candle_ids),
            "support_label_ids": list(dataset.selection.annotation_label_ids),
            "annotation_roles": list(dataset.selection.annotation_roles),
        }
        trace_payload = build_trace_scaffold(
            dataset=dataset,
            artifacts=artifacts,
            relations=relations,
            witness_symbolic={
                "type": "candlestick_body_threshold_witness",
                "direction": str(dataset.selection.trace["direction"]),
                "body_threshold": int(dataset.selection.trace["body_threshold"]),
                "candle_ids": list(dataset.selection.annotation_candle_ids),
                "roles": list(dataset.selection.annotation_roles),
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
            answer_gt=TypedValue(type="integer", value=answer_value),
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
            default_query_id=UP_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt == 0
                else int(hash64(int(instance_seed), "charts.candlestick.body_threshold_retry", int(attempt)))
            )
            try:
                return self._generate_once(
                    int(attempt_seed), params=task_params, selected_query_id=str(selected_query_id)
                )
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"failed to generate {self.task_id}: {last_error}")


__all__ = [
    "ChartsCandlestickDirectionalBodyThresholdCountTask",
    "DIRECTIONAL_BODY_THRESHOLD_COUNT_SUPPORT",
]
