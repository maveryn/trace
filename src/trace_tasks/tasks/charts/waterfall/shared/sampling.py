"""Sampling helpers for waterfall chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.charts.waterfall.shared.defaults import GENERATION_DEFAULTS, SCENE_NAMESPACE
from trace_tasks.tasks.charts.waterfall.shared.state import WaterfallDataset, WaterfallStep
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds


def _resolve_step_count_bounds(
    params: Mapping[str, Any],
    *,
    override_min: int | None,
    override_max: int | None,
) -> tuple[int, int]:
    if override_min is not None or override_max is not None:
        default_min, default_max = resolve_required_int_bounds(
            params,
            GENERATION_DEFAULTS,
            min_key="step_count_min",
            max_key="step_count_max",
            fallback_min=6,
            fallback_max=9,
            context="waterfall step count",
        )
        return (
            int(default_min if override_min is None else override_min),
            int(default_max if override_max is None else override_max),
        )
    return resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="step_count_min",
        max_key="step_count_max",
        fallback_min=6,
        fallback_max=9,
        context="waterfall step count",
    )


def sample_waterfall_dataset(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    step_count_min: int | None = None,
    step_count_max: int | None = None,
) -> WaterfallDataset:
    """Sample an ordered waterfall sequence with both positive and negative steps."""

    step_min, step_max = _resolve_step_count_bounds(
        params,
        override_min=step_count_min,
        override_max=step_count_max,
    )
    start_min, start_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="start_value_min",
        max_key="start_value_max",
        fallback_min=35,
        fallback_max=65,
        context="waterfall start value",
    )
    delta_min, delta_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="delta_abs_min",
        max_key="delta_abs_max",
        fallback_min=5,
        fallback_max=20,
        context="waterfall delta magnitude",
    )
    total_min, total_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="running_total_min",
        max_key="running_total_max",
        fallback_min=8,
        fallback_max=95,
        context="waterfall running total",
    )

    for attempt in range(200):
        local_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.steps.retry", int(attempt))
        step_count = int(step_min) + int(local_rng.randrange(int(step_max) - int(step_min) + 1))
        start_value = int(local_rng.randint(int(start_min), int(start_max)))
        labels = sample_chart_labels(
            count=int(step_count),
            instance_seed=int(instance_seed) + int(attempt) * 7919,
            namespace=f"{SCENE_NAMESPACE}.labels:{int(step_count)}",
        )
        current = int(start_value)
        steps: list[WaterfallStep] = []
        has_positive = False
        has_negative = False
        for index in range(int(step_count)):
            magnitude = int(local_rng.randint(int(delta_min), int(delta_max)))
            sign = 1 if local_rng.random() < 0.52 else -1
            if current + sign * magnitude > int(total_max):
                sign = -1
            if current + sign * magnitude < int(total_min):
                sign = 1
            delta = int(sign) * int(magnitude)
            next_value = int(current) + int(delta)
            if not (int(total_min) <= int(next_value) <= int(total_max)):
                break
            steps.append(
                WaterfallStep(
                    step_id=f"step_{index}",
                    label=str(labels[index]),
                    delta=int(delta),
                    running_before=int(current),
                    running_after=int(next_value),
                )
            )
            current = int(next_value)
            has_positive = bool(has_positive or delta > 0)
            has_negative = bool(has_negative or delta < 0)
        if len(steps) == int(step_count) and has_positive and has_negative:
            return WaterfallDataset(
                start_value=int(start_value),
                final_value=int(steps[-1].running_after),
                steps=tuple(steps),
            )

    raise ValueError("could not sample a valid waterfall sequence")


def choose_step_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    step_count: int,
    min_index: int = 0,
    max_index: int | None = None,
    max_from_end: int | None = None,
) -> int:
    """Choose one feasible contribution-step index from an ordered waterfall sequence."""

    resolved_max = int(step_count) - 1 if max_index is None else int(max_index)
    if max_from_end is not None:
        resolved_max = min(int(resolved_max), int(step_count) - 1 - max(0, int(max_from_end)))
    support = [index for index in range(int(step_count)) if int(min_index) <= int(index) <= int(resolved_max)]
    if not support:
        raise ValueError("no feasible waterfall target step")
    del params
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, tuple(support)))


def threshold_options(
    dataset: WaterfallDataset,
    *,
    direction: str,
) -> tuple[tuple[int, int], ...]:
    """Return feasible first-crossing targets as ``(step_index, threshold)`` pairs."""

    options: list[tuple[int, int]] = []
    previous_totals: list[int] = [int(dataset.start_value)]
    for index, step in enumerate(dataset.steps):
        if str(direction) == "at_least":
            low = max(previous_totals) + 1
            high = int(step.running_after)
        elif str(direction) == "at_most":
            low = int(step.running_after)
            high = min(previous_totals) - 1
        else:
            raise ValueError(f"unsupported waterfall threshold direction: {direction}")
        if int(low) <= int(high):
            options.append((int(index), int((int(low) + int(high)) // 2)))
        previous_totals.append(int(step.running_after))
    return tuple(options)


__all__ = [
    "choose_step_index",
    "sample_waterfall_dataset",
    "threshold_options",
]
