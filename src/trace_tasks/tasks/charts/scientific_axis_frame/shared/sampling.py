"""Neutral sampling helpers for scientific axis-frame chart scenes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.scientific_axis_frame.shared.defaults import (
    axis_start_bounds,
    tick_count_bounds,
    tick_step_bounds,
)
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import (
    AxisFrameBinding,
    AxisFrameDataset,
    AxisSpec,
    SCENE_NAMESPACE,
)


def normalize_axis(axis: str) -> str:
    resolved = str(axis).strip().lower()
    if resolved not in {"x", "y"}:
        raise ValueError(f"unsupported axis for scientific axis frame: {axis!r}")
    return resolved


def balanced_choice(values: Sequence[Any], params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> Any:
    """Sample uniformly from one explicit semantic support."""

    del params
    support = tuple(values)
    if not support:
        raise ValueError(f"empty support for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, support)


def axis_spec(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    axis: str,
    forced_step: int | None = None,
    forced_count: int | None = None,
) -> AxisSpec:
    """Build one uniformly spaced axis for span-readout tasks.

    Invariant: returned tick values are monotonic integer labels and the stored
    deltas exactly match the visible intervals used by rendering and metadata.
    """

    resolved_axis = normalize_axis(axis)
    count_min, count_max = tick_count_bounds(params)
    step_min, step_max = tick_step_bounds(params)
    count = int(forced_count) if forced_count is not None else int(
        balanced_choice(
            tuple(range(int(count_min), int(count_max) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{resolved_axis}.tick_count",
        )
    )
    step = int(forced_step) if forced_step is not None else int(
        balanced_choice(
            tuple(range(int(step_min), int(step_max) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{resolved_axis}.tick_step",
        )
    )
    if int(count) < 3:
        raise ValueError("axis tick count must be at least 3")
    if int(step) <= 0:
        raise ValueError("axis tick step must be positive")
    start = axis_start(params, instance_seed=int(instance_seed), axis=resolved_axis, step=int(step))
    values = tuple(int(start) + (int(index) * int(step)) for index in range(int(count)))
    return AxisSpec(
        axis=resolved_axis,
        values=values,
        start=int(start),
        step=int(step),
        count=int(count),
        deltas=tuple(int(step) for _ in range(max(0, int(count) - 1))),
    )


def uneven_axis_spec(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    axis: str,
    forced_first_delta: int | None = None,
    forced_last_delta: int | None = None,
    forced_count: int | None = None,
) -> AxisSpec:
    """Build one non-uniform axis for first/last interval readout tasks.

    Invariant: any forced endpoint interval is preserved, and at least one
    other visible interval has a different numeric spacing.
    """

    resolved_axis = normalize_axis(axis)
    count_min, count_max = tick_count_bounds(params)
    step_min, step_max = tick_step_bounds(params)
    count = int(forced_count) if forced_count is not None else int(
        balanced_choice(
            tuple(range(int(count_min), int(count_max) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{resolved_axis}.uneven_tick_count",
        )
    )
    if int(count) < 3:
        raise ValueError("uneven axis tick count must be at least 3")
    delta_support = tuple(range(int(step_min), int(step_max) + 1))
    if not delta_support:
        raise ValueError("axis tick delta support is empty")
    deltas: list[int] = []
    for gap_index in range(int(count) - 1):
        if int(gap_index) == 0 and forced_first_delta is not None:
            delta = int(forced_first_delta)
        elif int(gap_index) == int(count) - 2 and forced_last_delta is not None:
            delta = int(forced_last_delta)
        else:
            delta = int(
                balanced_choice(
                    delta_support,
                    params,
                    instance_seed=int(instance_seed),
                    namespace=f"{SCENE_NAMESPACE}.{resolved_axis}.uneven_delta.{int(gap_index)}",
                )
            )
        if int(delta) <= 0:
            raise ValueError("axis tick delta must be positive")
        deltas.append(int(delta))
    if len(set(deltas)) < 2:
        replacement = next((int(value) for value in delta_support if int(value) != int(deltas[-1])), None)
        if replacement is None:
            raise ValueError("uneven axis requires at least two supported deltas")
        replace_index = 1 if len(deltas) > 2 else len(deltas) - 1
        if forced_last_delta is not None and int(replace_index) == len(deltas) - 1:
            replace_index = 0
        if forced_first_delta is not None and int(replace_index) == 0:
            replace_index = len(deltas) - 1
        deltas[int(replace_index)] = int(replacement)
    start = axis_start(params, instance_seed=int(instance_seed), axis=resolved_axis, step=1)
    values = [int(start)]
    for delta in deltas:
        values.append(int(values[-1]) + int(delta))
    return AxisSpec(
        axis=resolved_axis,
        values=tuple(int(value) for value in values),
        start=int(start),
        step=int(deltas[0]),
        count=int(count),
        deltas=tuple(int(delta) for delta in deltas),
    )


def axis_start(params: Mapping[str, Any], *, instance_seed: int, axis: str, step: int) -> int:
    low, high = axis_start_bounds(params)
    support = [int(value) for value in range(int(low), int(high) + 1) if int(value) % max(1, int(step)) == 0]
    if not support:
        support = [0]
    return int(
        balanced_choice(
            support,
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{normalize_axis(axis)}.start.{int(step)}",
        )
    )


def span_support(params: Mapping[str, Any]) -> tuple[int, ...]:
    count_min, count_max = tick_count_bounds(params)
    step_min, step_max = tick_step_bounds(params)
    support = sorted(
        {
            int(step) * (int(count) - 1)
            for count in range(count_min, count_max + 1)
            for step in range(step_min, step_max + 1)
        }
    )
    return tuple(int(value) for value in support)


def span_pairs_for(params: Mapping[str, Any], span: int) -> tuple[tuple[int, int], ...]:
    count_min, count_max = tick_count_bounds(params)
    step_min, step_max = tick_step_bounds(params)
    pairs = [
        (int(count), int(step))
        for count in range(count_min, count_max + 1)
        for step in range(step_min, step_max + 1)
        if int(step) * (int(count) - 1) == int(span)
    ]
    return tuple(pairs)


def decorative_points(dataset_seed: int, x_axis: AxisSpec, y_axis: AxisSpec) -> tuple[tuple[float, float], ...]:
    rng = spawn_rng(int(dataset_seed), f"{SCENE_NAMESPACE}.decorative_series")
    points: list[tuple[float, float]] = []
    y_span = max(1, int(y_axis.values[-1]) - int(y_axis.values[0]))
    base = float(y_axis.values[0]) + (0.48 * float(y_span))
    amplitude = 0.26 * float(y_span)
    phase = float(rng.random()) * math.pi
    for index, x_value in enumerate(x_axis.values):
        y_value = base + (math.sin(float(index) * 0.9 + phase) * amplitude) + float(rng.randint(-2, 2))
        y_value = max(float(y_axis.values[0]), min(float(y_axis.values[-1]), y_value))
        points.append((float(x_value), float(y_value)))
    return tuple(points)


def assemble_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    target_axis: AxisSpec,
) -> tuple[AxisSpec, AxisSpec]:
    other_axis_name = "y" if str(target_axis.axis) == "x" else "x"
    other_axis = axis_spec(params, instance_seed=int(instance_seed) + 17, axis=other_axis_name)
    x_axis = target_axis if str(target_axis.axis) == "x" else other_axis
    y_axis = target_axis if str(target_axis.axis) == "y" else other_axis
    return x_axis, y_axis


def assemble_uneven_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    target_axis: AxisSpec,
) -> tuple[AxisSpec, AxisSpec]:
    other_axis_name = "y" if str(target_axis.axis) == "x" else "x"
    other_axis = uneven_axis_spec(params, instance_seed=int(instance_seed) + 17, axis=other_axis_name)
    x_axis = target_axis if str(target_axis.axis) == "x" else other_axis
    y_axis = target_axis if str(target_axis.axis) == "y" else other_axis
    return x_axis, y_axis


def build_tick_spacing_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    axis: str,
    pair_position: str,
) -> AxisFrameDataset:
    """Sample a frame where the requested axis has an unequal adjacent tick pair.

    Invariant: this builder returns semantic tick roles only; public task files
    decide which query id maps to this axis.
    """

    resolved_axis = normalize_axis(axis)
    resolved_position = str(pair_position).strip().lower()
    if resolved_position not in {"first", "last"}:
        raise ValueError(f"unsupported tick-pair position: {pair_position!r}")
    step_min, step_max = tick_step_bounds(params)
    target_delta = int(
        balanced_choice(
            tuple(range(int(step_min), int(step_max) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.tick_spacing.{resolved_position}.answer",
        )
    )
    target_axis = uneven_axis_spec(
        params,
        instance_seed=int(instance_seed),
        axis=resolved_axis,
        forced_first_delta=int(target_delta) if resolved_position == "first" else None,
        forced_last_delta=int(target_delta) if resolved_position == "last" else None,
    )
    x_axis, y_axis = assemble_uneven_dataset(params=params, instance_seed=int(instance_seed), target_axis=target_axis)
    target_axis_values = x_axis.values if resolved_axis == "x" else y_axis.values
    pair_index = 0 if resolved_position == "first" else len(target_axis_values) - 2
    first_value = int(target_axis_values[int(pair_index)])
    next_value = int(target_axis_values[int(pair_index) + 1])
    first_key = f"{resolved_axis}:{first_value}"
    next_key = f"{resolved_axis}:{next_value}"
    binding = AxisFrameBinding(
        axis=resolved_axis,
        answer=int(next_value - first_value),
        answer_type="integer",
        annotation_roles={"first_tick": str(first_key), "next_tick": str(next_key)},
        tick_values={"first_tick": int(first_value), "next_tick": int(next_value)},
        trace={
            "axis": str(resolved_axis),
            "axis_name": f"{resolved_axis}-axis",
            "first_tick_value": int(first_value),
            "next_tick_value": int(next_value),
            "tick_pair_index": int(pair_index),
            "tick_pair_position": str(resolved_position),
        },
    )
    return AxisFrameDataset(
        x_axis=x_axis,
        y_axis=y_axis,
        binding=binding,
        series_points=decorative_points(int(instance_seed), x_axis, y_axis),
    )


def build_axis_span_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    axis: str,
) -> AxisFrameDataset:
    """Sample a frame where the requested axis has a controlled min-max span.

    Invariant: the numeric answer is determined by visible tick labels, not by
    decorative plotted series points.
    """

    resolved_axis = normalize_axis(axis)
    support = span_support(params)
    if not support:
        raise ValueError("axis span support is empty")
    target_span = int(
        balanced_choice(
            support,
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.axis_span.answer",
        )
    )
    pairs = span_pairs_for(params, int(target_span))
    count, step = balanced_choice(
        pairs,
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.axis_span.pair.{int(target_span)}",
    )
    target_axis = axis_spec(
        params,
        instance_seed=int(instance_seed),
        axis=resolved_axis,
        forced_step=int(step),
        forced_count=int(count),
    )
    x_axis, y_axis = assemble_dataset(params=params, instance_seed=int(instance_seed), target_axis=target_axis)
    target_axis_values = x_axis.values if resolved_axis == "x" else y_axis.values
    min_value = int(min(target_axis_values))
    max_value = int(max(target_axis_values))
    min_key = f"{resolved_axis}:{min_value}"
    max_key = f"{resolved_axis}:{max_value}"
    binding = AxisFrameBinding(
        axis=resolved_axis,
        answer=int(max_value - min_value),
        answer_type="integer",
        annotation_roles={"min_tick": str(min_key), "max_tick": str(max_key)},
        tick_values={"min_tick": int(min_value), "max_tick": int(max_value)},
        trace={
            "axis": str(resolved_axis),
            "axis_name": f"{resolved_axis}-axis",
            "min_tick_value": int(min_value),
            "max_tick_value": int(max_value),
        },
    )
    return AxisFrameDataset(
        x_axis=x_axis,
        y_axis=y_axis,
        binding=binding,
        series_points=decorative_points(int(instance_seed), x_axis, y_axis),
    )


__all__ = [
    "balanced_choice",
    "build_axis_span_dataset",
    "build_tick_spacing_dataset",
    "decorative_points",
    "uneven_axis_spec",
    "normalize_axis",
    "span_pairs_for",
    "span_support",
]
