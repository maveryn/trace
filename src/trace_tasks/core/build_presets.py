"""Reusable build-config presets for common Trace dataset recipes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import trace_tasks.tasks  # noqa: F401  # Ensure task registration side effects run.

from ..tasks.registry import list_default_task_ids
from .config import BuildConfig, BuildTaskConfig


def resolve_equal_split_task_count(*, num_instances: int, task_count: int) -> int:
    """Resolve one exact per-task count for an equal-split build."""

    if int(num_instances) <= 0:
        raise ValueError("num_instances must be positive")
    if int(task_count) <= 0:
        raise ValueError("task_count must be positive")
    if int(num_instances) % int(task_count) != 0:
        raise ValueError(
            "num_instances must be divisible by task_count for an equal-split build "
            f"(got num_instances={num_instances}, task_count={task_count})"
        )
    return int(num_instances) // int(task_count)


def resolve_weighted_task_counts(*, num_instances: int, task_weights: Mapping[str, float]) -> Dict[str, int]:
    """Resolve exact integer task counts from positive task weights.

    Counts are assigned by largest remainder after flooring the ideal fractional counts.
    Ties are broken by task id for deterministic builds.
    """

    total = int(num_instances)
    if total <= 0:
        raise ValueError("num_instances must be positive")
    weights: Dict[str, float] = {}
    for task_id, raw_weight in task_weights.items():
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"task weight must be non-negative for {task_id}")
        weights[str(task_id)] = weight
    if not weights:
        raise ValueError("at least one task weight is required")
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("at least one task weight must be positive")

    fractional = {
        task_id: (float(total) * weight / weight_sum)
        for task_id, weight in weights.items()
    }
    counts = {task_id: int(value) for task_id, value in fractional.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        ranked = sorted(
            fractional,
            key=lambda task_id: (-(fractional[task_id] - counts[task_id]), str(task_id)),
        )
        for task_id in ranked[:remainder]:
            counts[task_id] += 1
    return dict(sorted(counts.items()))


def build_equal_split_all_tasks_config(
    *,
    output_root: str,
    dataset_name: str,
    num_instances: int,
    instance_version: str = "v0",
    image_format: str = "png",
    strict_repro: bool = False,
    max_attempts_per_instance: int = 100,
    sampling_seed: int = 0,
    workers: int = 0,
    max_in_flight: int = 0,
    task_params_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> BuildConfig:
    """Build a config that splits instances evenly across default-selected tasks."""

    task_ids = list_default_task_ids()
    per_task_count = resolve_equal_split_task_count(
        num_instances=int(num_instances),
        task_count=len(task_ids),
    )
    params_by_id = task_params_by_id or {}
    return BuildConfig(
        output_root=str(output_root),
        dataset_name=str(dataset_name),
        instance_version=str(instance_version),
        image_format=str(image_format).lower(),
        tasks=[
            BuildTaskConfig(
                task_id=str(task_id),
                count=per_task_count,
                params=dict(params_by_id.get(task_id, {})),
            )
            for task_id in task_ids
        ],
        num_instances=None,
        strict_repro=bool(strict_repro),
        max_attempts_per_instance=int(max_attempts_per_instance),
        sampling_seed=int(sampling_seed),
        workers=int(workers),
        max_in_flight=int(max_in_flight),
    )
