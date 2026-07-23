"""Sampling helpers for organic-structure scene-level axes."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.common import resolve_symbolic_axis_variant

from .state import SCENE_VARIANTS


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the organic scene style variant."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(sampling_scope),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def build_with_retries(
    builder: Any,
    *,
    instance_seed: int,
    max_attempts: int,
    failure_message: str,
) -> Any:
    """Run a caller-owned dataset builder with deterministic retry seeds."""

    result: Any = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            result = builder(int(instance_seed) + int(attempt_index))
            break
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    if result is None:
        raise RuntimeError(str(failure_message)) from last_error
    return result


__all__ = ["build_with_retries", "resolve_scene_variant"]
