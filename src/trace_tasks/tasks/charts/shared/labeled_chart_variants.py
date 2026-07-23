"""Variant-axis helpers for labeled chart task families."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .chart_scene_types import SUPPORTED_CHART_SCENE_VARIANTS

SceneVariant = str

SUPPORTED_LABELED_CHART_SCENE_VARIANTS: Tuple[str, ...] = tuple(SUPPORTED_CHART_SCENE_VARIANTS)

PIE_LIKE_SCENE_VARIANTS = frozenset({"pie", "donut"})


def is_pie_like_scene_variant(scene_variant: str) -> bool:
    """Return whether the chart variant uses composition-style pie slices."""

    return str(scene_variant) in PIE_LIKE_SCENE_VARIANTS


def apply_scene_variant_mark_count_cap(
    *,
    scene_variant: str,
    mark_count_min: int,
    mark_count_max: int,
) -> Tuple[int, int]:
    """Apply scene-specific mark-count caps for chart readability."""

    resolved_min = int(mark_count_min)
    resolved_max = int(mark_count_max)
    if str(scene_variant) in {"pie", "donut"}:
        resolved_max = min(int(resolved_max), 8)
    if str(scene_variant) == "radar":
        resolved_max = min(int(resolved_max), 7)
    if int(resolved_min) > int(resolved_max):
        raise ValueError(f"no feasible mark-count support for scene_variant={scene_variant}")
    return int(resolved_min), int(resolved_max)


def resolve_chart_axis_variant(
    *,
    params: Mapping[str, object],
    gen_defaults: Mapping[str, object],
    instance_seed: int,
    supported_variants: Sequence[str],
    task_id: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced chart task/scene variant axis."""

    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=supported_variants,
        namespace=f"{task_id}.{axis_namespace}",
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
    )


def resolve_chart_axis_variant_for_namespace(
    *,
    params: Mapping[str, object],
    gen_defaults: Mapping[str, object],
    instance_seed: int,
    supported_variants: Sequence[str],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced chart variant axis using a neutral sampling namespace."""

    sampling_namespace = str(namespace)
    variant_rng = spawn_rng(int(instance_seed), sampling_namespace)
    selected_variant, probabilities = resolve_variant(
        variant_rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=supported_variants,
        explicit_key=explicit_key,
        weights_key=weights_key,
    )
    variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=supported_variants,
        balance_flag_key=balance_flag_key,
        explicit_key=explicit_key,
        weights_key=weights_key,
        sampling_namespace=sampling_namespace,
    )
    return str(variant), {str(key): float(value) for key, value in sorted(probabilities.items())}


__all__ = [
    "PIE_LIKE_SCENE_VARIANTS",
    "SUPPORTED_LABELED_CHART_SCENE_VARIANTS",
    "SceneVariant",
    "apply_scene_variant_mark_count_cap",
    "is_pie_like_scene_variant",
    "resolve_chart_axis_variant",
    "resolve_chart_axis_variant_for_namespace",
]
