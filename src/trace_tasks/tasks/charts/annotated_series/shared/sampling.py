"""Identity-free sampling helpers for annotated-series charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.annotated_series.shared.defaults import (
    FALLBACK_CHART_DEFAULTS,
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    generation_bounds,
    generation_int,
    group_default,
)
from trace_tasks.tasks.charts.annotated_series.shared.state import SeriesSample
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.charts.shared.labeled_chart_values import balanced_choice_from_values, resolve_value_bounds


def _probability_map_from_weights(weights: Mapping[str, float]) -> dict[str, float]:
    positives = {str(key): float(value) for key, value in weights.items() if float(value) > 0.0}
    total = float(sum(positives.values()))
    if total <= 0.0:
        return {str(key): 0.0 for key in weights}
    return {str(key): float(value) / total for key, value in positives.items()}


def _choice_from_weight_map(
    weights: Mapping[str, float],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    probabilities = _probability_map_from_weights(weights)
    positives = [str(key) for key, value in probabilities.items() if float(value) > 0.0]
    if not positives:
        raise ValueError(f"no positive support for {namespace}")
    threshold = spawn_rng(instance_seed, namespace).random()
    cumulative = 0.0
    for key in positives:
        cumulative += float(probabilities[key])
        if threshold <= cumulative:
            return key, dict(probabilities)
    return positives[-1], dict(probabilities)


def choose_scene_variant(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> tuple[str, dict[str, float]]:
    explicit = params.get("scene_variant")
    if explicit is not None:
        variant = str(explicit)
        if variant not in SUPPORTED_SCENE_VARIANTS:
            raise ValueError(
                f"scene_variant must be one of {SUPPORTED_SCENE_VARIANTS}, got {variant!r}"
            )
        return variant, {name: 1.0 if name == variant else 0.0 for name in SUPPORTED_SCENE_VARIANTS}

    raw_weights = group_default(params, GENERATION_DEFAULTS, "scene_variant_weights", None)
    if isinstance(raw_weights, Mapping) and raw_weights:
        weights = {str(key): float(value) for key, value in raw_weights.items()}
        unsupported = sorted(set(weights) - set(SUPPORTED_SCENE_VARIANTS))
        if unsupported:
            raise ValueError(f"Unsupported scene_variant_weights entries: {unsupported}")
    else:
        weights = {variant: 1.0 for variant in SUPPORTED_SCENE_VARIANTS}

    if bool(group_default(params, GENERATION_DEFAULTS, "balanced_scene_variant_sampling", False)):
        support = tuple(name for name in SUPPORTED_SCENE_VARIANTS if weights.get(name, 0.0) > 0.0)
        if not support:
            raise ValueError("scene_variant_weights must leave at least one positive scene variant")
        selected, probabilities = uniform_choice_with_probabilities(
            spawn_rng(instance_seed, f"{SCENE_NAMESPACE}.scene_variant"),
            support,
        )
        return str(selected), dict(probabilities)

    return _choice_from_weight_map(
        weights,
        instance_seed=instance_seed,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
    )


def choose_mark_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    mark_min, mark_max = generation_bounds(
        params,
        "mark_count_min",
        "mark_count_max",
        FALLBACK_CHART_DEFAULTS.mark_count_min,
        FALLBACK_CHART_DEFAULTS.mark_count_max,
    )
    support = tuple(range(mark_min, mark_max + 1))
    return balanced_choice_from_values(
        support,
        params=params,
        namespace=f"{SCENE_NAMESPACE}.mark_count",
        instance_seed=instance_seed,
    )


def sample_values(
    params: Mapping[str, Any],
    *,
    count: int,
    instance_seed: int,
) -> tuple[int, ...]:
    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=GENERATION_DEFAULTS,
        defaults=FALLBACK_CHART_DEFAULTS,
        task_id=SCENE_NAMESPACE,
        instance_seed=int(instance_seed),
    )
    if value_max - value_min + 1 < count:
        value_max = value_min + count + 10
    rng = spawn_rng(instance_seed, f"{SCENE_NAMESPACE}.values")
    values = rng.sample(range(value_min, value_max + 1), count)
    if len(set(values)) < count:
        raise RuntimeError("Annotated-series values must be unique by construction")
    return tuple(int(value) for value in values)


def build_series_sample(params: Mapping[str, Any], *, instance_seed: int) -> SeriesSample:
    scene_variant, probabilities = choose_scene_variant(params, instance_seed=instance_seed)
    count = choose_mark_count(params, instance_seed=instance_seed)
    labels = tuple(
        sample_chart_labels(
            count=count,
            namespace=f"{SCENE_NAMESPACE}.labels.{count}",
            instance_seed=instance_seed,
        )
    )
    values = sample_values(params, count=count, instance_seed=instance_seed)
    return SeriesSample(
        scene_variant=scene_variant,
        labels=labels,
        values=values,
        scene_variant_probabilities=probabilities,
    )


def choose_semantic_branch(
    params: Mapping[str, Any],
    *,
    support: Sequence[str],
    branch_name: str,
    instance_seed: int,
) -> tuple[str, dict[str, float]]:
    if not support:
        raise ValueError("Semantic branch support cannot be empty")
    explicit = params.get(branch_name)
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{branch_name} must be one of {tuple(support)}, got {value!r}")
        return value, {name: 1.0 if name == value else 0.0 for name in support}

    selected, probabilities = uniform_choice_with_probabilities(
        spawn_rng(instance_seed, f"{SCENE_NAMESPACE}.{branch_name}"),
        tuple(str(value) for value in support),
    )
    return str(selected), dict(probabilities)


def uniform_probability_map(support: Sequence[str]) -> dict[str, float]:
    return _probability_map_from_weights({str(value): 1.0 for value in support})
