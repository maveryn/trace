"""Shared query-id and scene/query-axis sampling helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ...core.seed import spawn_rng
from ...core.sampling import (
    normalize_positive_weights,
    uniform_choice_with_probabilities,
    weighted_choice,
)
from .config_defaults import group_default


def is_uniform_probability_map(probabilities: Mapping[str, float], *, tol: float = 1e-9) -> bool:
    """Return true when all positive probabilities are approximately equal."""

    positives = [float(value) for value in probabilities.values() if float(value) > 0.0]
    if not positives:
        return False
    return max(positives) - min(positives) <= float(tol)


def has_non_null_param(params: Mapping[str, Any], key: str) -> bool:
    """Return true when a non-null override key is present."""

    return key in params and params.get(key) is not None


def resolve_variant(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    supported_variants: Sequence[str],
    explicit_key: str = "shape_variant",
    weights_key: str = "variant_weights",
) -> Tuple[str, Dict[str, float]]:
    """Resolve one variant with optional explicit override + weighted sampling."""

    supported = [str(item) for item in supported_variants]
    supported_set = set(supported)
    explicit_variant = params.get(str(explicit_key))
    if explicit_variant is not None:
        selected = str(explicit_variant).strip()
        if selected not in supported_set:
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return selected, {key: (1.0 if key == selected else 0.0) for key in sorted(supported_set)}

    raw_weights = params.get(
        str(weights_key),
        group_default(gen_defaults, str(weights_key), {key: 1.0 for key in supported}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError(f"{weights_key} must be a mapping when provided")
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
        if str(key) in supported_set
    }
    probabilities = normalize_positive_weights(weights, default_keys=supported)
    selected_variant = weighted_choice(rng, probabilities, sort_keys=True)
    return str(selected_variant), {str(key): float(value) for key, value in sorted(probabilities.items())}


def apply_balanced_variant_sampling(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    selected_variant: str,
    variant_probabilities: Mapping[str, float],
    supported_variants: Sequence[str],
    balance_flag_key: str = "balanced_variant_sampling",
    explicit_key: str = "shape_variant",
    weights_key: str = "variant_weights",
    sampling_namespace: str | None = None,
) -> str:
    """Sample a variant from uniform positive support when balancing is enabled."""

    enabled = bool(params.get(str(balance_flag_key), group_default(gen_defaults, str(balance_flag_key), True)))
    if not bool(enabled):
        return str(selected_variant)
    overridden = any(has_non_null_param(params, key) for key in (str(explicit_key), str(weights_key)))
    if overridden or (not is_uniform_probability_map(variant_probabilities)):
        return str(selected_variant)
    positive_variants = {
        str(key)
        for key, value in variant_probabilities.items()
        if float(value) > 0.0
    }
    values = [str(item) for item in supported_variants if str(item) in positive_variants]
    if not values:
        return str(selected_variant)
    namespace = str(sampling_namespace) if sampling_namespace is not None else "variant"
    rng = spawn_rng(int(instance_seed), namespace)
    selected, _probabilities = uniform_choice_with_probabilities(
        rng,
        values,
        sort_keys=False,
    )
    return str(selected)


def _full_probability_map(supported: Sequence[str], probabilities: Mapping[str, float]) -> Dict[str, float]:
    """Expand one restricted probability map over the full supported variant domain."""

    positive = {str(key): float(value) for key, value in probabilities.items()}
    return {
        str(key): float(positive.get(str(key), 0.0))
        for key in supported
    }


def resolve_compatible_scene_query_ids(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    supported_scene_variants: Sequence[str],
    supported_query_ids: Sequence[str],
    compatibility: Mapping[str, Sequence[str]],
    scene_sampling_namespace: str,
    query_sampling_namespace: str,
    decouple_scene_sampling: bool = False,
) -> Tuple[str, Dict[str, float], str, Dict[str, float]]:
    """Resolve one compatible `(scene_variant, query_id)` pair.

    Policy:
    - explicit `scene_variant` and `query_id` (or `query_id`) are both
      honored when compatible;
    - otherwise the explicit axis is fixed and the other axis is sampled from the
      compatible subset;
    - with neither axis fixed, `query_id` is selected first, then one
      compatible `scene_variant` is resolved within its feasible set.
    """

    scene_supported = [str(value) for value in supported_scene_variants]
    query_supported = [str(value) for value in supported_query_ids]
    compatibility_map = {
        str(scene): tuple(str(query) for query in queries)
        for scene, queries in compatibility.items()
    }
    scene_set = set(scene_supported)
    query_set = set(query_supported)

    explicit_scene = params.get("scene_variant")
    explicit_query = params.get("query_id")
    if explicit_scene is not None and str(explicit_scene) not in scene_set:
        raise ValueError(f"unsupported scene_variant: {explicit_scene}")
    if explicit_query is not None and str(explicit_query) not in query_set:
        raise ValueError(f"unsupported query_id: {explicit_query}")

    if explicit_scene is not None and explicit_query is not None:
        allowed_queries = set(compatibility_map.get(str(explicit_scene), ()))
        if str(explicit_query) not in allowed_queries:
            raise ValueError(
                f"incompatible scene/query combination: {explicit_scene} + {explicit_query}"
            )
        return (
            str(explicit_scene),
            _full_probability_map(scene_supported, {str(explicit_scene): 1.0}),
            str(explicit_query),
            _full_probability_map(query_supported, {str(explicit_query): 1.0}),
        )

    if explicit_query is not None:
        allowed_scenes = [
            scene for scene in scene_supported
            if str(explicit_query) in set(compatibility_map.get(scene, ()))
        ]
        selected_scene, restricted_scene_probs = resolve_variant(
            rng,
            params=params,
            gen_defaults=gen_defaults,
            supported_variants=allowed_scenes,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
        )
        selected_scene = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            selected_variant=str(selected_scene),
            variant_probabilities=restricted_scene_probs,
            supported_variants=allowed_scenes,
            balance_flag_key="balanced_scene_variant_sampling",
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            sampling_namespace=str(scene_sampling_namespace),
        )
        return (
            str(selected_scene),
            _full_probability_map(scene_supported, restricted_scene_probs),
            str(explicit_query),
            _full_probability_map(query_supported, {str(explicit_query): 1.0}),
        )

    selected_query, restricted_query_probs = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=query_supported,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    selected_query = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_query),
        variant_probabilities=restricted_query_probs,
        supported_variants=query_supported,
        balance_flag_key="balanced_query_id_sampling",
        explicit_key="query_id",
        weights_key="query_id_weights",
        sampling_namespace=str(query_sampling_namespace),
    )

    allowed_scenes = [
        scene for scene in scene_supported
        if str(selected_query) in set(compatibility_map.get(scene, ()))
    ]
    scene_params = params
    selected_scene, restricted_scene_probs = resolve_variant(
        rng,
        params=scene_params,
        gen_defaults=gen_defaults,
        supported_variants=allowed_scenes,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected_scene = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=scene_params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_scene),
        variant_probabilities=restricted_scene_probs,
        supported_variants=allowed_scenes,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=str(scene_sampling_namespace),
    )

    return (
        str(selected_scene),
        _full_probability_map(scene_supported, restricted_scene_probs),
        str(selected_query),
        _full_probability_map(query_supported, restricted_query_probs),
    )


__all__ = [
    "apply_balanced_variant_sampling",
    "has_non_null_param",
    "is_uniform_probability_map",
    "resolve_compatible_scene_query_ids",
    "resolve_variant",
]
