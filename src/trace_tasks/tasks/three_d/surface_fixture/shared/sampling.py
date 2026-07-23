"""Sampling helpers for surface-fixture scene axes and scalar counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant_for_namespace

from .state import (
    ELEMENT_TYPE_BY_SCENE_VARIANT,
    SCENE_VARIANT_BY_ELEMENT_TYPE,
    SEMANTIC_COLOR_SUPPORT,
)


@dataclass(frozen=True)
class ResolvedSurfaceFixtureAxes:
    """Resolved scene and element axes for one surface-fixture instance."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    element_type: str
    element_type_probabilities: Dict[str, float]


def one_hot_probability_map(values: Sequence[str], selected: str) -> Dict[str, float]:
    """Return a probability map pinned to one selected string."""

    return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}


def uniform_int_probability_map(values: Sequence[int], *, selected: int | None = None) -> Dict[str, float]:
    """Return a uniform or pinned probability map over integer support."""

    support = tuple(int(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if int(value) == int(selected) else 0.0) for value in support}
    probability = 1.0 / float(max(1, len(support)))
    return {str(value): float(probability) for value in support}


def configured_int(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], key: str, default: int) -> int:
    """Resolve an integer from explicit params, scene defaults, or fallback."""

    return int(params.get(str(key), group_default(gen_defaults, str(key), int(default))))


def resolve_int_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    min_key: str,
    max_key: str,
    default_min: int,
    default_max: int,
    explicit_keys: Sequence[str],
    lower_bound: int = 0,
    upper_bound: int = 64,
) -> Tuple[int, Dict[str, float]]:
    """Resolve an integer answer axis with stable support metadata."""

    minimum = max(int(lower_bound), min(int(upper_bound), configured_int(params, gen_defaults, min_key, int(default_min))))
    maximum = max(minimum, min(int(upper_bound), configured_int(params, gen_defaults, max_key, int(default_max))))
    support = tuple(range(int(minimum), int(maximum) + 1))
    explicit = None
    for key in explicit_keys:
        if key in params:
            explicit = params[key]
            break
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported {explicit_keys[0]}: {selected}")
        return int(selected), uniform_int_probability_map(support, selected=int(selected))
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = int(support[int(rng.randrange(len(support)))])
    return int(selected), uniform_int_probability_map(support)


def resolve_scene_and_element(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    supported_scenes: Sequence[str],
) -> ResolvedSurfaceFixtureAxes:
    """Resolve a compatible fixture scene variant and its repeated element type."""

    scene_support = tuple(str(scene) for scene in supported_scenes)
    element_support = tuple(str(ELEMENT_TYPE_BY_SCENE_VARIANT[str(scene)]) for scene in scene_support)
    explicit_element = params.get("target_element_type", params.get("element_type"))
    explicit_scene = params.get("scene_variant")
    if explicit_element is not None:
        element_type = str(explicit_element)
        if element_type not in set(element_support):
            raise ValueError(f"unsupported target_element_type: {element_type}")
        expected_scene = str(SCENE_VARIANT_BY_ELEMENT_TYPE[element_type])
        if expected_scene not in set(scene_support):
            raise ValueError(f"unsupported scene_variant: {expected_scene}")
        if explicit_scene is not None and str(explicit_scene) != expected_scene:
            raise ValueError(f"{element_type} fixtures require scene_variant={expected_scene}")
        return ResolvedSurfaceFixtureAxes(
            scene_variant=str(expected_scene),
            scene_variant_probabilities=one_hot_probability_map(scene_support, expected_scene),
            element_type=str(element_type),
            element_type_probabilities=one_hot_probability_map(element_support, element_type),
        )

    scene_variant, scene_probabilities = resolve_axis_variant_for_namespace(
        params,
        namespace=f"{namespace}.scene_variant",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=scene_support,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )
    element_type = str(ELEMENT_TYPE_BY_SCENE_VARIANT[str(scene_variant)])
    if explicit_scene is not None:
        element_probabilities = one_hot_probability_map(element_support, element_type)
    else:
        element_probabilities = {
            str(ELEMENT_TYPE_BY_SCENE_VARIANT[str(scene)]): float(scene_probabilities[str(scene)])
            for scene in scene_support
        }
    return ResolvedSurfaceFixtureAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probabilities),
        element_type=str(element_type),
        element_type_probabilities=dict(element_probabilities),
    )


def resolve_color(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    explicit_key: str,
    exclude: Sequence[str] = (),
) -> str:
    """Resolve a semantic fixture color."""

    explicit = params.get(str(explicit_key))
    support = tuple(color for color in SEMANTIC_COLOR_SUPPORT if str(color) not in set(str(item) for item in exclude))
    if not support:
        raise ValueError("empty color support")
    if explicit is not None:
        color = str(explicit)
        if color not in set(support):
            raise ValueError(f"unsupported {explicit_key}: {color}")
        return str(color)
    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(support[int(rng.randrange(len(support)))])


def sample_indices(rng: Any, support: Sequence[int], count: int) -> list[int]:
    """Sample sorted flat indices without replacement."""

    choices = list(int(value) for value in support)
    rng.shuffle(choices)
    return sorted(choices[: int(count)])


__all__ = [
    "ResolvedSurfaceFixtureAxes",
    "configured_int",
    "one_hot_probability_map",
    "resolve_color",
    "resolve_int_support",
    "resolve_scene_and_element",
    "sample_indices",
    "uniform_int_probability_map",
]
