"""Sampling helpers for bridge-circuit apparatus parameters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import (
    uniform_probability_map,
)

from .formulas import construct_balanced_values
from .state import (
    DEFAULT_TARGET_SUPPORT,
    RESISTOR_LABELS,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
    BridgeResistor,
    BridgeScenario,
)


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map over values."""

    resolved = tuple(str(value) for value in values)
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in resolved}
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {value: float(probability) for value in resolved}


def support_from_defaults(
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a non-empty sorted integer support from scene defaults."""

    raw = group_default(defaults, str(key), tuple(int(value) for value in fallback))
    support: List[int] = []
    for value in raw:
        selected = int(value)
        if selected not in support:
            support.append(selected)
    if not support:
        raise ValueError(f"{key} must contain at least one integer")
    return tuple(sorted(support))


def resolve_scene_variant(params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve the bridge scene variant."""

    explicit = str(params.get("scene_variant") or SCENE_VARIANT).strip()
    if explicit != SCENE_VARIANT:
        raise ValueError(f"unsupported bridge scene_variant: {explicit}")
    return SCENE_VARIANT, {SCENE_VARIANT: 1.0}


def resolve_accent_color(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select a visible accent color name for the meter needle."""

    supported = tuple(str(value) for value in SUPPORTED_PHYSICS_COLOR_NAMES)
    explicit = str(params.get("accent_color_name") or "").strip()
    if explicit:
        if explicit not in supported:
            raise ValueError(f"unsupported accent_color_name: {explicit}")
        return explicit, probability_map(supported, selected=explicit)

    weights = group_default(defaults, "accent_color_name_weights", {})
    if bool(
        params.get(
            "balanced_accent_color_name_sampling",
            group_default(defaults, "balanced_accent_color_name_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
        selected = str(rng.choice(supported))
    else:
        weighted: List[str] = []
        raw_weights = params.get("accent_color_name_weights", weights)
        for color in supported:
            weight = (
                float(raw_weights.get(color, 1.0))
                if isinstance(raw_weights, Mapping)
                else 1.0
            )
            weighted.extend([color] * max(0, int(round(weight))))
        if not weighted:
            weighted = list(supported)
        rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
        selected = weighted[int(rng.randrange(len(weighted)))]
    return str(selected), probability_map(supported)


def resolve_missing_resistor(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select the resistor slot hidden behind the question mark."""

    explicit = str(params.get("missing_resistor") or "").strip()
    if explicit:
        if explicit not in RESISTOR_LABELS:
            raise ValueError(f"unsupported missing_resistor: {explicit}")
        return explicit, probability_map(RESISTOR_LABELS, selected=explicit)
    if bool(
        params.get(
            "balanced_missing_resistor_sampling",
            group_default(defaults, "balanced_missing_resistor_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.missing_resistor")
        selected = str(rng.choice(RESISTOR_LABELS))
    else:
        weights = params.get(
            "missing_resistor_weights",
            group_default(defaults, "missing_resistor_weights", {}),
        )
        weighted: List[str] = []
        for label in RESISTOR_LABELS:
            weight = (
                float(weights.get(label, 1.0))
                if isinstance(weights, Mapping)
                else 1.0
            )
            weighted.extend([label] * max(0, int(round(weight))))
        if not weighted:
            weighted = list(RESISTOR_LABELS)
        rng = spawn_rng(int(instance_seed), f"{namespace}.missing_resistor")
        selected = weighted[int(rng.randrange(len(weighted)))]
    return str(selected), probability_map(RESISTOR_LABELS)


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Select the missing resistance value."""

    support = support_from_defaults(defaults, "target_answer_support", DEFAULT_TARGET_SUPPORT)
    explicit = params.get("target_resistance_ohm", params.get("target_answer"))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported target_answer: {selected}")
        return int(selected), uniform_probability_map(support, selected=int(selected))
    if bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(defaults, "balanced_target_answer_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
        selected = int(rng.choice(support))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
        selected = int(support[int(rng.randrange(len(support)))])
    return int(selected), uniform_probability_map(support)


def resolve_bridge_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> BridgeScenario:
    """Resolve all scene state needed by the bridge-circuit renderer."""

    scene_variant, scene_probabilities = resolve_scene_variant(params)
    accent_color, accent_probabilities = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    missing_resistor, missing_probabilities = resolve_missing_resistor(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    target_answer, target_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    values = construct_balanced_values(
        instance_seed=int(instance_seed),
        missing_resistor=str(missing_resistor),
        target_answer=int(target_answer),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    resistors = tuple(
        BridgeResistor(
            label=str(label),
            value_ohm=int(values[str(label)]),
            is_missing=str(label) == str(missing_resistor),
        )
        for label in RESISTOR_LABELS
    )
    return BridgeScenario(
        scene_variant=str(scene_variant),
        accent_color_name=str(accent_color),
        missing_resistor=str(missing_resistor),
        target_answer=int(target_answer),
        resistors=resistors,
        scene_variant_probabilities=dict(scene_probabilities),
        accent_color_name_probabilities=dict(accent_probabilities),
        missing_resistor_probabilities=dict(missing_probabilities),
        target_answer_probabilities=dict(target_probabilities),
    )


__all__ = [
    "probability_map",
    "resolve_accent_color",
    "resolve_bridge_scenario",
    "resolve_missing_resistor",
    "resolve_scene_variant",
    "resolve_target_answer",
    "support_from_defaults",
]
