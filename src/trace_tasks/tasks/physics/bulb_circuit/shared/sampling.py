"""Sampling helpers for bulb-circuit apparatus parameters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default

from .formulas import has_unique_powers, power_values
from .state import (
    BULB_LABELS,
    DEFAULT_RESISTANCE_OPTIONS,
    SCENE_NAMESPACE,
    SCENE_VARIANTS,
    TARGET_DIRECTIONS,
    BulbScenario,
    BulbSpec,
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


def _weighted_choice(
    *,
    values: Sequence[str],
    weights: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> str:
    weighted: List[str] = []
    for value in values:
        weight = float(weights.get(str(value), 1.0)) if isinstance(weights, Mapping) else 1.0
        weighted.extend([str(value)] * max(0, int(round(weight))))
    if not weighted:
        weighted = [str(value) for value in values]
    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(weighted[int(rng.randrange(len(weighted)))])


def resolve_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select the visible bulb-circuit topology variant."""

    explicit = str(params.get("scene_variant") or params.get("topology_variant") or "").strip()
    if explicit:
        if explicit not in set(SCENE_VARIANTS):
            raise ValueError(f"unsupported scene_variant: {explicit}")
        return explicit, probability_map(SCENE_VARIANTS, selected=explicit)

    if bool(params.get("balanced_scene_variant_sampling", group_default(defaults, "balanced_scene_variant_sampling", True))):
        cursor = params.get("_sample_cursor")
        choice_namespace = (
            f"{namespace}.scene_variant"
            if cursor is None
            else f"{namespace}.scene_variant.cursor.{int(cursor)}"
        )
        rng = spawn_rng(int(instance_seed), choice_namespace)
        selected = str(rng.choice(SCENE_VARIANTS))
    else:
        selected = _weighted_choice(
            values=SCENE_VARIANTS,
            weights=group_default(defaults, "scene_variant_weights", {}),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.scene_variant",
        )
    return str(selected), probability_map(SCENE_VARIANTS)


def resolve_accent_color(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select a visible accent color name for bulb filaments."""

    supported = tuple(str(value) for value in SUPPORTED_PHYSICS_COLOR_NAMES)
    explicit = str(params.get("accent_color_name") or "").strip()
    if explicit:
        if explicit not in supported:
            raise ValueError(f"unsupported accent_color_name: {explicit}")
        return explicit, probability_map(supported, selected=explicit)

    if bool(params.get("balanced_accent_color_name_sampling", group_default(defaults, "balanced_accent_color_name_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
        selected = str(rng.choice(supported))
    else:
        selected = _weighted_choice(
            values=supported,
            weights=group_default(defaults, "accent_color_name_weights", {}),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.accent_color_name",
        )
    return str(selected), probability_map(supported)


def resolve_target_label(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select which public bulb label will receive the answer slot."""

    explicit = str(params.get("target_label") or params.get("target_answer") or "").strip()
    if explicit:
        if explicit not in set(BULB_LABELS):
            raise ValueError(f"target label must be one of {BULB_LABELS}")
        return explicit, probability_map(BULB_LABELS, selected=explicit)

    if bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        cursor = params.get("_sample_cursor")
        choice_namespace = (
            f"{namespace}.target_label"
            if cursor is None
            else f"{namespace}.target_label.cursor.{int(cursor)}"
        )
        rng = spawn_rng(int(instance_seed), choice_namespace)
        selected = str(rng.choice(BULB_LABELS))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label")
        selected = BULB_LABELS[int(rng.randrange(len(BULB_LABELS)))]
    return str(selected), probability_map(BULB_LABELS)


def resistance_options(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve available positive resistance options."""

    raw = params.get(
        "resistance_options",
        group_default(defaults, "resistance_options", DEFAULT_RESISTANCE_OPTIONS),
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("resistance_options must be a sequence of positive integers")
    values = tuple(dict.fromkeys(int(value) for value in raw))
    if len(values) < len(BULB_LABELS) or any(value <= 0 for value in values):
        raise ValueError("resistance_options must contain at least five positive integers")
    return values


def resolve_resistances(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    scene_variant: str,
    namespace: str,
) -> Tuple[int, ...]:
    """Resolve five resistances that produce a unique brightness order."""

    explicit_raw = params.get("resistance_values")
    if explicit_raw is not None:
        if (
            isinstance(explicit_raw, (str, bytes))
            or not isinstance(explicit_raw, Sequence)
            or len(explicit_raw) != len(BULB_LABELS)
        ):
            raise ValueError("resistance_values must contain exactly five positive integers")
        values = tuple(int(value) for value in explicit_raw)
        if any(value <= 0 for value in values):
            raise ValueError("resistance_values must be positive")
        if not has_unique_powers(power_values(str(scene_variant), values)):
            raise ValueError("resistance_values must produce a unique brightness order")
        return values

    options = list(resistance_options(params, defaults))
    rng = spawn_rng(int(instance_seed), f"{namespace}.resistances.{scene_variant}")
    for _ in range(200):
        values = tuple(int(value) for value in rng.sample(options, k=len(BULB_LABELS)))
        if has_unique_powers(power_values(str(scene_variant), values)):
            return values
    raise RuntimeError(f"failed to sample unique bulb powers for {scene_variant}")


def resolve_bulb_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    target_direction: str,
    namespace: str = SCENE_NAMESPACE,
) -> BulbScenario:
    """Resolve all state needed by the bulb-circuit renderer."""

    direction = str(target_direction)
    if direction not in set(TARGET_DIRECTIONS):
        raise ValueError(f"unsupported target_direction: {direction}")
    scene_variant, scene_probabilities = resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    accent_color, accent_probabilities = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    target_label, target_label_probabilities = resolve_target_label(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    resistances = resolve_resistances(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        scene_variant=str(scene_variant),
        namespace=str(namespace),
    )
    powers = power_values(str(scene_variant), resistances)
    slots = tuple(f"slot_{index}" for index in range(len(BULB_LABELS)))
    sorted_slots = sorted(
        slots,
        key=lambda slot: float(powers[slots.index(slot)]),
        reverse=True,
    )
    brightest_slot = str(sorted_slots[0])
    dimmest_slot = str(sorted_slots[-1])
    target_slot = brightest_slot if direction == "brightest" else dimmest_slot

    label_by_slot: Dict[str, str] = {str(target_slot): str(target_label)}
    remaining_slots = [slot for slot in slots if str(slot) != str(target_slot)]
    remaining_labels = [label for label in BULB_LABELS if str(label) != str(target_label)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.label_assignment")
    rng.shuffle(remaining_slots)
    rng.shuffle(remaining_labels)
    for slot, label in zip(remaining_slots, remaining_labels):
        label_by_slot[str(slot)] = str(label)

    bulbs = tuple(
        BulbSpec(
            slot_id=str(slot),
            label=str(label_by_slot[str(slot)]),
            resistance_ohm=int(resistances[index]),
            relative_power=float(powers[index]),
        )
        for index, slot in enumerate(slots)
    )
    label_by_power = {str(spec.label): float(spec.relative_power) for spec in bulbs}
    brightest_label = max(label_by_power, key=label_by_power.get)
    dimmest_label = min(label_by_power, key=label_by_power.get)
    branch_single_position = (
        "top"
        if int(spawn_rng(int(instance_seed), f"{namespace}.single_branch_position").randrange(2)) == 0
        else "bottom"
    )
    if params.get("branch_single_position") in {"top", "bottom"}:
        branch_single_position = str(params["branch_single_position"])
    return BulbScenario(
        target_direction=str(direction),
        scene_variant=str(scene_variant),
        accent_color_name=str(accent_color),
        branch_single_position=str(branch_single_position),
        bulbs=bulbs,
        correct_label=str(brightest_label if direction == "brightest" else dimmest_label),
        brightest_label=str(brightest_label),
        dimmest_label=str(dimmest_label),
        scene_variant_probabilities=dict(scene_probabilities),
        accent_color_name_probabilities=dict(accent_probabilities),
        target_label_probabilities=dict(target_label_probabilities),
    )


__all__ = [
    "probability_map",
    "resistance_options",
    "resolve_accent_color",
    "resolve_bulb_scenario",
    "resolve_resistances",
    "resolve_scene_variant",
    "resolve_target_label",
]
