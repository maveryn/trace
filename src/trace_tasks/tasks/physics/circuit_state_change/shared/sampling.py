"""Sampling helpers for circuit state-change scenarios."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default

from .formulas import change_class, powers_for_state
from .state import (
    BULB_LABELS,
    BULB_ROLES,
    CHANGE_CLASSES,
    DEFAULT_RESISTANCE_OPTIONS,
    SCENE_NAMESPACE,
    BulbStateChangeSpec,
    CircuitStateChangeScenario,
)


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map over a finite support."""

    resolved = tuple(str(value) for value in values)
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in resolved}
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {value: float(probability) for value in resolved}


def resolve_switch_action(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    allowed_switch_actions: Sequence[str],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visible switch action from task-owned semantic constraints."""

    allowed = tuple(str(value) for value in allowed_switch_actions)
    if not allowed or any(value not in {"opens", "closes"} for value in allowed):
        raise ValueError("allowed_switch_actions must contain opens and/or closes")
    explicit = str(params.get("switch_action") or "").strip().lower()
    if explicit:
        if explicit in {"close", "closed"}:
            explicit = "closes"
        if explicit in {"open", "opened"}:
            explicit = "opens"
        if explicit not in set(allowed):
            raise ValueError(f"switch_action={explicit!r} is incompatible with the selected query")
        return str(explicit), probability_map(allowed, selected=str(explicit))
    if len(allowed) == 1:
        return str(allowed[0]), probability_map(allowed, selected=str(allowed[0]))
    rng = spawn_rng(int(instance_seed), f"{namespace}.switch_action")
    return str(rng.choice(allowed)), probability_map(allowed)


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
    if bool(
        params.get(
            "balanced_accent_color_name_sampling",
            group_default(defaults, "balanced_accent_color_name_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
        selected = str(rng.choice(supported))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
        selected = supported[int(rng.randrange(len(supported)))]
    return str(selected), probability_map(supported)


def resolve_target_label(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select which visible bulb label will receive the answer role."""

    explicit = str(params.get("target_label") or params.get("target_answer") or "").strip()
    if explicit:
        if explicit not in set(BULB_LABELS):
            raise ValueError(f"target label must be one of {BULB_LABELS}")
        return str(explicit), probability_map(BULB_LABELS, selected=str(explicit))
    if bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(defaults, "balanced_target_answer_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label")
        selected = str(rng.choice(BULB_LABELS))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label")
        selected = BULB_LABELS[int(rng.randrange(len(BULB_LABELS)))]
    return str(selected), probability_map(BULB_LABELS)


def resistance_options(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve available positive resistance values."""

    raw = params.get(
        "resistance_options",
        group_default(defaults, "resistance_options", DEFAULT_RESISTANCE_OPTIONS),
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("resistance_options must be a sequence of positive integers")
    values = tuple(dict.fromkeys(int(value) for value in raw))
    if len(values) < len(BULB_ROLES) or any(value <= 0 for value in values):
        raise ValueError(f"resistance_options must contain at least {len(BULB_ROLES)} positive integers")
    return values


def resolve_resistances(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Dict[str, int]:
    """Resolve one resistance value for each semantic bulb role."""

    explicit_raw = params.get("resistance_values")
    if explicit_raw is not None:
        if isinstance(explicit_raw, Mapping):
            values = {str(role): int(explicit_raw[str(role)]) for role in BULB_ROLES}
        else:
            if (
                isinstance(explicit_raw, (str, bytes))
                or not isinstance(explicit_raw, Sequence)
                or len(explicit_raw) != len(BULB_ROLES)
            ):
                raise ValueError(f"resistance_values must be a role mapping or a {len(BULB_ROLES)}-value sequence")
            values = {str(role): int(value) for role, value in zip(BULB_ROLES, explicit_raw)}
        if any(value <= 0 for value in values.values()):
            raise ValueError("resistance_values must be positive")
        return dict(values)
    options = list(resistance_options(params, defaults))
    rng = spawn_rng(int(instance_seed), f"{namespace}.resistances")
    sampled = rng.sample(options, k=len(BULB_ROLES))
    return {str(role): int(value) for role, value in zip(BULB_ROLES, sampled)}


def resolve_state_change_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    target_change_class: str,
    allowed_switch_actions: Sequence[str],
    namespace: str = SCENE_NAMESPACE,
) -> CircuitStateChangeScenario:
    """Resolve all symbolic state needed by the state-change circuit renderer."""

    change_target = str(target_change_class)
    if change_target not in set(CHANGE_CLASSES):
        raise ValueError(f"unsupported target_change_class: {change_target}")
    switch_action, switch_probabilities = resolve_switch_action(
        instance_seed=int(instance_seed),
        params=params,
        allowed_switch_actions=allowed_switch_actions,
        namespace=str(namespace),
    )
    accent_color_name, accent_probabilities = resolve_accent_color(
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
        namespace=str(namespace),
    )
    before_closed = str(switch_action) == "opens"
    after_closed = str(switch_action) == "closes"
    powers_before = powers_for_state(resistances, switch_closed=before_closed)
    powers_after = powers_for_state(resistances, switch_closed=after_closed)
    role_changes = {
        str(role): change_class(float(powers_before[str(role)]), float(powers_after[str(role)]))
        for role in BULB_ROLES
    }
    answer_roles = [
        str(role)
        for role, role_change in role_changes.items()
        if str(role_change) == str(change_target)
    ]
    if len(answer_roles) != 1:
        raise RuntimeError(f"state-change circuit must produce one answer, got {answer_roles}")
    answer_role = str(answer_roles[0])

    label_by_role = {str(answer_role): str(target_label)}
    remaining_roles = [str(role) for role in BULB_ROLES if str(role) != str(answer_role)]
    remaining_labels = [str(label) for label in BULB_LABELS if str(label) != str(target_label)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.label_assignment")
    rng.shuffle(remaining_roles)
    rng.shuffle(remaining_labels)
    for role, label in zip(remaining_roles, remaining_labels):
        label_by_role[str(role)] = str(label)

    bulbs = tuple(
        BulbStateChangeSpec(
            role=str(role),
            label=str(label_by_role[str(role)]),
            resistance_ohm=int(resistances[str(role)]),
            power_before=float(powers_before[str(role)]),
            power_after=float(powers_after[str(role)]),
            change_class=str(role_changes[str(role)]),
        )
        for role in BULB_ROLES
    )
    correct_label = next(
        str(spec.label)
        for spec in bulbs
        if str(spec.change_class) == str(change_target)
    )
    return CircuitStateChangeScenario(
        switch_action=str(switch_action),
        target_change_class=str(change_target),
        accent_color_name=str(accent_color_name),
        target_label=str(target_label),
        bulbs=bulbs,
        correct_label=str(correct_label),
        switch_action_probabilities=dict(switch_probabilities),
        accent_color_name_probabilities=dict(accent_probabilities),
        target_label_probabilities=dict(target_label_probabilities),
    )


__all__ = [
    "probability_map",
    "resolve_state_change_scenario",
]
