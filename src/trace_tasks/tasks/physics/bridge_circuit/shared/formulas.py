"""Balanced bridge-circuit formulas and value construction."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import DEFAULT_MAX_RESISTANCE, RESISTOR_LABELS


def bridge_balance_holds(values_by_label: Mapping[str, int]) -> bool:
    """Return whether the bridge satisfies R1*R4 = R2*R3."""

    return int(values_by_label["R1"]) * int(values_by_label["R4"]) == int(
        values_by_label["R2"]
    ) * int(values_by_label["R3"])


def explicit_resistor_values(params: Mapping[str, Any]) -> Dict[str, int] | None:
    """Return explicitly supplied balanced resistor values, if present."""

    raw = params.get("resistor_values")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        values = {label: int(raw[label]) for label in RESISTOR_LABELS}
    elif (
        not isinstance(raw, (str, bytes))
        and isinstance(raw, Sequence)
        and len(raw) == len(RESISTOR_LABELS)
    ):
        values = {label: int(raw[index]) for index, label in enumerate(RESISTOR_LABELS)}
    else:
        raise ValueError(
            "resistor_values must be a mapping with R1..R4 or a four-value sequence"
        )
    if any(int(value) <= 0 for value in values.values()):
        raise ValueError("resistor_values must contain positive integers")
    if not bridge_balance_holds(values):
        raise ValueError(
            "resistor_values must satisfy the balanced bridge equation R1*R4 = R2*R3"
        )
    return dict(values)


def construct_balanced_values(
    *,
    instance_seed: int,
    missing_resistor: str,
    target_answer: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Dict[str, int]:
    """Construct a balanced bridge with the selected missing resistor value."""

    explicit_values = explicit_resistor_values(params)
    if explicit_values is not None:
        if int(explicit_values[str(missing_resistor)]) != int(target_answer):
            raise ValueError(
                "target_answer must match the value of the selected missing_resistor"
            )
        return explicit_values

    max_resistance = int(
        params.get(
            "component_value_max",
            group_default(defaults, "component_value_max", DEFAULT_MAX_RESISTANCE),
        )
    )
    if int(max_resistance) < int(target_answer):
        raise ValueError("component_value_max must be at least the target answer")

    pair_options = (
        (2, 3),
        (2, 5),
        (2, 7),
        (2, 9),
        (3, 2),
        (3, 4),
        (3, 5),
        (3, 7),
        (3, 8),
    )
    rng = spawn_rng(
        int(instance_seed),
        f"{namespace}.bridge_values.{missing_resistor}.{int(target_answer)}",
    )
    shuffled = list(pair_options)
    rng.shuffle(shuffled)
    fallback: Dict[str, int] | None = None
    for multiplier, base in shuffled:
        t = int(target_answer)
        c = int(multiplier)
        r = int(base)
        if str(missing_resistor) == "R1":
            values = {"R1": t, "R2": c * t, "R3": r, "R4": c * r}
        elif str(missing_resistor) == "R2":
            values = {"R1": c * t, "R2": t, "R3": c * r, "R4": r}
        elif str(missing_resistor) == "R3":
            values = {"R1": r, "R2": c * r, "R3": t, "R4": c * t}
        elif str(missing_resistor) == "R4":
            values = {"R1": c * r, "R2": r, "R3": c * t, "R4": t}
        else:
            raise ValueError(f"unsupported missing resistor: {missing_resistor}")
        if max(values.values()) > int(max_resistance):
            continue
        if not bridge_balance_holds(values):
            raise RuntimeError("bridge construction produced an unbalanced resistor set")
        if fallback is None and len(set(values.values())) >= 3:
            fallback = dict(values)
        if len(set(values.values())) == 4:
            return dict(values)
    if fallback is not None:
        return dict(fallback)
    raise RuntimeError("failed to construct a balanced bridge resistor set")


__all__ = [
    "bridge_balance_holds",
    "construct_balanced_values",
    "explicit_resistor_values",
]
