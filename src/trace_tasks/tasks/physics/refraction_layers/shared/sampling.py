"""Sampling helpers for the refraction-layers scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng

from .state import (
    ALL_SPEED_ORDERS,
    MEDIUM_LABELS,
    OPTION_LABELS,
    SCENE_NAMESPACE,
    SPEED_VALUES_BY_RANK,
    RefractionScenario,
)


def order_text(order: Sequence[str]) -> str:
    """Return the visible speed-order option text."""

    return " > ".join(str(label) for label in order)


def parse_speed_order(raw_value: Any) -> Tuple[str, str, str] | None:
    """Parse a caller-specified medium speed order, if valid."""

    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        tokens = [part.strip() for part in raw_value.replace(">", ",").split(",") if part.strip()]
    elif isinstance(raw_value, Sequence):
        tokens = [str(part).strip() for part in raw_value]
    else:
        return None
    candidate = tuple(tokens)
    if candidate in ALL_SPEED_ORDERS:
        return candidate  # type: ignore[return-value]
    return None


def _weighted_choice(
    values: Sequence[str],
    weights: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> str:
    active: List[Tuple[str, float]] = []
    for value in values:
        weight = float(weights.get(str(value), 1.0))
        if weight > 0.0:
            active.append((str(value), float(weight)))
    rng = spawn_rng(int(instance_seed), str(namespace))
    if not active:
        return str(rng.choice(tuple(str(value) for value in values)))
    total = sum(weight for _, weight in active)
    raw = float(rng.random())
    cursor = 0.0
    for value, weight in active:
        cursor += float(weight) / float(total)
        if raw <= cursor:
            return str(value)
    return str(active[-1][0])


def make_refraction_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> RefractionScenario:
    """Resolve physical media, ray-entry direction, and visible option cards together."""

    orientation = str(params.get("layer_orientation") or params.get("orientation") or "").strip()
    if orientation not in {"horizontal", "vertical"}:
        orientation = _weighted_choice(
            ("horizontal", "vertical"),
            generation_defaults.get("layer_orientation_weights", {}),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.orientation",
        )
    if orientation == "horizontal":
        entry_options = ("top", "bottom")
    else:
        entry_options = ("left", "right")
    entry_side = str(params.get("entry_side") or "").strip()
    if entry_side not in set(entry_options):
        entry_side = str(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.entry_side").choice(entry_options))
    transverse_sign = 1 if int(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.transverse_sign").randrange(2)) == 0 else -1

    explicit_order = parse_speed_order(params.get("speed_order") or params.get("target_order"))
    if explicit_order is None:
        speed_order = tuple(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.speed_order").choice(ALL_SPEED_ORDERS))
    else:
        speed_order = explicit_order

    medium_speeds = {
        str(label): float(SPEED_VALUES_BY_RANK[rank])
        for rank, label in enumerate(speed_order)
    }
    max_angle_options = (40.0, 42.0, 44.0) if orientation == "horizontal" else (34.0, 36.0, 38.0)
    max_angle_deg = float(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.max_angle").choice(max_angle_options))
    invariant = math.sin(math.radians(max_angle_deg)) / max(medium_speeds.values())
    angle_by_medium_deg = {
        str(label): round(float(math.degrees(math.asin(max(-0.95, min(0.95, invariant * speed))))), 3)
        for label, speed in medium_speeds.items()
    }

    correct_index = int(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.correct_option").randrange(len(OPTION_LABELS)))
    distractor_orders = [order for order in ALL_SPEED_ORDERS if tuple(order) != tuple(speed_order)]
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.option_shuffle")
    rng.shuffle(distractor_orders)
    option_map: Dict[str, str] = {}
    distractor_index = 0
    for index, label in enumerate(OPTION_LABELS):
        if index == correct_index:
            option_map[str(label)] = order_text(speed_order)
        else:
            option_map[str(label)] = order_text(distractor_orders[distractor_index])
            distractor_index += 1

    return RefractionScenario(
        orientation=str(orientation),
        entry_side=str(entry_side),
        transverse_sign=int(transverse_sign),
        speed_order=tuple(speed_order),
        medium_speeds=dict(medium_speeds),
        angle_by_medium_deg=dict(angle_by_medium_deg),
        option_map=dict(option_map),
        correct_label=str(OPTION_LABELS[correct_index]),
    )
