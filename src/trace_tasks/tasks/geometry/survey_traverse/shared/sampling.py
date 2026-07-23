"""Scene-local sampling helpers for survey-traverse diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index

BEARING_SUPPORT: Tuple[int, ...] = (
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    200,
    210,
    220,
    230,
    240,
    250,
    260,
    280,
    290,
    300,
    310,
    320,
    330,
    340,
)
TURN_SUPPORT: Tuple[int, ...] = (35, 45, 55, 65, 75, 85, 95, 105, 115, 125)
LEVELING_CASES: Tuple[Tuple[int, int, int], ...] = (
    (120, 4, 7),
    (96, 6, 2),
    (142, 3, 8),
    (88, 5, 9),
    (135, 7, 4),
    (105, 2, 6),
    (160, 8, 3),
    (118, 9, 5),
    (175, 4, 10),
    (132, 6, 11),
)
OFFSET_TRAPEZOID_CASES: Tuple[Tuple[int, int, int, int, int, int, int], ...] = (
    (20, 40, 60, 3, 5, 4, 2),
    (30, 60, 90, 2, 4, 7, 5),
    (20, 50, 80, 4, 6, 3, 5),
    (25, 50, 75, 6, 2, 4, 8),
    (40, 80, 120, 3, 6, 5, 2),
    (30, 70, 100, 5, 3, 6, 4),
)


def support_probability_map(values: Sequence[Any], selected: Any | None = None) -> Dict[str, float]:
    """Return a JSON-stable probability map over a finite support."""

    keys = tuple(",".join(str(item) for item in value) if isinstance(value, tuple) else str(value) for value in values)
    if selected is not None:
        selected_key = ",".join(str(item) for item in selected) if isinstance(selected, tuple) else str(selected)
        return {key: (1.0 if key == selected_key else 0.0) for key in keys}
    probability = 1.0 / float(max(1, len(keys)))
    return {key: float(probability) for key in keys}


def choose_from_support(
    *,
    values: Sequence[Any],
    params: Mapping[str, Any],
    explicit_key: str,
    instance_seed: int,
    namespace: str,
) -> tuple[Any, Dict[str, float]]:
    """Select one finite construction case with explicit override validation."""

    explicit = params.get(str(explicit_key))
    if explicit is not None:
        if isinstance(values[0], tuple):
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise ValueError(f"{explicit_key} must be a numeric sequence")
            selected = tuple(int(value) for value in explicit)
        else:
            selected = int(explicit)
        if selected not in values:
            raise ValueError(f"{explicit_key}={selected} is not supported")
        return selected, support_probability_map(values, selected=selected)
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = uniform_choice(rng, tuple(values))
    return selected, support_probability_map(values)


def choose_turn(*, params: Mapping[str, Any], instance_seed: int, namespace: str) -> tuple[int, str, Dict[str, float]]:
    """Select a finite traverse turn angle and left/right direction."""

    turn_angle, turn_probabilities = choose_from_support(
        values=TURN_SUPPORT,
        params=params,
        explicit_key="turn_angle",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.turn_angle",
    )
    explicit_direction = params.get("turn_direction")
    if explicit_direction is not None:
        turn_direction = str(explicit_direction)
        if turn_direction not in {"left", "right"}:
            raise ValueError("turn_direction must be 'left' or 'right'")
    else:
        direction_index = resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.turn_direction",
        )
        turn_direction = "left" if int(direction_index) % 2 == 0 else "right"
    return int(turn_angle), str(turn_direction), dict(turn_probabilities)


def choose_station_labels3(*, params: Mapping[str, Any], instance_seed: int, namespace: str) -> Tuple[str, str, str]:
    """Select or validate three distinct station labels for a survey sketch."""

    explicit = params.get("station_labels")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)) or len(explicit) != 3:
            raise ValueError("station_labels must be a three-item sequence")
        labels = tuple(str(value).strip().upper() for value in explicit)
        if len(set(labels)) != 3 or any(len(label) != 1 or not label.isalpha() for label in labels):
            raise ValueError("station_labels must contain three distinct single letters")
        return labels  # type: ignore[return-value]
    pools = (
        ("A", "B", "C"),
        ("P", "Q", "R"),
        ("K", "L", "M"),
        ("S", "T", "U"),
        ("D", "E", "F"),
        ("G", "H", "J"),
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.station_labels")
    return tuple(uniform_choice(rng, pools))  # type: ignore[return-value]


def choose_station_labels4(*, params: Mapping[str, Any], instance_seed: int, namespace: str) -> Tuple[str, str, str, str]:
    """Select or validate four distinct station labels for area traverses."""

    explicit = params.get("station_labels")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)) or len(explicit) not in {3, 4}:
            raise ValueError("station_labels must be a three- or four-item sequence")
        labels = tuple(str(value).strip().upper() for value in explicit)
        if len(set(labels)) != len(labels) or any(len(label) != 1 or not label.isalpha() for label in labels):
            raise ValueError("station_labels must contain distinct single letters")
        if len(labels) == 4:
            return labels  # type: ignore[return-value]
    labels3 = choose_station_labels3(params=params, instance_seed=int(instance_seed), namespace=str(namespace))
    for candidate in ("D", "E", "F", "V", "W", "X", "Y", "Z"):
        if candidate not in labels3:
            return (labels3[0], labels3[1], labels3[2], candidate)
    raise ValueError("could not construct four unique station labels")


__all__ = [
    "BEARING_SUPPORT",
    "LEVELING_CASES",
    "OFFSET_TRAPEZOID_CASES",
    "TURN_SUPPORT",
    "choose_from_support",
    "choose_station_labels3",
    "choose_station_labels4",
    "choose_turn",
    "support_probability_map",
]
