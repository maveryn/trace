"""Sampling primitives for graduated-cylinder diagrams."""

from __future__ import annotations

from collections.abc import Sequence

from trace_tasks.core.seed import spawn_rng

from .state import LIQUID_PALETTE, SCALE_OPTIONS, SCENE_NAMESPACE, CylinderScale


def choose_scale(instance_seed: int, *, namespace: str = SCENE_NAMESPACE) -> CylinderScale:
    """Choose one graduated scale from the scene support."""

    return spawn_rng(int(instance_seed), f"{namespace}.scale").choice(SCALE_OPTIONS)


def volume_support(
    scale: CylinderScale,
    *,
    min_ml: int = 10,
    max_margin_ml: int = 10,
) -> tuple[int, ...]:
    """Return readable volume values for one scale."""

    support = [
        int(value)
        for value in range(
            int(min_ml),
            int(scale.capacity_ml - max_margin_ml) + 1,
            int(scale.minor_tick_ml),
        )
    ]
    support = [
        int(value)
        for value in support
        if int(value) % int(scale.major_tick_ml) != 0 or len(support) < 8
    ]
    if not support:
        raise ValueError("graduated-cylinder volume support is empty")
    return tuple(support)


def choose_from_support(
    instance_seed: int,
    support: Sequence[int],
    *,
    namespace: str,
) -> int:
    """Choose one integer from a finite support."""

    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError("cannot choose from an empty graduated-cylinder support")
    return int(spawn_rng(int(instance_seed), str(namespace)).choice(values))


def choose_volume(
    instance_seed: int,
    scale: CylinderScale,
    *,
    min_ml: int = 10,
    max_margin_ml: int = 10,
    namespace: str = SCENE_NAMESPACE,
) -> int:
    """Choose one readable liquid volume for one scale."""

    return choose_from_support(
        int(instance_seed),
        volume_support(scale, min_ml=int(min_ml), max_margin_ml=int(max_margin_ml)),
        namespace=f"{namespace}.volume",
    )


def displacement_support(scale: CylinderScale, *, before_ml: int) -> tuple[int, ...]:
    """Return feasible displaced-volume increments for a before reading."""

    support = [
        int(value)
        for value in range(int(scale.minor_tick_ml), 41, int(scale.minor_tick_ml))
        if int(before_ml) + int(value) <= int(scale.capacity_ml) - 8
    ]
    if not support:
        raise ValueError("graduated-cylinder displacement support is empty")
    return tuple(support)


def choose_displacement(
    instance_seed: int,
    scale: CylinderScale,
    *,
    before_ml: int,
    namespace: str = SCENE_NAMESPACE,
) -> int:
    """Choose one feasible displacement increment."""

    return choose_from_support(
        int(instance_seed),
        displacement_support(scale, before_ml=int(before_ml)),
        namespace=f"{namespace}.displacement",
    )


def choose_liquid_rgb(
    instance_seed: int,
    *,
    namespace: str = SCENE_NAMESPACE,
) -> tuple[int, int, int]:
    """Choose one readable liquid color."""

    selected = spawn_rng(int(instance_seed), f"{namespace}.liquid").choice(LIQUID_PALETTE)
    return tuple(int(value) for value in selected)
