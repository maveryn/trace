"""Identity-free sampling helpers for Pac-Man maze scenes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import (
    DEFAULTS,
    PACMAN_GHOST_COLOR_KEYS,
    SUPPORTED_PACMAN_SCENE_VARIANTS,
    SUPPORTED_PACMAN_STYLE_VARIANTS,
)
from .state import (
    Coord,
    PacmanGhost,
    all_grid_coords,
    ghost_entity_id,
    grid_neighbors,
    sorted_coords,
)


@dataclass(frozen=True)
class PacmanVisualAxes:
    """Resolved visual and board-size axes shared by Pac-Man tasks."""

    scene_variant: str
    style_variant: str
    row_count: int
    col_count: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    row_count_probabilities: Dict[str, float]
    col_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PacmanIntegerTargetAxis:
    """Resolved balanced integer support for one task-owned target."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PacmanLabelTargetAxis:
    """Resolved balanced label support for one task-owned target."""

    target_label: str
    target_label_support: Tuple[str, ...]
    target_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PacmanCountAxis:
    """Resolved balanced integer support for one count axis."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named Pac-Man visual axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.{namespace}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(value) for value in supported],
    )


def resolve_pacman_visual_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> PacmanVisualAxes:
    """Resolve visual and maze-size axes without task/objective branching."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_PACMAN_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_PACMAN_STYLE_VARIANTS,
    )
    row_count, row_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="row_count_support",
        explicit_key="row_count",
        fallback_support=DEFAULTS.row_count_support,
        namespace=f"{namespace}.row_count",
        balanced_flag_key="balanced_row_count_sampling",
        namespace_support_permutation=True,
    )
    col_count, col_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="col_count_support",
        explicit_key="col_count",
        fallback_support=DEFAULTS.col_count_support,
        namespace=f"{namespace}.col_count",
        balanced_flag_key="balanced_col_count_sampling",
        namespace_support_permutation=True,
    )
    return PacmanVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        row_count=int(row_count),
        col_count=int(col_count),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        row_count_probabilities=dict(row_count_probabilities),
        col_count_probabilities=dict(col_count_probabilities),
    )


def string_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve a string support list from params/defaults."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    values = (str(raw),) if isinstance(raw, str) else tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    return values


def resolve_pacman_integer_target(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> PacmanIntegerTargetAxis:
    """Resolve a task-owned integer target from config support."""

    target_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    target_answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return PacmanIntegerTargetAxis(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        target_answer_probabilities=dict(probabilities),
    )


def resolve_pacman_count_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> PacmanCountAxis:
    """Resolve a non-answer integer axis such as visible item count."""

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return PacmanCountAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_pacman_label_target(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
    balanced_flag_key: str,
) -> PacmanLabelTargetAxis:
    """Resolve one task-owned label-valued target."""

    support = string_support(params, gen_defaults=gen_defaults, key=str(support_key), fallback=fallback_support)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        return PacmanLabelTargetAxis(
            target_label=value,
            target_label_support=tuple(str(item) for item in support),
            target_label_probabilities={str(item): (1.0 if str(item) == value else 0.0) for item in support},
        )
    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), str(namespace))
    label = str(uniform_choice(rng, support))
    return PacmanLabelTargetAxis(
        target_label=label,
        target_label_support=tuple(str(item) for item in support),
        target_label_probabilities=probabilities,
    )


def sample_route(*, rng: Any, rows: int, cols: int, length: int) -> Tuple[Coord, ...]:
    """Sample one self-avoiding orthogonal route."""

    inner_rows = tuple(range(1, int(rows) - 1))
    inner_cols = tuple(range(1, int(cols) - 1))
    for _restart in range(96):
        start = (int(rng.choice(inner_rows)), 1)
        route = [start]
        seen = {start}
        while len(route) < int(length):
            current = route[-1]
            candidates = [coord for coord in grid_neighbors(current, rows=rows, cols=cols) if coord not in seen]
            if not candidates:
                break
            rng.shuffle(candidates)
            candidates.sort(key=lambda coord: (0 if int(coord[1]) >= int(current[1]) else 1, rng.random()))
            next_coord = tuple(candidates[0])
            route.append(next_coord)
            seen.add(next_coord)
        if len(route) >= int(length):
            return tuple(route[: int(length)])
    raise ValueError("failed to sample Pac-Man route")


def expand_open_cells(
    *,
    rng: Any,
    rows: int,
    cols: int,
    route_coords: Sequence[Coord],
    min_open_cells: int,
) -> Tuple[Coord, ...]:
    """Expand a connected open-cell set around the route."""

    open_set = {tuple(coord) for coord in route_coords}
    target = min(
        max(int(min_open_cells), len(open_set)),
        max(1, (int(rows) - 2) * (int(cols) - 2)),
    )
    guard = 0
    while len(open_set) < int(target) and guard < int(rows) * int(cols) * 8:
        guard += 1
        source = tuple(rng.choice(tuple(sorted(open_set))))
        candidates = [coord for coord in grid_neighbors(source, rows=rows, cols=cols) if coord not in open_set]
        if not candidates:
            continue
        open_set.add(tuple(rng.choice(candidates)))
    if len(open_set) < int(target):
        raise ValueError("failed to expand Pac-Man open cells")
    return sorted_coords(open_set)


def wall_cells(*, rows: int, cols: int, open_cells: Sequence[Coord]) -> Tuple[Coord, ...]:
    """Return wall cells as every grid cell not marked open."""

    open_set = {tuple(coord) for coord in open_cells}
    return sorted_coords(coord for coord in all_grid_coords(rows, cols) if coord not in open_set)


def available_open_cells(
    open_cells: Sequence[Coord],
    *,
    excluded: Sequence[Coord] = (),
) -> Tuple[Coord, ...]:
    """Return open cells excluding a set."""

    excluded_set = {tuple(coord) for coord in excluded}
    return sorted_coords(coord for coord in open_cells if tuple(coord) not in excluded_set)


def sample_decorative_ghosts(
    *,
    rng: Any,
    open_cells: Sequence[Coord],
    excluded: Sequence[Coord],
    start_index: int = 1,
    min_count: int = 1,
    max_count: int = 3,
) -> Tuple[PacmanGhost, ...]:
    """Sample off-objective ghosts used as Pac-Man scene decoration."""

    candidates = list(available_open_cells(open_cells, excluded=excluded))
    rng.shuffle(candidates)
    if not candidates:
        return tuple()
    count = min(len(candidates), int(rng.randint(int(min_count), int(max_count) + 1)))
    ghosts: list[PacmanGhost] = []
    color_cycle = cycle(shuffled_support(rng, PACMAN_GHOST_COLOR_KEYS))
    for offset, coord in enumerate(candidates[:count]):
        color_key = str(next(color_cycle))
        ghosts.append(
            PacmanGhost(
                ghost_id=ghost_entity_id(int(start_index) + int(offset)),
                coord=tuple(coord),
                color_key=color_key,
                is_stop_ghost=False,
            )
        )
    return tuple(ghosts)


__all__ = [
    "PacmanCountAxis",
    "PacmanIntegerTargetAxis",
    "PacmanLabelTargetAxis",
    "PacmanVisualAxes",
    "available_open_cells",
    "expand_open_cells",
    "resolve_pacman_count_axis",
    "resolve_pacman_integer_target",
    "resolve_pacman_label_target",
    "resolve_pacman_visual_axes",
    "sample_decorative_ghosts",
    "sample_route",
    "string_support",
    "wall_cells",
]
