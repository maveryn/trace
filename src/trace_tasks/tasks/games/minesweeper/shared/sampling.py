"""Identity-free Minesweeper board construction primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_MINESWEEPER_STYLE_VARIANTS

from .defaults import DEFAULTS, SCENE_VARIANTS
from .rules import (
    adjacent_flag_count,
    clue_number,
    forced_mine_supports,
    forced_safe_supports,
    neighbor_coords,
    validate_board_contract,
)
from .state import Coord, MinesweeperSample, all_coords, sorted_coords


@dataclass(frozen=True)
class MinesweeperAxes:
    """Resolved semantic and visual axes for one Minesweeper sample."""

    scene_variant: str
    style_variant: str
    board_size: int
    target_answer: int | None
    target_answer_support: Tuple[int, ...]
    branch_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    board_size_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    option_count: int = 0
    option_count_support: Tuple[int, ...] = tuple()
    option_count_probabilities: Dict[str, float] | None = None


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
    supported: Tuple[str, ...],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named Minesweeper generation/render axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.{namespace}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=supported,
    )


def resolve_minesweeper_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
    branch_probabilities: Mapping[str, float],
    board_size_support_key: str,
    board_size_fallback_support: Sequence[int],
    target_support_key: str,
    target_fallback_support: Sequence[int],
    option_count_support_key: str = "option_count_support",
    option_count_fallback_support: Sequence[int] = (),
) -> MinesweeperAxes:
    """Resolve common scene/style/board/target axes without public task routing."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SCENE_VARIANTS,
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
        supported=SUPPORTED_MINESWEEPER_STYLE_VARIANTS,
    )
    board_size, board_size_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(board_size_support_key),
        explicit_key="board_size",
        fallback_support=tuple(int(value) for value in board_size_fallback_support),
        namespace=f"{namespace}.board_size.{str(board_size_support_key)}",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(target_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in target_fallback_support),
        namespace=f"{namespace}.target_answer.{str(target_support_key)}",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_answer_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(target_support_key),
        fallback=tuple(int(value) for value in target_fallback_support),
    )
    option_count = 0
    option_count_support: Tuple[int, ...] = tuple()
    option_count_probabilities: Dict[str, float] = {}
    if option_count_fallback_support:
        option_count, option_count_probabilities = resolve_integer_choice(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key=str(option_count_support_key),
            explicit_key="option_count",
            fallback_support=tuple(int(value) for value in option_count_fallback_support),
            namespace=f"{namespace}.option_count",
            balanced_flag_key="balanced_option_count_sampling",
            namespace_support_permutation=True,
        )
        option_count_support = resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key=str(option_count_support_key),
            fallback=tuple(int(value) for value in option_count_fallback_support),
        )
    return MinesweeperAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_size=int(board_size),
        target_answer=None if target_answer is None else int(target_answer),
        target_answer_support=tuple(int(value) for value in target_answer_support),
        branch_probabilities=dict(branch_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        board_size_probabilities=dict(board_size_probabilities),
        target_answer_probabilities=dict(target_answer_probabilities),
        option_count=int(option_count),
        option_count_support=tuple(int(value) for value in option_count_support),
        option_count_probabilities=dict(option_count_probabilities),
    )


def _distractor_bounds(scene_variant: str) -> Tuple[int, int]:
    """Return hidden-distractor count bounds for one scene variant."""

    if str(scene_variant) == "mixed_grid":
        return int(DEFAULTS.mixed_min_distractor_hidden_count), int(DEFAULTS.mixed_max_distractor_hidden_count)
    return int(DEFAULTS.open_min_distractor_hidden_count), int(DEFAULTS.open_max_distractor_hidden_count)


def _choose_distractors(
    *,
    rng,
    size: int,
    clue_cell: Coord,
    occupied: set[Coord],
    scene_variant: str,
) -> Tuple[Coord, ...]:
    """Choose non-local hidden distractor cells away from the forcing clue."""

    local = set(neighbor_coords(clue_cell, size=int(size)))
    local.add(clue_cell)
    candidates = [coord for coord in all_coords(size=int(size)) if coord not in occupied and coord not in local]
    rng.shuffle(candidates)
    low, high = _distractor_bounds(str(scene_variant))
    count = int(rng.randint(min(int(low), len(candidates)), min(int(high), len(candidates)))) if candidates else 0
    return tuple(sorted(candidates[:count]))


def _build_state(
    *,
    size: int,
    mine_coords: set[Coord],
    flagged_coords: set[Coord],
    hidden_coords: set[Coord],
) -> Tuple[Tuple[Coord, ...], Tuple[Coord, ...], Tuple[Coord, ...], Tuple[Coord, ...]]:
    """Return canonical mine/revealed/flagged/hidden partitions."""

    revealed_coords = set(all_coords(size=int(size))) - set(hidden_coords) - set(flagged_coords)
    validate_board_contract(
        size=int(size),
        mine_coords=tuple(mine_coords),
        revealed_coords=tuple(revealed_coords),
        flagged_coords=tuple(flagged_coords),
        hidden_coords=tuple(hidden_coords),
    )
    return (
        sorted_coords(mine_coords),
        sorted_coords(revealed_coords),
        sorted_coords(flagged_coords),
        sorted_coords(hidden_coords),
    )


def _supporting_clues(
    *,
    forced_cells: Tuple[Coord, ...],
    support_map: Mapping[Coord, Tuple[Coord, ...]],
) -> Tuple[Coord, ...]:
    """Return clue witnesses for a homogeneous forced-cell set."""

    clue_coords: set[Coord] = set()
    for coord in forced_cells:
        clue_coords.update(support_map.get(coord, ()))
    return sorted_coords(clue_coords)


def sample_forced_cell_scene(
    *,
    rng,
    axes: MinesweeperAxes,
    force_kind: str,
    target_count: int,
) -> MinesweeperSample:
    """Construct one board around a local forced mine-or-safe clue."""

    size = int(axes.board_size)
    interior = [(row, col) for row in range(1, size - 1) for col in range(1, size - 1)]
    clue_cell = tuple(rng.choice(interior))
    neighbors = list(neighbor_coords(clue_cell, size=int(size)))
    rng.shuffle(neighbors)
    hidden_targets = set(tuple(coord) for coord in neighbors[: int(target_count)])
    remaining_neighbors = [coord for coord in neighbors if coord not in hidden_targets]
    if str(force_kind) == "safe":
        flag_count = int(rng.randint(1, max(1, min(3, len(remaining_neighbors)))))
    else:
        max_flags = min(2, len(remaining_neighbors))
        flag_count = int(rng.randint(0, max_flags)) if max_flags > 0 else 0
    flagged_coords = set(tuple(coord) for coord in remaining_neighbors[: int(flag_count)])
    occupied = set(hidden_targets) | set(flagged_coords) | {clue_cell}
    distractors = set(
        _choose_distractors(
            rng=rng,
            size=int(size),
            clue_cell=clue_cell,
            occupied=occupied,
            scene_variant=str(axes.scene_variant),
        )
    )
    hidden_coords = set(hidden_targets) | set(distractors)
    if str(force_kind) == "mine":
        mine_coords = set(hidden_targets) | set(flagged_coords)
    elif str(force_kind) == "safe":
        mine_coords = set(distractors) | set(flagged_coords)
    else:
        raise ValueError(f"unsupported Minesweeper force kind: {force_kind}")
    mine_coords_tuple, revealed_coords, flagged_coords_tuple, hidden_coords_tuple = _build_state(
        size=int(size),
        mine_coords=mine_coords,
        flagged_coords=set(flagged_coords),
        hidden_coords=hidden_coords,
    )
    mine_support = forced_mine_supports(
        size=int(size),
        mine_coords=mine_coords_tuple,
        revealed_coords=revealed_coords,
        flagged_coords=flagged_coords_tuple,
        hidden_coords=hidden_coords_tuple,
    )
    safe_support = forced_safe_supports(
        size=int(size),
        mine_coords=mine_coords_tuple,
        revealed_coords=revealed_coords,
        flagged_coords=flagged_coords_tuple,
        hidden_coords=hidden_coords_tuple,
    )
    if str(force_kind) == "mine":
        forced = sorted_coords(mine_support.keys())
        support_map = mine_support
    else:
        forced = sorted_coords(safe_support.keys())
        support_map = safe_support
    if set(forced) != set(hidden_targets):
        raise ValueError("constructed Minesweeper forced-cell count does not match target")
    return MinesweeperSample(
        size=int(size),
        answer=int(len(forced)),
        mine_coords=tuple(mine_coords_tuple),
        revealed_coords=tuple(revealed_coords),
        flagged_coords=tuple(flagged_coords_tuple),
        hidden_coords=tuple(hidden_coords_tuple),
        forced_mine_coords=sorted_coords(mine_support.keys()),
        forced_safe_coords=sorted_coords(safe_support.keys()),
        forcing_clue_coords=_supporting_clues(forced_cells=tuple(forced), support_map=support_map),
        annotation_coords=tuple(forced),
        target_answer=int(target_count),
        distractor_hidden_count=len(distractors),
        construction_mode=f"local_forced_{str(force_kind)}_cells",
    )


def sample_remaining_adjacent_mine_scene(*, rng, axes: MinesweeperAxes, target_count: int) -> MinesweeperSample:
    """Construct one board asking for remaining adjacent mines around one clue."""

    size = int(axes.board_size)
    target = int(target_count)
    if target < 0 or target > 5:
        raise ValueError(f"unsupported remaining adjacent mine target count: {target}")
    interior = [(row, col) for row in range(1, size - 1) for col in range(1, size - 1)]
    if not interior:
        raise ValueError("remaining adjacent mine count requires an interior clue cell")
    for _attempt in range(256):
        clue_cell = tuple(rng.choice(interior))
        neighbors = list(neighbor_coords(clue_cell, size=int(size)))
        rng.shuffle(neighbors)
        if int(target) == 0:
            max_flags = min(4, len(neighbors) - 1)
            if max_flags < 1:
                continue
            flag_count = int(rng.randint(1, max_flags))
            max_hidden_safe = min(4, len(neighbors) - int(flag_count))
            if max_hidden_safe < 1:
                continue
            hidden_safe_count = int(rng.randint(1, max_hidden_safe))
        else:
            max_flags = min(3, len(neighbors) - int(target) - 1)
            if max_flags < 0:
                continue
            flag_count = int(rng.randint(0, max_flags))
            max_extra_safe = min(3, len(neighbors) - int(flag_count) - int(target))
            if max_extra_safe < 1:
                continue
            hidden_safe_count = int(rng.randint(1, max_extra_safe))
        flagged_coords = set(tuple(coord) for coord in neighbors[:flag_count])
        hidden_mines = set(tuple(coord) for coord in neighbors[flag_count : flag_count + int(target)])
        hidden_safe_start = int(flag_count) + int(target)
        hidden_safe = set(tuple(coord) for coord in neighbors[hidden_safe_start : hidden_safe_start + int(hidden_safe_count)])
        occupied = set(flagged_coords) | set(hidden_mines) | set(hidden_safe) | {clue_cell}
        distractors = set(
            _choose_distractors(
                rng=rng,
                size=int(size),
                clue_cell=clue_cell,
                occupied=occupied,
                scene_variant=str(axes.scene_variant),
            )
        )
        mine_coords, revealed_coords, flagged_coords_tuple, hidden_coords_tuple = _build_state(
            size=int(size),
            mine_coords=set(flagged_coords) | set(hidden_mines),
            flagged_coords=set(flagged_coords),
            hidden_coords=set(hidden_mines) | set(hidden_safe) | set(distractors),
        )
        clue = clue_number(clue_cell, mine_coords=mine_coords, size=int(size))
        flags = adjacent_flag_count(clue_cell, flagged_coords=flagged_coords_tuple, size=int(size))
        if int(clue) <= 0 or int(clue) - int(flags) != int(target):
            continue
        mine_support = forced_mine_supports(
            size=int(size),
            mine_coords=mine_coords,
            revealed_coords=revealed_coords,
            flagged_coords=flagged_coords_tuple,
            hidden_coords=hidden_coords_tuple,
        )
        safe_support = forced_safe_supports(
            size=int(size),
            mine_coords=mine_coords,
            revealed_coords=revealed_coords,
            flagged_coords=flagged_coords_tuple,
            hidden_coords=hidden_coords_tuple,
        )
        return MinesweeperSample(
            size=int(size),
            answer=int(target),
            mine_coords=tuple(mine_coords),
            revealed_coords=tuple(revealed_coords),
            flagged_coords=tuple(flagged_coords_tuple),
            hidden_coords=tuple(hidden_coords_tuple),
            forced_mine_coords=sorted_coords(mine_support.keys()),
            forced_safe_coords=sorted_coords(safe_support.keys()),
            forcing_clue_coords=(clue_cell,),
            annotation_coords=(clue_cell,),
            target_answer=int(target),
            distractor_hidden_count=len(distractors),
            construction_mode="marked_number_remaining_adjacent_mines",
        )
    raise ValueError("failed to construct Minesweeper remaining-adjacent-mine board")


def sample_forced_mine_option_scene(
    *,
    rng,
    axes: MinesweeperAxes,
    target_label_index: int,
    option_labels: Sequence[str],
) -> MinesweeperSample:
    """Construct four labeled hidden cells with exactly one forced mine option."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) < 2:
        raise ValueError("Minesweeper option-label construction requires at least two labels")
    target_index = int(target_label_index)
    if target_index < 0 or target_index >= len(labels):
        raise ValueError(f"unsupported Minesweeper forced-mine option index: {target_index}")
    base = sample_forced_cell_scene(
        rng=rng,
        axes=axes,
        force_kind="mine",
        target_count=1,
    )
    if len(base.forced_mine_coords) != 1:
        raise ValueError("forced-mine option scene requires exactly one forced mine")
    correct_coord = tuple(base.forced_mine_coords[0])
    forced_mines = {tuple(coord) for coord in base.forced_mine_coords}
    distractor_pool = [tuple(coord) for coord in base.hidden_coords if tuple(coord) not in forced_mines]
    rng.shuffle(distractor_pool)
    if len(distractor_pool) < len(labels) - 1:
        raise ValueError("not enough non-forced hidden cells for Minesweeper option labels")

    option_pairs: list[tuple[str, Coord]] = []
    distractor_iter = iter(distractor_pool)
    for label_index, label in enumerate(labels):
        coord = correct_coord if int(label_index) == target_index else tuple(next(distractor_iter))
        option_pairs.append((str(label), coord))
    return replace(
        base,
        answer=str(labels[target_index]),
        annotation_coords=(correct_coord,),
        target_answer=int(target_index),
        candidate_option_coords=tuple(option_pairs),
        construction_mode="local_forced_mine_option_label",
    )


__all__ = [
    "MinesweeperAxes",
    "resolve_minesweeper_axes",
    "sample_forced_cell_scene",
    "sample_forced_mine_option_scene",
    "sample_remaining_adjacent_mine_scene",
]
