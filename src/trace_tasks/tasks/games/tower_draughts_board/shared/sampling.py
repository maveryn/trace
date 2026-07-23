"""Sampling primitives for tower draughts board scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    capture_paths,
    capture_targets,
    max_capture_count_for_board,
    max_controlled_count_for_board,
    opponent,
    playable_coords,
    player_from_name,
    player_name,
)
from .state import BLACK, PLAYER_SUPPORT, RED, SCENE_NAMESPACE, STYLE_VARIANTS, TOP_KIND_SUPPORT, Coord, StackSpec, TowerDraughtsAxes, TowerDraughtsSample


MaxCountResolver = Callable[[int], int]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def resolve_tower_draughts_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_answer_support_key: str,
    target_answer_fallback: Sequence[int],
    max_count_for_board: MaxCountResolver,
    force_crowned_for_large_target: bool = False,
) -> TowerDraughtsAxes:
    """Resolve shared generation axes for one public tower draughts objective."""

    style_variant, style_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    target_player_name, target_player_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="target_player",
        explicit_key="target_player",
        weights_key="target_player_weights",
        balance_flag_key="balanced_target_player_sampling",
        supported=PLAYER_SUPPORT,
    )
    marked_player_name, marked_player_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="marked_player",
        explicit_key="marked_player",
        weights_key="marked_player_weights",
        balance_flag_key="balanced_marked_player_sampling",
        supported=PLAYER_SUPPORT,
    )
    top_kind, top_kind_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="top_kind",
        explicit_key="top_kind",
        weights_key="top_kind_weights",
        balance_flag_key="balanced_top_kind_sampling",
        supported=TOP_KIND_SUPPORT,
    )
    target_answer, target_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(target_answer_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in target_answer_fallback),
        namespace=f"{SCENE_NAMESPACE}.{target_answer_support_key}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(target_answer_support_key),
        fallback=tuple(int(value) for value in target_answer_fallback),
    )
    board_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="board_size_support",
        fallback=DEFAULTS.board_size_support,
    )
    feasible_board_support = tuple(
        int(size)
        for size in board_support
        if int(max_count_for_board(int(size))) >= int(target_answer)
    )
    if not feasible_board_support:
        raise ValueError(f"target answer {target_answer} is infeasible for tower draughts board")
    board_params = dict(params)
    board_params["board_size_support"] = list(feasible_board_support)
    if "board_size" in params and int(params["board_size"]) not in feasible_board_support:
        raise ValueError(f"explicit board_size={params['board_size']} cannot realize target answer {target_answer}")
    board_size, board_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=board_params,
        gen_defaults=GEN_DEFAULTS,
        support_key="board_size_support",
        explicit_key="board_size",
        fallback_support=feasible_board_support,
        namespace=f"{SCENE_NAMESPACE}.{target_answer_support_key}.board_size",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    if bool(force_crowned_for_large_target) and int(target_answer) > 2:
        top_kind = "crowned"
    return TowerDraughtsAxes(
        target_player=player_from_name(str(target_player_name)),
        marked_player=player_from_name(str(marked_player_name)),
        top_kind=str(top_kind),
        style_variant=str(style_variant),
        board_size=int(board_size),
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        board_size_probabilities=dict(board_probs),
        target_answer_probabilities=dict(target_probs),
        target_player_probabilities=dict(target_player_probs),
        marked_player_probabilities=dict(marked_player_probs),
        top_kind_probabilities=dict(top_kind_probs),
        style_variant_probabilities=dict(style_probs),
    )


def axis_support_metadata(axes: TowerDraughtsAxes) -> dict[str, Any]:
    """Return common axis metadata for trace query specs."""

    return {
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size": int(axes.board_size),
        "board_size_probabilities": dict(axes.board_size_probabilities),
        "target_answer": int(axes.target_answer),
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
        "target_player": player_name(int(axes.target_player)),
        "target_player_probabilities": dict(axes.target_player_probabilities),
        "marked_player": player_name(int(axes.marked_player)),
        "marked_player_probabilities": dict(axes.marked_player_probabilities),
        "top_kind": str(axes.top_kind),
        "top_kind_probabilities": dict(axes.top_kind_probabilities),
    }


def random_stack_height(rng: Any) -> int:
    """Sample one stack height from scene defaults."""

    support = resolve_integer_support(
        {},
        gen_defaults=GEN_DEFAULTS,
        key="stack_height_support",
        fallback=DEFAULTS.stack_height_support,
    )
    return int(rng.choice(list(support)))


def make_stack(*, rng: Any, coord: Coord, owner: int, crowned: bool = False) -> StackSpec:
    """Build one stack with random lower disks and a controlled top disk."""

    height = random_stack_height(rng)
    disks = [int(owner)]
    for _ in range(max(0, int(height) - 1)):
        disks.insert(0, int(rng.choice([RED, BLACK])))
    disks[-1] = int(owner)
    return StackSpec(coord=tuple(coord), disks=tuple(int(value) for value in disks), top_crowned=bool(crowned))


def desired_occupied_count(*, rng: Any, board_size: int, minimum: int) -> int:
    """Sample the target number of occupied playable cells for visual density."""

    playable_count = len(playable_coords(int(board_size)))
    min_fraction = float(group_default(GEN_DEFAULTS, "min_occupied_fraction", DEFAULTS.min_occupied_fraction))
    max_fraction = float(group_default(GEN_DEFAULTS, "max_occupied_fraction", DEFAULTS.max_occupied_fraction))
    lo = max(int(minimum), int(round(float(playable_count) * min_fraction)))
    hi = max(lo, int(round(float(playable_count) * max_fraction)))
    return min(int(playable_count), int(rng.randint(lo, hi)))


def fill_extra_stacks(
    *,
    rng: Any,
    board_size: int,
    stacks: List[StackSpec],
    protected_empty: set[Coord],
    desired_count: int,
) -> None:
    """Fill additional cells while preserving required empty witness cells."""

    occupied = {tuple(stack.coord) for stack in stacks}
    candidates = [
        coord
        for coord in playable_coords(int(board_size))
        if tuple(coord) not in occupied and tuple(coord) not in protected_empty
    ]
    rng.shuffle(candidates)
    crown_prob = float(group_default(GEN_DEFAULTS, "crowned_top_probability", DEFAULTS.crowned_top_probability))
    for coord in candidates:
        if len(stacks) >= int(desired_count):
            break
        owner = int(rng.choice([RED, BLACK]))
        stacks.append(make_stack(rng=rng, coord=tuple(coord), owner=owner, crowned=bool(rng.random() < crown_prob)))


def sample_controlled_stack_count_scene(*, rng: Any, axes: TowerDraughtsAxes) -> TowerDraughtsSample:
    """Construct a board with exactly the target number of controlled stacks."""

    target = int(axes.target_answer)
    board_size = int(axes.board_size)
    playable = list(playable_coords(board_size))
    if target > len(playable):
        raise ValueError("controlled-stack target exceeds playable cells")
    rng.shuffle(playable)
    target_coords = playable[:target]
    remaining = playable[target:]
    desired_count = desired_occupied_count(rng=rng, board_size=board_size, minimum=max(1, target))
    desired_count = max(desired_count, target)
    stacks: list[StackSpec] = [
        make_stack(rng=rng, coord=coord, owner=int(axes.target_player), crowned=bool(rng.random() < 0.2))
        for coord in target_coords
    ]
    enemy = opponent(int(axes.target_player))
    for coord in remaining:
        if len(stacks) >= desired_count:
            break
        stacks.append(make_stack(rng=rng, coord=coord, owner=enemy, crowned=bool(rng.random() < 0.2)))
    annotation = tuple(sorted(coord for coord in target_coords))
    return TowerDraughtsSample(
        board_size=int(board_size),
        style_variant=str(axes.style_variant),
        stacks=tuple(sorted(stacks, key=lambda stack: stack.coord)),
        marked_coord=None,
        target_player=int(axes.target_player),
        marked_player=int(axes.marked_player),
        top_kind=str(axes.top_kind),
        annotation_coords=tuple(annotation),
        answer=int(len(annotation)),
        construction_mode="target_conditioned_top_owner_count",
        metadata={},
    )


def viable_marked_coords(
    *,
    board_size: int,
    owner: int,
    crowned: bool,
    min_captures: int = 0,
) -> Tuple[Coord, ...]:
    """Return marked-stack coordinates with enough geometric move/capture slots."""

    out: list[Coord] = []
    for coord in playable_coords(int(board_size)):
        captures = capture_paths(coord=coord, owner=int(owner), crowned=bool(crowned), board_size=int(board_size))
        if len(captures) >= int(min_captures):
            out.append(coord)
    return tuple(out)


def sample_marked_stack_capture_count_scene(*, rng: Any, axes: TowerDraughtsAxes) -> TowerDraughtsSample:
    """Construct a board with exactly the target number of immediate captures."""

    target = int(axes.target_answer)
    board_size = int(axes.board_size)
    crowned = str(axes.top_kind) == "crowned"
    viable = list(
        viable_marked_coords(
            board_size=board_size,
            owner=int(axes.marked_player),
            crowned=bool(crowned),
            min_captures=target,
        )
    )
    if not viable:
        raise ValueError("no marked stack can realize capture target")
    rng.shuffle(viable)
    marked_coord = tuple(viable[0])
    paths = list(
        capture_paths(
            coord=marked_coord,
            owner=int(axes.marked_player),
            crowned=bool(crowned),
            board_size=board_size,
        )
    )
    rng.shuffle(paths)
    selected = paths[:target]
    selected_captured = {tuple(captured) for captured, _landing in selected}
    selected_landings = {tuple(landing) for _captured, landing in selected}
    stacks: list[StackSpec] = [
        make_stack(rng=rng, coord=marked_coord, owner=int(axes.marked_player), crowned=bool(crowned))
    ]
    for captured, _landing in selected:
        stacks.append(make_stack(rng=rng, coord=tuple(captured), owner=opponent(int(axes.marked_player)), crowned=bool(rng.random() < 0.2)))
    for captured, landing in paths[target:]:
        if tuple(captured) in selected_captured or tuple(landing) in selected_landings:
            continue
        mode = str(rng.choice(["own_piece", "blocked_landing", "empty_middle"]))
        if mode == "own_piece":
            stacks.append(make_stack(rng=rng, coord=tuple(captured), owner=int(axes.marked_player), crowned=bool(rng.random() < 0.2)))
        elif mode == "blocked_landing":
            stacks.append(make_stack(rng=rng, coord=tuple(captured), owner=opponent(int(axes.marked_player)), crowned=bool(rng.random() < 0.2)))
            stacks.append(make_stack(rng=rng, coord=tuple(landing), owner=int(rng.choice([RED, BLACK])), crowned=bool(rng.random() < 0.2)))
    unique: dict[Coord, StackSpec] = {}
    for stack in stacks:
        unique[tuple(stack.coord)] = stack
    stacks = list(unique.values())
    desired_count = desired_occupied_count(rng=rng, board_size=board_size, minimum=len(stacks))
    fill_extra_stacks(
        rng=rng,
        board_size=board_size,
        stacks=stacks,
        protected_empty={tuple(coord) for coord in selected_landings},
        desired_count=desired_count,
    )
    actual = capture_targets(stacks=tuple(stacks), marked_coord=marked_coord, board_size=board_size)
    if set(actual) != selected_captured:
        raise ValueError("constructed capture count mismatch")
    return TowerDraughtsSample(
        board_size=int(board_size),
        style_variant=str(axes.style_variant),
        stacks=tuple(sorted(stacks, key=lambda stack: stack.coord)),
        marked_coord=marked_coord,
        target_player=int(axes.target_player),
        marked_player=int(axes.marked_player),
        top_kind=str(axes.top_kind),
        annotation_coords=tuple(sorted(actual)),
        answer=int(len(actual)),
        construction_mode="target_conditioned_marked_stack_captures",
        metadata={"captured_stacks": [[int(coord[0]), int(coord[1])] for coord in sorted(actual)]},
    )


__all__ = [
    "axis_support_metadata",
    "desired_occupied_count",
    "fill_extra_stacks",
    "make_stack",
    "max_capture_count_for_board",
    "max_controlled_count_for_board",
    "random_stack_height",
    "resolve_tower_draughts_axes",
    "sample_controlled_stack_count_scene",
    "sample_marked_stack_capture_count_scene",
    "viable_marked_coords",
]
