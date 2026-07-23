"""Identity-free sampling primitives for Snake board construction."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    all_coords,
    candidate_move_sequences,
    coord_to_cell_id,
    neighbor_coords,
    simulate_snake_moves,
    step_coord,
)
from .state import (
    Coord,
    PLANNED_MOVE_OUTCOMES,
    SCENE_VARIANTS,
    STYLE_VARIANTS,
    SnakeSceneAxes,
    SnakeSimulation,
    SnakeState,
)


def resolve_scene_axes(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> SnakeSceneAxes:
    """Resolve scene/render axes used by all Snake objectives."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=STYLE_VARIANTS,
    )
    board_size, board_size_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="board_size_support",
        explicit_key="board_size",
        fallback_support=DEFAULTS.board_size_support,
        namespace=f"{namespace}.board_size",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    body_length, body_length_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="body_length_support",
        explicit_key="body_length",
        fallback_support=DEFAULTS.body_length_support,
        namespace=f"{namespace}.body_length",
        balanced_flag_key="balanced_body_length_sampling",
        namespace_support_permutation=True,
    )
    planned_move_count, planned_move_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="planned_move_count_support",
        explicit_key="planned_move_count",
        fallback_support=DEFAULTS.planned_move_count_support,
        namespace=f"{namespace}.planned_move_count",
        balanced_flag_key="balanced_planned_move_count_sampling",
        namespace_support_permutation=True,
    )
    obstacle_count, obstacle_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="obstacle_count_support",
        explicit_key="obstacle_count",
        fallback_support=DEFAULTS.obstacle_count_support,
        namespace=f"{namespace}.obstacle_count",
        balanced_flag_key="balanced_obstacle_count_sampling",
        namespace_support_permutation=True,
    )
    return SnakeSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_size=int(board_size),
        body_length=int(body_length),
        planned_move_count=int(planned_move_count),
        obstacle_count=int(obstacle_count),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        board_size_probabilities=dict(board_size_probabilities),
        body_length_probabilities=dict(body_length_probabilities),
        planned_move_count_probabilities=dict(planned_move_count_probabilities),
        obstacle_count_probabilities=dict(obstacle_count_probabilities),
    )


def integer_support(params: Mapping[str, Any], *, key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve one integer support from params/defaults."""

    return resolve_integer_support(params, gen_defaults=GEN_DEFAULTS, key=str(key), fallback=tuple(int(v) for v in fallback))


def select_integer_target(
    params: Mapping[str, Any],
    *,
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    instance_seed: int,
    namespace: str,
    balance_flag_key: str,
) -> tuple[int, dict[str, float]]:
    """Resolve a target integer value for one public objective."""

    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balance_flag_key),
        namespace_support_permutation=True,
    )


def string_support(params: Mapping[str, Any], *, key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    """Resolve one string-valued support from params/defaults."""

    raw = params.get(str(key), group_default(GEN_DEFAULTS, str(key), tuple(fallback)))
    values = (str(raw),) if isinstance(raw, str) else tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one value")
    return values


def select_planned_outcome_target(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve point-vs-game-over outcomes with roughly 25% game-over targets."""

    support = string_support(
        params,
        key="planned_move_outcome_support",
        fallback=PLANNED_MOVE_OUTCOMES,
    )
    explicit = params.get("target_planned_outcome", params.get("target_path_result"))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"target_planned_outcome={value!r} is not in planned_move_outcome_support")
        return value, {str(item): (1.0 if str(item) == value else 0.0) for item in support}

    if "point" in support and "game_over" in support:
        probabilities = {"point": 0.75, "game_over": 0.25}
        sampling_index = params.get("_sample_cursor")
        balanced = bool(
            params.get(
                "balanced_planned_move_outcome_sampling",
                group_default(GEN_DEFAULTS, "balanced_planned_move_outcome_sampling", True),
            )
        )
        if balanced and sampling_index is not None:
            return ("game_over" if abs(int(sampling_index)) % 4 == 0 else "point"), probabilities
        rng = spawn_rng(int(instance_seed), f"{namespace}.path_result")
        return ("game_over" if float(rng.random()) < 0.25 else "point"), probabilities

    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), f"{namespace}.path_result")
    return str(rng.choice(tuple(support))), probabilities


def random_snake_state(
    *,
    rng: Any,
    board_size: int,
    body_length: int,
    prefer_edge_head: bool = False,
) -> SnakeState:
    """Construct a random connected visible snake with food on an open cell."""

    size = int(board_size)
    length = max(2, int(body_length) + 1)
    for _attempt in range(260):
        if bool(prefer_edge_head):
            edge = int(rng.randrange(4))
            if edge == 0:
                head = (0, int(rng.randrange(size)))
            elif edge == 1:
                head = (size - 1, int(rng.randrange(size)))
            elif edge == 2:
                head = (int(rng.randrange(size)), 0)
            else:
                head = (int(rng.randrange(size)), size - 1)
        else:
            head = (int(rng.randrange(size)), int(rng.randrange(size)))
        path = [head]
        while len(path) < length:
            candidates = [coord for coord in neighbor_coords(path[-1], size=size) if coord not in set(path)]
            if not candidates:
                break
            rng.shuffle(candidates)
            path.append(candidates[0])
        if len(path) != length:
            continue
        open_cells = [coord for coord in all_coords(size) if coord not in set(path)]
        if not open_cells:
            continue
        food = rng.choice(open_cells)
        return SnakeState(board_size=size, head=head, body=tuple(path[1:]), food=food, obstacles=tuple())
    raise ValueError("failed to construct random Snake state")


def with_food(state: SnakeState, food: Coord) -> SnakeState:
    """Return `state` with a different food coordinate."""

    return SnakeState(
        board_size=int(state.board_size),
        head=state.head,
        body=tuple(state.body),
        food=(int(food[0]), int(food[1])),
        obstacles=tuple(state.obstacles),
    )


def with_obstacles(state: SnakeState, obstacles: Sequence[Coord]) -> SnakeState:
    """Return `state` with visible blocked wall cells."""

    return SnakeState(
        board_size=int(state.board_size),
        head=state.head,
        body=tuple(state.body),
        food=state.food,
        obstacles=tuple((int(row), int(col)) for row, col in obstacles),
    )


def open_cells(state: SnakeState, *, exclude: Iterable[Coord] = ()) -> Tuple[Coord, ...]:
    """Return open board cells not occupied by snake and optional exclusions."""

    excluded = set((int(row), int(col)) for row, col in exclude)
    occupied = {state.head} | set(state.body) | set(state.obstacles) | excluded
    return tuple(coord for coord in all_coords(state.board_size) if coord not in occupied)


def sample_obstacles(
    *,
    rng: Any,
    state: SnakeState,
    count: int,
    exclude: Iterable[Coord] = (),
) -> Tuple[Coord, ...]:
    """Sample blocked wall cells away from the snake, food, and optional cells."""

    excluded = set((int(row), int(col)) for row, col in exclude)
    excluded.add((int(state.food[0]), int(state.food[1])))
    candidates = list(open_cells(state, exclude=excluded))
    if len(candidates) < int(count):
        raise ValueError("not enough open cells for Snake wall obstacles")
    rng.shuffle(candidates)
    return tuple((int(row), int(col)) for row, col in candidates[: int(count)])


def dummy_food_state(state: SnakeState) -> SnakeState:
    """Place food on a deterministic open cell away from sampled movement paths."""

    candidates = open_cells(state)
    if not candidates:
        raise ValueError("no open cell for dummy food")
    return with_food(state, candidates[-1])


def find_sequence_for_path_result(
    *,
    rng: Any,
    state: SnakeState,
    length: int,
    target_result: str,
) -> Tuple[SnakeState, Tuple[str, ...], SnakeSimulation] | None:
    """Search move sequences for either a final point or a game-over event."""

    search_state = dummy_food_state(state)
    sequences = list(candidate_move_sequences(int(length)))
    rng.shuffle(sequences)
    for sequence in sequences:
        simulation = simulate_snake_moves(search_state, sequence)
        if simulation.outcome == "food":
            continue
        is_game_over = str(simulation.outcome) in {"body", "wall"}
        is_point = (not is_game_over) and len(simulation.traversed_coords) == int(length)
        if str(target_result) == "game_over" and not is_game_over:
            continue
        if str(target_result) == "point" and not is_point:
            continue

        traversed = set(simulation.traversed_coords)
        open_food = [coord for coord in open_cells(state) if coord not in traversed]
        if not open_food:
            continue
        final_state = with_food(state, rng.choice(open_food))
        final_simulation = simulate_snake_moves(final_state, sequence)
        if final_simulation.outcome == "food":
            continue
        final_game_over = str(final_simulation.outcome) in {"body", "wall"}
        final_point = (not final_game_over) and len(final_simulation.traversed_coords) == int(length)
        if str(target_result) == "game_over" and final_game_over:
            return final_state, tuple(sequence), final_simulation
        if str(target_result) == "point" and final_point:
            return final_state, tuple(sequence), final_simulation
    return None


def point_option_coords(
    *,
    rng: Any,
    state: SnakeState,
    count: int,
    exclude: Iterable[Coord] = (),
) -> Tuple[Coord, ...]:
    """Sample visible in-board cells for path-result point options."""

    excluded = set((int(row), int(col)) for row, col in exclude)
    candidates = [coord for coord in open_cells(state) if coord not in excluded]
    if len(candidates) < int(count):
        raise ValueError("not enough open cells for Snake point options")
    rng.shuffle(candidates)
    return tuple(candidates[: int(count)])


def planned_annotation_ids(simulation: SnakeSimulation, *, fallback_state: SnakeState) -> Tuple[str, ...]:
    """Return public annotation cell ids for one planned move sequence."""

    coords = tuple(dict.fromkeys(simulation.traversed_coords))
    if coords:
        return tuple(coord_to_cell_id(coord) for coord in coords)
    return (coord_to_cell_id(fallback_state.head),)


__all__ = [
    "dummy_food_state",
    "find_sequence_for_path_result",
    "integer_support",
    "open_cells",
    "planned_annotation_ids",
    "point_option_coords",
    "random_snake_state",
    "resolve_scene_axes",
    "sample_obstacles",
    "select_integer_target",
    "select_planned_outcome_target",
    "string_support",
    "with_food",
    "with_obstacles",
]
