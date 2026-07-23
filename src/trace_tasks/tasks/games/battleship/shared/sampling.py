"""Identity-free sampling helpers for the Battleship games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.style import SUPPORTED_BATTLESHIP_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import FALLBACK_GENERATION_DEFAULTS
from .rules import (
    candidate_completes_target_shape,
    shape_orientations,
    validate_battleship_scene,
)
from .state import (
    FLEET_SHAPES,
    LAST_CELL_OPTION_LABELS,
    SCENE_ID,
    SHAPE_OPTION_LABELS,
    SUPPORTED_BATTLESHIP_SCENE_VARIANTS,
    BattleshipCandidateOption,
    BattleshipSample,
    BattleshipShapeOption,
    BattleshipShipPlacement,
    Coord,
    all_coords,
    sorted_coords,
)


_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class ResolvedBattleshipSceneAxes:
    """Resolved query-neutral semantic and visual axes for one Battleship scene."""

    scene_variant: str
    style_variant: str
    board_size: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    board_size_probabilities: Dict[str, float]


def resolve_battleship_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Tuple[str, ...],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named Battleship axis without public task identity."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(item) for item in supported],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), dict(probabilities)


def resolve_battleship_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
) -> ResolvedBattleshipSceneAxes:
    """Resolve scene/style/board axes that are shared by Battleship objectives."""

    defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    fallback = FALLBACK_GENERATION_DEFAULTS
    scene_variant, scene_variant_probabilities = resolve_battleship_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        namespace="games.battleship.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BATTLESHIP_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_battleship_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        namespace="games.battleship.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BATTLESHIP_STYLE_VARIANTS,
    )
    board_size, board_size_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        support_key="board_size_support",
        explicit_key="board_size",
        fallback_support=fallback["board_size_support"],
        namespace="games.battleship.board_size",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    return ResolvedBattleshipSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_size=int(board_size),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        board_size_probabilities=dict(board_size_probabilities),
    )


def resolve_battleship_target_ship_id(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    supported_shape_ids: Tuple[str, ...],
) -> Tuple[str, Dict[str, float]]:
    """Resolve a target fleet shape id for an objective-owned task."""

    return resolve_battleship_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key="target_ship_id",
        weights_key="target_ship_id_weights",
        balance_flag_key="balanced_target_ship_id_sampling",
        supported=tuple(str(item) for item in supported_shape_ids),
    )


def resolve_battleship_option_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    balanced_flag_key: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve labeled candidate-cell option count for visual option tasks."""

    option_count, probabilities = resolve_integer_choice(
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
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    if int(option_count) < 2 or int(option_count) > len(LAST_CELL_OPTION_LABELS):
        raise ValueError("Battleship option count must be between 2 and the label pool size")
    return int(option_count), tuple(int(value) for value in support), dict(probabilities)


def inflated_cells(coords: Iterable[Coord], *, board_size: int) -> set[Coord]:
    """Return a one-cell Chebyshev margin around a ship."""

    inflated: set[Coord] = set()
    for row, col in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(row) + int(dr)
                nc = int(col) + int(dc)
                if 0 <= nr < int(board_size) and 0 <= nc < int(board_size):
                    inflated.add((nr, nc))
    return inflated


def place_offsets(
    *,
    rng: Any,
    board_size: int,
    offsets: Sequence[Coord],
    forbidden: set[Coord],
) -> Tuple[Coord, ...]:
    """Place one ship on the board without touching existing ships."""

    orientations = list(shape_orientations(offsets))
    rng.shuffle(orientations)
    for oriented in orientations:
        max_row = max(row for row, _col in oriented)
        max_col = max(col for _row, col in oriented)
        starts = [
            (row, col)
            for row in range(0, int(board_size) - int(max_row))
            for col in range(0, int(board_size) - int(max_col))
        ]
        rng.shuffle(starts)
        for start_row, start_col in starts:
            candidate = tuple((int(start_row) + int(row), int(start_col) + int(col)) for row, col in oriented)
            if not (set(candidate) & forbidden):
                return sorted_coords(candidate)
    raise ValueError("failed to place Battleship ship")


def place_fleet(*, rng: Any, board_size: int) -> Tuple[BattleshipShipPlacement, ...]:
    """Place the full Battleship fleet with no hits assigned yet."""

    forbidden: set[Coord] = set()
    placements: list[BattleshipShipPlacement] = []
    for shape in FLEET_SHAPES:
        coords = place_offsets(
            rng=rng,
            board_size=int(board_size),
            offsets=shape.offsets,
            forbidden=forbidden,
        )
        forbidden.update(inflated_cells(coords, board_size=int(board_size)))
        placements.append(
            BattleshipShipPlacement(
                ship_id=str(shape.shape_id),
                shape_id=str(shape.shape_id),
                display_name=str(shape.display_name),
                coords=tuple(coords),
                hit_coords=tuple(),
                is_sunk=False,
            )
        )
    return tuple(placements)


def build_ship_placements_with_hits(
    placements: Sequence[BattleshipShipPlacement],
    *,
    hit_coords_by_ship_id: Mapping[str, Sequence[Coord]],
) -> Tuple[BattleshipShipPlacement, ...]:
    """Return placements with per-ship hit coordinates applied."""

    updated: list[BattleshipShipPlacement] = []
    for ship in placements:
        hit_coords = sorted_coords(hit_coords_by_ship_id.get(str(ship.ship_id), tuple()))
        if not set(hit_coords) <= set(ship.coords):
            raise ValueError("Battleship hit coordinates must belong to their ship")
        updated.append(
            BattleshipShipPlacement(
                ship_id=str(ship.ship_id),
                shape_id=str(ship.shape_id),
                display_name=str(ship.display_name),
                coords=tuple(ship.coords),
                hit_coords=hit_coords,
                is_sunk=bool(set(hit_coords) == set(ship.coords)),
            )
        )
    return tuple(updated)


def miss_count_bounds(params: Mapping[str, Any], *, gen_defaults: Mapping[str, Any] | None = None) -> Tuple[int, int]:
    """Resolve miss-marker count bounds."""

    defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    fallback = FALLBACK_GENERATION_DEFAULTS
    low = int(params.get("min_miss_count", group_default(defaults, "min_miss_count", fallback["min_miss_count"])))
    high = int(params.get("max_miss_count", group_default(defaults, "max_miss_count", fallback["max_miss_count"])))
    if low > high:
        raise ValueError("min_miss_count must be <= max_miss_count")
    return max(0, int(low)), max(0, int(high))


def sample_miss_coords(
    *,
    rng: Any,
    board_size: int,
    occupied_ship_coords: Iterable[Coord],
    excluded_coords: Iterable[Coord],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[Coord, ...]:
    """Sample water-cell miss markers for a completed fleet placement."""

    occupied = {(int(row), int(col)) for row, col in occupied_ship_coords}
    excluded = {(int(row), int(col)) for row, col in excluded_coords}
    available_for_misses = [
        coord
        for coord in all_coords(int(board_size))
        if coord not in occupied and coord not in excluded
    ]
    rng.shuffle(available_for_misses)
    miss_low, miss_high = miss_count_bounds(params, gen_defaults=gen_defaults)
    miss_count = min(len(available_for_misses), int(rng.randint(int(miss_low), int(miss_high))))
    return sorted_coords(available_for_misses[:miss_count])


def fleet_status_counts(placements: Sequence[BattleshipShipPlacement]) -> Tuple[int, int, int]:
    """Return `(sunk, partial, untouched)` ship counts."""

    sunk_count = len([ship for ship in placements if bool(ship.is_sunk)])
    partial_count = len([ship for ship in placements if bool(ship.hit_coords) and not bool(ship.is_sunk)])
    untouched_count = len([ship for ship in placements if not bool(ship.hit_coords)])
    return int(sunk_count), int(partial_count), int(untouched_count)


def build_battleship_scene_state(
    *,
    board_size: int,
    scene_variant: str,
    placements: Sequence[BattleshipShipPlacement],
    miss_coords: Sequence[Coord],
    construction_mode: str,
    candidate_options: Sequence[BattleshipCandidateOption] = tuple(),
    shape_options: Sequence[BattleshipShapeOption] = tuple(),
) -> BattleshipSample:
    """Build and validate a query-neutral Battleship scene state."""

    sunk_count, partial_count, untouched_count = fleet_status_counts(placements)
    sample = BattleshipSample(
        board_size=int(board_size),
        scene_variant=str(scene_variant),
        ship_placements=tuple(placements),
        hit_coords=sorted_coords(coord for ship in placements for coord in ship.hit_coords),
        miss_coords=sorted_coords(miss_coords),
        sunk_ship_count=int(sunk_count),
        partial_ship_count=int(partial_count),
        untouched_ship_count=int(untouched_count),
        construction_mode=str(construction_mode),
        candidate_options=tuple(candidate_options),
        shape_options=tuple(shape_options),
    )
    validate_battleship_scene(sample)
    return sample


def sample_shape_options(
    *,
    rng: Any,
    answer_shape_id: str,
    answer_label_index: int,
    option_count: int = 5,
) -> Tuple[BattleshipShapeOption, ...]:
    """Return labeled fleet-shape options with exactly one answer."""

    labels = tuple(str(label) for label in SHAPE_OPTION_LABELS[: int(option_count)])
    if len(labels) != int(option_count) or int(option_count) != len(SHAPE_OPTION_LABELS):
        raise ValueError("Battleship shape-option tasks currently require five options")
    answer_shape = next((shape for shape in FLEET_SHAPES if str(shape.shape_id) == str(answer_shape_id)), None)
    if answer_shape is None:
        raise ValueError("answer_shape_id must belong to the Battleship fleet")
    answer_index = int(answer_label_index)
    if answer_index < 0 or answer_index >= len(labels):
        raise ValueError("answer_label_index must be inside the visible Battleship shape-option labels")
    distractors = [shape for shape in FLEET_SHAPES if str(shape.shape_id) != str(answer_shape_id)]
    rng.shuffle(distractors)
    option_shapes: list[Any] = []
    distractor_index = 0
    for label_index in range(len(labels)):
        if int(label_index) == int(answer_index):
            option_shapes.append(answer_shape)
        else:
            option_shapes.append(distractors[distractor_index])
            distractor_index += 1
    return tuple(
        BattleshipShapeOption(
            label=str(label),
            shape_id=str(shape.shape_id),
            display_name=str(shape.display_name),
            is_answer=bool(str(shape.shape_id) == str(answer_shape_id)),
        )
        for label, shape in zip(labels, option_shapes)
    )


def sample_last_cell_candidate_options(
    *,
    rng: Any,
    board_size: int,
    target_ship: BattleshipShipPlacement,
    missing_coord: Coord,
    ship_cells: set[Coord],
    hit_coords: Sequence[Coord],
    answer_label_index: int,
    option_count: int,
) -> Tuple[BattleshipCandidateOption, ...]:
    """Return labeled candidate cells with exactly one valid missing-cell answer."""

    correct = (int(missing_coord[0]), int(missing_coord[1]))
    labels = tuple(str(label) for label in LAST_CELL_OPTION_LABELS[: int(option_count)])
    if len(labels) < 2:
        raise ValueError("Battleship last-cell option count must be at least 2")
    hit_set = {(int(row), int(col)) for row, col in hit_coords}
    target_hits = tuple((int(row), int(col)) for row, col in target_ship.hit_coords)
    invalid_pool: list[Coord] = []
    seen: set[Coord] = set()

    def add_candidate(coord: Coord) -> None:
        row, col = int(coord[0]), int(coord[1])
        item = (row, col)
        if item in seen or item == correct or item in hit_set or item in ship_cells:
            return
        if not (0 <= row < int(board_size) and 0 <= col < int(board_size)):
            return
        if candidate_completes_target_shape(
            candidate=item,
            target_hit_coords=target_hits,
            target_shape_id=str(target_ship.shape_id),
        ):
            return
        seen.add(item)
        invalid_pool.append(item)

    target_rows = [int(row) for row, _col in target_hits]
    target_cols = [int(col) for _row, col in target_hits]
    if target_rows and target_cols:
        for row in range(min(target_rows) - 2, max(target_rows) + 3):
            for col in range(min(target_cols) - 2, max(target_cols) + 3):
                add_candidate((row, col))

    for row, col in target_hits:
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if abs(int(dr)) + abs(int(dc)) <= 3:
                    add_candidate((int(row) + int(dr), int(col) + int(dc)))

    all_water = [
        coord
        for coord in all_coords(int(board_size))
        if coord not in ship_cells and coord not in hit_set and coord != correct
    ]
    rng.shuffle(all_water)
    for coord in all_water:
        add_candidate(coord)

    if len(invalid_pool) < len(labels) - 1:
        raise ValueError("failed to sample enough invalid Battleship last-cell candidates")

    rng.shuffle(invalid_pool)
    selected_label_index = int(answer_label_index)
    if selected_label_index < 0 or selected_label_index >= len(labels):
        raise ValueError("answer_label_index must be inside the visible Battleship option labels")
    distractor_coords = list(invalid_pool[: len(labels) - 1])
    option_coords: list[Coord] = []
    distractor_index = 0
    for label_index in range(len(labels)):
        if int(label_index) == int(selected_label_index):
            option_coords.append(correct)
        else:
            option_coords.append(distractor_coords[distractor_index])
            distractor_index += 1
    options = tuple(
        BattleshipCandidateOption(
            label=str(label),
            coord=(int(coord[0]), int(coord[1])),
            is_answer=bool((int(coord[0]), int(coord[1])) == correct),
        )
        for label, coord in zip(labels, option_coords)
    )
    if sum(1 for option in options if bool(option.is_answer)) != 1:
        raise ValueError("failed to place exactly one Battleship answer option")
    return options


__all__ = [
    "ResolvedBattleshipSceneAxes",
    "build_battleship_scene_state",
    "build_ship_placements_with_hits",
    "fleet_status_counts",
    "inflated_cells",
    "miss_count_bounds",
    "place_fleet",
    "place_offsets",
    "resolve_battleship_named_axis",
    "resolve_battleship_option_count",
    "resolve_battleship_scene_axes",
    "resolve_battleship_target_ship_id",
    "sample_last_cell_candidate_options",
    "sample_miss_coords",
    "sample_shape_options",
]
