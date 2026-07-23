"""Identity-free sampling helpers for lane-crossing games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    FALLBACK_GENERATION_DEFAULTS,
    SCENE_ID,
    START_LABELS,
    SUPPORTED_CROSSING_SCENE_VARIANTS,
    SUPPORTED_CROSSING_STYLE_VARIANTS,
    VEHICLE_OPTION_LABELS,
)
from .rules import (
    route_collision_vehicle_ids,
    route_first_collision_tick,
    validate_crossing_sample,
    vehicle_col_at_tick,
    vehicle_exit_tick,
)
from .state import CrossingRouteOption, CrossingSample, CrossingSceneAxes, CrossingVehicle, route_entity_id, vehicle_entity_id


_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def _generation_default(key: str) -> Any:
    return group_default(_GEN_DEFAULTS, str(key), FALLBACK_GENERATION_DEFAULTS[str(key)])


def _full_probability_map(supported: Sequence[str], probabilities: Mapping[str, float]) -> dict[str, float]:
    return {str(key): float(probabilities.get(str(key), 0.0)) for key in supported}


def resolve_scene_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
    gen_defaults: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, float]]:
    """Resolve one named scene axis."""

    active_defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=active_defaults,
        supported_variants=tuple(str(value) for value in supported_values),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported_values),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), _full_probability_map(supported_values, probabilities)


def resolve_crossing_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
    min_lane_count: int = 0,
    min_row_count: int = 0,
    namespace_suffix: str = "",
) -> CrossingSceneAxes:
    """Resolve scene-level axes for one crossing task."""

    active_defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    suffix = f".{namespace_suffix}" if str(namespace_suffix) else ""
    scene_variant, scene_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        namespace=f"games.crossing.scene_variant{suffix}",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_CROSSING_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        namespace=f"games.crossing.style_variant{suffix}",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_CROSSING_STYLE_VARIANTS,
    )
    lane_count, lane_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        support_key="lane_count_support",
        explicit_key="lane_count",
        fallback_support=tuple(int(value) for value in _generation_default("lane_count_support")),
        namespace=f"games.crossing.lane_count{suffix}",
        balanced_flag_key="balanced_lane_count_sampling",
        namespace_support_permutation=True,
    )
    row_count, row_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        support_key="row_count_support",
        explicit_key="row_count",
        fallback_support=tuple(int(value) for value in _generation_default("row_count_support")),
        namespace=f"games.crossing.row_count{suffix}",
        balanced_flag_key="balanced_row_count_sampling",
        namespace_support_permutation=True,
    )
    return CrossingSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        lane_count=max(int(lane_count), int(min_lane_count)),
        row_count=max(int(row_count), int(min_row_count)),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        lane_count_probabilities=dict(lane_count_probabilities),
        row_count_probabilities=dict(row_count_probabilities),
    )


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve task-owned integer answer support."""

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
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return int(target_answer), tuple(int(value) for value in support), dict(probabilities)


def _row_directions(rng: Any, row_count: int) -> list[int]:
    first = -1 if int(rng.randrange(2)) == 0 else 1
    return [int(first if row % 2 == 0 else -first) for row in range(int(row_count))]


def _collision_start_for_col(rng: Any, *, lane_count: int, col: int, tick: int) -> tuple[int, int] | None:
    directions = [-1, 1]
    rng.shuffle(directions)
    for direction in directions:
        start_col = int(col) - (int(direction) * int(tick))
        if 0 <= int(start_col) < int(lane_count):
            return int(direction), int(start_col)
    return None


def _collision_reachable_cols(*, lane_count: int, tick: int) -> tuple[int, ...]:
    return tuple(
        int(col)
        for col in range(int(lane_count))
        if any(0 <= int(col - (direction * int(tick))) < int(lane_count) for direction in (-1, 1))
    )


def _random_route_path(
    rng: Any,
    *,
    lane_count: int,
    row_count: int,
    required_cols_by_row: Mapping[int, Sequence[int]] | None = None,
) -> tuple[int, ...] | None:
    """Sample a valid upward route while satisfying per-row column constraints."""

    constraints: dict[int, set[int]] = {}
    if isinstance(required_cols_by_row, Mapping):
        for raw_row, raw_cols in required_cols_by_row.items():
            row = int(raw_row)
            if row < 0 or row >= int(row_count):
                return None
            allowed = {int(col) for col in raw_cols if 0 <= int(col) < int(lane_count)}
            if not allowed:
                return None
            constraints[int(row)] = set(allowed)

    suffix_ok: list[set[int]] = [set() for _ in range(int(row_count))]
    for row in reversed(range(int(row_count))):
        allowed_cols = constraints.get(int(row), set(range(int(lane_count))))
        if int(row) == int(row_count) - 1:
            suffix_ok[int(row)] = set(allowed_cols)
            continue
        next_ok = suffix_ok[int(row) + 1]
        suffix_ok[int(row)] = {
            int(col)
            for col in allowed_cols
            if any(int(next_col) in next_ok for next_col in (int(col) - 1, int(col), int(col) + 1))
        }
    if not suffix_ok or not suffix_ok[0]:
        return None

    candidates = sorted(suffix_ok[0])
    col = int(candidates[int(rng.randrange(len(candidates)))])
    path = [int(col)]
    for row in range(1, int(row_count)):
        candidates = [
            int(next_col)
            for next_col in (int(col) - 1, int(col), int(col) + 1)
            if int(next_col) in suffix_ok[int(row)]
        ]
        if not candidates:
            return None
        rng.shuffle(candidates)
        col = int(candidates[0])
        path.append(int(col))
    return tuple(path)


def _add_vehicle(
    vehicles: list[CrossingVehicle],
    *,
    row: int,
    start_col: int,
    direction: int,
    color_index: int,
    option_label: str | None = None,
) -> CrossingVehicle:
    vehicle = CrossingVehicle(
        vehicle_id=vehicle_entity_id(len(vehicles)),
        row=int(row),
        start_col=int(start_col),
        direction=int(direction),
        color_index=int(color_index),
        option_label=None if option_label is None else str(option_label),
    )
    vehicles.append(vehicle)
    return vehicle


def _add_clutter(
    rng: Any,
    *,
    vehicles: list[CrossingVehicle],
    lane_count: int,
    row_count: int,
    row_directions: Sequence[int],
    avoid_cols_by_row: Mapping[int, set[int]],
    max_extra_per_row: int,
    avoid_start_cols_by_row: Mapping[int, set[int]] | None = None,
) -> None:
    """Add unlabeled moving objects while preserving route-collision invariants."""

    occupied = {(int(vehicle.row), int(vehicle.start_col)) for vehicle in vehicles}
    for row in range(int(row_count)):
        extra_count = int(rng.randint(0, int(max_extra_per_row)))
        direction = int(row_directions[int(row)])
        tick = int(row + 1)
        avoid_cols = set(int(value) for value in avoid_cols_by_row.get(int(row), set()))
        avoid_start_cols = set(int(value) for value in (avoid_start_cols_by_row or {}).get(int(row), set()))
        candidates = [
            col
            for col in range(int(lane_count))
            if (int(row), int(col)) not in occupied
            and int(col) not in avoid_start_cols
            and (
                vehicle_col_at_tick(
                    CrossingVehicle("_tmp", int(row), int(col), int(direction), 0),
                    tick=tick,
                    lane_count=int(lane_count),
                )
                not in avoid_cols
            )
        ]
        rng.shuffle(candidates)
        for col in candidates[:extra_count]:
            _add_vehicle(
                vehicles,
                row=int(row),
                start_col=int(col),
                direction=int(direction),
                color_index=int(rng.randrange(5)),
            )
            occupied.add((int(row), int(col)))


def _sample_route_intersection_count(
    rng: Any,
    *,
    axes: CrossingSceneAxes,
    target_answer: int,
    max_extra_per_row: int,
) -> CrossingSample | None:
    """Construct a marked route with exactly the requested collision count."""

    lane_count = int(axes.lane_count)
    target_count = int(target_answer)
    row_count = max(int(axes.row_count), int(target_count))
    rows = list(range(row_count))
    rng.shuffle(rows)
    target_rows = sorted(rows[:target_count])
    route_path = _random_route_path(
        rng,
        lane_count=lane_count,
        row_count=row_count,
        required_cols_by_row={
            int(row): _collision_reachable_cols(lane_count=lane_count, tick=int(row + 1))
            for row in target_rows
        },
    )
    if route_path is None:
        return None
    row_directions = _row_directions(rng, row_count)
    vehicles: list[CrossingVehicle] = []
    for row in target_rows:
        hit = _collision_start_for_col(rng, lane_count=lane_count, col=int(route_path[row]), tick=int(row + 1))
        if hit is None:
            return None
        direction, start_col = hit
        row_directions[int(row)] = int(direction)
        _add_vehicle(vehicles, row=int(row), start_col=int(start_col), direction=int(direction), color_index=int(rng.randrange(5)))
    _add_clutter(
        rng,
        vehicles=vehicles,
        lane_count=lane_count,
        row_count=row_count,
        row_directions=row_directions,
        avoid_cols_by_row={row: {int(route_path[row])} for row in range(row_count)},
        max_extra_per_row=int(max_extra_per_row),
    )
    route = CrossingRouteOption(route_id=route_entity_id("M"), label="M", path_cols=tuple(route_path), color_index=0)
    hit_ids = route_collision_vehicle_ids(route, tuple(vehicles), lane_count=lane_count)
    if len(hit_ids) != int(target_count):
        return None
    sample = CrossingSample(
        lane_count=lane_count,
        row_count=row_count,
        count_mode="route_intersections",
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        answer=int(target_count),
        row_directions=tuple(int(value) for value in row_directions),
        vehicles=tuple(vehicles),
        start_labels=tuple(START_LABELS[:lane_count]),
        route_options=(route,),
        marked_route_label="M",
        target_start_label=None,
        target_route_label=None,
        target_object_label=None,
        first_collision_tick=route_first_collision_tick(route, tuple(vehicles), lane_count=lane_count),
        intersecting_vehicle_ids=tuple(hit_ids),
        annotation_entity_ids=tuple(hit_ids),
        target_answer=int(target_count),
        target_label_index=None,
        construction_mode="marked_route_intersection_count",
    )
    validate_crossing_sample(sample)
    return sample


def _non_colliding_slots_for_route(
    *,
    lane_count: int,
    row_count: int,
    row_directions: Sequence[int],
    route_path: Sequence[int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Return vehicle start slots that do not collide with a route at the row tick."""

    slots: list[tuple[int, int, int]] = []
    for row in range(int(row_count)):
        tick = int(row + 1)
        route_col = int(route_path[int(row)])
        direction = int(row_directions[int(row)])
        for start_col in range(int(lane_count)):
            if (int(row), int(start_col)) in occupied:
                continue
            if int(start_col) == int(route_col):
                continue
            vehicle_col = vehicle_col_at_tick(
                CrossingVehicle("_tmp", int(row), int(start_col), int(direction), 0),
                tick=tick,
                lane_count=int(lane_count),
            )
            if vehicle_col is None or int(vehicle_col) != int(route_col):
                slots.append((int(row), int(start_col), int(direction)))
    return slots


def sample_labeled_route_collision_scene(
    *,
    rng: Any,
    axes: CrossingSceneAxes,
    target_label: str,
    max_extra_per_row: int,
) -> CrossingSample:
    """Construct a straight-route scene where exactly one labeled object collides."""

    labels = tuple(str(label) for label in VEHICLE_OPTION_LABELS)
    target_label = str(target_label)
    if target_label not in labels:
        raise ValueError(f"unsupported crossing vehicle option label: {target_label}")

    lane_count = int(axes.lane_count)
    row_count = int(axes.row_count)
    for _attempt in range(1200):
        target_row = int(rng.randrange(row_count))
        reachable_cols = list(_collision_reachable_cols(lane_count=lane_count, tick=int(target_row + 1)))
        if not reachable_cols:
            continue
        route_col = int(reachable_cols[int(rng.randrange(len(reachable_cols)))])
        route_path = tuple(int(route_col) for _row in range(row_count))
        row_directions = _row_directions(rng, row_count)
        hit = _collision_start_for_col(
            rng,
            lane_count=lane_count,
            col=int(route_col),
            tick=int(target_row + 1),
        )
        if hit is None:
            continue
        target_direction, target_start_col = hit
        row_directions[int(target_row)] = int(target_direction)

        occupied = {(int(target_row), int(target_start_col))}
        available_slots = _non_colliding_slots_for_route(
            lane_count=lane_count,
            row_count=row_count,
            row_directions=row_directions,
            route_path=route_path,
            occupied=occupied,
        )
        rng.shuffle(available_slots)
        other_labels = [label for label in labels if str(label) != str(target_label)]
        if len(available_slots) < len(other_labels):
            continue

        vehicle_specs = [
            {
                "row": int(target_row),
                "start_col": int(target_start_col),
                "direction": int(target_direction),
                "option_label": str(target_label),
            }
        ]
        for label, (row, start_col, direction) in zip(other_labels, available_slots):
            vehicle_specs.append(
                {
                    "row": int(row),
                    "start_col": int(start_col),
                    "direction": int(direction),
                    "option_label": str(label),
                }
            )
            occupied.add((int(row), int(start_col)))
        rng.shuffle(vehicle_specs)

        vehicles: list[CrossingVehicle] = []
        target_vehicle_id: str | None = None
        for spec in vehicle_specs:
            vehicle = _add_vehicle(
                vehicles,
                row=int(spec["row"]),
                start_col=int(spec["start_col"]),
                direction=int(spec["direction"]),
                color_index=int(rng.randrange(5)),
                option_label=str(spec["option_label"]),
            )
            if str(spec["option_label"]) == str(target_label):
                target_vehicle_id = str(vehicle.vehicle_id)

        _add_clutter(
            rng,
            vehicles=vehicles,
            lane_count=lane_count,
            row_count=row_count,
            row_directions=row_directions,
            avoid_cols_by_row={row: {int(route_path[row])} for row in range(row_count)},
            max_extra_per_row=int(max_extra_per_row),
            avoid_start_cols_by_row={row: {int(route_path[row])} for row in range(row_count)},
        )
        route = CrossingRouteOption(route_id=route_entity_id("M"), label="M", path_cols=tuple(route_path), color_index=0)
        hit_ids = route_collision_vehicle_ids(route, tuple(vehicles), lane_count=lane_count)
        if target_vehicle_id is None or tuple(hit_ids) != (str(target_vehicle_id),):
            continue

        sample = CrossingSample(
            lane_count=lane_count,
            row_count=row_count,
            count_mode="labeled_route_collision",
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=str(target_label),
            row_directions=tuple(int(value) for value in row_directions),
            vehicles=tuple(vehicles),
            start_labels=tuple(START_LABELS[:lane_count]),
            route_options=(route,),
            marked_route_label="M",
            target_start_label=str(START_LABELS[int(route_col)]),
            target_route_label=None,
            target_object_label=str(target_label),
            first_collision_tick=route_first_collision_tick(route, tuple(vehicles), lane_count=lane_count),
            intersecting_vehicle_ids=tuple(hit_ids),
            annotation_entity_ids=(str(target_vehicle_id),),
            target_answer=None,
            target_label_index=int(labels.index(str(target_label))),
            construction_mode="straight_labeled_route_collision",
        )
        validate_crossing_sample(sample)
        return sample
    raise ValueError("could not construct crossing hit-object label scene")


def sample_labeled_first_exit_scene(
    *,
    rng: Any,
    axes: CrossingSceneAxes,
    target_label: str,
) -> CrossingSample:
    """Construct a no-route scene where one labeled object exits first."""

    labels = tuple(str(label) for label in VEHICLE_OPTION_LABELS)
    target_label = str(target_label)
    if target_label not in labels:
        raise ValueError(f"unsupported crossing vehicle option label: {target_label}")

    lane_count = int(axes.lane_count)
    row_count = max(int(axes.row_count), len(labels))
    exit_candidates = [
        (
            vehicle_exit_tick(
                CrossingVehicle("_tmp", 0, int(start_col), int(direction), 0),
                lane_count=int(lane_count),
            ),
            int(direction),
            int(start_col),
        )
        for direction in (-1, 1)
        for start_col in range(int(lane_count))
    ]
    target_candidates = [
        candidate
        for candidate in exit_candidates
        if 2 <= int(candidate[0]) <= max(2, min(3, int(lane_count) - 1))
    ] or [candidate for candidate in exit_candidates if int(candidate[0]) == 1]
    if not target_candidates:
        raise ValueError("could not resolve crossing first-exit target candidates")

    for _attempt in range(300):
        target_tick, target_direction, target_start_col = target_candidates[int(rng.randrange(len(target_candidates)))]
        later_candidates = [candidate for candidate in exit_candidates if int(candidate[0]) > int(target_tick)]
        rng.shuffle(later_candidates)
        if len(later_candidates) < len(labels) - 1:
            continue

        rows = list(range(int(row_count)))
        rng.shuffle(rows)
        chosen_rows = [int(row) for row in rows[: len(labels)]]
        row_directions = _row_directions(rng, row_count)
        other_labels = [str(label) for label in labels if str(label) != str(target_label)]
        rng.shuffle(other_labels)
        vehicle_specs: list[dict[str, int | str]] = [
            {
                "row": int(chosen_rows[0]),
                "start_col": int(target_start_col),
                "direction": int(target_direction),
                "option_label": str(target_label),
            }
        ]
        for label, row, (_tick, direction, start_col) in zip(other_labels, chosen_rows[1:], later_candidates):
            vehicle_specs.append(
                {
                    "row": int(row),
                    "start_col": int(start_col),
                    "direction": int(direction),
                    "option_label": str(label),
                }
            )
        for spec in vehicle_specs:
            row_directions[int(spec["row"])] = int(spec["direction"])
        rng.shuffle(vehicle_specs)

        vehicles: list[CrossingVehicle] = []
        label_to_vehicle_id: dict[str, str] = {}
        for spec in vehicle_specs:
            vehicle = _add_vehicle(
                vehicles,
                row=int(spec["row"]),
                start_col=int(spec["start_col"]),
                direction=int(spec["direction"]),
                color_index=int(rng.randrange(5)),
                option_label=str(spec["option_label"]),
            )
            label_to_vehicle_id[str(spec["option_label"])] = str(vehicle.vehicle_id)

        target_vehicle_id = label_to_vehicle_id.get(str(target_label))
        if target_vehicle_id is None:
            continue
        exit_ticks = {
            str(vehicle.vehicle_id): vehicle_exit_tick(vehicle, lane_count=lane_count)
            for vehicle in vehicles
        }
        earliest_tick = min(exit_ticks.values())
        earliest_ids = tuple(sorted(vehicle_id for vehicle_id, tick in exit_ticks.items() if int(tick) == int(earliest_tick)))
        if tuple(earliest_ids) != (str(target_vehicle_id),):
            continue

        sample = CrossingSample(
            lane_count=lane_count,
            row_count=row_count,
            count_mode="labeled_first_exit",
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=str(target_label),
            row_directions=tuple(int(value) for value in row_directions),
            vehicles=tuple(vehicles),
            start_labels=(),
            route_options=(),
            marked_route_label=None,
            target_start_label=None,
            target_route_label=None,
            target_object_label=str(target_label),
            first_collision_tick=None,
            intersecting_vehicle_ids=tuple(),
            annotation_entity_ids=(str(target_vehicle_id),),
            target_answer=None,
            target_label_index=int(labels.index(str(target_label))),
            construction_mode="labeled_first_exit",
        )
        validate_crossing_sample(sample)
        return sample
    raise ValueError("could not construct crossing first-exit label scene")


def _sample_direction_count(
    rng: Any,
    *,
    axes: CrossingSceneAxes,
    target_answer: int,
    direction: int,
) -> CrossingSample | None:
    """Construct a scene with exactly the requested number of same-direction vehicles."""

    target_direction = int(direction)
    lane_count = int(axes.lane_count)
    row_count = int(axes.row_count)
    row_directions = _row_directions(rng, row_count)
    target_slots = [
        (int(row), int(col))
        for row, row_direction in enumerate(row_directions)
        if int(row_direction) == int(target_direction)
        for col in range(lane_count)
    ]
    if len(target_slots) < int(target_answer):
        return None
    rng.shuffle(target_slots)
    vehicles: list[CrossingVehicle] = []
    occupied: set[tuple[int, int]] = set()
    annotation_ids: list[str] = []
    for row, col in target_slots[: int(target_answer)]:
        vehicle = _add_vehicle(
            vehicles,
            row=int(row),
            start_col=int(col),
            direction=int(target_direction),
            color_index=int(rng.randrange(5)),
        )
        occupied.add((int(row), int(col)))
        annotation_ids.append(str(vehicle.vehicle_id))

    opposite_direction = -target_direction
    distractor_slots = [
        (int(row), int(col))
        for row, row_direction in enumerate(row_directions)
        if int(row_direction) == int(opposite_direction)
        for col in range(lane_count)
        if (int(row), int(col)) not in occupied
    ]
    rng.shuffle(distractor_slots)
    max_distractors = min(len(distractor_slots), max(1, min(6, int(target_answer) + 2)))
    distractor_count = int(rng.randint(1, int(max_distractors))) if max_distractors > 0 else 0
    for row, col in distractor_slots[:distractor_count]:
        _add_vehicle(
            vehicles,
            row=int(row),
            start_col=int(col),
            direction=int(opposite_direction),
            color_index=int(rng.randrange(5)),
        )

    count_mode = "left_movers" if int(target_direction) < 0 else "right_movers"
    sample = CrossingSample(
        lane_count=lane_count,
        row_count=row_count,
        count_mode=count_mode,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        answer=int(target_answer),
        row_directions=tuple(int(value) for value in row_directions),
        vehicles=tuple(vehicles),
        start_labels=tuple(START_LABELS[:lane_count]),
        route_options=(),
        marked_route_label=None,
        target_start_label=None,
        target_route_label=None,
        target_object_label=None,
        first_collision_tick=None,
        intersecting_vehicle_ids=(),
        annotation_entity_ids=tuple(annotation_ids),
        target_answer=int(target_answer),
        target_label_index=None,
        construction_mode=f"{count_mode}_exact_direction_count",
    )
    validate_crossing_sample(sample)
    return sample


def sample_crossing_scene(
    *,
    rng: Any,
    axes: CrossingSceneAxes,
    count_mode: str,
    target_answer: int,
    gen_defaults: Mapping[str, Any],
) -> CrossingSample:
    """Construct one exact-answer crossing sample for a semantic count mode."""

    mode = str(count_mode)
    max_extra_per_row = int(group_default(gen_defaults, "moving_object_max_extra_per_row", 1))
    for _attempt in range(1200):
        if mode == "route_intersections":
            sample = _sample_route_intersection_count(
                rng,
                axes=axes,
                target_answer=int(target_answer),
                max_extra_per_row=int(max_extra_per_row),
            )
        elif mode == "left_movers":
            sample = _sample_direction_count(rng, axes=axes, target_answer=int(target_answer), direction=-1)
        elif mode == "right_movers":
            sample = _sample_direction_count(rng, axes=axes, target_answer=int(target_answer), direction=1)
        else:
            raise ValueError(f"unsupported crossing count mode: {mode}")
        if sample is not None:
            return sample
    raise ValueError(f"could not construct crossing sample for count mode {mode}")


__all__ = [
    "resolve_crossing_scene_axes",
    "resolve_scene_axis",
    "resolve_target_answer",
    "sample_labeled_route_collision_scene",
    "sample_labeled_first_exit_scene",
    "sample_crossing_scene",
]
