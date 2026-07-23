"""Scene-neutral sampling helpers for radial hunt board tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import all_coords, capture_destinations, capture_paths, legal_destinations, neighbors
from .state import (
    SCENE_ID,
    SCENE_NAMESPACE,
    Coord,
    RadialHuntBoardSample,
    RadialHuntBoardTargetAxis,
    RadialHuntBoardVisualAxes,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced visual axis for this scene."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def resolve_radial_hunt_board_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str = SCENE_NAMESPACE,
) -> RadialHuntBoardVisualAxes:
    """Resolve scene/style axes without objective-specific branching."""

    scene_variant, scene_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace_root),
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace_root),
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_STYLE_VARIANTS,
    )
    return RadialHuntBoardVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
    )


def resolve_radial_hunt_board_target_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> RadialHuntBoardTargetAxis:
    """Resolve a task-owned integer target-answer axis."""

    target_answer, target_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.target_answer_support,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="target_answer_support",
        fallback=DEFAULTS.target_answer_support,
    )
    return RadialHuntBoardTargetAxis(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        target_answer_probabilities=dict(target_probs),
    )


def piece_count_bounds(scene_variant: str, *, gen_defaults: Mapping[str, Any]) -> Tuple[int, int]:
    """Resolve total piece-count bounds for a visual crowding variant."""

    if str(scene_variant) == "open_position":
        fallback_min, fallback_max = (5, 8)
    elif str(scene_variant) == "crowded_position":
        fallback_min, fallback_max = (11, 16)
    else:
        fallback_min, fallback_max = (8, 12)
    min_pieces = int(group_default(gen_defaults, "min_total_piece_count", fallback_min))
    max_pieces = int(group_default(gen_defaults, "max_total_piece_count", fallback_max))
    if min_pieces > max_pieces:
        raise ValueError(f"{SCENE_ID} min_total_piece_count must be <= max_total_piece_count")
    return min_pieces, max_pieces


def fill_extra_occupied_coords(
    *,
    rng: Any,
    occupied_coords: set[Coord],
    protected_empty_coords: set[Coord],
    desired_piece_count: int,
) -> set[Coord]:
    """Fill random extra pieces while respecting protected empty witness cells."""

    occupied = {tuple(coord) for coord in occupied_coords}
    protected_empty = {tuple(coord) for coord in protected_empty_coords}
    candidates = [
        coord
        for coord in all_coords()
        if tuple(coord) not in occupied and tuple(coord) not in protected_empty
    ]
    rng.shuffle(candidates)
    for coord in candidates:
        if len(occupied) >= int(desired_piece_count):
            break
        occupied.add(tuple(coord))
    return occupied


def sample_destination_scene(
    *,
    rng: Any,
    axes: RadialHuntBoardVisualAxes,
    target_axis: RadialHuntBoardTargetAxis,
    gen_defaults: Mapping[str, Any],
) -> RadialHuntBoardSample:
    """Construct one board with a fixed empty-adjacent destination count."""

    target = int(target_axis.target_answer)
    if target < 0 or target > 6:
        raise ValueError("radial hunt destination target answer must be in 0..6")
    viable = [coord for coord in all_coords() if len(neighbors(coord)) >= target]
    if not viable:
        raise ValueError("no radial hunt point can support requested destination count")
    rng.shuffle(viable)
    marked_coord = tuple(viable[0])
    neighbor_coords = list(neighbors(marked_coord))
    rng.shuffle(neighbor_coords)
    destinations: set[Coord] = {tuple(coord) for coord in neighbor_coords[:target]}
    occupied: set[Coord] = {marked_coord}
    for coord in neighbor_coords:
        if tuple(coord) not in destinations:
            occupied.add(tuple(coord))

    min_pieces, max_pieces = piece_count_bounds(str(axes.scene_variant), gen_defaults=gen_defaults)
    desired_piece_count = max(len(occupied), int(rng.randint(min_pieces, max_pieces)))
    occupied = fill_extra_occupied_coords(
        rng=rng,
        occupied_coords=occupied,
        protected_empty_coords=destinations,
        desired_piece_count=int(desired_piece_count),
    )
    annotation = legal_destinations(
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
    )
    if len(annotation) != target:
        raise ValueError("constructed radial destination count mismatch")
    return RadialHuntBoardSample(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        annotation_coords=tuple(annotation),
        answer=int(len(annotation)),
        construction_mode="target_conditioned_radial_destination_board",
    )


def sample_capture_scene(
    *,
    rng: Any,
    axes: RadialHuntBoardVisualAxes,
    target_axis: RadialHuntBoardTargetAxis,
    gen_defaults: Mapping[str, Any],
) -> RadialHuntBoardSample:
    """Construct one board with a fixed jump-capture landing count."""

    target = int(target_axis.target_answer)
    if target < 0 or target > 6:
        raise ValueError("radial hunt capture target answer must be in 0..6")
    viable = [coord for coord in all_coords() if len(capture_paths(coord)) >= target]
    if not viable:
        raise ValueError("no radial hunt point can support requested capture count")
    rng.shuffle(viable)
    marked_coord = tuple(viable[0])
    available_paths = list(capture_paths(marked_coord))
    rng.shuffle(available_paths)
    selected_paths = tuple(available_paths[:target])
    selected_destinations: set[Coord] = {tuple(path[0]) for path in selected_paths}
    selected_captured: set[Coord] = {tuple(path[1]) for path in selected_paths}
    if selected_destinations & selected_captured:
        raise ValueError("selected radial capture paths conflict on occupied and empty points")

    occupied: set[Coord] = {marked_coord, *selected_captured}
    protected_empty: set[Coord] = set(selected_destinations)
    for destination, captured in available_paths:
        destination = tuple(destination)
        captured = tuple(captured)
        if destination in selected_destinations and (destination, captured) not in selected_paths:
            if captured in selected_captured:
                raise ValueError("selected radial capture paths create an accidental extra capture")
            protected_empty.add(captured)
    for destination, captured in available_paths:
        destination = tuple(destination)
        captured = tuple(captured)
        if destination in selected_destinations:
            continue
        if captured in selected_captured:
            occupied.add(destination)
        elif rng.random() < 0.62:
            occupied.add(destination)
        else:
            protected_empty.add(captured)

    min_pieces, max_pieces = piece_count_bounds(str(axes.scene_variant), gen_defaults=gen_defaults)
    desired_piece_count = max(len(occupied), int(rng.randint(min_pieces, max_pieces)))
    occupied = fill_extra_occupied_coords(
        rng=rng,
        occupied_coords=occupied,
        protected_empty_coords=protected_empty,
        desired_piece_count=int(desired_piece_count),
    )
    annotation = capture_destinations(
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
    )
    if len(annotation) != target:
        raise ValueError("constructed radial capture count mismatch")
    return RadialHuntBoardSample(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        annotation_coords=tuple(annotation),
        answer=int(len(annotation)),
        construction_mode="target_conditioned_radial_capture_board",
    )


__all__ = [
    "fill_extra_occupied_coords",
    "piece_count_bounds",
    "resolve_radial_hunt_board_target_axis",
    "resolve_radial_hunt_board_visual_axes",
    "sample_capture_scene",
    "sample_destination_scene",
]
