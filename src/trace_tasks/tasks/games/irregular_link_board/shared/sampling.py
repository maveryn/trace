"""Sampling helpers for irregular-link-board scene tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import all_coords, all_possible_edges, capture_destinations, capture_paths, edge, legal_destinations, neighbors
from .state import (
    SCENE_NAMESPACE,
    SCENE_VARIANTS,
    STYLE_VARIANTS,
    Coord,
    Edge,
    IrregularLinkBoardAxes,
    IrregularLinkBoardSample,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
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


def resolve_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
    board_size_support_key: str = "board_size_support",
    fallback_board_size_support: Sequence[int] | None = None,
) -> IrregularLinkBoardAxes:
    """Resolve scene/render-independent axes for one task-owned sample."""

    scene_variant, scene_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SCENE_VARIANTS,
    )
    style_variant, style_probs = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    board_size, board_size_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(board_size_support_key),
        explicit_key="board_size",
        fallback_support=tuple(int(value) for value in (fallback_board_size_support or DEFAULTS.board_size_support)),
        namespace=f"{namespace}.board_size",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    target_answer, target_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.target_answer_support,
        namespace=f"{namespace}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="target_answer_support",
        fallback=DEFAULTS.target_answer_support,
    )
    return IrregularLinkBoardAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_size=int(board_size),
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        board_size_probabilities=dict(board_size_probs),
        target_answer_probabilities=dict(target_probs),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
    )


def _link_density(scene_variant: str) -> float:
    if str(scene_variant) == "sparse_links":
        return 0.42
    if str(scene_variant) == "dense_links":
        return 0.72
    return 0.56


def sample_destination_scene(
    *,
    rng: Any,
    axes: IrregularLinkBoardAxes,
    gen_defaults: Mapping[str, Any],
) -> IrregularLinkBoardSample:
    """Construct a board with exactly the requested one-step destination count."""

    board_size = int(axes.board_size)
    target = int(axes.target_answer)
    if target < 0 or target > 6:
        raise ValueError("irregular link board target answer must be in 0..6")
    viable = [coord for coord in all_coords(board_size) if len(neighbors(coord, board_size)) >= target]
    if not viable:
        raise ValueError(f"no marked point can realize target answer {target}")
    rng.shuffle(viable)
    marked_coord = viable[0]
    neighbor_coords = list(neighbors(marked_coord, board_size))
    rng.shuffle(neighbor_coords)
    destinations = set(neighbor_coords[:target])
    non_dest_neighbors = [coord for coord in neighbor_coords if coord not in destinations]

    edges: set[Edge] = {edge(marked_coord, coord) for coord in destinations}
    occupied: set[Coord] = {marked_coord}

    for coord in non_dest_neighbors:
        connect_and_block = bool(rng.random() < 0.58)
        if connect_and_block:
            edges.add(edge(marked_coord, coord))
            occupied.add(coord)
        elif rng.random() < 0.38:
            occupied.add(coord)

    possible_edges = set(all_possible_edges(board_size))
    marked_incident = {edge(marked_coord, coord) for coord in neighbor_coords}
    density = _link_density(str(axes.scene_variant))
    for link in possible_edges:
        if link in marked_incident:
            continue
        if rng.random() < float(density):
            edges.add(link)

    min_pieces = int(group_default(gen_defaults, "min_total_piece_count", DEFAULTS.min_total_piece_count))
    max_pieces = int(group_default(gen_defaults, "max_total_piece_count", DEFAULTS.max_total_piece_count))
    desired_piece_count = max(len(occupied), int(rng.randint(min_pieces, max_pieces)))
    protected_empty = set(destinations)
    candidates = [coord for coord in all_coords(board_size) if coord not in occupied and coord not in protected_empty]
    rng.shuffle(candidates)
    for coord in candidates:
        if len(occupied) >= desired_piece_count:
            break
        occupied.add(coord)

    annotation = legal_destinations(
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        edges=tuple(sorted(edges)),
        board_size=board_size,
    )
    if len(annotation) != target:
        raise ValueError("constructed legal destination count mismatch")
    return IrregularLinkBoardSample(
        board_size=int(board_size),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        edges=tuple(sorted(edges)),
        annotation_coords=tuple(annotation),
        answer=int(len(annotation)),
        construction_mode="target_conditioned_variable_link_board",
    )


def sample_capture_scene(
    *,
    rng: Any,
    axes: IrregularLinkBoardAxes,
    gen_defaults: Mapping[str, Any],
) -> IrregularLinkBoardSample:
    """Construct a board with exactly the requested jump-capture count."""

    board_size = int(axes.board_size)
    target = int(axes.target_answer)
    if target < 0 or target > 6:
        raise ValueError("irregular link capture target answer must be in 0..6")
    viable = [coord for coord in all_coords(board_size) if len(capture_paths(coord, board_size)) >= target]
    if not viable:
        raise ValueError(f"no marked point can realize capture target answer {target}")
    rng.shuffle(viable)
    marked_coord = viable[0]
    possible_paths = list(capture_paths(marked_coord, board_size))
    rng.shuffle(possible_paths)
    selected_paths = tuple(possible_paths[:target])
    selected_destinations = {path[0] for path in selected_paths}
    non_selected_paths = [path for path in possible_paths if path[0] not in selected_destinations]

    edges: set[Edge] = set()
    blocked_edges: set[Edge] = set()
    occupied: set[Coord] = {marked_coord}
    protected_empty: set[Coord] = set(selected_destinations)

    for destination, captured in selected_paths:
        edges.add(edge(marked_coord, captured))
        edges.add(edge(captured, destination))
        occupied.add(captured)

    for destination, captured in non_selected_paths:
        if rng.random() < 0.42:
            edges.add(edge(marked_coord, captured))
            occupied.add(captured)
            protected_empty.add(destination)
            blocked_edges.add(edge(captured, destination))
        elif rng.random() < 0.48:
            occupied.add(destination)
            if rng.random() < 0.58:
                edges.add(edge(marked_coord, captured))
                edges.add(edge(captured, destination))
        else:
            occupied.add(captured)
            protected_empty.add(destination)
            blocked_edges.add(edge(marked_coord, captured))

    possible_edges = set(all_possible_edges(board_size))
    marked_incident = {edge(marked_coord, coord) for coord in neighbors(marked_coord, board_size)}
    density = _link_density(str(axes.scene_variant))
    for link in possible_edges:
        if link in blocked_edges:
            continue
        if link in marked_incident and link not in edges:
            continue
        if link in edges:
            continue
        if rng.random() < float(density):
            edges.add(link)

    min_pieces = int(group_default(gen_defaults, "min_total_piece_count", DEFAULTS.min_total_piece_count))
    max_pieces = int(group_default(gen_defaults, "max_total_piece_count", DEFAULTS.max_total_piece_count))
    desired_piece_count = max(len(occupied), int(rng.randint(min_pieces, max_pieces)))
    candidates = [coord for coord in all_coords(board_size) if coord not in occupied and coord not in protected_empty]
    rng.shuffle(candidates)
    for coord in candidates:
        if len(occupied) >= desired_piece_count:
            break
        occupied.add(coord)

    annotation = capture_destinations(
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        edges=tuple(sorted(edges)),
        board_size=board_size,
    )
    if len(annotation) != target:
        raise ValueError("constructed capture move count mismatch")
    return IrregularLinkBoardSample(
        board_size=int(board_size),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        marked_coord=marked_coord,
        occupied_coords=tuple(sorted(occupied)),
        edges=tuple(sorted(edges)),
        annotation_coords=tuple(annotation),
        answer=int(len(annotation)),
        construction_mode="target_conditioned_capture_move_board",
    )


__all__ = ["resolve_axes", "sample_capture_scene", "sample_destination_scene"]
