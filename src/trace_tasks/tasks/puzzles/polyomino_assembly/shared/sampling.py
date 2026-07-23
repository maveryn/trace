"""Scene-local sampling primitives for polyomino assembly puzzles."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from trace_tasks.tasks.shared.mcq import option_label_for_index

from .state import DEFAULTS, Cell, Cells
from .rules import (
    can_two_pieces_tile_target,
    canonicalize_cells,
    edge_neighbors,
    is_connected,
    json_cells,
    pair_rotation_signature,
    rotation_signature,
    shape_bbox_dims,
)


def option_labels(option_count: int = 4) -> tuple[str, ...]:
    """Return canonical visible option labels."""

    return tuple(option_label_for_index(index) for index in range(int(option_count)))


def option_index_for_label(labels: Iterable[str], answer_label: str) -> int:
    """Return the index for one selected option label."""

    label_tuple = tuple(str(label) for label in labels)
    if str(answer_label) not in set(label_tuple):
        raise ValueError(f"answer label must be one of {label_tuple}")
    return int(label_tuple.index(str(answer_label)))


def sample_connected_shape(
    *,
    rng,
    area: int,
    max_dim: int,
    preferred_dims: tuple[int, int] | None = None,
) -> Cells:
    """Sample one connected polyomino with a bounded footprint."""

    for _attempt in range(1500):
        cells: set[Cell] = {(0, 0)}
        while len(cells) < int(area):
            frontier = {
                neighbor
                for cell in cells
                for neighbor in edge_neighbors(cell)
                if neighbor not in cells
            }
            if not frontier:
                break
            frontier_list = list(frontier)
            rng.shuffle(frontier_list)
            cells.add(frontier_list[0])
        canonical = canonicalize_cells(cells)
        width, height = shape_bbox_dims(canonical)
        if max(int(width), int(height)) > int(max_dim):
            continue
        if preferred_dims is not None:
            if abs(int(width) - int(preferred_dims[0])) > 1:
                continue
            if abs(int(height) - int(preferred_dims[1])) > 1:
                continue
        return canonical
    raise RuntimeError("failed to sample connected polyomino")


def sample_split_target(
    *,
    rng,
    total_area: int,
    min_piece_area: int,
    max_piece_area: int,
    max_dim: int,
    split_attempts: int,
) -> dict[str, Cells]:
    """Sample a connected target split into two connected pieces."""

    for _attempt in range(max(1, int(split_attempts))):
        target = sample_connected_shape(
            rng=rng,
            area=int(total_area),
            max_dim=int(max_dim),
        )
        lower = max(int(min_piece_area), int(total_area) - int(max_piece_area))
        upper = min(int(max_piece_area), int(total_area) - int(min_piece_area))
        if lower > upper:
            continue
        piece_area = int(rng.randint(int(lower), int(upper)))
        split = _sample_connected_subset_split(
            source_cells=target,
            subset_area=int(piece_area),
            rng=rng,
        )
        if split is None:
            continue
        piece_a, piece_b = split
        if can_two_pieces_tile_target(piece_a, piece_b, target):
            return {
                "target": canonicalize_cells(target),
                "piece_a": canonicalize_cells(piece_a),
                "piece_b": canonicalize_cells(piece_b),
            }
    raise RuntimeError("failed to split target into two connected pieces")


def sample_pair_distractor(
    *,
    rng,
    total_area: int,
    min_piece_area: int,
    max_piece_area: int,
    max_dim: int,
) -> tuple[Cells, Cells]:
    """Sample a same-area two-piece distractor candidate."""

    for _attempt in range(2000):
        lower = max(int(min_piece_area), int(total_area) - int(max_piece_area))
        upper = min(int(max_piece_area), int(total_area) - int(min_piece_area))
        if lower > upper:
            raise ValueError("invalid distractor piece-size bounds")
        first_area = int(rng.randint(int(lower), int(upper)))
        second_area = int(total_area - first_area)
        piece_a = sample_connected_shape(
            rng=rng,
            area=int(first_area),
            max_dim=int(max_dim),
            preferred_dims=None,
        )
        piece_b = sample_connected_shape(
            rng=rng,
            area=int(second_area),
            max_dim=int(max_dim),
            preferred_dims=None,
        )
        return canonicalize_cells(piece_a), canonicalize_cells(piece_b)
    raise RuntimeError("failed to sample pair distractor")


def sample_shape_distractor(
    *,
    rng,
    total_area: int,
    max_dim: int,
    preferred_dims: tuple[int, int],
) -> Cells:
    """Sample a same-area shape distractor candidate."""

    for attempt in range(2000):
        preferred = preferred_dims if int(attempt) < 1000 else None
        candidate = sample_connected_shape(
            rng=rng,
            area=int(total_area),
            max_dim=int(max_dim),
            preferred_dims=preferred,
        )
        return canonicalize_cells(candidate)
    raise RuntimeError("failed to sample shape distractor")


def serialize_pair_option_specs(
    *,
    option_pairs: list[tuple[Cells, Cells]],
    correct_option_index: int,
) -> list[dict[str, Any]]:
    """Serialize two-piece option panels."""

    specs: list[dict[str, Any]] = []
    for option_index, pair in enumerate(option_pairs):
        piece_a, piece_b = pair
        specs.append(
            {
                "option_panel_id": f"option_panel_{option_index + 1}",
                "option_choice_id": f"option_choice_{option_index + 1}",
                "option_label": option_label_for_index(option_index),
                "pieces": [
                    {"piece_id": "piece_a", "cells": json_cells(piece_a)},
                    {"piece_id": "piece_b", "cells": json_cells(piece_b)},
                ],
                "total_cell_count": int(len(piece_a) + len(piece_b)),
                "is_correct": bool(option_index == int(correct_option_index)),
                "signature": [
                    json_cells(shape)
                    for shape in pair_rotation_signature(piece_a, piece_b)
                ],
            }
        )
    return specs


def serialize_shape_option_specs(
    *,
    option_shapes: list[Cells],
    correct_option_index: int,
) -> list[dict[str, Any]]:
    """Serialize one-shape option panels."""

    specs: list[dict[str, Any]] = []
    for option_index, cells in enumerate(option_shapes):
        specs.append(
            {
                "option_panel_id": f"option_panel_{option_index + 1}",
                "option_choice_id": f"option_choice_{option_index + 1}",
                "option_label": option_label_for_index(option_index),
                "cells": json_cells(cells),
                "cell_count": int(len(cells)),
                "bbox_dims": list(shape_bbox_dims(cells)),
                "is_correct": bool(option_index == int(correct_option_index)),
                "signature": json_cells(rotation_signature(cells)),
            }
        )
    return specs


def _sample_connected_subset_split(
    *,
    source_cells: Cells,
    subset_area: int,
    rng,
) -> tuple[Cells, Cells] | None:
    """Sample a connected subset whose complement is also connected."""

    source_set = set(source_cells)
    if int(subset_area) <= 0 or int(subset_area) >= len(source_set):
        return None
    for _attempt in range(500):
        start = rng.choice(list(source_set))
        subset = {start}
        while len(subset) < int(subset_area):
            frontier = {
                neighbor
                for cell in subset
                for neighbor in edge_neighbors(cell)
                if neighbor in source_set and neighbor not in subset
            }
            if not frontier:
                break
            frontier_list = list(frontier)
            rng.shuffle(frontier_list)
            subset.add(frontier_list[0])
        if len(subset) != int(subset_area):
            continue
        complement = source_set - subset
        if is_connected(subset) and is_connected(complement):
            return canonicalize_cells(subset), canonicalize_cells(complement)
    return None


def generation_bounds(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, int]:
    """Resolve non-random generation bounds used by both task constructors."""

    def _int_value(key: str, fallback: int) -> int:
        return int(params.get(str(key), defaults.get(str(key), int(fallback))))

    return {
        "piece_cell_count_min": _int_value(
            "piece_cell_count_min",
            DEFAULTS.piece_cell_count_min,
        ),
        "piece_cell_count_max": _int_value(
            "piece_cell_count_max",
            DEFAULTS.piece_cell_count_max,
        ),
        "shape_bbox_max_dim": _int_value(
            "shape_bbox_max_dim",
            DEFAULTS.shape_bbox_max_dim,
        ),
        "split_attempts": _int_value("split_attempts", DEFAULTS.split_attempts),
        "distractor_attempts": _int_value(
            "distractor_attempts",
            DEFAULTS.distractor_attempts,
        ),
    }


__all__ = [
    "generation_bounds",
    "option_index_for_label",
    "option_labels",
    "sample_pair_distractor",
    "sample_shape_distractor",
    "sample_split_target",
    "serialize_pair_option_specs",
    "serialize_shape_option_specs",
]
