"""Reusable color-board construction families for cell-board objectives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng

from .sampling import (
    sample_answer,
    sample_dimensions,
    sample_palette_size,
    sample_unique_largest_component_board,
)
from .state import CellBoardCase
from .topology import sort_coords


def build_largest_component_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    prompt_query_key: str,
) -> CellBoardCase:
    """Build a largest-component case with exactly one winning target group."""

    rows, cols = sample_dimensions(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="puzzles.cell_board.largest_component.dimensions",
        fallback_rows_min=6,
        fallback_rows_max=9,
        fallback_cols_min=6,
        fallback_cols_max=9,
    )
    answer, support = sample_answer(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="puzzles.cell_board.largest_component.answer",
        fallback_min=2,
        fallback_max=7,
        min_key="target_largest_component_size_min",
        max_key="target_largest_component_size_max",
    )
    rng = spawn_rng(int(instance_seed), "puzzles.cell_board.largest_component.case")
    target_filler_min_color_distance = float(
        params.get(
            "target_filler_min_color_distance",
            gen_defaults.get("target_filler_min_color_distance", 0.0),
        )
    )
    color_distance_space = str(
        params.get(
            "target_filler_color_distance_space",
            gen_defaults.get("target_filler_color_distance_space", "lab"),
        )
    )
    sample = sample_unique_largest_component_board(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        palette_size=sample_palette_size(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            namespace="puzzles.cell_board.largest_component.palette",
            fallback_min=2,
            fallback_max=3,
        ),
        largest_size=int(answer),
        target_filler_min_color_distance=float(target_filler_min_color_distance),
        color_distance_space=str(color_distance_space),
    )
    return CellBoardCase(
        rows=int(rows),
        cols=int(cols),
        board_colors=sample.board,
        answer_value=int(answer),
        annotation_kind="bbox_set",
        annotation_coords=tuple(sample.witness_coords),
        prompt_task_key="cell_board_topology_query",
        prompt_query_key=str(prompt_query_key),
        prompt_slots={"query_color": str(sample.target_color[0])},
        execution_trace={
            "query_color": str(sample.target_color[0]),
            "target_filler_min_color_distance": float(
                target_filler_min_color_distance
            ),
            "target_filler_color_distance_space": str(color_distance_space),
            "components": [
                [[int(r), int(c)] for r, c in sort_coords(component)]
                for component in sample.components
            ],
            "answer_support": [int(value) for value in support],
        },
    )


__all__ = [
    "build_largest_component_case",
]
