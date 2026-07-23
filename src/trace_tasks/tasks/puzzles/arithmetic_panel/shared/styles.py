"""Style resolution for arithmetic-constraint puzzle panels."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    puzzle_scene_style_metadata,
    resolve_puzzle_scene_style,
)


def resolve_arithmetic_panel_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Resolve the shared puzzle/game panel treatment for this arithmetic scene."""

    treatment_weights = params.get("treatment_weights")
    palette_weights = params.get("palette_weights")
    style, style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace="puzzles.arithmetic_panel.panel_style",
        treatment_weights=(
            treatment_weights if isinstance(treatment_weights, Mapping) else None
        ),
        palette_weights=(
            palette_weights if isinstance(palette_weights, Mapping) else None
        ),
    )
    return style, style_meta


__all__ = [
    "make_puzzle_scene_background",
    "puzzle_scene_style_metadata",
    "resolve_arithmetic_panel_style",
]
