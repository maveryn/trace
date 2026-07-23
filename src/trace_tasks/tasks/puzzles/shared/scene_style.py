"""Puzzle-domain adapter for globally shared panel-style renderers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.visual_style.panel import (
    DEFAULT_PANEL_SCENE_STYLE,
    PANEL_SCENE_TREATMENTS,
    PanelSceneStyle,
    draw_panel_chrome_by_mode,
    draw_panel_grid_cell,
    draw_panel_option_card,
    draw_panel_plain_chrome,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    panel_scene_style_metadata,
    resolve_panel_chrome_mode,
    resolve_panel_scene_style,
)

PuzzleSceneStyle = PanelSceneStyle
DEFAULT_PUZZLE_SCENE_STYLE = DEFAULT_PANEL_SCENE_STYLE
PUZZLE_SCENE_TREATMENTS = PANEL_SCENE_TREATMENTS


def resolve_puzzle_scene_style(
    *,
    instance_seed: int,
    namespace: str,
    treatments: Sequence[str] | None = None,
    treatment_weights: Mapping[str, float] | None = None,
    style_pack_weights: Mapping[str, float] | None = None,
    palette_weights: Mapping[str, float] | None = None,
) -> tuple[PuzzleSceneStyle, dict[str, Any]]:
    """Resolve one puzzle scene style via the shared panel-style layer."""

    return resolve_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        treatments=treatments,
        treatment_weights=treatment_weights,
        palette_weights=palette_weights or style_pack_weights,
    )


def puzzle_scene_style_metadata(style: PuzzleSceneStyle) -> dict[str, Any]:
    """Serialize one puzzle scene style using the shared metadata schema."""

    return panel_scene_style_metadata(style)


draw_puzzle_grid_cell = draw_panel_grid_cell
draw_puzzle_option_card = draw_panel_option_card
draw_puzzle_chrome_by_mode = draw_panel_chrome_by_mode
draw_puzzle_plain_chrome = draw_panel_plain_chrome
draw_puzzle_panel_chrome = draw_panel_scene_chrome
make_puzzle_scene_background = make_panel_scene_background


__all__ = [
    "DEFAULT_PUZZLE_SCENE_STYLE",
    "PUZZLE_SCENE_TREATMENTS",
    "PuzzleSceneStyle",
    "draw_puzzle_chrome_by_mode",
    "draw_puzzle_grid_cell",
    "draw_puzzle_option_card",
    "draw_puzzle_panel_chrome",
    "draw_puzzle_plain_chrome",
    "make_puzzle_scene_background",
    "puzzle_scene_style_metadata",
    "resolve_panel_chrome_mode",
    "resolve_puzzle_scene_style",
]
