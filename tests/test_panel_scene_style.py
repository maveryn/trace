"""Regression tests for shared panel/canvas visual treatments."""

from __future__ import annotations

from trace_tasks.tasks.shared.visual_style.palette import PANEL_SCENE_PALETTES
from trace_tasks.tasks.shared.visual_style.panel import PANEL_SCENE_TREATMENTS
from trace_tasks.tasks.shared.visual_style.panel import _compatible_palette_ids


def test_panel_scene_treatments_match_target_light_dark_split() -> None:
    """Panel-style scenes should expose 20 light and 5 dark treatments."""

    light_palette_ids = {
        palette_id
        for palette_id, palette in PANEL_SCENE_PALETTES.items()
        if "light" in palette.compatibility
    }
    dark_palette_ids = {
        palette_id
        for palette_id, palette in PANEL_SCENE_PALETTES.items()
        if "dark" in palette.compatibility
    }
    light_treatments = []
    dark_treatments = []
    compatible_style_pack_count = 0

    for treatment in PANEL_SCENE_TREATMENTS:
        compatible = set(_compatible_palette_ids(str(treatment)))
        compatible_style_pack_count += len(compatible)
        if compatible and compatible.issubset(dark_palette_ids):
            dark_treatments.append(str(treatment))
        elif compatible and compatible.issubset(light_palette_ids):
            light_treatments.append(str(treatment))
        else:
            raise AssertionError(
                f"{treatment} has mixed or unknown panel palette compatibility: "
                f"{sorted(compatible)}"
            )

    assert len(PANEL_SCENE_TREATMENTS) == 25
    assert len(light_treatments) == 20
    assert len(dark_treatments) == 5
    assert compatible_style_pack_count == 325
    assert set(dark_treatments) == {
        "arcade_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
        "terminal_screen",
    }
