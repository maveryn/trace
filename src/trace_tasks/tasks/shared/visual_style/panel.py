"""Shared panel/canvas treatments for puzzle, game, and icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ..drawing import draw_rounded_rect
from ..text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    resolve_readable_text_style,
    text_legibility_summary,
)
from .metadata import color_separation_metadata
from .palette import PANEL_SCENE_PALETTES, Color, PanelScenePalette

PANEL_CHROME_MODES: Tuple[str, ...] = ("none", "plain_panel", "accent_frame")
DEFAULT_PANEL_CHROME_WEIGHTS = {
    "none": 0.5,
    "plain_panel": 0.25,
    "accent_frame": 0.25,
}

PANEL_SCENE_TREATMENTS: Tuple[str, ...] = (
    "bare_canvas",
    "plain_sheet",
    "matte_sheet",
    "thin_frame",
    "soft_panel",
    "margin_sheet",
    "dot_sheet",
    "worksheet_panel",
    "notebook_grid",
    "index_card",
    "clipboard_sheet",
    "printout_panel",
    "puzzle_card",
    "inset_board",
    "tabletop_mat",
    "corkboard_sheet",
    "game_table",
    "lab_panel",
    "whiteboard_panel",
    "tile_tray",
    "arcade_screen",
    "terminal_screen",
    "dark_game_table",
    "neon_grid_screen",
    "scoreboard_panel",
)

_DARK_TREATMENTS = {
    "arcade_screen",
    "terminal_screen",
    "dark_game_table",
    "neon_grid_screen",
    "scoreboard_panel",
}


@dataclass(frozen=True)
class PanelSceneStyle:
    """Resolved shared panel style after independent treatment/palette sampling."""

    treatment: str
    palette_id: str
    style_pack: str
    background_rgb: Color
    background_accent_rgb: Color
    state_colors: Tuple[Color, ...]
    panel_fill_rgb: Color
    panel_border_rgb: Color
    panel_accent_rgb: Color
    grid_rgb: Color
    text_rgb: Color
    text_stroke_rgb: Color
    mark_rgb: Color
    agent_rgb: Color
    agent_inner_rgb: Color
    option_fill_rgb: Color
    option_marker_fill_rgb: Color
    step_fill_rgb: Color
    notebook_line_rgb: Color
    notebook_margin_rgb: Color


def _style_from_palette(
    *, treatment: str, palette: PanelScenePalette
) -> PanelSceneStyle:
    return PanelSceneStyle(
        treatment=str(treatment),
        palette_id=str(palette.palette_id),
        style_pack=f"{str(treatment)}:{str(palette.palette_id)}",
        background_rgb=tuple(palette.background_rgb),
        background_accent_rgb=tuple(palette.background_accent_rgb),
        state_colors=tuple(tuple(color) for color in palette.state_colors),
        panel_fill_rgb=tuple(palette.panel_fill_rgb),
        panel_border_rgb=tuple(palette.panel_border_rgb),
        panel_accent_rgb=tuple(palette.panel_accent_rgb),
        grid_rgb=tuple(palette.grid_rgb),
        text_rgb=tuple(palette.text_rgb),
        text_stroke_rgb=tuple(palette.text_stroke_rgb),
        mark_rgb=tuple(palette.mark_rgb),
        agent_rgb=tuple(palette.agent_rgb),
        agent_inner_rgb=tuple(palette.agent_inner_rgb),
        option_fill_rgb=tuple(palette.option_fill_rgb),
        option_marker_fill_rgb=tuple(palette.option_marker_fill_rgb),
        step_fill_rgb=tuple(palette.step_fill_rgb),
        notebook_line_rgb=tuple(palette.notebook_line_rgb),
        notebook_margin_rgb=tuple(palette.notebook_margin_rgb),
    )


DEFAULT_PANEL_SCENE_STYLE = _style_from_palette(
    treatment="plain_sheet",
    palette=PANEL_SCENE_PALETTES["plain_neutral"],
)


def _weighted_choice(
    *,
    values: Sequence[str],
    weights: Mapping[str, float],
    instance_seed: int,
    namespace: str,
) -> str:
    if not values:
        raise ValueError("values must not be empty")
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=str(namespace))
    parsed = [max(0.0, float(weights.get(str(value), 1.0))) for value in values]
    total = sum(parsed)
    if total <= 0.0:
        parsed = [1.0 for _ in values]
        total = float(len(values))
    threshold = rng.random() * total
    cursor = 0.0
    for value, weight in zip(values, parsed):
        cursor += float(weight)
        if threshold <= cursor:
            return str(value)
    return str(values[-1])


def _palette_modes_for_treatment(treatment: str) -> Tuple[str, ...]:
    if str(treatment) in _DARK_TREATMENTS:
        return ("dark", "both")
    return ("light", "both")


def _compatible_palette_ids(treatment: str) -> Tuple[str, ...]:
    modes = set(_palette_modes_for_treatment(str(treatment)))
    out = [
        palette_id
        for palette_id, palette in sorted(PANEL_SCENE_PALETTES.items())
        if modes.intersection(set(str(item) for item in palette.compatibility))
    ]
    return tuple(out) or ("plain_neutral",)


def resolve_panel_scene_style(
    *,
    instance_seed: int,
    namespace: str,
    treatments: Sequence[str] | None = None,
    treatment_weights: Mapping[str, float] | None = None,
    palette_weights: Mapping[str, float] | None = None,
) -> tuple[PanelSceneStyle, dict[str, Any]]:
    """Resolve a treatment/palette pair for one panel-like scene."""

    allowed_treatments = tuple(
        str(item) for item in (treatments or PANEL_SCENE_TREATMENTS)
    )
    treatment = _weighted_choice(
        values=allowed_treatments,
        weights=treatment_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.treatment",
    )
    palette_ids = _compatible_palette_ids(str(treatment))
    palette_id = _weighted_choice(
        values=palette_ids,
        weights=palette_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.palette.{treatment}",
    )
    palette = PANEL_SCENE_PALETTES.get(
        str(palette_id), PANEL_SCENE_PALETTES["plain_neutral"]
    )
    style = _style_from_palette(treatment=str(treatment), palette=palette)
    text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.text_legibility",
        role="read_required_panel_text",
        surface_rgbs=(
            style.background_rgb,
            style.panel_fill_rgb,
            style.option_fill_rgb,
        ),
        preferred_rgbs=(style.text_rgb, style.grid_rgb, style.panel_border_rgb),
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        required=True,
    )
    style = replace(
        style,
        text_rgb=tuple(text_style.fill_rgb),
        text_stroke_rgb=tuple(text_style.stroke_rgb),
    )
    metadata = panel_scene_style_metadata(style)
    metadata["text_legibility"] = text_legibility_summary((text_style,))
    metadata["text_color_policy"] = (
        "read_required_text_uses_random_nonsemantic_readable_ink"
    )
    return style, metadata


def panel_scene_style_metadata(style: PanelSceneStyle) -> dict[str, Any]:
    """Serialize one resolved shared panel style into trace-friendly metadata."""

    return {
        "treatment": str(style.treatment),
        "palette_id": str(style.palette_id),
        "style_pack": str(style.style_pack),
        "background_rgb": list(style.background_rgb),
        "background_accent_rgb": list(style.background_accent_rgb),
        "state_colors_rgb": [list(color) for color in style.state_colors],
        "panel_fill_rgb": list(style.panel_fill_rgb),
        "panel_border_rgb": list(style.panel_border_rgb),
        "panel_accent_rgb": list(style.panel_accent_rgb),
        "grid_rgb": list(style.grid_rgb),
        "text_rgb": list(style.text_rgb),
        "text_stroke_rgb": list(style.text_stroke_rgb),
        "mark_rgb": list(style.mark_rgb),
        "agent_rgb": list(style.agent_rgb),
        "option_fill_rgb": list(style.option_fill_rgb),
        "notebook_line_rgb": list(style.notebook_line_rgb),
        **color_separation_metadata(
            anchor_rgb=style.background_rgb,
            compared_rgbs=style.state_colors,
            prefix="background_state",
        ),
    }


def resolve_panel_chrome_mode(
    *,
    instance_seed: int,
    namespace: str,
    modes: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the optional whole-board/panel chrome independently from cell style."""

    allowed_modes = tuple(str(item) for item in (modes or PANEL_CHROME_MODES))
    mode = _weighted_choice(
        values=allowed_modes,
        weights=weights or DEFAULT_PANEL_CHROME_WEIGHTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_chrome",
    )
    probabilities = {
        str(item): float((weights or DEFAULT_PANEL_CHROME_WEIGHTS).get(str(item), 1.0))
        for item in allowed_modes
    }
    total = sum(max(0.0, value) for value in probabilities.values())
    if total <= 0.0:
        probabilities = {
            str(item): 1.0 / float(len(allowed_modes)) for item in allowed_modes
        }
    else:
        probabilities = {
            str(item): max(0.0, float(value)) / float(total)
            for item, value in probabilities.items()
        }
    return str(mode), {
        "mode": str(mode),
        "available_modes": list(allowed_modes),
        "mode_probabilities": dict(probabilities),
    }


def make_panel_scene_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: PanelSceneStyle,
) -> tuple[Image.Image, dict[str, Any]]:
    """Create a shared style-coordinated panel-scene background canvas."""

    width = max(1, int(canvas_width))
    height = max(1, int(canvas_height))
    image = Image.new(
        "RGB", (width, height), tuple(int(value) for value in style.background_rgb)
    )
    draw = ImageDraw.Draw(image)
    accent = tuple(int(value) for value in style.background_accent_rgb)
    treatment = str(style.treatment)

    if treatment == "bare_canvas":
        pass
    elif treatment == "matte_sheet":
        draw.rectangle((0, 0, width - 1, height - 1), outline=accent, width=1)
    elif treatment == "thin_frame":
        draw.rectangle((14, 14, width - 15, height - 15), outline=accent, width=1)
    elif treatment == "soft_panel":
        draw.rounded_rectangle(
            (18, 18, width - 18, height - 18), radius=18, fill=style.panel_fill_rgb
        )
    elif treatment == "margin_sheet":
        draw.line([(34, 0), (34, height)], fill=accent, width=1)
        draw.line([(0, 38), (width, 38)], fill=accent, width=1)
    elif treatment == "dot_sheet":
        for y in range(24, height, 36):
            for x in range(24, width, 36):
                draw.point((x, y), fill=accent)
    elif treatment == "worksheet_panel":
        for y in range(28, height, 42):
            draw.line([(24, y), (width - 24, y)], fill=accent, width=1)
    elif treatment == "lab_panel":
        for x in range(18, width, 72):
            draw.line([(x, 0), (x, height)], fill=accent, width=1)
        draw.rounded_rectangle(
            (18, 18, min(width - 18, 190), 34), radius=8, fill=accent
        )
    elif treatment == "whiteboard_panel":
        draw.rounded_rectangle(
            (16, 16, width - 16, height - 16),
            radius=16,
            fill=style.panel_fill_rgb,
            outline=style.panel_border_rgb,
            width=2,
        )
        draw.line([(36, 44), (width - 36, 44)], fill=accent, width=2)
        for x in range(42, min(width - 42, 168), 34):
            draw.rounded_rectangle((x, 24, x + 22, 30), radius=4, fill=accent)
    elif treatment == "tile_tray":
        draw.rounded_rectangle(
            (18, 18, width - 18, height - 18),
            radius=22,
            fill=style.panel_fill_rgb,
            outline=style.panel_border_rgb,
            width=3,
        )
        for x in range(34, width - 18, 64):
            draw.line([(x, 22), (x, height - 22)], fill=accent, width=1)
        for y in range(34, height - 18, 64):
            draw.line([(22, y), (width - 22, y)], fill=accent, width=1)
    elif treatment == "notebook_grid":
        for y in range(24, height, 34):
            draw.line([(0, y), (width, y)], fill=accent, width=1)
        draw.line([(42, 0), (42, height)], fill=style.notebook_margin_rgb, width=1)
    elif treatment == "puzzle_card":
        for offset in range(-height, width, 46):
            draw.line([(offset, height), (offset + height, 0)], fill=accent, width=1)
    elif treatment == "inset_board":
        draw.rounded_rectangle(
            (14, 14, width - 14, height - 14), radius=18, outline=accent, width=2
        )
        draw.rounded_rectangle(
            (26, 26, width - 26, height - 26), radius=12, outline=accent, width=1
        )
    elif treatment == "tabletop_mat":
        for offset in range(-height, width, 56):
            draw.line([(offset, 0), (offset + height, height)], fill=accent, width=1)
        draw.rounded_rectangle(
            (20, 20, width - 20, height - 20),
            radius=20,
            outline=style.panel_border_rgb,
            width=2,
        )
    elif treatment == "corkboard_sheet":
        for y in range(18, height, 32):
            for x in range(18, width, 32):
                draw.point((x, y), fill=accent)
        draw.rectangle(
            (10, 10, width - 11, height - 11), outline=style.panel_border_rgb, width=2
        )
    elif treatment == "game_table":
        for x in range(0, width, 52):
            draw.line([(x, 0), (x + height, height)], fill=accent, width=1)
        draw.rounded_rectangle(
            (12, 12, width - 12, height - 12),
            radius=24,
            outline=style.panel_border_rgb,
            width=3,
        )
    elif treatment == "clipboard_sheet":
        draw.rounded_rectangle(
            (width // 2 - 80, 10, width // 2 + 80, 34), radius=10, fill=accent
        )
        draw.rounded_rectangle(
            (width // 2 - 42, 4, width // 2 + 42, 22),
            radius=8,
            fill=style.panel_border_rgb,
        )
    elif treatment == "index_card":
        for y in range(34, height, 30):
            draw.line([(28, y), (width - 28, y)], fill=accent, width=1)
        draw.rectangle((0, 0, width, 18), fill=accent)
    elif treatment == "printout_panel":
        for y in range(18, height, 36):
            draw.line([(18, y), (width - 18, y)], fill=accent, width=1)
        for x in range(30, width, 120):
            draw.point((x, (x * 7) % max(1, height)), fill=accent)
    elif treatment == "arcade_screen":
        for y in range(0, height, 4):
            draw.line([(0, y), (width, y)], fill=accent, width=1)
        draw.rounded_rectangle(
            (10, 10, width - 10, height - 10),
            radius=24,
            outline=style.panel_border_rgb,
            width=3,
        )
    elif treatment == "terminal_screen":
        for y in range(0, height, 6):
            draw.line([(0, y), (width, y)], fill=accent, width=1)
        draw.rectangle((0, 0, width, 24), fill=accent)
        for x in range(12, min(width, 84), 22):
            draw.ellipse((x, 8, x + 8, 16), fill=style.panel_border_rgb)
    elif treatment == "dark_game_table":
        for offset in range(-height, width, 58):
            draw.line([(offset, 0), (offset + height, height)], fill=accent, width=1)
        draw.rounded_rectangle(
            (16, 16, width - 16, height - 16),
            radius=28,
            outline=style.panel_border_rgb,
            width=4,
        )
        draw.rounded_rectangle(
            (30, 30, width - 30, height - 30),
            radius=20,
            outline=accent,
            width=1,
        )
    elif treatment == "neon_grid_screen":
        for x in range(0, width, 54):
            draw.line([(x, 0), (x, height)], fill=accent, width=1)
        for y in range(0, height, 54):
            draw.line([(0, y), (width, y)], fill=accent, width=1)
        draw.rounded_rectangle(
            (12, 12, width - 12, height - 12),
            radius=22,
            outline=style.panel_border_rgb,
            width=3,
        )
    elif treatment == "scoreboard_panel":
        for y in range(0, height, 10):
            draw.line([(0, y), (width, y)], fill=accent, width=1)
        draw.rounded_rectangle(
            (14, 14, width - 14, height - 14),
            radius=12,
            outline=style.panel_border_rgb,
            width=3,
        )
        for x in range(28, width - 24, 56):
            draw.rectangle((x, 22, x + 18, 28), fill=accent)

    metadata = {
        "enabled": True,
        "selected_style": f"panel_scene_style:{style.style_pack}",
        "available_treatments": list(PANEL_SCENE_TREATMENTS),
        "available_palettes": sorted(PANEL_SCENE_PALETTES.keys()),
        "style_spec": {
            "kind": "panel_scene_style",
            "treatment": str(style.treatment),
            "palette_id": str(style.palette_id),
            "style_pack": str(style.style_pack),
            "background_rgb": list(style.background_rgb),
            "background_accent_rgb": list(style.background_accent_rgb),
            "min_background_state_lab_distance": panel_scene_style_metadata(style)[
                "min_background_state_lab_distance"
            ],
            "min_background_state_rgb_distance": panel_scene_style_metadata(style)[
                "min_background_state_rgb_distance"
            ],
        },
    }
    return image, metadata


def draw_panel_scene_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    style: PanelSceneStyle,
    radius: int,
    border_width: int,
) -> None:
    """Draw reusable full treatment-specific panel chrome for one shared treatment."""

    treatment = str(style.treatment)
    effective_radius = max(0, int(radius))
    if treatment == "bare_canvas":
        return
    if treatment in {
        "arcade_screen",
        "terminal_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
    }:
        effective_radius = max(effective_radius, 18)
    elif treatment in {"printout_panel", "plain_sheet", "matte_sheet", "thin_frame"}:
        effective_radius = max(3, min(effective_radius, 8))

    draw_rounded_rect(
        draw,
        bbox=bbox,
        radius=effective_radius,
        fill=style.panel_fill_rgb,
        outline=style.panel_border_rgb,
        width=int(border_width),
    )
    if treatment in {"plain_sheet", "matte_sheet", "soft_panel"}:
        return
    if treatment == "thin_frame":
        draw.rectangle(
            (bbox[0] + 7, bbox[1] + 7, bbox[2] - 7, bbox[3] - 7),
            outline=style.panel_accent_rgb,
            width=1,
        )
    elif treatment == "margin_sheet":
        draw.line(
            [(bbox[0] + 26, bbox[1] + 10), (bbox[0] + 26, bbox[3] - 10)],
            fill=style.notebook_margin_rgb,
            width=1,
        )
        draw.line(
            [(bbox[0] + 10, bbox[1] + 30), (bbox[2] - 10, bbox[1] + 30)],
            fill=style.notebook_line_rgb,
            width=1,
        )
    elif treatment == "dot_sheet":
        for y in range(bbox[1] + 18, bbox[3] - 10, 28):
            for x in range(bbox[0] + 18, bbox[2] - 10, 28):
                draw.point((x, y), fill=style.panel_accent_rgb)
    elif treatment == "lab_panel":
        strip = (bbox[0] + 16, bbox[1] + 12, bbox[2] - 16, bbox[1] + 28)
        draw.rounded_rectangle(strip, radius=8, fill=style.panel_accent_rgb)
        for x in range(bbox[0] + 28, bbox[2] - 20, 54):
            draw.line(
                [(x, bbox[1] + 36), (x, min(bbox[3] - 18, bbox[1] + 46))],
                fill=style.panel_accent_rgb,
                width=2,
            )
    elif treatment == "whiteboard_panel":
        draw.line(
            [(bbox[0] + 18, bbox[1] + 30), (bbox[2] - 18, bbox[1] + 30)],
            fill=style.panel_accent_rgb,
            width=2,
        )
        tray_y = bbox[3] - 18
        draw.rounded_rectangle(
            (bbox[0] + 34, tray_y, bbox[2] - 34, tray_y + 5),
            radius=3,
            fill=style.panel_accent_rgb,
        )
    elif treatment == "tile_tray":
        draw.rounded_rectangle(
            (bbox[0] + 9, bbox[1] + 9, bbox[2] - 9, bbox[3] - 9),
            radius=max(4, effective_radius - 7),
            outline=style.panel_accent_rgb,
            width=1,
        )
        for x in range(bbox[0] + 28, bbox[2] - 20, 48):
            draw.line(
                [(x, bbox[1] + 12), (x, bbox[3] - 12)],
                fill=style.panel_accent_rgb,
                width=1,
            )
        for y in range(bbox[1] + 28, bbox[3] - 20, 48):
            draw.line(
                [(bbox[0] + 12, y), (bbox[2] - 12, y)],
                fill=style.panel_accent_rgb,
                width=1,
            )
    elif treatment == "notebook_grid":
        margin_x = int(bbox[0] + 34)
        draw.line(
            [(margin_x, bbox[1] + 14), (margin_x, bbox[3] - 14)],
            fill=style.notebook_margin_rgb,
            width=1,
        )
        for y in range(int(bbox[1]) + 30, int(bbox[3]) - 12, 30):
            draw.line(
                [(bbox[0] + 12, y), (bbox[2] - 12, y)],
                fill=style.notebook_line_rgb,
                width=1,
            )
    elif treatment == "puzzle_card":
        inner = (bbox[0] + 12, bbox[1] + 12, bbox[2] - 12, bbox[3] - 12)
        draw.rounded_rectangle(
            inner,
            radius=max(4, effective_radius - 8),
            outline=style.panel_accent_rgb,
            width=2,
        )
        tab = (
            bbox[0] + 24,
            bbox[1] + 10,
            min(bbox[2] - 24, bbox[0] + 142),
            bbox[1] + 28,
        )
        draw.rounded_rectangle(tab, radius=8, fill=style.panel_accent_rgb)
    elif treatment == "inset_board":
        draw.rounded_rectangle(
            (bbox[0] + 9, bbox[1] + 9, bbox[2] - 9, bbox[3] - 9),
            radius=max(4, effective_radius - 7),
            outline=style.panel_accent_rgb,
            width=2,
        )
    elif treatment == "tabletop_mat":
        for offset in range(bbox[0] - (bbox[3] - bbox[1]), bbox[2], 42):
            draw.line(
                [(offset, bbox[1] + 8), (offset + (bbox[3] - bbox[1]), bbox[3] - 8)],
                fill=style.panel_accent_rgb,
                width=1,
            )
    elif treatment == "corkboard_sheet":
        for y in range(bbox[1] + 18, bbox[3] - 12, 34):
            for x in range(bbox[0] + 18, bbox[2] - 12, 34):
                draw.point((x, y), fill=style.panel_accent_rgb)
    elif treatment == "game_table":
        draw.rounded_rectangle(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[3] - 10),
            radius=max(4, effective_radius - 8),
            outline=style.panel_accent_rgb,
            width=2,
        )
    elif treatment == "clipboard_sheet":
        draw.rounded_rectangle(
            (bbox[0] + 24, bbox[1] + 10, bbox[2] - 24, bbox[1] + 30),
            radius=8,
            fill=style.panel_accent_rgb,
        )
    elif treatment == "index_card":
        for y in range(bbox[1] + 30, bbox[3] - 14, 28):
            draw.line(
                [(bbox[0] + 12, y), (bbox[2] - 12, y)],
                fill=style.notebook_line_rgb,
                width=1,
            )
    elif treatment == "printout_panel":
        draw.rectangle(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[3] - 10),
            outline=style.panel_accent_rgb,
            width=1,
        )
    elif treatment == "arcade_screen":
        draw.rounded_rectangle(
            (bbox[0] + 9, bbox[1] + 9, bbox[2] - 9, bbox[3] - 9),
            radius=max(8, effective_radius - 8),
            outline=style.panel_accent_rgb,
            width=2,
        )
    elif treatment == "terminal_screen":
        draw.rectangle(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[1] + 28),
            fill=style.panel_accent_rgb,
        )
    elif treatment == "dark_game_table":
        draw.rounded_rectangle(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[3] - 10),
            radius=max(8, effective_radius - 8),
            outline=style.panel_accent_rgb,
            width=2,
        )
        for offset in range(bbox[0] - (bbox[3] - bbox[1]), bbox[2], 54):
            draw.line(
                [(offset, bbox[1] + 8), (offset + (bbox[3] - bbox[1]), bbox[3] - 8)],
                fill=style.panel_accent_rgb,
                width=1,
            )
    elif treatment == "neon_grid_screen":
        for x in range(bbox[0] + 20, bbox[2] - 12, 42):
            draw.line(
                [(x, bbox[1] + 10), (x, bbox[3] - 10)],
                fill=style.panel_accent_rgb,
                width=1,
            )
        for y in range(bbox[1] + 20, bbox[3] - 12, 42):
            draw.line(
                [(bbox[0] + 10, y), (bbox[2] - 10, y)],
                fill=style.panel_accent_rgb,
                width=1,
            )
        draw.rounded_rectangle(
            (bbox[0] + 8, bbox[1] + 8, bbox[2] - 8, bbox[3] - 8),
            radius=max(6, effective_radius - 8),
            outline=style.panel_accent_rgb,
            width=2,
        )
    elif treatment == "scoreboard_panel":
        draw.rectangle(
            (bbox[0] + 10, bbox[1] + 10, bbox[2] - 10, bbox[1] + 26),
            fill=style.panel_accent_rgb,
        )
        for x in range(bbox[0] + 18, bbox[2] - 20, 34):
            draw.rectangle(
                (x, bbox[1] + 32, x + 10, bbox[1] + 38), fill=style.panel_accent_rgb
            )
    else:
        underline_y = int(bbox[1] + 26)
        draw.line(
            [(bbox[0] + 18, underline_y), (bbox[2] - 18, underline_y)],
            fill=style.panel_accent_rgb,
            width=2,
        )


def draw_panel_plain_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    style: PanelSceneStyle,
    radius: int,
    border_width: int,
) -> None:
    """Draw a quiet backing panel without treatment-specific decoration."""

    treatment = str(style.treatment)
    effective_radius = max(0, int(radius))
    if treatment in {
        "arcade_screen",
        "terminal_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
    }:
        effective_radius = max(effective_radius, 14)
    elif treatment in {"printout_panel", "plain_sheet", "matte_sheet", "thin_frame"}:
        effective_radius = max(3, min(effective_radius, 8))
    draw_rounded_rect(
        draw,
        bbox=bbox,
        radius=effective_radius,
        fill=style.panel_fill_rgb,
        outline=style.panel_border_rgb,
        width=max(1, min(int(border_width), 1)),
    )


def draw_panel_chrome_by_mode(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    style: PanelSceneStyle,
    radius: int,
    border_width: int,
    mode: str,
) -> None:
    """Draw optional whole-panel chrome decoupled from cell treatment."""

    resolved_mode = str(mode)
    if resolved_mode == "none":
        return
    if resolved_mode == "plain_panel":
        draw_panel_plain_chrome(
            draw,
            bbox=bbox,
            style=style,
            radius=int(radius),
            border_width=int(border_width),
        )
        return
    draw_panel_scene_chrome(
        draw,
        bbox=bbox,
        style=style,
        radius=int(radius),
        border_width=int(border_width),
    )


def draw_panel_grid_cell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    fill: Sequence[int],
    style: PanelSceneStyle,
    outline: Sequence[int] | None = None,
    width: int = 1,
    selected: bool = False,
    selected_width: int = 3,
) -> None:
    """Draw one repeated-unit cell using a shared panel treatment."""

    x0, y0, x1, y1 = [int(value) for value in bbox]
    outline_rgb = tuple(int(value) for value in (outline or style.grid_rgb))
    fill_rgb = tuple(int(value) for value in fill)
    cell_w = max(1, x1 - x0)
    cell_h = max(1, y1 - y0)
    radius = max(3, int(min(cell_w, cell_h) * 0.10))
    treatment = str(style.treatment)

    dimensional_treatments = {
        "lab_panel",
        "whiteboard_panel",
        "tile_tray",
        "puzzle_card",
        "inset_board",
        "tabletop_mat",
        "game_table",
        "arcade_screen",
        "terminal_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
    }
    dark_screen_treatments = {
        "arcade_screen",
        "terminal_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
    }
    if treatment in dimensional_treatments:
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius + (2 if treatment in dark_screen_treatments else 0),
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(width),
        )
        inset = max(3, int(min(cell_w, cell_h) * 0.12))
        if treatment in {
            "whiteboard_panel",
            "tile_tray",
            "puzzle_card",
            "inset_board",
            "tabletop_mat",
            "game_table",
            "arcade_screen",
            "terminal_screen",
            "dark_game_table",
            "neon_grid_screen",
            "scoreboard_panel",
        }:
            draw.rounded_rectangle(
                (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                radius=max(2, radius - 2),
                outline=style.panel_accent_rgb,
                width=1,
            )
        else:
            draw.line(
                [(x0 + inset, y0 + inset), (x1 - inset, y0 + inset)],
                fill=style.panel_accent_rgb,
                width=max(1, int(width)),
            )
    else:
        draw.rectangle(
            (x0, y0, x1, y1), fill=fill_rgb, outline=outline_rgb, width=int(width)
        )
        if treatment in {"notebook_grid", "index_card"}:
            for y in range(y0 + max(6, cell_h // 3), y1 - 3, max(10, cell_h // 3)):
                draw.line(
                    [(x0 + 4, y), (x1 - 4, y)], fill=style.notebook_line_rgb, width=1
                )
        elif treatment not in {
            "bare_canvas",
            "plain_sheet",
            "matte_sheet",
            "soft_panel",
        }:
            draw.line(
                [(x0 + 3, y0 + 3), (x1 - 3, y0 + 3)],
                fill=style.panel_accent_rgb,
                width=1,
            )

    if selected:
        draw.rectangle(
            (x0, y0, x1, y1), outline=style.mark_rgb, width=max(1, int(selected_width))
        )


def draw_panel_option_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    style: PanelSceneStyle,
    fill: Sequence[int] | None = None,
    radius: int = 14,
    border_width: int = 2,
) -> None:
    """Draw reusable option-card chrome for panel-like scenes."""

    x0, y0, x1, y1 = [int(value) for value in bbox]
    fill_rgb = tuple(int(value) for value in (fill or style.option_fill_rgb))
    treatment = str(style.treatment)
    effective_radius = max(3, int(radius))
    if treatment in {"printout_panel", "plain_sheet", "matte_sheet", "thin_frame"}:
        effective_radius = min(effective_radius, 8)

    draw_rounded_rect(
        draw,
        bbox=(x0, y0, x1, y1),
        radius=effective_radius,
        fill=fill_rgb,
        outline=style.panel_border_rgb,
        width=int(border_width),
    )
    if treatment in {"bare_canvas", "plain_sheet", "matte_sheet", "soft_panel"}:
        return
    if treatment == "thin_frame":
        draw.rectangle(
            (x0 + 7, y0 + 7, x1 - 7, y1 - 7), outline=style.panel_accent_rgb, width=1
        )
    elif treatment in {"margin_sheet", "dot_sheet"}:
        draw.line(
            [(x0 + 12, y0 + 18), (x1 - 12, y0 + 18)],
            fill=style.panel_accent_rgb,
            width=1,
        )
    elif treatment == "lab_panel":
        draw.rounded_rectangle(
            (x0 + 10, y0 + 8, x1 - 10, y0 + 20), radius=6, fill=style.panel_accent_rgb
        )
    elif treatment in {"notebook_grid", "index_card"}:
        margin_x = x0 + 26
        draw.line(
            [(margin_x, y0 + 8), (margin_x, y1 - 8)],
            fill=style.notebook_margin_rgb,
            width=1,
        )
        for y in range(y0 + 26, y1 - 8, 22):
            draw.line([(x0 + 8, y), (x1 - 8, y)], fill=style.notebook_line_rgb, width=1)
    elif treatment in {
        "whiteboard_panel",
        "tile_tray",
        "puzzle_card",
        "inset_board",
        "tabletop_mat",
        "game_table",
        "arcade_screen",
        "terminal_screen",
        "dark_game_table",
        "neon_grid_screen",
        "scoreboard_panel",
    }:
        inset = 7
        draw.rounded_rectangle(
            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            radius=max(4, effective_radius - 5),
            outline=style.panel_accent_rgb,
            width=2,
        )
    elif treatment == "corkboard_sheet":
        for px in (x0 + 16, x1 - 20):
            draw.ellipse((px, y0 + 10, px + 8, y0 + 18), fill=style.panel_accent_rgb)
    elif treatment == "clipboard_sheet":
        draw.rounded_rectangle(
            (x0 + 12, y0 + 8, x1 - 12, y0 + 20), radius=6, fill=style.panel_accent_rgb
        )
    elif treatment == "printout_panel":
        draw.rectangle(
            (x0 + 8, y0 + 8, x1 - 8, y1 - 8), outline=style.panel_accent_rgb, width=1
        )
    else:
        draw.line(
            [(x0 + 12, y0 + 18), (x1 - 12, y0 + 18)],
            fill=style.panel_accent_rgb,
            width=1,
        )


__all__ = [
    "DEFAULT_PANEL_SCENE_STYLE",
    "DEFAULT_PANEL_CHROME_WEIGHTS",
    "PANEL_CHROME_MODES",
    "PANEL_SCENE_TREATMENTS",
    "PanelSceneStyle",
    "draw_panel_chrome_by_mode",
    "draw_panel_grid_cell",
    "draw_panel_option_card",
    "draw_panel_plain_chrome",
    "draw_panel_scene_chrome",
    "make_panel_scene_background",
    "panel_scene_style_metadata",
    "resolve_panel_chrome_mode",
    "resolve_panel_scene_style",
]
