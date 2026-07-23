"""Style helpers for symbolic abacus rendering."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.scene_style import SymbolicSceneStyle

from .state import AbacusOptionPanelRenderParams, AbacusReadoutRenderParams


def resolve_readout_render_params(defaults: Mapping[str, Any]) -> AbacusReadoutRenderParams:
    """Resolve readout render params from scene config defaults."""

    return AbacusReadoutRenderParams(
        canvas_width=int(defaults.get("canvas_width", 980)),
        canvas_height=int(defaults.get("canvas_height", 760)),
        panel_width_px=int(defaults.get("panel_width_px", 800)),
        panel_height_px=int(defaults.get("panel_height_px", 540)),
        panel_corner_radius_px=int(defaults.get("panel_corner_radius_px", 24)),
        frame_width_px=int(defaults.get("frame_width_px", 8)),
        rod_width_px=int(defaults.get("rod_width_px", 5)),
        beam_height_px=int(defaults.get("beam_height_px", 22)),
        bead_width_px=int(defaults.get("bead_width_px", 58)),
        bead_height_px=int(defaults.get("bead_height_px", 34)),
        title_font_size_px=int(defaults.get("title_font_size_px", 25)),
        label_font_size_px=int(defaults.get("label_font_size_px", 23)),
        small_font_size_px=int(defaults.get("small_font_size_px", 16)),
        readout_option_card_width_px=int(defaults.get("readout_option_card_width_px", 130)),
        readout_option_card_height_px=int(defaults.get("readout_option_card_height_px", 58)),
        readout_option_card_gap_px=int(defaults.get("readout_option_card_gap_px", 12)),
        readout_option_card_margin_top_px=int(defaults.get("readout_option_card_margin_top_px", 24)),
        readout_option_label_font_size_px=int(defaults.get("readout_option_label_font_size_px", 22)),
        readout_option_value_font_size_px=int(defaults.get("readout_option_value_font_size_px", 24)),
    )


def resolve_option_panel_render_params(defaults: Mapping[str, Any]) -> AbacusOptionPanelRenderParams:
    """Resolve option-panel render params from scene config defaults."""

    return AbacusOptionPanelRenderParams(
        canvas_width=int(defaults.get("canvas_width", 1200)),
        canvas_height=int(defaults.get("canvas_height", 760)),
        option_card_width_px=int(defaults.get("option_card_width_px", 340)),
        option_card_height_px=int(defaults.get("option_card_height_px", 280)),
        option_card_gap_x_px=int(defaults.get("option_card_gap_x_px", 44)),
        option_card_gap_y_px=int(defaults.get("option_card_gap_y_px", 52)),
        option_card_corner_radius_px=int(defaults.get("option_card_corner_radius_px", 18)),
        option_label_font_size_px=int(defaults.get("option_label_font_size_px", 26)),
        option_place_label_font_size_px=int(defaults.get("option_place_label_font_size_px", 16)),
        option_bead_width_px=int(defaults.get("option_bead_width_px", 32)),
        option_bead_height_px=int(defaults.get("option_bead_height_px", 20)),
        option_rod_width_px=int(defaults.get("option_rod_width_px", 3)),
        option_beam_height_px=int(defaults.get("option_beam_height_px", 10)),
    )


def variant_colors(scene_variant: str, style: SymbolicSceneStyle) -> dict[str, tuple[int, int, int]]:
    """Return high-contrast abacus colors for one non-semantic scene variant."""

    if str(scene_variant) == "wood_frame":
        bead_fill = (202, 125, 64)
        return {
            "panel_fill": (246, 235, 212),
            "panel_outline": (116, 76, 43),
            "frame": (139, 91, 49),
            "rod": (83, 69, 57),
            "beam": (112, 72, 39),
            "active_bead": bead_fill,
            "inactive_bead": bead_fill,
            "bead_outline": (82, 58, 43),
            "label": (46, 38, 31),
            "guide": (211, 194, 161),
            "shadow": (204, 194, 178),
        }
    if str(scene_variant) == "worksheet":
        bead_fill = (71, 122, 184)
        return {
            "panel_fill": (252, 250, 242),
            "panel_outline": (128, 139, 151),
            "frame": (54, 67, 82),
            "rod": (74, 85, 99),
            "beam": (62, 73, 87),
            "active_bead": bead_fill,
            "inactive_bead": bead_fill,
            "bead_outline": (45, 57, 72),
            "label": (35, 45, 58),
            "guide": (220, 224, 230),
            "shadow": (209, 214, 220),
        }
    bead_fill = tuple(int(value) for value in style.panel_accent_rgb)
    return {
        "panel_fill": tuple(int(value) for value in style.panel_fill_rgb),
        "panel_outline": tuple(int(value) for value in style.panel_border_rgb),
        "frame": tuple(int(value) for value in style.text_rgb),
        "rod": (72, 79, 88),
        "beam": (46, 52, 60),
        "active_bead": bead_fill,
        "inactive_bead": bead_fill,
        "bead_outline": (49, 56, 66),
        "label": tuple(int(value) for value in style.text_rgb),
        "guide": tuple(int(value) for value in style.grid_rgb),
        "shadow": (208, 214, 222),
    }


__all__ = [
    "resolve_option_panel_render_params",
    "resolve_readout_render_params",
    "variant_colors",
]
