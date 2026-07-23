"""Style and render-parameter helpers for symbolic chemical equations."""

from __future__ import annotations

from typing import Any, Mapping

from .state import ChemicalEquationRenderParams


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> ChemicalEquationRenderParams:
    """Resolve chemical-equation rendering dimensions from config and params."""

    merged = {**dict(render_defaults), **dict(params)}
    return ChemicalEquationRenderParams(
        canvas_width=int(merged.get("canvas_width", 1240)),
        canvas_height=int(merged.get("canvas_height", 820)),
        panel_left_px=int(merged.get("panel_left_px", 68)),
        panel_top_px=int(merged.get("panel_top_px", 72)),
        panel_width_px=int(merged.get("panel_width_px", 1104)),
        panel_height_px=int(merged.get("panel_height_px", 676)),
        coefficient_box_width_px=int(merged.get("coefficient_box_width_px", 48)),
        coefficient_box_height_px=int(merged.get("coefficient_box_height_px", 56)),
        molecule_card_width_px=int(merged.get("molecule_card_width_px", 160)),
        molecule_card_height_px=int(merged.get("molecule_card_height_px", 214)),
        term_gap_px=int(merged.get("term_gap_px", 8)),
        operator_gap_px=int(merged.get("operator_gap_px", 48)),
        card_corner_radius_px=int(merged.get("card_corner_radius_px", 12)),
        card_border_width_px=int(merged.get("card_border_width_px", 2)),
        coefficient_font_size_px=int(merged.get("coefficient_font_size_px", 28)),
        atom_font_size_px=int(merged.get("atom_font_size_px", 15)),
        option_label_font_size_px=int(merged.get("option_label_font_size_px", 26)),
        option_text_font_size_px=int(merged.get("option_text_font_size_px", 27)),
        operator_font_size_px=int(merged.get("operator_font_size_px", 30)),
        atom_chip_diameter_px=int(merged.get("atom_chip_diameter_px", 33)),
    )


def chemical_variant_palette(scene_variant: str) -> dict[str, tuple[int, int, int]]:
    """Return non-semantic colors for one chemical-equation visual variant."""

    variant = str(scene_variant)
    if variant == "worksheet":
        return {
            "panel_fill": (255, 252, 242),
            "panel_border": (104, 116, 132),
            "card_fill": (255, 255, 250),
            "slot_fill": (247, 250, 255),
            "option_fill": (255, 254, 248),
            "option_border": (91, 108, 130),
            "text": (31, 38, 48),
            "muted_text": (83, 94, 110),
            "accent": (44, 113, 126),
            "line": (92, 109, 128),
        }
    if variant == "notebook_scan":
        return {
            "panel_fill": (250, 248, 236),
            "panel_border": (113, 112, 100),
            "card_fill": (255, 253, 243),
            "slot_fill": (248, 246, 236),
            "option_fill": (255, 253, 244),
            "option_border": (98, 100, 91),
            "text": (34, 36, 33),
            "muted_text": (86, 87, 80),
            "accent": (126, 72, 54),
            "line": (132, 131, 119),
        }
    return {
        "panel_fill": (247, 251, 250),
        "panel_border": (74, 99, 112),
        "card_fill": (255, 255, 255),
        "slot_fill": (240, 248, 250),
        "option_fill": (253, 255, 255),
        "option_border": (65, 95, 110),
        "text": (28, 36, 44),
        "muted_text": (80, 93, 104),
        "accent": (32, 117, 135),
        "line": (69, 100, 113),
    }


ELEMENT_COLORS: dict[str, tuple[int, int, int]] = {
    "Al": (179, 185, 191),
    "C": (88, 93, 101),
    "Cl": (95, 172, 96),
    "Fe": (179, 106, 83),
    "H": (236, 241, 246),
    "K": (151, 113, 198),
    "Mg": (132, 169, 178),
    "N": (89, 124, 206),
    "Na": (235, 189, 84),
    "O": (220, 81, 80),
    "P": (224, 143, 72),
}


__all__ = [
    "ELEMENT_COLORS",
    "chemical_variant_palette",
    "resolve_render_params",
]
