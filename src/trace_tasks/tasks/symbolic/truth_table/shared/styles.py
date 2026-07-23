"""Style and render-parameter helpers for symbolic truth tables."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state import TruthTableRenderParams


def _tuple_rgb(raw: Sequence[int] | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if raw is None:
        return tuple(fallback)
    values = tuple(int(value) for value in raw[:3])
    return values if len(values) == 3 else tuple(fallback)


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> TruthTableRenderParams:
    """Resolve truth-table rendering dimensions from scene defaults and params."""

    merged = {**dict(render_defaults), **dict(params)}
    return TruthTableRenderParams(
        canvas_width=int(merged.get("canvas_width", 1120)),
        canvas_height=int(merged.get("canvas_height", 780)),
        table_left_px=int(merged.get("table_left_px", 88)),
        table_top_px=int(merged.get("table_top_px", 102)),
        row_label_width_px=int(merged.get("row_label_width_px", 58)),
        variable_cell_width_px=int(merged.get("variable_cell_width_px", 66)),
        output_cell_width_px=int(merged.get("output_cell_width_px", 210)),
        compact_output_cell_width_px=int(merged.get("compact_output_cell_width_px", 82)),
        row_height_px=int(merged.get("row_height_px", 52)),
        header_height_px=int(merged.get("header_height_px", 72)),
        card_corner_radius_px=int(merged.get("card_corner_radius_px", 14)),
        card_border_width_px=int(merged.get("card_border_width_px", 2)),
        grid_line_width_px=int(merged.get("grid_line_width_px", 2)),
        expression_font_size_px=int(merged.get("expression_font_size_px", 24)),
        header_font_size_px=int(merged.get("header_font_size_px", 22)),
        cell_font_size_px=int(merged.get("cell_font_size_px", 24)),
        option_label_font_size_px=int(merged.get("option_label_font_size_px", 25)),
        pattern_font_size_px=int(merged.get("pattern_font_size_px", 28)),
        title_font_size_px=int(merged.get("title_font_size_px", 22)),
    )


def truth_table_variant_palette(scene_variant: str) -> dict[str, tuple[int, int, int]]:
    """Return non-semantic table colors for one visual variant."""

    variant = str(scene_variant)
    if variant == "notebook_table":
        return {
            "panel_fill": (255, 252, 238),
            "header_fill": (232, 241, 255),
            "cell_fill": (255, 254, 247),
            "grid": (92, 111, 143),
            "accent": (37, 105, 164),
            "text": (35, 40, 48),
        }
    if variant == "exam_scan":
        return {
            "panel_fill": (248, 249, 247),
            "header_fill": (232, 235, 233),
            "cell_fill": (252, 253, 250),
            "grid": (82, 86, 82),
            "accent": (115, 48, 48),
            "text": (32, 34, 32),
        }
    return {
        "panel_fill": (249, 251, 255),
        "header_fill": (230, 238, 250),
        "cell_fill": (255, 255, 255),
        "grid": (67, 82, 104),
        "accent": (35, 105, 135),
        "text": (28, 35, 45),
    }


__all__ = ["resolve_render_params", "truth_table_variant_palette"]
