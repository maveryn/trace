"""Shared state constants for the styled table chart scene."""

from __future__ import annotations

from dataclasses import dataclass


DOMAIN = "charts"
SCENE_ID = "table"
SCENE_NAMESPACE = "charts.table"

SUPPORTED_TABLE_SCENE_VARIANTS = (
    "spreadsheet",
    "zebra",
    "ledger",
    "card_table",
)


@dataclass(frozen=True)
class TableDefaults:
    """Stable fallback defaults shared by table scene primitives."""

    row_count_min: int = 5
    row_count_max: int = 10
    numeric_column_count_min: int = 3
    numeric_column_count_max: int = 5
    value_min: int = 1
    value_max: int = 32
    canvas_width: int = 940
    canvas_height: int = 640
    table_margin_left_px: int = 52
    table_margin_right_px: int = 52
    table_margin_top_px: int = 56
    table_margin_bottom_px: int = 52
    row_label_width_fraction: float = 0.28
    row_label_min_width_px: int = 132
    label_font_size_px: int = 24
    value_font_size_px: int = 22
    border_width_px: int = 2
    grid_width_px: int = 1
    rounded_corner_radius_px: int = 18
    cell_padding_px: int = 14
    header_style: str = "light"
    frame_style: str = "flat"
    inner_rule_style: str = "solid"
    numeric_alignment: str = "center"
    shadow_offset_px: int = 5


__all__ = [
    "DOMAIN",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_TABLE_SCENE_VARIANTS",
    "TableDefaults",
]
