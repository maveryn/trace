"""Passive state containers for profile-card-grid page scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


Color = Tuple[int, int, int]
BBox = List[float]

SCENE_VARIANTS: Tuple[str, ...] = ("directory_grid", "compact_cards")
PROFILE_TEXT_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Role",
        (
            "Analyst",
            "Curator",
            "Planner",
            "Auditor",
            "Designer",
            "Coordinator",
            "Navigator",
            "Archivist",
            "Reviewer",
            "Strategist",
            "Operator",
            "Liaison",
        ),
    ),
    (
        "Region",
        (
            "North Pier",
            "West Loop",
            "Cedar Bay",
            "East Ridge",
            "South Gate",
            "River Bend",
            "Hill Yard",
            "Lake Point",
            "Mesa Park",
            "Harbor Row",
            "Pine Flats",
            "Stone Quay",
        ),
    ),
    (
        "Signal",
        ("Amber", "Cobalt", "Indigo", "Violet", "Copper", "Silver", "Teal", "Crimson", "Olive", "Maroon", "Saffron", "Azure"),
    ),
    ("Code", ("K-17", "M-42", "R-08", "T-63", "V-29", "X-54", "B-31", "D-76", "H-90", "J-21", "L-68", "Q-35")),
)
PROFILE_NUMERIC_FIELDS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("Score", (42, 55, 61, 68, 73, 81, 89, 94, 101, 108, 116, 123)),
    ("Cases", (6, 9, 12, 15, 18, 22, 26, 31, 35, 39, 44, 50)),
    ("Hours", (18, 24, 29, 34, 41, 47, 53, 58, 64, 70, 76, 82)),
)
PROFILE_FILTER_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Team", ("Atlas", "Beacon", "Cedar", "Delta", "Ember", "Harbor")),
    ("Track", ("North", "East", "South", "West", "Central", "Coastal")),
    ("Unit", ("Aster", "Birch", "Canyon", "Dune", "Elm", "Fjord")),
)
PROFILE_RANK_POSITION_SUPPORT: Tuple[int, ...] = (2, 3)
PROFILE_RANK_ORDINALS: Dict[int, str] = {
    2: "second",
    3: "third",
}
ACCENTS: Tuple[Color, ...] = (
    (57, 118, 172),
    (48, 143, 112),
    (188, 92, 77),
    (129, 101, 183),
    (198, 144, 62),
    (75, 129, 95),
    (173, 89, 132),
    (76, 130, 151),
)


@dataclass(frozen=True)
class ProfileCardGridDefaults:
    """Stable fallback defaults for profile-card-grid pages."""

    card_count_support: Tuple[int, ...] = (9, 12)
    rank_position_support: Tuple[int, ...] = PROFILE_RANK_POSITION_SUPPORT
    canvas_width: int = 1120
    canvas_height: int = 860
    outer_margin_px: int = 34
    header_height_px: int = 88
    gap_px: int = 14
    corner_radius_px: int = 14
    outline_width_px: int = 2
    title_font_size_px: int = 30
    subtitle_font_size_px: int = 17
    card_title_font_size_px: int = 22
    label_font_size_px: int = 15
    value_font_size_px: int = 17


@dataclass(frozen=True)
class ProfileCardRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    gap_px: int
    corner_radius_px: int
    outline_width_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    card_title_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int


@dataclass(frozen=True)
class ProfileCard:
    profile_id: str
    name: str
    fields: Dict[str, str]
    numeric_fields: Dict[str, int]
    accent_rgb: Color


@dataclass(frozen=True)
class ProfileCardGridSpec:
    cards: Tuple[ProfileCard, ...]
    title: str
    subtitle: str
    text_resource_metadata: Dict[str, Any]
    filter_field_label: str = ""


@dataclass(frozen=True)
class ProfileCardGridCase:
    spec: ProfileCardGridSpec
    scene_variant: str
    card_count: int
    card_count_support: Tuple[int, ...]
    card_count_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedProfileCardGrid:
    image: Any
    entities: Tuple[Dict[str, Any], ...]
    card_traces: Tuple[Dict[str, Any], ...]
    panel_bbox_px: BBox
    title_bbox_px: BBox
    layout_meta: Dict[str, Any]
    card_bboxes_px: Dict[str, BBox]
    name_bboxes_px: Dict[str, BBox]
    field_label_bboxes_px: Dict[str, Dict[str, BBox]]
    field_value_bboxes_px: Dict[str, Dict[str, BBox]]


@dataclass(frozen=True)
class RenderedProfileCardGridBundle:
    image: Any
    rendered_grid: RenderedProfileCardGrid
    render_params: ProfileCardRenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
