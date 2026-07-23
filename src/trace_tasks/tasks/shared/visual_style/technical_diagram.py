"""Shared technical-diagram treatments for geometry and physics renderers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ..color_distance import color_distance, rgb_euclidean_distance
from ..text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    resolve_readable_text_style,
    text_legibility_summary,
)
from .metadata import color_separation_metadata


Color = Tuple[int, int, int]

TECHNICAL_DIAGRAM_FRAME_MODES: Tuple[str, ...] = ("none", "plain_outline", "matching_outline")
DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS = {
    "none": 0.5,
    "plain_outline": 0.25,
    "matching_outline": 0.25,
}


@dataclass(frozen=True)
class TechnicalDiagramPalette:
    """One non-semantic color palette for technical diagram scenes."""

    palette_id: str
    compatibility: Tuple[str, ...]
    canvas_rgb: Color
    canvas_accent_rgb: Color
    paper_rgb: Color
    panel_fill_rgb: Color
    panel_alt_fill_rgb: Color
    panel_border_rgb: Color
    grid_minor_rgb: Color
    grid_major_rgb: Color
    axis_rgb: Color
    guide_rgb: Color
    stroke_rgb: Color
    secondary_stroke_rgb: Color
    label_rgb: Color
    label_stroke_rgb: Color
    label_fill_rgb: Color
    label_border_rgb: Color
    accent_rgb: Color
    secondary_accent_rgb: Color
    fill_rgb: Color
    muted_fill_rgb: Color
    highlight_rgb: Color
    option_fill_rgb: Color


@dataclass(frozen=True)
class TechnicalDiagramTreatment:
    """One structural treatment for a math/physics technical diagram."""

    treatment_id: str
    compatibility: Tuple[str, ...]
    background_kind: str
    grid_kind: str
    grid_spacing_px: int
    major_every: int
    minor_line_width_px: int
    major_line_width_px: int
    axis_stroke_width_px: int
    panel_border_width_px: int
    label_stroke_width_px: int
    texture: str


@dataclass(frozen=True)
class TechnicalDiagramTheme:
    """One curated treatment/palette/frame combination for a technical profile."""

    theme_id: str
    technical_profile: str
    compatibility: Tuple[str, ...]
    treatment_id: str
    palette_id: str
    frame_mode: str


@dataclass(frozen=True)
class TechnicalDiagramStyle:
    """Resolved technical-diagram style after treatment and palette sampling."""

    treatment: str
    palette_id: str
    style_pack: str
    technical_profile: str | None
    theme_id: str | None
    theme_compatibility: Tuple[str, ...]
    background_kind: str
    grid_kind: str
    frame_mode: str
    canvas_rgb: Color
    canvas_accent_rgb: Color
    paper_rgb: Color
    panel_fill_rgb: Color
    panel_alt_fill_rgb: Color
    panel_border_rgb: Color
    grid_minor_rgb: Color
    grid_major_rgb: Color
    axis_rgb: Color
    guide_rgb: Color
    stroke_rgb: Color
    secondary_stroke_rgb: Color
    label_rgb: Color
    label_stroke_rgb: Color
    label_fill_rgb: Color
    label_border_rgb: Color
    accent_rgb: Color
    secondary_accent_rgb: Color
    fill_rgb: Color
    muted_fill_rgb: Color
    highlight_rgb: Color
    option_fill_rgb: Color
    grid_spacing_px: int
    major_every: int
    grid_minor_width_px: int
    grid_major_width_px: int
    axis_stroke_width_px: int
    panel_border_width_px: int
    label_stroke_width_px: int
    texture: str
    protected_colors: Tuple[Color, ...]


TECHNICAL_DIAGRAM_TREATMENTS: dict[str, TechnicalDiagramTreatment] = {
    "bare_canvas": TechnicalDiagramTreatment(
        "bare_canvas", ("light",), "bare_canvas", "none", 40, 0, 1, 1, 3, 2, 1, "none"
    ),
    "off_white_paper": TechnicalDiagramTreatment(
        "off_white_paper", ("light",), "paper_sheet", "none", 40, 0, 1, 1, 3, 2, 1, "paper_fiber"
    ),
    "graph_paper_light": TechnicalDiagramTreatment(
        "graph_paper_light", ("light",), "graph_paper", "square", 32, 5, 1, 2, 3, 2, 1, "none"
    ),
    "engineering_grid": TechnicalDiagramTreatment(
        "engineering_grid", ("light",), "engineering_grid", "square", 28, 4, 1, 2, 4, 2, 1, "none"
    ),
    "millimeter_paper": TechnicalDiagramTreatment(
        "millimeter_paper", ("light",), "millimeter_paper", "square", 20, 5, 1, 2, 3, 2, 1, "none"
    ),
    "blueprint_grid": TechnicalDiagramTreatment(
        "blueprint_grid", ("dark",), "blueprint", "square", 34, 4, 1, 2, 4, 2, 2, "blueprint"
    ),
    "dark_blueprint": TechnicalDiagramTreatment(
        "dark_blueprint", ("dark",), "dark_blueprint", "square", 30, 5, 1, 2, 4, 2, 2, "blueprint"
    ),
    "lab_notebook": TechnicalDiagramTreatment(
        "lab_notebook", ("light",), "lab_notebook", "lab_grid", 30, 5, 1, 2, 3, 2, 1, "paper_fiber"
    ),
    "ruled_notebook": TechnicalDiagramTreatment(
        "ruled_notebook", ("light",), "ruled_notebook", "ruled", 34, 0, 1, 1, 3, 2, 1, "paper_fiber"
    ),
    "worksheet_panel": TechnicalDiagramTreatment(
        "worksheet_panel", ("light",), "worksheet", "ruled", 42, 0, 1, 1, 3, 2, 1, "none"
    ),
    "textbook_figure": TechnicalDiagramTreatment(
        "textbook_figure", ("light",), "textbook_figure", "none", 40, 0, 1, 1, 3, 2, 1, "none"
    ),
    "presentation_slide": TechnicalDiagramTreatment(
        "presentation_slide", ("light",), "presentation_slide", "none", 48, 0, 1, 1, 3, 2, 1, "none"
    ),
    "whiteboard": TechnicalDiagramTreatment(
        "whiteboard", ("light",), "whiteboard", "none", 40, 0, 1, 1, 4, 3, 1, "wipe_marks"
    ),
    "clean_panel": TechnicalDiagramTreatment(
        "clean_panel", ("light",), "plain_sheet", "none", 44, 0, 1, 1, 3, 2, 1, "none"
    ),
    "chalkboard_dark": TechnicalDiagramTreatment(
        "chalkboard_dark", ("dark",), "chalkboard", "none", 42, 0, 1, 1, 4, 3, 2, "chalk_dust"
    ),
    "dark_slide": TechnicalDiagramTreatment(
        "dark_slide", ("dark",), "presentation_slide", "none", 48, 0, 1, 1, 3, 2, 2, "none"
    ),
    "slate_problem_box": TechnicalDiagramTreatment(
        "slate_problem_box", ("dark",), "exam_problem_box", "none", 40, 0, 1, 1, 3, 2, 2, "none"
    ),
    "drafting_vellum": TechnicalDiagramTreatment(
        "drafting_vellum", ("light",), "drafting_vellum", "crosshair", 44, 4, 1, 1, 3, 2, 1, "paper_fiber"
    ),
    "isometric_grid": TechnicalDiagramTreatment(
        "isometric_grid", ("light",), "isometric_grid", "isometric", 34, 4, 1, 2, 3, 2, 1, "none"
    ),
    "subtle_scan_sheet": TechnicalDiagramTreatment(
        "subtle_scan_sheet", ("light",), "scan_sheet", "none", 40, 0, 1, 1, 3, 2, 1, "scan"
    ),
    "exam_problem_box": TechnicalDiagramTreatment(
        "exam_problem_box", ("light",), "exam_problem_box", "none", 40, 0, 1, 1, 3, 2, 1, "none"
    ),
    "lab_card": TechnicalDiagramTreatment(
        "lab_card", ("light",), "lab_card", "lab_grid", 36, 4, 1, 2, 3, 2, 1, "none"
    ),
    "dark_lab_grid": TechnicalDiagramTreatment(
        "dark_lab_grid", ("dark",), "lab_card", "lab_grid", 36, 4, 1, 2, 4, 2, 2, "none"
    ),
    "monochrome_print": TechnicalDiagramTreatment(
        "monochrome_print", ("light",), "monochrome_print", "none", 40, 0, 1, 1, 3, 2, 1, "print"
    ),
}

TECHNICAL_DIAGRAM_TREATMENT_IDS: Tuple[str, ...] = tuple(TECHNICAL_DIAGRAM_TREATMENTS.keys())


TECHNICAL_DIAGRAM_PALETTES: dict[str, TechnicalDiagramPalette] = {
    "neutral_ink": TechnicalDiagramPalette(
        "neutral_ink",
        ("light",),
        (248, 250, 252),
        (230, 235, 242),
        (255, 255, 255),
        (255, 255, 255),
        (250, 252, 255),
        (80, 88, 102),
        (222, 228, 237),
        (192, 202, 216),
        (53, 61, 74),
        (128, 142, 164),
        (35, 42, 53),
        (74, 86, 104),
        (28, 33, 42),
        (255, 255, 255),
        (255, 255, 255),
        (92, 105, 126),
        (39, 116, 183),
        (86, 150, 123),
        (222, 232, 244),
        (236, 242, 249),
        (244, 207, 83),
        (255, 255, 255),
    ),
    "cool_gray_blue": TechnicalDiagramPalette(
        "cool_gray_blue",
        ("light",),
        (239, 244, 250),
        (218, 228, 240),
        (251, 253, 255),
        (252, 254, 255),
        (246, 251, 255),
        (69, 88, 112),
        (212, 224, 238),
        (181, 199, 221),
        (43, 76, 113),
        (99, 126, 159),
        (32, 55, 79),
        (61, 88, 118),
        (24, 42, 60),
        (255, 255, 255),
        (252, 254, 255),
        (76, 103, 137),
        (36, 119, 181),
        (85, 147, 154),
        (220, 235, 248),
        (234, 243, 250),
        (248, 206, 104),
        (250, 253, 255),
    ),
    "warm_paper_ink": TechnicalDiagramPalette(
        "warm_paper_ink",
        ("light",),
        (250, 247, 239),
        (232, 224, 206),
        (255, 252, 244),
        (255, 253, 247),
        (252, 247, 236),
        (101, 88, 70),
        (229, 220, 201),
        (203, 190, 163),
        (70, 61, 48),
        (133, 116, 89),
        (45, 39, 31),
        (91, 78, 60),
        (38, 33, 27),
        (255, 255, 255),
        (255, 254, 250),
        (126, 105, 73),
        (49, 117, 164),
        (117, 139, 84),
        (236, 226, 203),
        (246, 239, 224),
        (234, 196, 79),
        (255, 253, 247),
    ),
    "graphite_blue": TechnicalDiagramPalette(
        "graphite_blue",
        ("light",),
        (244, 247, 250),
        (226, 233, 241),
        (255, 255, 255),
        (253, 254, 255),
        (247, 250, 253),
        (70, 83, 101),
        (218, 225, 234),
        (188, 198, 213),
        (44, 65, 91),
        (111, 126, 147),
        (35, 45, 58),
        (65, 78, 97),
        (27, 35, 46),
        (255, 255, 255),
        (255, 255, 255),
        (91, 104, 124),
        (56, 101, 181),
        (85, 130, 144),
        (224, 234, 248),
        (240, 245, 251),
        (238, 195, 88),
        (255, 255, 255),
    ),
    "engineering_cyan": TechnicalDiagramPalette(
        "engineering_cyan",
        ("light",),
        (236, 248, 250),
        (208, 232, 238),
        (249, 254, 255),
        (250, 254, 255),
        (244, 252, 253),
        (43, 101, 116),
        (203, 229, 235),
        (164, 205, 216),
        (27, 81, 101),
        (79, 132, 146),
        (21, 61, 75),
        (52, 103, 119),
        (18, 48, 59),
        (255, 255, 255),
        (250, 255, 255),
        (64, 122, 139),
        (22, 137, 159),
        (96, 142, 103),
        (212, 238, 243),
        (233, 247, 250),
        (239, 203, 79),
        (247, 253, 254),
    ),
    "blueprint_white": TechnicalDiagramPalette(
        "blueprint_white",
        ("dark",),
        (19, 49, 82),
        (29, 68, 110),
        (22, 58, 96),
        (24, 60, 100),
        (19, 53, 91),
        (188, 216, 242),
        (55, 94, 136),
        (94, 139, 181),
        (225, 241, 255),
        (158, 202, 237),
        (235, 247, 255),
        (198, 222, 244),
        (244, 249, 255),
        (12, 36, 65),
        (26, 69, 112),
        (170, 208, 241),
        (244, 249, 255),
        (121, 206, 219),
        (62, 98, 138),
        (43, 77, 119),
        (246, 218, 116),
        (28, 68, 108),
    ),
    "blueprint_amber": TechnicalDiagramPalette(
        "blueprint_amber",
        ("dark",),
        (22, 45, 72),
        (34, 66, 99),
        (26, 52, 82),
        (27, 56, 88),
        (22, 49, 80),
        (218, 202, 150),
        (57, 88, 121),
        (104, 130, 154),
        (250, 229, 163),
        (203, 180, 119),
        (253, 236, 178),
        (222, 207, 156),
        (255, 240, 187),
        (15, 35, 58),
        (42, 70, 103),
        (198, 183, 135),
        (237, 186, 76),
        (125, 203, 204),
        (69, 93, 119),
        (48, 73, 101),
        (245, 202, 88),
        (31, 59, 91),
    ),
    "lab_teal": TechnicalDiagramPalette(
        "lab_teal",
        ("light",),
        (240, 250, 247),
        (215, 236, 229),
        (252, 255, 253),
        (252, 255, 254),
        (246, 253, 250),
        (51, 106, 96),
        (207, 231, 223),
        (173, 207, 198),
        (31, 83, 76),
        (82, 131, 119),
        (24, 62, 57),
        (57, 96, 89),
        (20, 50, 47),
        (255, 255, 255),
        (252, 255, 254),
        (73, 123, 112),
        (34, 139, 127),
        (83, 127, 185),
        (215, 238, 231),
        (235, 247, 243),
        (240, 198, 91),
        (250, 255, 253),
    ),
    "physics_orange": TechnicalDiagramPalette(
        "physics_orange",
        ("light",),
        (250, 247, 242),
        (234, 223, 208),
        (255, 253, 248),
        (255, 254, 250),
        (252, 248, 241),
        (105, 86, 65),
        (230, 220, 206),
        (204, 187, 166),
        (75, 61, 48),
        (136, 111, 82),
        (49, 41, 33),
        (94, 76, 58),
        (37, 32, 27),
        (255, 255, 255),
        (255, 254, 250),
        (132, 101, 67),
        (201, 111, 32),
        (69, 127, 163),
        (240, 226, 208),
        (248, 240, 229),
        (244, 199, 72),
        (255, 253, 247),
    ),
    "geometry_indigo": TechnicalDiagramPalette(
        "geometry_indigo",
        ("light",),
        (245, 244, 251),
        (226, 223, 241),
        (253, 252, 255),
        (253, 252, 255),
        (248, 247, 253),
        (82, 75, 119),
        (221, 219, 238),
        (194, 190, 219),
        (57, 51, 94),
        (119, 111, 151),
        (42, 38, 68),
        (76, 70, 111),
        (34, 31, 52),
        (255, 255, 255),
        (253, 252, 255),
        (102, 94, 137),
        (94, 86, 178),
        (76, 141, 142),
        (229, 226, 246),
        (241, 239, 250),
        (240, 204, 85),
        (252, 251, 255),
    ),
    "chalk_soft": TechnicalDiagramPalette(
        "chalk_soft",
        ("dark",),
        (30, 49, 46),
        (43, 65, 60),
        (35, 56, 52),
        (37, 60, 56),
        (31, 52, 49),
        (186, 211, 201),
        (62, 87, 82),
        (91, 123, 114),
        (232, 242, 235),
        (159, 190, 181),
        (238, 246, 240),
        (205, 224, 216),
        (247, 250, 245),
        (22, 38, 36),
        (45, 69, 65),
        (174, 207, 196),
        (225, 238, 229),
        (238, 210, 117),
        (68, 92, 86),
        (50, 73, 68),
        (242, 219, 130),
        (42, 64, 60),
    ),
    "whiteboard_marker": TechnicalDiagramPalette(
        "whiteboard_marker",
        ("light",),
        (247, 249, 249),
        (226, 232, 232),
        (255, 255, 255),
        (255, 255, 255),
        (249, 252, 252),
        (70, 88, 91),
        (222, 229, 230),
        (191, 205, 208),
        (35, 60, 67),
        (98, 128, 135),
        (22, 45, 52),
        (56, 84, 92),
        (20, 35, 39),
        (255, 255, 255),
        (255, 255, 255),
        (83, 110, 117),
        (31, 115, 180),
        (38, 142, 96),
        (223, 236, 244),
        (238, 246, 249),
        (245, 210, 84),
        (255, 255, 255),
    ),
    "green_grid": TechnicalDiagramPalette(
        "green_grid",
        ("light",),
        (242, 249, 244),
        (218, 234, 222),
        (252, 255, 252),
        (253, 255, 253),
        (247, 253, 248),
        (65, 99, 76),
        (211, 229, 215),
        (181, 207, 188),
        (42, 78, 55),
        (95, 132, 103),
        (31, 58, 41),
        (65, 96, 75),
        (24, 45, 34),
        (255, 255, 255),
        (252, 255, 252),
        (86, 124, 95),
        (45, 133, 75),
        (80, 124, 172),
        (224, 239, 227),
        (239, 248, 241),
        (235, 202, 88),
        (252, 255, 253),
    ),
    "violet_annotation": TechnicalDiagramPalette(
        "violet_annotation",
        ("light",),
        (248, 245, 252),
        (231, 224, 242),
        (255, 253, 255),
        (255, 254, 255),
        (250, 247, 253),
        (98, 74, 116),
        (228, 220, 238),
        (203, 189, 220),
        (68, 47, 88),
        (133, 108, 151),
        (50, 36, 66),
        (88, 66, 105),
        (39, 30, 50),
        (255, 255, 255),
        (255, 254, 255),
        (122, 91, 139),
        (134, 75, 176),
        (61, 135, 151),
        (236, 226, 246),
        (246, 240, 251),
        (240, 203, 88),
        (255, 253, 255),
    ),
    "sepia_print": TechnicalDiagramPalette(
        "sepia_print",
        ("light",),
        (248, 243, 234),
        (229, 217, 198),
        (255, 251, 242),
        (255, 252, 246),
        (250, 244, 232),
        (99, 83, 63),
        (225, 213, 193),
        (199, 181, 154),
        (71, 56, 40),
        (130, 108, 78),
        (49, 39, 29),
        (91, 73, 52),
        (37, 30, 23),
        (255, 255, 255),
        (255, 252, 247),
        (124, 98, 68),
        (137, 100, 56),
        (66, 117, 147),
        (236, 222, 197),
        (246, 237, 222),
        (227, 186, 72),
        (255, 251, 244),
    ),
    "slate_mint": TechnicalDiagramPalette(
        "slate_mint",
        ("light",),
        (236, 246, 243),
        (210, 228, 224),
        (250, 255, 253),
        (251, 255, 254),
        (245, 252, 250),
        (54, 87, 89),
        (205, 226, 222),
        (170, 202, 198),
        (31, 68, 72),
        (82, 123, 126),
        (24, 50, 54),
        (59, 91, 94),
        (19, 41, 44),
        (255, 255, 255),
        (250, 255, 253),
        (75, 113, 116),
        (30, 127, 116),
        (92, 113, 167),
        (215, 237, 232),
        (235, 247, 244),
        (239, 202, 87),
        (249, 254, 253),
    ),
    "navy_lab": TechnicalDiagramPalette(
        "navy_lab",
        ("dark",),
        (24, 32, 48),
        (36, 48, 68),
        (29, 39, 58),
        (31, 42, 63),
        (26, 36, 54),
        (178, 196, 224),
        (52, 65, 89),
        (85, 104, 135),
        (221, 235, 255),
        (148, 177, 214),
        (232, 241, 255),
        (194, 212, 236),
        (242, 247, 255),
        (17, 25, 39),
        (38, 50, 72),
        (166, 191, 224),
        (111, 192, 224),
        (238, 196, 90),
        (61, 77, 103),
        (43, 58, 81),
        (244, 211, 116),
        (34, 46, 67),
    ),
    "steel_crimson": TechnicalDiagramPalette(
        "steel_crimson",
        ("light",),
        (246, 247, 249),
        (226, 230, 235),
        (255, 255, 255),
        (254, 255, 255),
        (248, 250, 253),
        (79, 88, 100),
        (219, 225, 233),
        (190, 199, 211),
        (48, 61, 78),
        (108, 123, 142),
        (35, 45, 58),
        (72, 84, 99),
        (28, 36, 47),
        (255, 255, 255),
        (255, 255, 255),
        (93, 106, 123),
        (150, 57, 72),
        (44, 125, 153),
        (229, 234, 242),
        (241, 245, 249),
        (238, 199, 80),
        (253, 254, 255),
    ),
    "olive_field": TechnicalDiagramPalette(
        "olive_field",
        ("light",),
        (246, 248, 241),
        (226, 232, 210),
        (254, 255, 248),
        (254, 255, 250),
        (249, 252, 241),
        (82, 96, 64),
        (220, 228, 205),
        (191, 205, 171),
        (54, 73, 42),
        (109, 128, 84),
        (40, 55, 33),
        (76, 91, 58),
        (31, 43, 26),
        (255, 255, 255),
        (254, 255, 250),
        (100, 119, 76),
        (93, 130, 62),
        (45, 119, 160),
        (231, 239, 214),
        (243, 248, 233),
        (238, 202, 88),
        (253, 255, 248),
    ),
    "magenta_cyan": TechnicalDiagramPalette(
        "magenta_cyan",
        ("light",),
        (248, 246, 251),
        (229, 224, 238),
        (255, 254, 255),
        (255, 254, 255),
        (249, 247, 252),
        (84, 75, 102),
        (225, 220, 235),
        (197, 188, 217),
        (55, 47, 77),
        (113, 104, 137),
        (39, 34, 55),
        (77, 68, 99),
        (31, 28, 43),
        (255, 255, 255),
        (255, 254, 255),
        (101, 91, 124),
        (173, 63, 135),
        (30, 142, 165),
        (235, 226, 246),
        (245, 240, 250),
        (240, 204, 88),
        (254, 253, 255),
    ),
}


TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL = "analytical_diagram"
TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER = "graph_paper"
TECHNICAL_DIAGRAM_PROFILES: Tuple[str, ...] = (
    TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL,
    TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER,
)


def _theme(
    theme_id: str,
    technical_profile: str,
    compatibility: str,
    treatment_id: str,
    palette_id: str,
    frame_mode: str = "none",
) -> TechnicalDiagramTheme:
    return TechnicalDiagramTheme(
        theme_id=str(theme_id),
        technical_profile=str(technical_profile),
        compatibility=(str(compatibility),),
        treatment_id=str(treatment_id),
        palette_id=str(palette_id),
        frame_mode=str(frame_mode),
    )


TECHNICAL_DIAGRAM_THEMES: dict[str, TechnicalDiagramTheme] = {
    theme.theme_id: theme
    for theme in (
        _theme("analytical_plain_neutral", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "bare_canvas", "neutral_ink"),
        _theme("analytical_plain_cool", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "bare_canvas", "cool_gray_blue", "plain_outline"),
        _theme("analytical_paper_warm", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "off_white_paper", "warm_paper_ink"),
        _theme("analytical_paper_sepia", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "off_white_paper", "sepia_print", "matching_outline"),
        _theme("analytical_textbook_indigo", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "textbook_figure", "geometry_indigo"),
        _theme("analytical_textbook_graphite", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "textbook_figure", "graphite_blue", "plain_outline"),
        _theme("analytical_slide_blue", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "presentation_slide", "cool_gray_blue"),
        _theme("analytical_slide_steel", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "presentation_slide", "steel_crimson"),
        _theme("analytical_whiteboard_marker", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "whiteboard", "whiteboard_marker"),
        _theme("analytical_whiteboard_green", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "whiteboard", "green_grid"),
        _theme("analytical_clean_lab", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "clean_panel", "lab_teal", "plain_outline"),
        _theme("analytical_clean_violet", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "clean_panel", "violet_annotation"),
        _theme("analytical_scan_neutral", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "subtle_scan_sheet", "neutral_ink"),
        _theme("analytical_scan_olive", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "subtle_scan_sheet", "olive_field"),
        _theme("analytical_exam_graphite", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "exam_problem_box", "graphite_blue"),
        _theme("analytical_exam_physics", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "exam_problem_box", "physics_orange"),
        _theme("analytical_print_neutral", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "monochrome_print", "neutral_ink"),
        _theme("analytical_print_steel", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "monochrome_print", "steel_crimson"),
        _theme("analytical_clean_magenta", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "clean_panel", "magenta_cyan", "matching_outline"),
        _theme("analytical_paper_slate", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "light", "off_white_paper", "slate_mint", "plain_outline"),
        _theme("analytical_chalk_soft", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "dark", "chalkboard_dark", "chalk_soft"),
        _theme("analytical_chalk_blueprint", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "dark", "chalkboard_dark", "blueprint_white"),
        _theme("analytical_dark_slide_navy", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "dark", "dark_slide", "navy_lab"),
        _theme("analytical_slate_amber", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "dark", "slate_problem_box", "blueprint_amber"),
        _theme("analytical_slate_chalk", TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, "dark", "slate_problem_box", "chalk_soft", "plain_outline"),
        _theme("graph_square_neutral", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "graph_paper_light", "neutral_ink"),
        _theme("graph_square_green", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "graph_paper_light", "green_grid"),
        _theme("graph_square_indigo", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "graph_paper_light", "geometry_indigo"),
        _theme("graph_square_violet", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "graph_paper_light", "violet_annotation"),
        _theme("graph_square_orange", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "graph_paper_light", "physics_orange"),
        _theme("graph_engineering_cyan", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "engineering_grid", "engineering_cyan"),
        _theme("graph_engineering_graphite", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "engineering_grid", "graphite_blue"),
        _theme("graph_engineering_steel", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "engineering_grid", "steel_crimson"),
        _theme("graph_engineering_slate", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "engineering_grid", "slate_mint"),
        _theme("graph_millimeter_neutral", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "millimeter_paper", "neutral_ink"),
        _theme("graph_millimeter_cool", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "millimeter_paper", "cool_gray_blue"),
        _theme("graph_millimeter_sepia", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "millimeter_paper", "sepia_print"),
        _theme("graph_millimeter_olive", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "millimeter_paper", "olive_field"),
        _theme("graph_lab_notebook_lab", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_notebook", "lab_teal"),
        _theme("graph_lab_notebook_warm", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_notebook", "warm_paper_ink"),
        _theme("graph_lab_notebook_olive", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_notebook", "olive_field"),
        _theme("graph_lab_card_lab", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_card", "lab_teal"),
        _theme("graph_lab_card_slate", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_card", "slate_mint"),
        _theme("graph_lab_card_magenta", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_card", "magenta_cyan"),
        _theme("graph_lab_card_steel", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "light", "lab_card", "steel_crimson"),
        _theme("graph_blueprint_white", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "dark", "blueprint_grid", "blueprint_white"),
        _theme("graph_blueprint_amber", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "dark", "blueprint_grid", "blueprint_amber"),
        _theme("graph_dark_blueprint_white", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "dark", "dark_blueprint", "blueprint_white"),
        _theme("graph_dark_blueprint_navy", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "dark", "dark_blueprint", "navy_lab"),
        _theme("graph_dark_lab_navy", TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER, "dark", "dark_lab_grid", "navy_lab", "plain_outline"),
    )
}
TECHNICAL_DIAGRAM_PROFILE_THEME_IDS: dict[str, Tuple[str, ...]] = {
    profile: tuple(
        theme_id
        for theme_id, theme in TECHNICAL_DIAGRAM_THEMES.items()
        if str(theme.technical_profile) == str(profile)
    )
    for profile in TECHNICAL_DIAGRAM_PROFILES
}
TECHNICAL_DIAGRAM_THEME_IDS: Tuple[str, ...] = tuple(TECHNICAL_DIAGRAM_THEMES.keys())


DEFAULT_TECHNICAL_DIAGRAM_STYLE: TechnicalDiagramStyle


def _normalize_color(color: Sequence[int]) -> Color:
    if len(color) < 3:
        raise ValueError("RGB colors require three channels")
    return (
        max(0, min(255, int(color[0]))),
        max(0, min(255, int(color[1]))),
        max(0, min(255, int(color[2]))),
    )


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], *, b_weight: float) -> Color:
    weight = max(0.0, min(1.0, float(b_weight)))
    a = _normalize_color(color_a)
    b = _normalize_color(color_b)
    return (
        max(0, min(255, int(round((float(a[0]) * (1.0 - weight)) + (float(b[0]) * weight))))),
        max(0, min(255, int(round((float(a[1]) * (1.0 - weight)) + (float(b[1]) * weight))))),
        max(0, min(255, int(round((float(a[2]) * (1.0 - weight)) + (float(b[2]) * weight))))),
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
    total = float(sum(parsed))
    if total <= 0.0:
        parsed = [1.0 for _ in values]
        total = float(len(values))
    threshold = float(rng.random()) * total
    cursor = 0.0
    for value, weight in zip(values, parsed):
        cursor += float(weight)
        if threshold <= cursor:
            return str(value)
    return str(values[-1])


def _coerce_id_sequence(raw: Sequence[str] | str | None, *, default: Sequence[str]) -> Tuple[str, ...]:
    """Normalize optional string/list style-id controls."""

    if raw is None:
        return tuple(str(item) for item in default)
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else tuple(str(item) for item in default)
    return tuple(str(item) for item in raw)


def _treatment_modes(treatment: TechnicalDiagramTreatment) -> Tuple[str, ...]:
    return tuple(str(mode) for mode in treatment.compatibility)


def _palette_is_compatible(*, palette: TechnicalDiagramPalette, treatment: TechnicalDiagramTreatment) -> bool:
    treatment_modes = set(_treatment_modes(treatment))
    palette_modes = {str(item) for item in palette.compatibility}
    return bool(treatment_modes.intersection(palette_modes))


def _nonsemantic_palette_colors(palette: TechnicalDiagramPalette) -> Tuple[Color, ...]:
    return (
        palette.panel_border_rgb,
        palette.grid_minor_rgb,
        palette.grid_major_rgb,
        palette.axis_rgb,
        palette.guide_rgb,
        palette.stroke_rgb,
        palette.secondary_stroke_rgb,
        palette.label_rgb,
        palette.label_border_rgb,
        palette.accent_rgb,
        palette.secondary_accent_rgb,
        palette.fill_rgb,
        palette.muted_fill_rgb,
        palette.highlight_rgb,
    )


def _protected_color_distances(
    *,
    palette: TechnicalDiagramPalette,
    protected_colors: Sequence[Color],
) -> tuple[float, float]:
    protected = [_normalize_color(color) for color in protected_colors]
    compared = list(_nonsemantic_palette_colors(palette))
    if not protected or not compared:
        return (0.0, 0.0)
    lab_distances = [
        color_distance(color, protected_color, distance_space="lab")
        for color in compared
        for protected_color in protected
    ]
    rgb_distances = [
        rgb_euclidean_distance(color, protected_color)
        for color in compared
        for protected_color in protected
    ]
    return (round(float(min(lab_distances)), 3), round(float(min(rgb_distances)), 3))


def _normalize_technical_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = str(profile).strip().lower()
    if not normalized:
        return None
    if normalized not in set(TECHNICAL_DIAGRAM_PROFILES):
        raise ValueError(f"unknown technical diagram profile: {profile!r}")
    return normalized


def _theme_is_eligible(
    *,
    theme: TechnicalDiagramTheme,
    allow_dark: bool,
    require_grid: bool | None,
    requested_treatments: Sequence[str],
    requested_palettes: Sequence[str],
) -> bool:
    if theme.treatment_id not in TECHNICAL_DIAGRAM_TREATMENTS:
        return False
    if theme.palette_id not in TECHNICAL_DIAGRAM_PALETTES:
        return False
    if theme.treatment_id not in set(requested_treatments):
        return False
    if theme.palette_id not in set(requested_palettes):
        return False
    if not bool(allow_dark) and "dark" in set(str(item) for item in theme.compatibility):
        return False
    treatment = TECHNICAL_DIAGRAM_TREATMENTS[str(theme.treatment_id)]
    palette = TECHNICAL_DIAGRAM_PALETTES[str(theme.palette_id)]
    if not _palette_is_compatible(palette=palette, treatment=treatment):
        return False
    if require_grid is not None and (treatment.grid_kind != "none") != bool(require_grid):
        return False
    return True


def _resolve_profile_theme(
    *,
    instance_seed: int,
    namespace: str,
    theme_profile: str,
    themes: Sequence[str] | str | None,
    theme_weights: Mapping[str, float] | None,
    treatments: Sequence[str] | str | None,
    palettes: Sequence[str] | str | None,
    allow_dark: bool,
    require_grid: bool | None,
    protected_colors: Sequence[Color],
    min_protected_lab_distance: float,
) -> tuple[TechnicalDiagramTheme, dict[str, Any]]:
    profile_theme_ids = TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[str(theme_profile)]
    requested_theme_ids = _coerce_id_sequence(themes, default=profile_theme_ids)
    requested_treatments = _coerce_id_sequence(treatments, default=TECHNICAL_DIAGRAM_TREATMENT_IDS)
    requested_palettes = _coerce_id_sequence(palettes, default=tuple(sorted(TECHNICAL_DIAGRAM_PALETTES.keys())))
    candidate_theme_ids = [
        theme_id
        for theme_id in requested_theme_ids
        if theme_id in TECHNICAL_DIAGRAM_THEMES
        and TECHNICAL_DIAGRAM_THEMES[str(theme_id)].technical_profile == str(theme_profile)
        and _theme_is_eligible(
            theme=TECHNICAL_DIAGRAM_THEMES[str(theme_id)],
            allow_dark=bool(allow_dark),
            require_grid=require_grid,
            requested_treatments=requested_treatments,
            requested_palettes=requested_palettes,
        )
    ]
    if not candidate_theme_ids:
        candidate_theme_ids = [
            theme_id
            for theme_id in profile_theme_ids
            if _theme_is_eligible(
                theme=TECHNICAL_DIAGRAM_THEMES[str(theme_id)],
                allow_dark=bool(allow_dark),
                require_grid=require_grid,
                requested_treatments=TECHNICAL_DIAGRAM_TREATMENT_IDS,
                requested_palettes=tuple(sorted(TECHNICAL_DIAGRAM_PALETTES.keys())),
            )
        ]
    protected = tuple(_normalize_color(color) for color in (protected_colors or ()))
    protected_threshold = max(0.0, float(min_protected_lab_distance))
    protected_safe_theme_ids = [
        theme_id
        for theme_id in candidate_theme_ids
        if _protected_color_distances(
            palette=TECHNICAL_DIAGRAM_PALETTES[TECHNICAL_DIAGRAM_THEMES[str(theme_id)].palette_id],
            protected_colors=protected,
        )[0]
        >= protected_threshold
    ]
    protected_filter_fallback = False
    if protected and protected_safe_theme_ids:
        candidate_theme_ids = protected_safe_theme_ids
    elif protected and not protected_safe_theme_ids:
        protected_filter_fallback = True
    if not candidate_theme_ids:
        fallback_ids = [
            theme_id
            for theme_id in profile_theme_ids
            if _theme_is_eligible(
                theme=TECHNICAL_DIAGRAM_THEMES[str(theme_id)],
                allow_dark=False,
                require_grid=require_grid,
                requested_treatments=TECHNICAL_DIAGRAM_TREATMENT_IDS,
                requested_palettes=tuple(sorted(TECHNICAL_DIAGRAM_PALETTES.keys())),
            )
        ]
        candidate_theme_ids = fallback_ids or [profile_theme_ids[0]]
    theme_id = _weighted_choice(
        values=tuple(candidate_theme_ids),
        weights=theme_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.theme.{theme_profile}",
    )
    metadata = {
        "technical_profile": str(theme_profile),
        "requested_themes": list(requested_theme_ids),
        "eligible_themes": list(candidate_theme_ids),
        "available_theme_ids": list(profile_theme_ids),
        "requested_treatments": list(requested_treatments),
        "requested_palettes": list(requested_palettes),
        "theme_weights": {
            str(theme_id): float((theme_weights or {}).get(str(theme_id), 1.0))
            for theme_id in candidate_theme_ids
        },
        "protected_palette_filter_fallback": bool(protected_filter_fallback),
    }
    return TECHNICAL_DIAGRAM_THEMES[str(theme_id)], metadata


def _style_from_treatment_palette(
    *,
    treatment: TechnicalDiagramTreatment,
    palette: TechnicalDiagramPalette,
    frame_mode: str = "none",
    protected_colors: Sequence[Color],
    technical_profile: str | None = None,
    theme_id: str | None = None,
    theme_compatibility: Sequence[str] = (),
) -> TechnicalDiagramStyle:
    resolved_frame_mode = str(frame_mode)
    if resolved_frame_mode not in set(TECHNICAL_DIAGRAM_FRAME_MODES):
        resolved_frame_mode = "none"
    return TechnicalDiagramStyle(
        treatment=str(treatment.treatment_id),
        palette_id=str(palette.palette_id),
        style_pack=f"{str(treatment.treatment_id)}:{str(palette.palette_id)}:{resolved_frame_mode}",
        technical_profile=str(technical_profile) if technical_profile is not None else None,
        theme_id=str(theme_id) if theme_id is not None else None,
        theme_compatibility=tuple(str(item) for item in theme_compatibility),
        background_kind=str(treatment.background_kind),
        grid_kind=str(treatment.grid_kind),
        frame_mode=str(resolved_frame_mode),
        canvas_rgb=tuple(palette.canvas_rgb),
        canvas_accent_rgb=tuple(palette.canvas_accent_rgb),
        paper_rgb=tuple(palette.paper_rgb),
        panel_fill_rgb=tuple(palette.panel_fill_rgb),
        panel_alt_fill_rgb=tuple(palette.panel_alt_fill_rgb),
        panel_border_rgb=tuple(palette.panel_border_rgb),
        grid_minor_rgb=tuple(palette.grid_minor_rgb),
        grid_major_rgb=tuple(palette.grid_major_rgb),
        axis_rgb=tuple(palette.axis_rgb),
        guide_rgb=tuple(palette.guide_rgb),
        stroke_rgb=tuple(palette.stroke_rgb),
        secondary_stroke_rgb=tuple(palette.secondary_stroke_rgb),
        label_rgb=tuple(palette.label_rgb),
        label_stroke_rgb=tuple(palette.label_stroke_rgb),
        label_fill_rgb=tuple(palette.label_fill_rgb),
        label_border_rgb=tuple(palette.label_border_rgb),
        accent_rgb=tuple(palette.accent_rgb),
        secondary_accent_rgb=tuple(palette.secondary_accent_rgb),
        fill_rgb=tuple(palette.fill_rgb),
        muted_fill_rgb=tuple(palette.muted_fill_rgb),
        highlight_rgb=tuple(palette.highlight_rgb),
        option_fill_rgb=tuple(palette.option_fill_rgb),
        grid_spacing_px=max(4, int(treatment.grid_spacing_px)),
        major_every=max(0, int(treatment.major_every)),
        grid_minor_width_px=max(1, int(treatment.minor_line_width_px)),
        grid_major_width_px=max(1, int(treatment.major_line_width_px)),
        axis_stroke_width_px=max(1, int(treatment.axis_stroke_width_px)),
        panel_border_width_px=max(1, int(treatment.panel_border_width_px)),
        label_stroke_width_px=max(0, int(treatment.label_stroke_width_px)),
        texture=str(treatment.texture),
        protected_colors=tuple(_normalize_color(color) for color in protected_colors),
    )


DEFAULT_TECHNICAL_DIAGRAM_STYLE = _style_from_treatment_palette(
    treatment=TECHNICAL_DIAGRAM_TREATMENTS["bare_canvas"],
    palette=TECHNICAL_DIAGRAM_PALETTES["neutral_ink"],
    frame_mode="none",
    protected_colors=(),
    )


def _resolve_technical_text_legibility(
    style: TechnicalDiagramStyle,
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[TechnicalDiagramStyle, dict[str, Any]]:
    """Resolve readable non-semantic label colors for technical diagrams."""

    label_surfaces = (
        style.canvas_rgb,
        style.paper_rgb,
        style.panel_fill_rgb,
        style.panel_alt_fill_rgb,
        style.option_fill_rgb,
    )
    label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.text_legibility.label",
        role="read_required_diagram_label",
        surface_rgbs=label_surfaces,
        preferred_rgbs=(style.label_rgb, style.stroke_rgb, style.axis_rgb),
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        required=True,
    )
    adjusted = replace(
        style,
        label_rgb=tuple(label_style.fill_rgb),
        label_stroke_rgb=tuple(label_style.stroke_rgb),
        label_stroke_width_px=max(1, int(style.label_stroke_width_px)),
    )
    return adjusted, text_legibility_summary((label_style,))


def resolve_technical_diagram_style(
    *,
    instance_seed: int,
    namespace: str,
    theme_profile: str | None = None,
    themes: Sequence[str] | str | None = None,
    theme_weights: Mapping[str, float] | None = None,
    treatments: Sequence[str] | None = None,
    treatment_weights: Mapping[str, float] | None = None,
    palettes: Sequence[str] | None = None,
    palette_weights: Mapping[str, float] | None = None,
    frame_modes: Sequence[str] | str | None = None,
    frame_mode_weights: Mapping[str, float] | None = None,
    allow_dark: bool = False,
    require_grid: bool | None = None,
    protected_colors: Sequence[Color] | None = None,
    min_protected_lab_distance: float = 18.0,
) -> tuple[TechnicalDiagramStyle, dict[str, Any]]:
    """Resolve one non-semantic technical-diagram style for a scene."""

    normalized_theme_profile = _normalize_technical_profile(theme_profile)
    if normalized_theme_profile is not None:
        theme, theme_meta = _resolve_profile_theme(
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            theme_profile=str(normalized_theme_profile),
            themes=themes,
            theme_weights=theme_weights or {},
            treatments=treatments,
            palettes=palettes,
            allow_dark=bool(allow_dark),
            require_grid=require_grid,
            protected_colors=protected_colors or (),
            min_protected_lab_distance=float(min_protected_lab_distance),
        )
        treatment = TECHNICAL_DIAGRAM_TREATMENTS[str(theme.treatment_id)]
        palette = TECHNICAL_DIAGRAM_PALETTES[str(theme.palette_id)]
        protected = tuple(_normalize_color(color) for color in (protected_colors or ()))
        style = _style_from_treatment_palette(
            treatment=treatment,
            palette=palette,
            frame_mode=str(theme.frame_mode),
            protected_colors=protected,
            technical_profile=str(theme.technical_profile),
            theme_id=str(theme.theme_id),
            theme_compatibility=tuple(theme.compatibility),
        )
        style, text_legibility = _resolve_technical_text_legibility(
            style,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        metadata = technical_diagram_style_metadata(style)
        metadata["text_legibility"] = dict(text_legibility)
        metadata["selection"] = {
            "namespace": str(namespace),
            "allow_dark": bool(allow_dark),
            "require_grid": None if require_grid is None else bool(require_grid),
            "selected_theme_id": str(theme.theme_id),
            "selected_theme_compatibility": list(theme.compatibility),
            "selected_theme_treatment": str(theme.treatment_id),
            "selected_theme_palette": str(theme.palette_id),
            "selected_theme_frame_mode": str(theme.frame_mode),
            "requested_frame_modes": [],
            "eligible_frame_modes": [str(theme.frame_mode)],
            "frame_mode_weights": {str(theme.frame_mode): 1.0},
            "min_protected_lab_distance_required": max(0.0, float(min_protected_lab_distance)),
            **dict(theme_meta),
        }
        return style, metadata

    requested_treatments = _coerce_id_sequence(treatments, default=TECHNICAL_DIAGRAM_TREATMENT_IDS)
    valid_treatments = [
        treatment_id
        for treatment_id in requested_treatments
        if treatment_id in TECHNICAL_DIAGRAM_TREATMENTS
    ]
    if not bool(allow_dark):
        valid_treatments = [
            treatment_id
            for treatment_id in valid_treatments
            if "dark" not in set(TECHNICAL_DIAGRAM_TREATMENTS[str(treatment_id)].compatibility)
        ]
    if require_grid is not None:
        valid_treatments = [
            treatment_id
            for treatment_id in valid_treatments
            if (TECHNICAL_DIAGRAM_TREATMENTS[str(treatment_id)].grid_kind != "none") == bool(require_grid)
        ]
    if not valid_treatments:
        valid_treatments = ["graph_paper_light"] if bool(require_grid) else ["bare_canvas"]

    treatment_id = _weighted_choice(
        values=tuple(valid_treatments),
        weights=treatment_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.treatment",
    )
    treatment = TECHNICAL_DIAGRAM_TREATMENTS[str(treatment_id)]

    requested_palettes = _coerce_id_sequence(palettes, default=tuple(sorted(TECHNICAL_DIAGRAM_PALETTES.keys())))
    compatible_palette_ids = [
        palette_id
        for palette_id in requested_palettes
        if palette_id in TECHNICAL_DIAGRAM_PALETTES
        and _palette_is_compatible(palette=TECHNICAL_DIAGRAM_PALETTES[str(palette_id)], treatment=treatment)
    ]
    if not compatible_palette_ids:
        compatible_palette_ids = ["neutral_ink"]

    protected = tuple(_normalize_color(color) for color in (protected_colors or ()))
    protected_threshold = max(0.0, float(min_protected_lab_distance))
    protected_safe_palette_ids = [
        palette_id
        for palette_id in compatible_palette_ids
        if _protected_color_distances(
            palette=TECHNICAL_DIAGRAM_PALETTES[str(palette_id)],
            protected_colors=protected,
        )[0]
        >= protected_threshold
    ]
    protected_filter_fallback = False
    if protected and protected_safe_palette_ids:
        compatible_palette_ids = protected_safe_palette_ids
    elif protected and not protected_safe_palette_ids:
        protected_filter_fallback = True

    palette_id = _weighted_choice(
        values=tuple(compatible_palette_ids),
        weights=palette_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.palette.{treatment_id}",
    )
    palette = TECHNICAL_DIAGRAM_PALETTES.get(str(palette_id), TECHNICAL_DIAGRAM_PALETTES["neutral_ink"])
    requested_frame_modes = _coerce_id_sequence(frame_modes, default=TECHNICAL_DIAGRAM_FRAME_MODES)
    eligible_frame_modes = tuple(mode for mode in requested_frame_modes if mode in set(TECHNICAL_DIAGRAM_FRAME_MODES))
    if not eligible_frame_modes:
        eligible_frame_modes = ("none",)
    frame_mode = _weighted_choice(
        values=eligible_frame_modes,
        weights=frame_mode_weights or DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.frame_mode.{treatment_id}.{palette_id}",
    )
    style = _style_from_treatment_palette(
        treatment=treatment,
        palette=palette,
        frame_mode=str(frame_mode),
        protected_colors=protected,
    )
    style, text_legibility = _resolve_technical_text_legibility(
        style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    metadata = technical_diagram_style_metadata(style)
    metadata["text_legibility"] = dict(text_legibility)
    metadata["selection"] = {
        "namespace": str(namespace),
        "allow_dark": bool(allow_dark),
        "require_grid": None if require_grid is None else bool(require_grid),
        "requested_treatments": list(requested_treatments),
        "eligible_treatments": list(valid_treatments),
        "requested_palettes": list(requested_palettes),
        "eligible_palettes": list(compatible_palette_ids),
        "requested_frame_modes": list(requested_frame_modes),
        "eligible_frame_modes": list(eligible_frame_modes),
        "frame_mode_weights": {
            str(mode): float((frame_mode_weights or DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS).get(str(mode), 1.0))
            for mode in eligible_frame_modes
        },
        "protected_palette_filter_fallback": bool(protected_filter_fallback),
        "min_protected_lab_distance_required": float(protected_threshold),
    }
    return style, metadata


def technical_diagram_style_metadata(style: TechnicalDiagramStyle) -> dict[str, Any]:
    """Serialize one technical-diagram style into trace-friendly metadata."""

    protected = tuple(_normalize_color(color) for color in style.protected_colors)
    synthetic_palette = TechnicalDiagramPalette(
        palette_id=str(style.palette_id),
        compatibility=("light",),
        canvas_rgb=style.canvas_rgb,
        canvas_accent_rgb=style.canvas_accent_rgb,
        paper_rgb=style.paper_rgb,
        panel_fill_rgb=style.panel_fill_rgb,
        panel_alt_fill_rgb=style.panel_alt_fill_rgb,
        panel_border_rgb=style.panel_border_rgb,
        grid_minor_rgb=style.grid_minor_rgb,
        grid_major_rgb=style.grid_major_rgb,
        axis_rgb=style.axis_rgb,
        guide_rgb=style.guide_rgb,
        stroke_rgb=style.stroke_rgb,
        secondary_stroke_rgb=style.secondary_stroke_rgb,
        label_rgb=style.label_rgb,
        label_stroke_rgb=style.label_stroke_rgb,
        label_fill_rgb=style.label_fill_rgb,
        label_border_rgb=style.label_border_rgb,
        accent_rgb=style.accent_rgb,
        secondary_accent_rgb=style.secondary_accent_rgb,
        fill_rgb=style.fill_rgb,
        muted_fill_rgb=style.muted_fill_rgb,
        highlight_rgb=style.highlight_rgb,
        option_fill_rgb=style.option_fill_rgb,
    )
    min_protected_lab, min_protected_rgb = _protected_color_distances(
        palette=synthetic_palette,
        protected_colors=protected,
    )
    return {
        "kind": "technical_diagram_style",
        "treatment": str(style.treatment),
        "palette_id": str(style.palette_id),
        "technical_profile": style.technical_profile,
        "theme_id": style.theme_id,
        "theme_compatibility": list(style.theme_compatibility),
        "available_theme_ids": (
            list(TECHNICAL_DIAGRAM_PROFILE_THEME_IDS.get(str(style.technical_profile), ()))
            if style.technical_profile is not None
            else []
        ),
        "style_pack": str(style.style_pack),
        "frame_mode": str(style.frame_mode),
        "background_style": {
            "kind": str(style.background_kind),
            "texture": str(style.texture),
            "canvas_rgb": list(style.canvas_rgb),
            "canvas_accent_rgb": list(style.canvas_accent_rgb),
            "paper_rgb": list(style.paper_rgb),
        },
        "grid_style": {
            "kind": str(style.grid_kind),
            "spacing_px": int(style.grid_spacing_px),
            "major_every": int(style.major_every),
            "minor_line_rgb": list(style.grid_minor_rgb),
            "major_line_rgb": list(style.grid_major_rgb),
            "minor_line_width_px": int(style.grid_minor_width_px),
            "major_line_width_px": int(style.grid_major_width_px),
        },
        "frame_style": {
            "mode": str(style.frame_mode),
            "available_modes": list(TECHNICAL_DIAGRAM_FRAME_MODES),
            "default_mode_weights": dict(DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS),
            "plain_outline_rgb": list(style.panel_border_rgb),
            "matching_outline_rgb": list(style.accent_rgb),
            "width_px": int(style.panel_border_width_px),
        },
        "stroke_widths": {
            "axis_px": int(style.axis_stroke_width_px),
            "panel_border_px": int(style.panel_border_width_px),
            "label_stroke_px": int(style.label_stroke_width_px),
        },
        "roles_rgb": {
            "canvas": list(style.canvas_rgb),
            "canvas_accent": list(style.canvas_accent_rgb),
            "paper": list(style.paper_rgb),
            "panel_fill": list(style.panel_fill_rgb),
            "panel_alt_fill": list(style.panel_alt_fill_rgb),
            "panel_border": list(style.panel_border_rgb),
            "grid_minor": list(style.grid_minor_rgb),
            "grid_major": list(style.grid_major_rgb),
            "axis": list(style.axis_rgb),
            "guide": list(style.guide_rgb),
            "stroke": list(style.stroke_rgb),
            "secondary_stroke": list(style.secondary_stroke_rgb),
            "label": list(style.label_rgb),
            "label_stroke": list(style.label_stroke_rgb),
            "label_fill": list(style.label_fill_rgb),
            "label_border": list(style.label_border_rgb),
            "accent": list(style.accent_rgb),
            "secondary_accent": list(style.secondary_accent_rgb),
            "fill": list(style.fill_rgb),
            "muted_fill": list(style.muted_fill_rgb),
            "highlight": list(style.highlight_rgb),
            "option_fill": list(style.option_fill_rgb),
        },
        "protected_colors_rgb": [list(color) for color in protected],
        "contrast_checks": {
            **color_separation_metadata(
                anchor_rgb=style.canvas_rgb,
                compared_rgbs=(style.stroke_rgb, style.label_rgb, style.axis_rgb, style.accent_rgb),
                prefix="canvas_ink",
            ),
            **color_separation_metadata(
                anchor_rgb=style.paper_rgb,
                compared_rgbs=(style.stroke_rgb, style.label_rgb, style.axis_rgb, style.accent_rgb),
                prefix="paper_ink",
            ),
            "min_protected_nonsemantic_lab_distance": float(min_protected_lab),
            "min_protected_nonsemantic_rgb_distance": float(min_protected_rgb),
        },
        "available_treatments": list(TECHNICAL_DIAGRAM_TREATMENT_IDS),
        "available_palettes": sorted(TECHNICAL_DIAGRAM_PALETTES.keys()),
        "text_color_policy": "read_required_text_uses_random_nonsemantic_readable_ink",
    }


def _draw_square_grid(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    spacing = max(4, int(style.grid_spacing_px))
    major_every = max(0, int(style.major_every))
    for x in range(0, int(width), spacing):
        is_major = bool(major_every and (x // spacing) % major_every == 0)
        draw.line(
            [(x, 0), (x, int(height))],
            fill=style.grid_major_rgb if is_major else style.grid_minor_rgb,
            width=style.grid_major_width_px if is_major else style.grid_minor_width_px,
        )
    for y in range(0, int(height), spacing):
        is_major = bool(major_every and (y // spacing) % major_every == 0)
        draw.line(
            [(0, y), (int(width), y)],
            fill=style.grid_major_rgb if is_major else style.grid_minor_rgb,
            width=style.grid_major_width_px if is_major else style.grid_minor_width_px,
        )


def _draw_ruled_lines(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    spacing = max(10, int(style.grid_spacing_px))
    for y in range(spacing, int(height), spacing):
        draw.line([(24, y), (int(width) - 24, y)], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)


def _draw_dots(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    spacing = max(10, int(style.grid_spacing_px))
    for y in range(spacing, int(height), spacing):
        for x in range(spacing, int(width), spacing):
            draw.point((x, y), fill=style.grid_major_rgb)


def _draw_isometric_grid(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    spacing = max(12, int(style.grid_spacing_px))
    for x in range(-int(height), int(width) + int(height), spacing):
        draw.line([(x, 0), (x + int(height), int(height))], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)
        draw.line([(x, int(height)), (x + int(height), 0)], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)
    for y in range(0, int(height), spacing):
        draw.line([(0, y), (int(width), y)], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)


def _draw_crosshair_grid(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    spacing = max(12, int(style.grid_spacing_px))
    for x in range(spacing, int(width), spacing):
        draw.line([(x, 0), (x, int(height))], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)
    for y in range(spacing, int(height), spacing):
        draw.line([(0, y), (int(width), y)], fill=style.grid_minor_rgb, width=style.grid_minor_width_px)
    draw.line([(int(width) // 2, 0), (int(width) // 2, int(height))], fill=style.grid_major_rgb, width=style.grid_major_width_px)
    draw.line([(0, int(height) // 2), (int(width), int(height) // 2)], fill=style.grid_major_rgb, width=style.grid_major_width_px)


def _draw_texture(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    style: TechnicalDiagramStyle,
    instance_seed: int,
    namespace: str,
) -> None:
    texture = str(style.texture)
    if texture == "none":
        return
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=f"{namespace}.texture.{style.style_pack}")
    if texture == "paper_fiber":
        line_rgb = _blend_rgb(style.canvas_rgb, style.canvas_accent_rgb, b_weight=0.45)
        for y in range(11, int(height), 37):
            draw.line([(0, y), (int(width), y + int(rng.randrange(-1, 2)))], fill=line_rgb, width=1)
        return
    if texture == "scan":
        line_rgb = _blend_rgb(style.canvas_rgb, style.canvas_accent_rgb, b_weight=0.35)
        for y in range(0, int(height), 5):
            draw.line([(0, y), (int(width), y)], fill=line_rgb, width=1)
        speck_count = max(40, int((int(width) * int(height)) // 18000))
        for _ in range(speck_count):
            x = int(rng.randrange(max(1, int(width))))
            y = int(rng.randrange(max(1, int(height))))
            draw.point((x, y), fill=style.canvas_accent_rgb)
        return
    if texture == "print":
        for y in range(0, int(height), 7):
            draw.line([(0, y), (int(width), y)], fill=style.canvas_accent_rgb, width=1)
        return
    if texture == "blueprint":
        for x in range(0, int(width), 9):
            draw.point((x, (x * 17) % max(1, int(height))), fill=style.canvas_accent_rgb)
        return
    if texture == "chalk_dust":
        dust_count = max(80, int((int(width) * int(height)) // 12000))
        for _ in range(dust_count):
            x = int(rng.randrange(max(1, int(width))))
            y = int(rng.randrange(max(1, int(height))))
            color = _blend_rgb(style.canvas_rgb, style.canvas_accent_rgb, b_weight=float(rng.random()) * 0.45)
            draw.point((x, y), fill=color)
        return
    if texture == "wipe_marks":
        mark_rgb = _blend_rgb(style.canvas_rgb, style.canvas_accent_rgb, b_weight=0.28)
        for y in range(28, int(height), 74):
            draw.arc((20, y - 18, int(width) - 20, y + 28), start=4, end=176, fill=mark_rgb, width=1)


def _draw_frame(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: TechnicalDiagramStyle) -> None:
    """Draw optional canvas-level technical diagram frame chrome."""

    mode = str(style.frame_mode)
    if mode == "none":
        return
    inset = 10
    frame_width = max(1, int(style.panel_border_width_px))
    if mode == "plain_outline":
        draw.rectangle(
            (inset, inset, int(width) - inset - 1, int(height) - inset - 1),
            outline=style.panel_border_rgb,
            width=frame_width,
        )
        return
    draw.rectangle(
        (inset, inset, int(width) - inset - 1, int(height) - inset - 1),
        outline=style.panel_border_rgb,
        width=frame_width,
    )
    inner_inset = inset + 8
    draw.rectangle(
        (inner_inset, inner_inset, int(width) - inner_inset - 1, int(height) - inner_inset - 1),
        outline=style.canvas_accent_rgb,
        width=1,
    )
    corner = 32
    for x0, y0, sx, sy in (
        (inner_inset, inner_inset, 1, 1),
        (int(width) - inner_inset - 1, inner_inset, -1, 1),
        (inner_inset, int(height) - inner_inset - 1, 1, -1),
        (int(width) - inner_inset - 1, int(height) - inner_inset - 1, -1, -1),
    ):
        draw.line([(x0, y0), (x0 + (sx * corner), y0)], fill=style.accent_rgb, width=1)
        draw.line([(x0, y0), (x0, y0 + (sy * corner))], fill=style.accent_rgb, width=1)


def make_technical_diagram_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: TechnicalDiagramStyle,
    instance_seed: int = 0,
    namespace: str = "technical_diagram_background",
) -> tuple[Image.Image, dict[str, Any]]:
    """Create a technical-diagram background before scene rendering."""

    width = max(1, int(canvas_width))
    height = max(1, int(canvas_height))
    image = Image.new("RGB", (width, height), tuple(int(value) for value in style.canvas_rgb))
    draw = ImageDraw.Draw(image)

    if style.background_kind in {
        "plain_sheet",
        "paper_sheet",
        "graph_paper",
        "engineering_grid",
        "millimeter_paper",
        "lab_notebook",
        "ruled_notebook",
        "worksheet",
        "textbook_figure",
        "drafting_vellum",
        "isometric_grid",
        "scan_sheet",
        "exam_problem_box",
        "lab_card",
        "monochrome_print",
        "whiteboard",
        "presentation_slide",
    }:
        draw.rectangle((0, 0, width, height), fill=style.canvas_rgb)

    if style.grid_kind in {"square", "lab_grid"}:
        _draw_square_grid(draw, width=width, height=height, style=style)
    elif style.grid_kind == "ruled":
        _draw_ruled_lines(draw, width=width, height=height, style=style)
    elif style.grid_kind == "dots":
        _draw_dots(draw, width=width, height=height, style=style)
    elif style.grid_kind == "isometric":
        _draw_isometric_grid(draw, width=width, height=height, style=style)
    elif style.grid_kind == "crosshair":
        _draw_crosshair_grid(draw, width=width, height=height, style=style)

    if style.background_kind in {"lab_notebook", "ruled_notebook"}:
        margin_x = 46
        draw.line([(margin_x, 0), (margin_x, height)], fill=style.secondary_accent_rgb, width=1)
    elif style.background_kind == "worksheet":
        draw.rectangle((14, 14, width - 15, height - 15), outline=style.canvas_accent_rgb, width=1)
    elif style.background_kind == "textbook_figure":
        draw.rectangle((18, 18, width - 19, height - 19), outline=style.canvas_accent_rgb, width=1)
        draw.line([(18, 48), (width - 19, 48)], fill=style.canvas_accent_rgb, width=1)
    elif style.background_kind == "presentation_slide":
        draw.rectangle((0, 0, width, 16), fill=style.canvas_accent_rgb)
        draw.rectangle((0, height - 10, width, height), fill=style.canvas_accent_rgb)
    elif style.background_kind == "whiteboard":
        draw.rectangle((10, 10, width - 11, height - 11), outline=style.canvas_accent_rgb, width=2)
    elif style.background_kind == "exam_problem_box":
        draw.rectangle((18, 18, width - 19, height - 19), outline=style.canvas_accent_rgb, width=2)
        draw.rectangle((30, 30, 104, 58), outline=style.canvas_accent_rgb, width=1)
    elif style.background_kind == "lab_card":
        draw.rectangle((0, 0, width, 28), fill=style.canvas_accent_rgb)
        for x in range(28, width, 96):
            draw.line([(x, 34), (x, 46)], fill=style.canvas_accent_rgb, width=1)
    elif style.background_kind == "monochrome_print":
        draw.rectangle((12, 12, width - 13, height - 13), outline=style.canvas_accent_rgb, width=1)

    _draw_texture(
        draw,
        width=width,
        height=height,
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    _draw_frame(draw, width=width, height=height, style=style)
    metadata = {
        "enabled": True,
        "selected_style": f"technical_diagram_style:{style.style_pack}",
        "style_spec": technical_diagram_style_metadata(style),
    }
    return image, metadata


__all__ = [
    "Color",
    "DEFAULT_TECHNICAL_DIAGRAM_STYLE",
    "DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS",
    "TECHNICAL_DIAGRAM_FRAME_MODES",
    "TECHNICAL_DIAGRAM_PALETTES",
    "TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL",
    "TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER",
    "TECHNICAL_DIAGRAM_PROFILE_THEME_IDS",
    "TECHNICAL_DIAGRAM_PROFILES",
    "TECHNICAL_DIAGRAM_THEME_IDS",
    "TECHNICAL_DIAGRAM_THEMES",
    "TECHNICAL_DIAGRAM_TREATMENTS",
    "TECHNICAL_DIAGRAM_TREATMENT_IDS",
    "TechnicalDiagramPalette",
    "TechnicalDiagramStyle",
    "TechnicalDiagramTheme",
    "TechnicalDiagramTreatment",
    "make_technical_diagram_background",
    "resolve_technical_diagram_style",
    "technical_diagram_style_metadata",
]
