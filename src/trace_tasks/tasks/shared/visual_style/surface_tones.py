"""Shared neutral surface tone pool for visual task renderers."""

from __future__ import annotations

from typing import Mapping, Tuple


RGB = Tuple[int, int, int]
LIGHT_SURFACE_TEXT_RGB: RGB = (30, 34, 42)
LIGHT_SURFACE_TEXT_STROKE_RGB: RGB = (255, 255, 255)
DARK_SURFACE_TONE_IDS = frozenset(
    {
        "charcoal_concrete",
        "deep_warehouse",
        "midnight_steel",
        "dark_sage_floor",
        "graphite_plaster",
    }
)

DEFAULT_SURFACE_TONES: Mapping[str, Mapping[str, RGB]] = {
    "concrete": {
        "floor_rgb": (238, 241, 240),
        "grid_rgb": (184, 193, 191),
        "edge_rgb": (96, 108, 113),
        "surface_accent_rgb": (214, 221, 219),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "cool_concrete": {
        "floor_rgb": (235, 241, 244),
        "grid_rgb": (179, 190, 198),
        "edge_rgb": (89, 103, 116),
        "surface_accent_rgb": (210, 220, 225),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "warm_concrete": {
        "floor_rgb": (241, 238, 229),
        "grid_rgb": (190, 183, 168),
        "edge_rgb": (113, 104, 88),
        "surface_accent_rgb": (219, 214, 199),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "warehouse_gray": {
        "floor_rgb": (226, 231, 229),
        "grid_rgb": (174, 184, 181),
        "edge_rgb": (91, 102, 105),
        "surface_accent_rgb": (205, 214, 211),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "pale_sand": {
        "floor_rgb": (242, 238, 224),
        "grid_rgb": (193, 184, 164),
        "edge_rgb": (121, 109, 87),
        "surface_accent_rgb": (222, 215, 194),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "green_gray": {
        "floor_rgb": (229, 235, 228),
        "grid_rgb": (174, 186, 174),
        "edge_rgb": (88, 105, 91),
        "surface_accent_rgb": (207, 218, 207),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "blue_gray": {
        "floor_rgb": (229, 236, 241),
        "grid_rgb": (171, 184, 196),
        "edge_rgb": (83, 99, 113),
        "surface_accent_rgb": (205, 217, 225),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "clay_gray": {
        "floor_rgb": (235, 229, 222),
        "grid_rgb": (184, 173, 164),
        "edge_rgb": (111, 96, 86),
        "surface_accent_rgb": (215, 205, 196),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "off_white": {
        "floor_rgb": (246, 247, 244),
        "grid_rgb": (192, 197, 193),
        "edge_rgb": (102, 111, 108),
        "surface_accent_rgb": (224, 228, 224),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "industrial_dim": {
        "floor_rgb": (216, 222, 225),
        "grid_rgb": (164, 174, 181),
        "edge_rgb": (78, 90, 101),
        "surface_accent_rgb": (196, 205, 210),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "limestone": {
        "floor_rgb": (240, 241, 235),
        "grid_rgb": (188, 190, 181),
        "edge_rgb": (104, 108, 96),
        "surface_accent_rgb": (219, 221, 211),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "silver_mist": {
        "floor_rgb": (234, 238, 240),
        "grid_rgb": (179, 187, 191),
        "edge_rgb": (91, 101, 106),
        "surface_accent_rgb": (211, 218, 221),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "porcelain_gray": {
        "floor_rgb": (243, 245, 246),
        "grid_rgb": (188, 194, 197),
        "edge_rgb": (98, 106, 111),
        "surface_accent_rgb": (222, 226, 228),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "dusted_plaster": {
        "floor_rgb": (239, 235, 228),
        "grid_rgb": (188, 179, 168),
        "edge_rgb": (111, 98, 86),
        "surface_accent_rgb": (219, 210, 199),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "light_sage": {
        "floor_rgb": (232, 238, 231),
        "grid_rgb": (176, 188, 174),
        "edge_rgb": (90, 107, 91),
        "surface_accent_rgb": (211, 222, 208),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "mist_blue": {
        "floor_rgb": (232, 239, 244),
        "grid_rgb": (176, 189, 202),
        "edge_rgb": (86, 102, 118),
        "surface_accent_rgb": (210, 222, 230),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "sandstone": {
        "floor_rgb": (241, 236, 221),
        "grid_rgb": (193, 181, 158),
        "edge_rgb": (120, 106, 82),
        "surface_accent_rgb": (223, 213, 190),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "soft_taupe": {
        "floor_rgb": (235, 232, 226),
        "grid_rgb": (183, 176, 166),
        "edge_rgb": (105, 96, 85),
        "surface_accent_rgb": (214, 207, 197),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "lab_tile": {
        "floor_rgb": (245, 247, 247),
        "grid_rgb": (190, 198, 199),
        "edge_rgb": (96, 108, 111),
        "surface_accent_rgb": (224, 229, 230),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "pale_zinc": {
        "floor_rgb": (230, 234, 233),
        "grid_rgb": (173, 183, 181),
        "edge_rgb": (84, 98, 100),
        "surface_accent_rgb": (207, 216, 214),
        "text_rgb": LIGHT_SURFACE_TEXT_RGB,
        "text_stroke_rgb": LIGHT_SURFACE_TEXT_STROKE_RGB,
    },
    "charcoal_concrete": {
        "floor_rgb": (43, 50, 60),
        "grid_rgb": (92, 105, 121),
        "edge_rgb": (185, 199, 216),
        "surface_accent_rgb": (57, 65, 77),
        "text_rgb": (232, 238, 247),
        "text_stroke_rgb": (16, 20, 28),
    },
    "deep_warehouse": {
        "floor_rgb": (34, 39, 43),
        "grid_rgb": (82, 93, 98),
        "edge_rgb": (190, 203, 209),
        "surface_accent_rgb": (49, 56, 61),
        "text_rgb": (235, 240, 244),
        "text_stroke_rgb": (18, 22, 26),
    },
    "midnight_steel": {
        "floor_rgb": (25, 33, 48),
        "grid_rgb": (68, 86, 116),
        "edge_rgb": (207, 224, 244),
        "surface_accent_rgb": (39, 51, 73),
        "text_rgb": (235, 241, 250),
        "text_stroke_rgb": (16, 22, 34),
    },
    "dark_sage_floor": {
        "floor_rgb": (25, 39, 35),
        "grid_rgb": (70, 103, 91),
        "edge_rgb": (204, 229, 219),
        "surface_accent_rgb": (38, 58, 52),
        "text_rgb": (229, 243, 238),
        "text_stroke_rgb": (15, 24, 22),
    },
    "graphite_plaster": {
        "floor_rgb": (39, 38, 42),
        "grid_rgb": (89, 86, 93),
        "edge_rgb": (207, 201, 211),
        "surface_accent_rgb": (55, 53, 59),
        "text_rgb": (238, 235, 242),
        "text_stroke_rgb": (22, 20, 26),
    },
}


__all__ = [
    "DARK_SURFACE_TONE_IDS",
    "DEFAULT_SURFACE_TONES",
    "LIGHT_SURFACE_TEXT_RGB",
    "LIGHT_SURFACE_TEXT_STROKE_RGB",
    "RGB",
]
