"""Shared structured-information scene styles for charts, pages, and graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ..color_distance import rgb_euclidean_distance
from ..text_legibility import (
    LARGE_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    resolve_readable_text_style,
    text_legibility_summary,
)
from .metadata import color_separation_metadata
from .request import VisualStyleRequest, visual_style_request_metadata


Color = Tuple[int, int, int]

INFORMATION_SCENE_CHROME_MODES: Tuple[str, ...] = ("none", "thin_frame", "accent_frame")
DEFAULT_INFORMATION_SCENE_CHROME_WEIGHTS = {
    "none": 0.5,
    "thin_frame": 0.25,
    "accent_frame": 0.25,
}


@dataclass(frozen=True)
class InformationScenePalette:
    """One non-semantic palette for structured information scenes."""

    palette_id: str
    compatibility: Tuple[str, ...]
    canvas_rgb: Color
    canvas_accent_rgb: Color
    surface_rgb: Color
    surface_alt_rgb: Color
    panel_fill_rgb: Color
    panel_border_rgb: Color
    header_rgb: Color
    header_text_rgb: Color
    text_rgb: Color
    muted_text_rgb: Color
    text_stroke_rgb: Color
    grid_rgb: Color
    axis_rgb: Color
    guide_rgb: Color
    connector_rgb: Color
    neutral_mark_rgb: Color
    accent_rgb: Color
    secondary_accent_rgb: Color
    highlight_rgb: Color
    shadow_rgb: Color
    callout_fill_rgb: Color
    callout_border_rgb: Color


@dataclass(frozen=True)
class InformationSceneTreatment:
    """One structural treatment for a structured information scene."""

    treatment_id: str
    compatibility: Tuple[str, ...]
    surface_kind: str
    chrome_kind: str
    texture: str
    content_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    frame_width_px: int
    corner_radius_px: int
    shadow_offset_px: int
    guide_density: str
    context_density: str
    typography_scale: float


@dataclass(frozen=True)
class InformationSceneStyle:
    """Resolved style after independent treatment, palette, and chrome sampling."""

    treatment: str
    palette_id: str
    style_pack: str
    surface_kind: str
    chrome_kind: str
    texture: str
    chrome_mode: str
    canvas_rgb: Color
    canvas_accent_rgb: Color
    surface_rgb: Color
    surface_alt_rgb: Color
    panel_fill_rgb: Color
    panel_border_rgb: Color
    header_rgb: Color
    header_text_rgb: Color
    text_rgb: Color
    muted_text_rgb: Color
    text_stroke_rgb: Color
    grid_rgb: Color
    axis_rgb: Color
    guide_rgb: Color
    connector_rgb: Color
    neutral_mark_rgb: Color
    accent_rgb: Color
    secondary_accent_rgb: Color
    highlight_rgb: Color
    shadow_rgb: Color
    callout_fill_rgb: Color
    callout_border_rgb: Color
    content_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    frame_width_px: int
    corner_radius_px: int
    shadow_offset_px: int
    guide_density: str
    context_density: str
    typography_scale: float
    protected_colors: Tuple[Color, ...]


INFORMATION_SCENE_TREATMENTS: dict[str, InformationSceneTreatment] = {
    "clean_default": InformationSceneTreatment("clean_default", ("light",), "plain", "none", "none", 24, 18, 0, 1, 4, 0, "normal", "none", 1.0),
    "report_card": InformationSceneTreatment("report_card", ("light",), "card", "panel", "none", 28, 20, 42, 1, 6, 4, "normal", "light", 1.0),
    "dashboard_tile": InformationSceneTreatment("dashboard_tile", ("light",), "dashboard", "panel", "none", 22, 18, 38, 1, 6, 3, "dense", "light", 0.96),
    "executive_dashboard": InformationSceneTreatment("executive_dashboard", ("light",), "dashboard", "accent_header", "none", 26, 20, 48, 2, 6, 4, "normal", "medium", 1.02),
    "news_graphic": InformationSceneTreatment("news_graphic", ("light",), "editorial", "rule_header", "paper", 30, 20, 46, 1, 2, 2, "normal", "medium", 1.0),
    "academic_figure": InformationSceneTreatment("academic_figure", ("light",), "publication", "thin_frame", "none", 26, 18, 32, 1, 0, 0, "normal", "light", 0.94),
    "journal_appendix": InformationSceneTreatment("journal_appendix", ("light",), "publication", "thin_frame", "scan", 26, 16, 30, 1, 0, 0, "dense", "light", 0.92),
    "infographic_panel": InformationSceneTreatment("infographic_panel", ("light",), "infographic", "accent_frame", "none", 32, 22, 52, 2, 8, 4, "sparse", "medium", 1.04),
    "poster_explainer": InformationSceneTreatment("poster_explainer", ("light",), "poster", "accent_header", "paper", 34, 24, 58, 2, 8, 5, "sparse", "heavy", 1.06),
    "annotated_callout": InformationSceneTreatment("annotated_callout", ("light",), "annotated", "side_rule", "none", 30, 20, 44, 1, 6, 3, "normal", "heavy", 1.0),
    "caption_heavy_figure": InformationSceneTreatment("caption_heavy_figure", ("light",), "captioned", "footer_rule", "paper", 30, 18, 34, 1, 3, 2, "normal", "heavy", 0.96),
    "source_note_sheet": InformationSceneTreatment("source_note_sheet", ("light",), "source_note", "footer_rule", "scan", 28, 18, 32, 1, 2, 1, "normal", "medium", 0.94),
    "compact_small_multiples": InformationSceneTreatment("compact_small_multiples", ("light",), "compact_grid", "panel", "none", 20, 14, 32, 1, 4, 2, "dense", "light", 0.90),
    "data_table_report": InformationSceneTreatment("data_table_report", ("light",), "table_report", "rule_header", "none", 24, 16, 38, 1, 2, 1, "dense", "medium", 0.94),
    "web_article_embed": InformationSceneTreatment("web_article_embed", ("light",), "web_embed", "panel", "none", 30, 20, 44, 1, 6, 3, "normal", "medium", 1.0),
    "desktop_app_window": InformationSceneTreatment("desktop_app_window", ("light",), "app_window", "accent_header", "none", 22, 16, 36, 1, 5, 3, "dense", "medium", 0.94),
    "control_console": InformationSceneTreatment("control_console", ("light",), "console", "accent_frame", "none", 24, 16, 40, 2, 4, 3, "dense", "medium", 0.96),
    "lab_report_sheet": InformationSceneTreatment("lab_report_sheet", ("light",), "publication", "rule_header", "paper", 28, 18, 40, 1, 1, 1, "normal", "medium", 0.96),
    "presentation_slide": InformationSceneTreatment("presentation_slide", ("light",), "slide", "accent_header", "none", 36, 26, 52, 1, 0, 0, "sparse", "light", 1.08),
    "print_scan_sheet": InformationSceneTreatment("print_scan_sheet", ("light",), "printout", "thin_frame", "scan", 28, 18, 34, 1, 1, 1, "normal", "light", 0.96),
    "dark_analytics_board": InformationSceneTreatment("dark_analytics_board", ("dark",), "dark_board", "accent_header", "none", 28, 20, 44, 1, 6, 0, "dense", "light", 0.98),
    "dark_report_card": InformationSceneTreatment("dark_report_card", ("dark",), "card", "panel", "none", 28, 20, 42, 1, 6, 0, "normal", "light", 1.0),
    "dark_publication_figure": InformationSceneTreatment("dark_publication_figure", ("dark",), "publication", "thin_frame", "none", 26, 18, 32, 1, 0, 0, "normal", "light", 0.94),
    "dark_dashboard_tile": InformationSceneTreatment("dark_dashboard_tile", ("dark",), "dashboard", "accent_header", "none", 22, 18, 38, 1, 6, 0, "dense", "light", 0.96),
    "dark_console_panel": InformationSceneTreatment("dark_console_panel", ("dark",), "console", "accent_frame", "none", 24, 16, 40, 2, 4, 0, "dense", "medium", 0.96),
}

INFORMATION_SCENE_TREATMENT_IDS: Tuple[str, ...] = tuple(INFORMATION_SCENE_TREATMENTS.keys())


INFORMATION_SCENE_PALETTES: dict[str, InformationScenePalette] = {
    "neutral_report": InformationScenePalette("neutral_report", ("light",), (247, 249, 252), (225, 231, 240), (255, 255, 255), (249, 251, 254), (255, 255, 255), (196, 205, 218), (48, 55, 68), (255, 255, 255), (31, 36, 46), (89, 101, 119), (255, 255, 255), (224, 229, 237), (67, 75, 90), (143, 154, 172), (111, 122, 141), (74, 116, 170), (42, 105, 168), (94, 145, 120), (239, 193, 73), (218, 224, 234), (252, 253, 255), (166, 179, 198)),
    "publication_gray": InformationScenePalette("publication_gray", ("light",), (246, 246, 244), (224, 224, 220), (255, 255, 252), (248, 248, 245), (255, 255, 252), (180, 180, 174), (39, 41, 43), (255, 255, 255), (26, 28, 30), (91, 93, 95), (255, 255, 255), (226, 226, 222), (58, 60, 62), (142, 144, 146), (98, 100, 102), (88, 108, 138), (66, 94, 132), (126, 126, 118), (224, 190, 76), (214, 214, 208), (252, 252, 249), (170, 170, 164)),
    "cool_business": InformationScenePalette("cool_business", ("light",), (241, 246, 252), (213, 226, 241), (253, 255, 255), (247, 251, 255), (253, 255, 255), (183, 200, 220), (34, 69, 105), (255, 255, 255), (25, 45, 67), (75, 100, 128), (255, 255, 255), (216, 228, 241), (41, 77, 113), (121, 151, 184), (90, 119, 151), (52, 122, 184), (36, 116, 181), (79, 146, 161), (244, 200, 82), (210, 222, 235), (250, 253, 255), (156, 180, 204)),
    "warm_editorial": InformationScenePalette("warm_editorial", ("light",), (250, 247, 240), (232, 223, 207), (255, 253, 247), (250, 246, 235), (255, 253, 247), (204, 190, 166), (78, 62, 44), (255, 255, 252), (47, 37, 28), (112, 93, 70), (255, 255, 255), (232, 222, 202), (84, 68, 50), (151, 130, 101), (117, 99, 74), (156, 92, 62), (164, 80, 54), (88, 137, 112), (233, 183, 69), (222, 213, 195), (255, 253, 247), (187, 170, 145)),
    "soft_mint": InformationScenePalette("soft_mint", ("light",), (242, 249, 247), (214, 232, 226), (253, 255, 254), (246, 252, 250), (253, 255, 254), (180, 211, 201), (34, 89, 76), (255, 255, 255), (25, 58, 51), (74, 115, 103), (255, 255, 255), (211, 229, 224), (39, 92, 80), (111, 157, 145), (83, 130, 118), (46, 134, 119), (35, 128, 113), (95, 130, 176), (234, 193, 82), (205, 222, 217), (251, 255, 253), (150, 189, 178)),
    "slate_amber": InformationScenePalette("slate_amber", ("light",), (245, 247, 249), (223, 228, 233), (255, 255, 255), (248, 250, 252), (255, 255, 255), (186, 196, 207), (50, 61, 73), (255, 255, 255), (30, 38, 47), (84, 96, 110), (255, 255, 255), (220, 226, 232), (57, 69, 83), (132, 145, 158), (102, 113, 126), (78, 99, 124), (62, 93, 130), (194, 132, 45), (241, 185, 70), (213, 220, 227), (252, 253, 255), (162, 174, 188)),
    "ink_teal": InformationScenePalette("ink_teal", ("light",), (245, 249, 250), (219, 232, 236), (255, 255, 255), (247, 252, 253), (255, 255, 255), (184, 205, 211), (31, 75, 83), (255, 255, 255), (22, 45, 50), (73, 104, 111), (255, 255, 255), (213, 229, 233), (35, 82, 91), (113, 151, 160), (84, 120, 129), (40, 122, 139), (30, 126, 147), (112, 101, 172), (235, 194, 83), (208, 224, 228), (252, 255, 255), (153, 185, 193)),
    "burgundy_sage": InformationScenePalette("burgundy_sage", ("light",), (250, 246, 246), (236, 221, 223), (255, 253, 253), (251, 246, 247), (255, 253, 253), (207, 185, 190), (112, 45, 58), (255, 255, 255), (54, 31, 36), (124, 80, 89), (255, 255, 255), (234, 220, 223), (98, 51, 61), (166, 122, 131), (126, 90, 98), (126, 81, 97), (131, 54, 76), (113, 142, 98), (230, 184, 77), (222, 210, 213), (255, 253, 253), (190, 168, 173)),
    "indigo_ochre": InformationScenePalette("indigo_ochre", ("light",), (246, 245, 252), (225, 223, 241), (253, 252, 255), (248, 247, 253), (253, 252, 255), (194, 190, 219), (60, 52, 101), (255, 255, 255), (34, 31, 52), (88, 80, 127), (255, 255, 255), (221, 219, 238), (57, 51, 94), (127, 120, 161), (98, 91, 130), (82, 94, 150), (94, 86, 178), (157, 116, 46), (238, 190, 69), (214, 211, 232), (253, 252, 255), (166, 159, 197)),
    "coastal_blue": InformationScenePalette("coastal_blue", ("light",), (240, 248, 252), (210, 230, 240), (252, 255, 255), (246, 252, 255), (252, 255, 255), (181, 205, 219), (31, 82, 112), (255, 255, 255), (22, 50, 68), (72, 111, 135), (255, 255, 255), (210, 229, 239), (37, 89, 119), (112, 157, 181), (82, 127, 151), (41, 117, 161), (37, 129, 180), (85, 151, 125), (234, 196, 83), (204, 223, 234), (251, 255, 255), (150, 187, 205)),
    "atlas_map": InformationScenePalette("atlas_map", ("light",), (243, 248, 245), (214, 228, 218), (255, 254, 247), (250, 251, 241), (255, 254, 247), (182, 199, 176), (53, 86, 64), (255, 255, 250), (36, 54, 42), (84, 111, 91), (255, 255, 255), (219, 229, 211), (61, 92, 70), (129, 154, 134), (97, 123, 103), (77, 126, 105), (48, 116, 135), (154, 118, 72), (231, 191, 84), (212, 224, 213), (255, 254, 247), (163, 184, 166)),
    "metro_bright": InformationScenePalette("metro_bright", ("light",), (247, 249, 250), (225, 229, 233), (255, 255, 255), (248, 250, 252), (255, 255, 255), (190, 198, 208), (35, 45, 58), (255, 255, 255), (25, 31, 39), (82, 94, 108), (255, 255, 255), (222, 226, 232), (44, 56, 70), (132, 145, 160), (101, 113, 128), (54, 118, 182), (30, 131, 197), (223, 79, 87), (246, 192, 63), (214, 220, 228), (255, 255, 255), (162, 174, 188)),
    "data_viz_classic": InformationScenePalette("data_viz_classic", ("light",), (248, 249, 250), (227, 231, 235), (255, 255, 255), (249, 250, 252), (255, 255, 255), (196, 203, 213), (43, 55, 70), (255, 255, 255), (28, 36, 46), (83, 96, 112), (255, 255, 255), (223, 228, 234), (50, 67, 86), (133, 149, 166), (101, 116, 134), (67, 107, 158), (76, 120, 188), (226, 112, 70), (239, 188, 70), (216, 222, 230), (253, 254, 255), (164, 176, 192)),
    "okabe_ito_light": InformationScenePalette("okabe_ito_light", ("light",), (248, 249, 248), (226, 229, 226), (255, 255, 255), (249, 251, 249), (255, 255, 255), (194, 203, 197), (39, 52, 56), (255, 255, 255), (25, 32, 35), (82, 96, 98), (255, 255, 255), (221, 228, 224), (47, 65, 70), (129, 146, 149), (101, 118, 122), (65, 110, 155), (0, 114, 178), (0, 158, 115), (240, 180, 62), (215, 222, 218), (253, 255, 254), (161, 176, 172)),
    "pastel_dashboard": InformationScenePalette("pastel_dashboard", ("light",), (248, 248, 252), (229, 228, 240), (255, 255, 255), (250, 250, 255), (255, 255, 255), (204, 202, 224), (72, 71, 111), (255, 255, 255), (41, 40, 66), (102, 101, 139), (255, 255, 255), (229, 228, 241), (74, 73, 116), (151, 149, 181), (121, 119, 153), (117, 103, 174), (124, 112, 198), (96, 156, 171), (241, 199, 92), (224, 222, 238), (254, 254, 255), (177, 174, 207)),
    "monochrome_news": InformationScenePalette("monochrome_news", ("light",), (247, 247, 245), (226, 226, 222), (255, 255, 253), (249, 249, 246), (255, 255, 253), (190, 190, 184), (31, 31, 31), (255, 255, 255), (20, 20, 20), (82, 82, 78), (255, 255, 255), (224, 224, 220), (45, 45, 44), (136, 136, 132), (101, 101, 98), (80, 80, 80), (58, 58, 58), (115, 115, 110), (221, 184, 69), (215, 215, 210), (253, 253, 250), (166, 166, 160)),
    "high_contrast_light": InformationScenePalette("high_contrast_light", ("light",), (255, 255, 255), (226, 230, 235), (255, 255, 255), (248, 250, 252), (255, 255, 255), (164, 172, 184), (20, 26, 34), (255, 255, 255), (0, 0, 0), (65, 72, 83), (255, 255, 255), (211, 216, 224), (20, 28, 38), (120, 130, 144), (82, 92, 106), (33, 88, 150), (0, 92, 170), (0, 135, 125), (235, 174, 43), (209, 215, 224), (255, 255, 255), (142, 152, 168)),
    "dark_analytics": InformationScenePalette("dark_analytics", ("dark",), (26, 31, 40), (45, 55, 70), (32, 38, 49), (38, 45, 58), (34, 41, 54), (82, 96, 116), (40, 51, 68), (232, 238, 247), (232, 238, 247), (178, 190, 207), (22, 27, 35), (68, 81, 99), (218, 228, 241), (113, 130, 153), (144, 160, 182), (92, 164, 222), (78, 161, 226), (118, 200, 174), (245, 204, 94), (15, 18, 24), (41, 50, 64), (107, 126, 151)),
    "dark_mint": InformationScenePalette("dark_mint", ("dark",), (22, 36, 34), (40, 65, 60), (28, 47, 44), (35, 57, 53), (31, 51, 48), (78, 116, 106), (35, 70, 62), (230, 246, 241), (228, 244, 239), (174, 204, 196), (18, 27, 26), (62, 97, 89), (212, 235, 229), (105, 151, 140), (138, 183, 171), (83, 180, 162), (70, 178, 161), (121, 165, 226), (241, 203, 91), (14, 23, 22), (37, 61, 57), (96, 136, 126)),
    "dark_blue_orange": InformationScenePalette("dark_blue_orange", ("dark",), (24, 31, 46), (45, 57, 82), (31, 40, 59), (38, 48, 70), (34, 44, 65), (79, 96, 128), (39, 54, 82), (235, 241, 250), (235, 241, 250), (178, 192, 214), (18, 24, 35), (63, 79, 106), (216, 229, 246), (104, 126, 162), (137, 158, 193), (93, 157, 224), (83, 152, 224), (239, 142, 68), (246, 203, 93), (14, 19, 29), (39, 51, 74), (96, 119, 156)),
}


def _normalize_color(value: Sequence[int]) -> Color:
    if len(value) < 3:
        raise ValueError("RGB colors require three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])


def _weighted_choice(*, values: Sequence[str], weights: Mapping[str, float], instance_seed: int, namespace: str) -> str:
    if not values:
        raise ValueError("values must not be empty")
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=str(namespace))
    parsed = [max(0.0, float(weights.get(str(value), 1.0))) for value in values]
    total = float(sum(parsed))
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


def _coerce_id_sequence(value: Sequence[str] | str | None, *, default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    if isinstance(value, str):
        return (str(value),)
    return tuple(str(item) for item in value)


def _minimum_protected_distance(palette: InformationScenePalette, protected_colors: Sequence[Color]) -> float:
    if not protected_colors:
        return 0.0
    role_colors = (
        palette.canvas_rgb,
        palette.canvas_accent_rgb,
        palette.surface_rgb,
        palette.surface_alt_rgb,
        palette.panel_fill_rgb,
        palette.panel_border_rgb,
        palette.header_rgb,
        palette.text_rgb,
        palette.muted_text_rgb,
        palette.grid_rgb,
        palette.axis_rgb,
        palette.guide_rgb,
        palette.connector_rgb,
        palette.neutral_mark_rgb,
        palette.accent_rgb,
        palette.secondary_accent_rgb,
        palette.highlight_rgb,
        palette.callout_fill_rgb,
        palette.callout_border_rgb,
    )
    return min(
        float(rgb_euclidean_distance(role_color, protected_color))
        for role_color in role_colors
        for protected_color in protected_colors
    )


def _style_from_parts(
    *,
    treatment: InformationSceneTreatment,
    palette: InformationScenePalette,
    chrome_mode: str,
    protected_colors: Sequence[Color],
) -> InformationSceneStyle:
    style_pack = f"{treatment.treatment_id}:{palette.palette_id}:{str(chrome_mode)}"
    return InformationSceneStyle(
        treatment=str(treatment.treatment_id),
        palette_id=str(palette.palette_id),
        style_pack=str(style_pack),
        surface_kind=str(treatment.surface_kind),
        chrome_kind=str(treatment.chrome_kind),
        texture=str(treatment.texture),
        chrome_mode=str(chrome_mode),
        canvas_rgb=tuple(palette.canvas_rgb),
        canvas_accent_rgb=tuple(palette.canvas_accent_rgb),
        surface_rgb=tuple(palette.surface_rgb),
        surface_alt_rgb=tuple(palette.surface_alt_rgb),
        panel_fill_rgb=tuple(palette.panel_fill_rgb),
        panel_border_rgb=tuple(palette.panel_border_rgb),
        header_rgb=tuple(palette.header_rgb),
        header_text_rgb=tuple(palette.header_text_rgb),
        text_rgb=tuple(palette.text_rgb),
        muted_text_rgb=tuple(palette.muted_text_rgb),
        text_stroke_rgb=tuple(palette.text_stroke_rgb),
        grid_rgb=tuple(palette.grid_rgb),
        axis_rgb=tuple(palette.axis_rgb),
        guide_rgb=tuple(palette.guide_rgb),
        connector_rgb=tuple(palette.connector_rgb),
        neutral_mark_rgb=tuple(palette.neutral_mark_rgb),
        accent_rgb=tuple(palette.accent_rgb),
        secondary_accent_rgb=tuple(palette.secondary_accent_rgb),
        highlight_rgb=tuple(palette.highlight_rgb),
        shadow_rgb=tuple(palette.shadow_rgb),
        callout_fill_rgb=tuple(palette.callout_fill_rgb),
        callout_border_rgb=tuple(palette.callout_border_rgb),
        content_margin_px=int(treatment.content_margin_px),
        panel_padding_px=int(treatment.panel_padding_px),
        title_band_height_px=int(treatment.title_band_height_px),
        frame_width_px=int(treatment.frame_width_px),
        corner_radius_px=int(treatment.corner_radius_px),
        shadow_offset_px=int(treatment.shadow_offset_px),
        guide_density=str(treatment.guide_density),
        context_density=str(treatment.context_density),
        typography_scale=float(treatment.typography_scale),
        protected_colors=tuple(_normalize_color(color) for color in protected_colors),
    )


def _resolve_information_text_legibility(
    style: InformationSceneStyle,
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[InformationSceneStyle, dict[str, Any]]:
    """Resolve readable non-semantic text colors for information scenes."""

    main_surfaces = (
        style.canvas_rgb,
        style.surface_rgb,
        style.surface_alt_rgb,
        style.panel_fill_rgb,
        style.callout_fill_rgb,
    )
    header_surfaces = (style.header_rgb,)
    main_text = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.text_legibility.main_text",
        role="read_required_text",
        surface_rgbs=main_surfaces,
        preferred_rgbs=(style.text_rgb, style.axis_rgb, style.accent_rgb),
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        required=True,
    )
    muted_text = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.text_legibility.muted_text",
        role="read_required_secondary_text",
        surface_rgbs=main_surfaces,
        preferred_rgbs=(style.muted_text_rgb, style.text_rgb),
        min_contrast_ratio=LARGE_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=28.0,
        required=True,
    )
    header_text = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.text_legibility.header_text",
        role="read_required_header_text",
        surface_rgbs=header_surfaces,
        preferred_rgbs=(style.header_text_rgb, style.text_rgb),
        min_contrast_ratio=LARGE_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=28.0,
        required=True,
    )
    adjusted = replace(
        style,
        header_text_rgb=tuple(header_text.fill_rgb),
        text_rgb=tuple(main_text.fill_rgb),
        muted_text_rgb=tuple(muted_text.fill_rgb),
        text_stroke_rgb=tuple(main_text.stroke_rgb),
    )
    return adjusted, text_legibility_summary((main_text, muted_text, header_text))


def resolve_information_scene_style(
    *,
    instance_seed: int,
    namespace: str,
    treatments: Sequence[str] | str | None = None,
    treatment_weights: Mapping[str, float] | None = None,
    palettes: Sequence[str] | str | None = None,
    palette_weights: Mapping[str, float] | None = None,
    chrome_modes: Sequence[str] | str | None = None,
    chrome_mode_weights: Mapping[str, float] | None = None,
    allow_dark: bool = False,
    protected_colors: Sequence[Color] | None = None,
    min_protected_rgb_distance: float = 42.0,
) -> tuple[InformationSceneStyle, dict[str, Any]]:
    """Resolve one structured-information style without consuming semantic RNG."""

    requested_treatments = _coerce_id_sequence(treatments, default=INFORMATION_SCENE_TREATMENT_IDS)
    valid_treatments = [item for item in requested_treatments if item in INFORMATION_SCENE_TREATMENTS]
    if not bool(allow_dark):
        valid_treatments = [
            item
            for item in valid_treatments
            if "dark" not in set(INFORMATION_SCENE_TREATMENTS[str(item)].compatibility)
        ]
    if not valid_treatments:
        valid_treatments = ["clean_default"]
    treatment_id = _weighted_choice(
        values=tuple(valid_treatments),
        weights=treatment_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.treatment",
    )
    treatment = INFORMATION_SCENE_TREATMENTS[str(treatment_id)]

    requested_palettes = _coerce_id_sequence(palettes, default=tuple(sorted(INFORMATION_SCENE_PALETTES.keys())))
    compatible = [
        item
        for item in requested_palettes
        if item in INFORMATION_SCENE_PALETTES
        and set(INFORMATION_SCENE_PALETTES[str(item)].compatibility).intersection(set(treatment.compatibility))
    ]
    if not compatible:
        compatible = ["neutral_report"]
    protected = tuple(_normalize_color(color) for color in (protected_colors or ()))
    protected_safe = [
        item
        for item in compatible
        if _minimum_protected_distance(INFORMATION_SCENE_PALETTES[str(item)], protected) >= float(min_protected_rgb_distance)
    ]
    protected_filter_fallback = False
    if protected and protected_safe:
        compatible = protected_safe
    elif protected and not protected_safe:
        protected_filter_fallback = True

    palette_id = _weighted_choice(
        values=tuple(compatible),
        weights=palette_weights or {},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.palette.{treatment_id}",
    )
    requested_chrome = _coerce_id_sequence(chrome_modes, default=INFORMATION_SCENE_CHROME_MODES)
    eligible_chrome = tuple(mode for mode in requested_chrome if mode in set(INFORMATION_SCENE_CHROME_MODES))
    if not eligible_chrome:
        eligible_chrome = ("none",)
    chrome_mode = _weighted_choice(
        values=eligible_chrome,
        weights=chrome_mode_weights or DEFAULT_INFORMATION_SCENE_CHROME_WEIGHTS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.chrome.{treatment_id}.{palette_id}",
    )
    style = _style_from_parts(
        treatment=treatment,
        palette=INFORMATION_SCENE_PALETTES[str(palette_id)],
        chrome_mode=str(chrome_mode),
        protected_colors=protected,
    )
    style, text_legibility = _resolve_information_text_legibility(
        style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    metadata = information_scene_style_metadata(style)
    metadata["text_legibility"] = dict(text_legibility)
    metadata["selection"] = {
        "namespace": str(namespace),
        "allow_dark": bool(allow_dark),
        "requested_treatments": list(requested_treatments),
        "eligible_treatments": list(valid_treatments),
        "requested_palettes": list(requested_palettes),
        "eligible_palettes": list(compatible),
        "requested_chrome_modes": list(requested_chrome),
        "eligible_chrome_modes": list(eligible_chrome),
        "chrome_mode_weights": dict(chrome_mode_weights or DEFAULT_INFORMATION_SCENE_CHROME_WEIGHTS),
        "protected_palette_filter_fallback": bool(protected_filter_fallback),
        "min_protected_rgb_distance_required": float(min_protected_rgb_distance),
    }
    return style, metadata


def resolve_information_scene_style_from_request(
    request: VisualStyleRequest,
    *,
    treatments: Sequence[str] | str | None = None,
    treatment_weights: Mapping[str, float] | None = None,
    palettes: Sequence[str] | str | None = None,
    palette_weights: Mapping[str, float] | None = None,
    chrome_modes: Sequence[str] | str | None = None,
    chrome_mode_weights: Mapping[str, float] | None = None,
    min_protected_rgb_distance: float = 42.0,
) -> tuple[InformationSceneStyle, dict[str, Any]]:
    """Resolve structured-information style from the shared request contract."""

    if str(request.style_family) != "information_scene":
        raise ValueError(f"expected information_scene request, got {request.style_family!r}")
    if palettes is None and not bool(request.allow_colored_surface):
        palettes = (
            "neutral_report",
            "publication_gray",
            "monochrome_news",
            "high_contrast_light",
        )
    style, metadata = resolve_information_scene_style(
        instance_seed=int(request.instance_seed),
        namespace=f"{request.domain}.{request.routing_key}.{request.scene_id}.information_scene_style",
        treatments=treatments,
        treatment_weights=treatment_weights,
        palettes=palettes,
        palette_weights=palette_weights,
        chrome_modes=chrome_modes,
        chrome_mode_weights=chrome_mode_weights,
        allow_dark=bool(request.allow_dark),
        protected_colors=request.protected_colors,
        min_protected_rgb_distance=float(min_protected_rgb_distance),
    )
    metadata["style_request"] = visual_style_request_metadata(request)
    return style, metadata


def information_scene_style_metadata(style: InformationSceneStyle) -> dict[str, Any]:
    """Serialize a structured-information style into trace-ready metadata."""

    protected = tuple(_normalize_color(color) for color in style.protected_colors)
    protected_distances = [
        float(rgb_euclidean_distance(role_color, protected_color))
        for role_color in (
            style.canvas_rgb,
            style.surface_rgb,
            style.panel_fill_rgb,
            style.panel_border_rgb,
            style.header_rgb,
            style.text_rgb,
            style.grid_rgb,
            style.axis_rgb,
            style.guide_rgb,
            style.accent_rgb,
            style.secondary_accent_rgb,
        )
        for protected_color in protected
    ]
    return {
        "kind": "information_scene_style",
        "treatment": str(style.treatment),
        "palette_id": str(style.palette_id),
        "style_pack": str(style.style_pack),
        "surface_style": {
            "kind": str(style.surface_kind),
            "chrome_kind": str(style.chrome_kind),
            "texture": str(style.texture),
            "chrome_mode": str(style.chrome_mode),
        },
        "layout_style": {
            "content_margin_px": int(style.content_margin_px),
            "panel_padding_px": int(style.panel_padding_px),
            "title_band_height_px": int(style.title_band_height_px),
            "frame_width_px": int(style.frame_width_px),
            "corner_radius_px": int(style.corner_radius_px),
            "shadow_offset_px": int(style.shadow_offset_px),
            "guide_density": str(style.guide_density),
            "context_density": str(style.context_density),
            "typography_scale": float(style.typography_scale),
        },
        "roles_rgb": {
            "canvas": list(style.canvas_rgb),
            "canvas_accent": list(style.canvas_accent_rgb),
            "surface": list(style.surface_rgb),
            "surface_alt": list(style.surface_alt_rgb),
            "panel_fill": list(style.panel_fill_rgb),
            "panel_border": list(style.panel_border_rgb),
            "header": list(style.header_rgb),
            "header_text": list(style.header_text_rgb),
            "text": list(style.text_rgb),
            "muted_text": list(style.muted_text_rgb),
            "text_stroke": list(style.text_stroke_rgb),
            "grid": list(style.grid_rgb),
            "axis": list(style.axis_rgb),
            "guide": list(style.guide_rgb),
            "connector": list(style.connector_rgb),
            "neutral_mark": list(style.neutral_mark_rgb),
            "accent": list(style.accent_rgb),
            "secondary_accent": list(style.secondary_accent_rgb),
            "highlight": list(style.highlight_rgb),
            "shadow": list(style.shadow_rgb),
            "callout_fill": list(style.callout_fill_rgb),
            "callout_border": list(style.callout_border_rgb),
        },
        "protected_colors_rgb": [list(color) for color in protected],
        "contrast_checks": {
            **color_separation_metadata(
                anchor_rgb=style.canvas_rgb,
                compared_rgbs=(style.text_rgb, style.axis_rgb, style.accent_rgb),
                prefix="canvas_foreground",
            ),
            **color_separation_metadata(
                anchor_rgb=style.surface_rgb,
                compared_rgbs=(style.text_rgb, style.axis_rgb, style.accent_rgb),
                prefix="surface_foreground",
            ),
            "min_protected_nonsemantic_rgb_distance": round(min(protected_distances), 3)
            if protected_distances
            else 0.0,
        },
        "semantic_color_policy": "style_nonsemantic_roles_only",
        "text_color_policy": "read_required_text_uses_random_nonsemantic_readable_ink",
        "context_text_policy": {
            "supported": True,
            "default_enabled": False,
            "requires_trace_ids_and_bboxes": True,
        },
        "available_treatments": list(INFORMATION_SCENE_TREATMENT_IDS),
        "available_palettes": sorted(INFORMATION_SCENE_PALETTES.keys()),
        "available_chrome_modes": list(INFORMATION_SCENE_CHROME_MODES),
    }


def information_scene_style_from_metadata(metadata: Mapping[str, Any]) -> InformationSceneStyle:
    """Rebuild a resolved style from trace metadata for downstream renderers."""

    roles = metadata.get("roles_rgb")
    if not isinstance(roles, Mapping):
        raise ValueError("information scene metadata missing roles_rgb")
    layout = metadata.get("layout_style")
    if not isinstance(layout, Mapping):
        raise ValueError("information scene metadata missing layout_style")
    surface = metadata.get("surface_style")
    if not isinstance(surface, Mapping):
        raise ValueError("information scene metadata missing surface_style")

    def _role(name: str) -> Color:
        value = roles.get(str(name))
        if value is None:
            raise ValueError(f"information scene metadata missing role: {name}")
        return _normalize_color(value)

    return InformationSceneStyle(
        treatment=str(metadata.get("treatment", "")),
        palette_id=str(metadata.get("palette_id", "")),
        style_pack=str(metadata.get("style_pack", "")),
        surface_kind=str(surface.get("kind", "plain")),
        chrome_kind=str(surface.get("chrome_kind", "none")),
        texture=str(surface.get("texture", "none")),
        chrome_mode=str(surface.get("chrome_mode", "none")),
        canvas_rgb=_role("canvas"),
        canvas_accent_rgb=_role("canvas_accent"),
        surface_rgb=_role("surface"),
        surface_alt_rgb=_role("surface_alt"),
        panel_fill_rgb=_role("panel_fill"),
        panel_border_rgb=_role("panel_border"),
        header_rgb=_role("header"),
        header_text_rgb=_role("header_text"),
        text_rgb=_role("text"),
        muted_text_rgb=_role("muted_text"),
        text_stroke_rgb=_role("text_stroke"),
        grid_rgb=_role("grid"),
        axis_rgb=_role("axis"),
        guide_rgb=_role("guide"),
        connector_rgb=_role("connector"),
        neutral_mark_rgb=_role("neutral_mark"),
        accent_rgb=_role("accent"),
        secondary_accent_rgb=_role("secondary_accent"),
        highlight_rgb=_role("highlight"),
        shadow_rgb=_role("shadow"),
        callout_fill_rgb=_role("callout_fill"),
        callout_border_rgb=_role("callout_border"),
        content_margin_px=int(layout.get("content_margin_px", 24)),
        panel_padding_px=int(layout.get("panel_padding_px", 18)),
        title_band_height_px=int(layout.get("title_band_height_px", 0)),
        frame_width_px=int(layout.get("frame_width_px", 1)),
        corner_radius_px=int(layout.get("corner_radius_px", 0)),
        shadow_offset_px=int(layout.get("shadow_offset_px", 0)),
        guide_density=str(layout.get("guide_density", "normal")),
        context_density=str(layout.get("context_density", "none")),
        typography_scale=float(layout.get("typography_scale", 1.0)),
        protected_colors=tuple(_normalize_color(color) for color in metadata.get("protected_colors_rgb", ())),
    )


def _draw_subtle_texture(draw: ImageDraw.ImageDraw, *, width: int, height: int, style: InformationSceneStyle, instance_seed: int, namespace: str) -> None:
    if style.texture == "none":
        return
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=f"{namespace}.texture")
    if style.texture in {"paper", "scan"}:
        count = max(80, int(width * height // 7000))
        for _ in range(count):
            x = int(rng.randrange(max(1, width)))
            y = int(rng.randrange(max(1, height)))
            delta = int(rng.choice([-5, -3, 3, 5]))
            base = style.canvas_rgb
            color = tuple(max(0, min(255, int(channel) + int(delta))) for channel in base)
            draw.point((x, y), fill=color)
    if style.texture == "scan":
        for y in range(0, int(height), 34):
            draw.line([(0, y), (int(width), y)], fill=style.canvas_accent_rgb, width=1)


def make_information_scene_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: InformationSceneStyle,
    instance_seed: int = 0,
    namespace: str = "information_scene_background",
) -> tuple[Image.Image, dict[str, Any]]:
    """Create a deterministic structured-information background."""

    width = max(1, int(canvas_width))
    height = max(1, int(canvas_height))
    image = Image.new("RGB", (width, height), tuple(int(value) for value in style.canvas_rgb))
    draw = ImageDraw.Draw(image)

    if style.surface_kind in {"dashboard", "app_window", "console", "dark_board"}:
        band_height = max(18, int(style.title_band_height_px))
        draw.rectangle((0, 0, width, band_height), fill=style.canvas_accent_rgb)
    elif style.surface_kind in {"editorial", "publication", "captioned", "source_note", "printout"}:
        inset = max(10, int(style.content_margin_px // 2))
        draw.rectangle((inset, inset, width - inset - 1, height - inset - 1), outline=style.canvas_accent_rgb, width=1)
    elif style.surface_kind in {"infographic", "poster", "annotated"}:
        side = max(8, int(style.content_margin_px // 2))
        draw.rectangle((0, 0, side, height), fill=style.canvas_accent_rgb)

    if style.guide_density == "dense":
        step = 32
        for x in range(step, width, step):
            draw.line([(x, 0), (x, height)], fill=style.canvas_accent_rgb, width=1)
        for y in range(step, height, step):
            draw.line([(0, y), (width, y)], fill=style.canvas_accent_rgb, width=1)

    if style.chrome_mode != "none":
        inset = max(8, int(style.content_margin_px // 2))
        frame_color = style.panel_border_rgb if style.chrome_mode == "thin_frame" else style.accent_rgb
        draw.rectangle(
            (inset, inset, width - inset - 1, height - inset - 1),
            outline=frame_color,
            width=max(1, int(style.frame_width_px)),
        )

    _draw_subtle_texture(
        draw,
        width=width,
        height=height,
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )

    metadata = {
        "enabled": True,
        "selected_style": f"information_scene_style:{style.style_pack}",
        "style_spec": information_scene_style_metadata(style),
    }
    return image, metadata


__all__ = [
    "Color",
    "DEFAULT_INFORMATION_SCENE_CHROME_WEIGHTS",
    "INFORMATION_SCENE_CHROME_MODES",
    "INFORMATION_SCENE_PALETTES",
    "INFORMATION_SCENE_TREATMENTS",
    "INFORMATION_SCENE_TREATMENT_IDS",
    "InformationScenePalette",
    "InformationSceneStyle",
    "InformationSceneTreatment",
    "information_scene_style_from_metadata",
    "information_scene_style_metadata",
    "make_information_scene_background",
    "resolve_information_scene_style",
    "resolve_information_scene_style_from_request",
]
