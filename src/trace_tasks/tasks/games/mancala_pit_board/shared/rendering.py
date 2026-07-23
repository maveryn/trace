"""Rendering primitives for Mancala-style pit-board tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.marking import draw_optional_marker_x
from trace_tasks.tasks.games.shared.scene_style import draw_panel_scene_chrome, make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_role_trace, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import pit_label, visual_row_col
from .state import DEFAULTS, PIT_COUNT, PITS_PER_ROW, STYLE_VARIANTS, MancalaSample, MancalaSceneAxes, MancalaTheme, PitBBox, RenderedMancalaScene


def _theme_for_style(style_variant: str) -> Tuple[MancalaTheme, Dict[str, Any]]:
    """Return a board palette while preserving the style id in trace metadata."""

    themes: dict[str, MancalaTheme] = {
        "wood_tray": MancalaTheme(
            tray_fill_rgb=(189, 128, 74),
            tray_border_rgb=(90, 56, 34),
            pit_fill_rgb=(123, 75, 43),
            pit_shadow_rgb=(80, 48, 30),
            pit_outline_rgb=(236, 190, 126),
            seed_rgbs=((241, 232, 196), (142, 78, 55), (64, 82, 106)),
            seed_outline_rgb=(54, 38, 32),
            label_fill_rgb=(250, 232, 185),
            label_text_rgb=(62, 42, 25),
            arrow_rgb=(56, 39, 28),
            source_marker_rgb=(224, 38, 42),
            target_marker_rgb=(25, 127, 194),
        ),
        "sand_stone": MancalaTheme(
            tray_fill_rgb=(219, 202, 162),
            tray_border_rgb=(106, 94, 68),
            pit_fill_rgb=(175, 157, 116),
            pit_shadow_rgb=(114, 101, 76),
            pit_outline_rgb=(249, 239, 202),
            seed_rgbs=((48, 55, 69), (198, 80, 62), (246, 236, 202)),
            seed_outline_rgb=(53, 48, 39),
            label_fill_rgb=(66, 72, 82),
            label_text_rgb=(248, 244, 229),
            arrow_rgb=(72, 67, 54),
            source_marker_rgb=(218, 43, 50),
            target_marker_rgb=(26, 117, 190),
        ),
        "slate_bowls": MancalaTheme(
            tray_fill_rgb=(58, 70, 86),
            tray_border_rgb=(213, 220, 228),
            pit_fill_rgb=(32, 41, 55),
            pit_shadow_rgb=(14, 22, 34),
            pit_outline_rgb=(154, 170, 190),
            seed_rgbs=((240, 244, 247), (255, 188, 74), (86, 205, 189)),
            seed_outline_rgb=(13, 20, 31),
            label_fill_rgb=(228, 235, 242),
            label_text_rgb=(25, 32, 43),
            arrow_rgb=(236, 242, 249),
            source_marker_rgb=(255, 82, 94),
            target_marker_rgb=(72, 191, 255),
        ),
        "cloth_pits": MancalaTheme(
            tray_fill_rgb=(65, 119, 96),
            tray_border_rgb=(225, 232, 216),
            pit_fill_rgb=(38, 86, 69),
            pit_shadow_rgb=(23, 58, 47),
            pit_outline_rgb=(174, 218, 185),
            seed_rgbs=((252, 235, 169), (112, 43, 82), (43, 50, 77)),
            seed_outline_rgb=(24, 35, 33),
            label_fill_rgb=(245, 243, 217),
            label_text_rgb=(33, 74, 58),
            arrow_rgb=(235, 241, 220),
            source_marker_rgb=(227, 43, 59),
            target_marker_rgb=(31, 128, 203),
        ),
        "arcade_pits": MancalaTheme(
            tray_fill_rgb=(43, 42, 83),
            tray_border_rgb=(255, 213, 87),
            pit_fill_rgb=(28, 24, 55),
            pit_shadow_rgb=(10, 9, 25),
            pit_outline_rgb=(96, 219, 232),
            seed_rgbs=((251, 90, 154), (96, 231, 168), (255, 235, 99)),
            seed_outline_rgb=(11, 12, 28),
            label_fill_rgb=(255, 230, 98),
            label_text_rgb=(29, 27, 61),
            arrow_rgb=(96, 219, 232),
            source_marker_rgb=(255, 78, 95),
            target_marker_rgb=(98, 215, 255),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in themes else "wood_tray"
    return themes[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(STYLE_VARIANTS),
        "board_style_policy": "scene_local_mancala_pit_board_palette",
    }


def _bbox_pad(bbox: Sequence[float], pad: float) -> PitBBox:
    return (
        round(float(bbox[0]) - float(pad), 3),
        round(float(bbox[1]) - float(pad), 3),
        round(float(bbox[2]) + float(pad), 3),
        round(float(bbox[3]) + float(pad), 3),
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    text: str,
    *,
    font: Any,
    fill: Sequence[int],
    surface_rgb: Sequence[int],
    instance_seed: int,
    namespace: str,
    role: str,
    stroke_width: int = 1,
) -> Dict[str, Any]:
    """Draw centered text with game text contrast tracing."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    bbox_center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
    bbox_center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
    text_center_x = 0.5 * (float(text_bbox[0]) + float(text_bbox[2]))
    text_center_y = 0.5 * (float(text_bbox[1]) + float(text_bbox[3]))
    x = bbox_center_x - text_center_x
    y = bbox_center_y - text_center_y
    return draw_text_traced(
        draw,
        (float(x), float(y)),
        str(text),
        font=font,
        fill=tuple(int(value) for value in fill[:3]),
        stroke_width=max(0, int(stroke_width)),
        role=str(role),
        required=True,
        surface_rgbs=(tuple(int(value) for value in surface_rgb[:3]),),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def _pit_badge_bbox(
    pit_bbox: Sequence[float],
    *,
    row: int,
    badge_size: float,
    side: str,
) -> PitBBox:
    """Place a small badge outside a pit so it does not cover seeds."""

    if str(side) == "left":
        cx = float(pit_bbox[0]) + (0.18 * float(badge_size))
    elif str(side) == "right":
        cx = float(pit_bbox[2]) - (0.18 * float(badge_size))
    else:
        cx = 0.5 * (float(pit_bbox[0]) + float(pit_bbox[2]))
    if int(row) == 0:
        cy = float(pit_bbox[1]) - (0.18 * float(badge_size))
    else:
        cy = float(pit_bbox[3]) + (0.18 * float(badge_size))
    return (
        round(cx - (0.5 * float(badge_size)), 3),
        round(cy - (0.5 * float(badge_size)), 3),
        round(cx + (0.5 * float(badge_size)), 3),
        round(cy + (0.5 * float(badge_size)), 3),
    )


def _draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float], *, fill: Sequence[int], width: int) -> None:
    """Draw one direction arrow between adjacent visual pit positions."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    draw.line((sx, sy, ex, ey), fill=tuple(int(value) for value in fill[:3]) + (235,), width=max(2, int(width)))
    angle = math.atan2(ey - sy, ex - sx)
    head_len = max(10.0, float(width) * 3.0)
    head_angle = math.radians(30.0)
    for sign in (-1.0, 1.0):
        hx = ex - head_len * math.cos(angle + sign * head_angle)
        hy = ey - head_len * math.sin(angle + sign * head_angle)
        draw.line((ex, ey, hx, hy), fill=tuple(int(value) for value in fill[:3]) + (235,), width=max(2, int(width)))


def _seed_offsets(count: int, *, pit_width: float, pit_height: float, seed_diameter: float, rng: Any) -> Tuple[Tuple[float, float], ...]:
    """Place visible seeds in one or two loose rows inside a pit."""

    count = int(count)
    if count <= 0:
        return ()
    rows = 1 if count <= 4 else 2
    first_row = int(math.ceil(float(count) / float(rows)))
    row_counts = [first_row]
    if rows == 2:
        row_counts.append(count - first_row)
    offsets: list[Tuple[float, float]] = []
    y_positions = [0.0] if rows == 1 else [-0.32 * pit_height, 0.32 * pit_height]
    for row_index, row_count in enumerate(row_counts):
        if row_count <= 0:
            continue
        if row_count == 1:
            xs = [0.0]
        else:
            span = min(float(pit_width) - (1.55 * float(seed_diameter)), (float(row_count) - 1.0) * float(seed_diameter) * 1.18)
            xs = [-0.5 * span + (span * float(col) / float(row_count - 1)) for col in range(row_count)]
        for x in xs:
            offsets.append(
                (
                    round(float(x) + float(rng.uniform(-1.2, 1.2)), 3),
                    round(float(y_positions[row_index]) + float(rng.uniform(-1.1, 1.1)), 3),
                )
            )
    return tuple(offsets[:count])


def render_mancala_scene(
    *,
    sample: MancalaSample,
    axes: MancalaSceneAxes,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedMancalaScene:
    """Render the full pit board while preserving pit ids and geometry maps."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    pit_width = int(params.get("pit_width_px", group_default(render_defaults, "pit_width_px", DEFAULTS.pit_width_px)))
    pit_height = int(params.get("pit_height_px", group_default(render_defaults, "pit_height_px", DEFAULTS.pit_height_px)))
    pit_gap = int(params.get("pit_gap_px", group_default(render_defaults, "pit_gap_px", DEFAULTS.pit_gap_px)))
    row_gap = int(params.get("row_gap_px", group_default(render_defaults, "row_gap_px", DEFAULTS.row_gap_px)))
    board_padding = int(params.get("board_padding_px", group_default(render_defaults, "board_padding_px", DEFAULTS.board_padding_px)))
    seed_diameter_min = int(params.get("seed_diameter_min_px", group_default(render_defaults, "seed_diameter_min_px", DEFAULTS.seed_diameter_min_px)))
    seed_diameter_max = int(params.get("seed_diameter_max_px", group_default(render_defaults, "seed_diameter_max_px", DEFAULTS.seed_diameter_max_px)))
    seed_diameter = int(rng.randint(min(seed_diameter_min, seed_diameter_max), max(seed_diameter_min, seed_diameter_max)))
    pit_outline_width = int(params.get("pit_outline_width_px", group_default(render_defaults, "pit_outline_width_px", DEFAULTS.pit_outline_width_px)))
    marker_width = int(params.get("marker_width_px", group_default(render_defaults, "marker_width_px", DEFAULTS.marker_width_px)))
    board_width = (PITS_PER_ROW * pit_width) + ((PITS_PER_ROW - 1) * pit_gap)
    board_height = (2 * pit_height) + row_gap
    side_padding = int(params.get("canvas_side_padding_px", group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)))
    vertical_padding = int(params.get("canvas_vertical_padding_px", group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px)))
    canvas_width = min(
        int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        max(
            int(params.get("canvas_min_width_px", group_default(render_defaults, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
            int(board_width + side_padding),
        ),
    )
    canvas_height = min(
        int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        max(
            int(params.get("canvas_min_height_px", group_default(render_defaults, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
            int(board_height + vertical_padding),
        ),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene_style",
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=panel_style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme, theme_meta = _theme_for_style(str(axes.style_variant))
    board_bbox = (
        round(0.5 * (float(canvas_width) - float(board_width)), 3),
        round(0.5 * (float(canvas_height) - float(board_height)), 3),
        round(0.5 * (float(canvas_width) + float(board_width)), 3),
        round(0.5 * (float(canvas_height) + float(board_height)), 3),
    )
    layout_jitter = resolve_games_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    padded_board_bbox = _bbox_pad(board_bbox, float(board_padding))
    tray_bbox, dx, dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=padded_board_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    board_bbox = (
        round(float(board_bbox[0]) + float(dx), 3),
        round(float(board_bbox[1]) + float(dy), 3),
        round(float(board_bbox[2]) + float(dx), 3),
        round(float(board_bbox[3]) + float(dy), 3),
    )
    draw_panel_scene_chrome(
        draw,
        bbox=tuple(int(round(value)) for value in tray_bbox),
        style=panel_style,
        radius=32,
        border_width=max(2, int(round(float(pit_outline_width) * 0.8))),
    )
    draw.rounded_rectangle(
        tray_bbox,
        radius=max(28, int(round(float(pit_height) * 0.55))),
        fill=tuple(theme.tray_fill_rgb) + (242,),
        outline=tuple(theme.tray_border_rgb) + (255,),
        width=max(2, int(pit_outline_width)),
    )

    pit_bboxes: Dict[str, List[float]] = {}
    pit_centers: Dict[str, List[float]] = {}
    seed_centers: Dict[str, List[List[float]]] = {}
    entities: List[Dict[str, Any]] = []
    label_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.markers",
        params=params,
        explicit_key="marker_font_family",
        weights_key="marker_font_family_weights",
    )
    option_font = load_font(max(18, int(round(float(pit_height) * 0.34))), bold=True, font_family=label_font_family)
    board_left = float(board_bbox[0])
    board_top = float(board_bbox[1])
    pit_rows: Dict[str, int] = {}
    for pit_index in range(PIT_COUNT):
        row, col = visual_row_col(pit_index)
        x0 = board_left + (float(col) * float(pit_width + pit_gap))
        y0 = board_top + (float(row) * float(pit_height + row_gap))
        pit_bbox: PitBBox = (round(x0, 3), round(y0, 3), round(x0 + pit_width, 3), round(y0 + pit_height, 3))
        center = [round(0.5 * (pit_bbox[0] + pit_bbox[2]), 3), round(0.5 * (pit_bbox[1] + pit_bbox[3]), 3)]
        label = pit_label(pit_index)
        pit_id = f"pit_{label}"
        shadow_bbox = (pit_bbox[0], pit_bbox[1] + 5.0, pit_bbox[2], pit_bbox[3] + 5.0)
        draw.ellipse(shadow_bbox, fill=tuple(theme.pit_shadow_rgb) + (120,))
        draw.ellipse(
            pit_bbox,
            fill=tuple(theme.pit_fill_rgb) + (255,),
            outline=tuple(theme.pit_outline_rgb) + (255,),
            width=max(2, int(pit_outline_width)),
        )
        pit_bboxes[pit_id] = [float(value) for value in pit_bbox]
        pit_centers[pit_id] = [float(value) for value in center]
        pit_rows[pit_id] = int(row)
        seed_rng = spawn_rng(int(instance_seed), f"{namespace}.pit.{label}.seeds")
        offsets = _seed_offsets(
            int(sample.initial_counts[pit_index]),
            pit_width=float(pit_width) * 0.78,
            pit_height=float(pit_height) * 0.50,
            seed_diameter=float(seed_diameter),
            rng=seed_rng,
        )
        pit_seed_centers: list[list[float]] = []
        for seed_index, offset in enumerate(offsets):
            sx = float(center[0]) + float(offset[0]) + 8.0
            sy = float(center[1]) + float(offset[1]) + 5.0
            seed_bbox = (
                sx - (0.5 * float(seed_diameter)),
                sy - (0.5 * float(seed_diameter)),
                sx + (0.5 * float(seed_diameter)),
                sy + (0.5 * float(seed_diameter)),
            )
            seed_rgb = tuple(theme.seed_rgbs[(pit_index + seed_index) % len(theme.seed_rgbs)])
            draw.ellipse(
                seed_bbox,
                fill=seed_rgb + (255,),
                outline=tuple(theme.seed_outline_rgb) + (235,),
                width=max(1, int(round(float(seed_diameter) * 0.10))),
            )
            highlight_radius = max(2.0, float(seed_diameter) * 0.15)
            draw.ellipse(
                (
                    sx - (0.20 * float(seed_diameter)),
                    sy - (0.24 * float(seed_diameter)),
                    sx - (0.20 * float(seed_diameter)) + highlight_radius,
                    sy - (0.24 * float(seed_diameter)) + highlight_radius,
                ),
                fill=(255, 255, 255, 84),
            )
            pit_seed_centers.append([round(float(sx), 3), round(float(sy), 3)])
        seed_centers[pit_id] = pit_seed_centers
        entities.append(
            {
                "entity_id": str(pit_id),
                "entity_type": "mancala_pit",
                "pit_index": int(pit_index),
                "label": str(label),
                "row": int(row),
                "col": int(col),
                "initial_seed_count": int(sample.initial_counts[pit_index]),
                "final_seed_count": int(sample.final_counts[pit_index]),
                "center_px": list(center),
                "bbox_px": [float(value) for value in pit_bbox],
            }
        )

    arrow_width = max(3, int(round(float(pit_outline_width) * 0.85)))
    top_y = float(board_bbox[1]) - 20.0
    bottom_y = float(board_bbox[3]) + 20.0
    left_x = float(board_bbox[0]) - 24.0
    right_x = float(board_bbox[2]) + 24.0
    for pit_index in range(PITS_PER_ROW - 1):
        start_pit = pit_centers[f"pit_{pit_label(pit_index)}"]
        end_pit = pit_centers[f"pit_{pit_label(pit_index + 1)}"]
        _draw_arrow(draw, (start_pit[0] + 0.36 * pit_width, top_y), (end_pit[0] - 0.36 * pit_width, top_y), fill=theme.arrow_rgb, width=arrow_width)
    last_top_pit = f"pit_{pit_label(PITS_PER_ROW - 1)}"
    first_bottom_pit = f"pit_{pit_label(PITS_PER_ROW)}"
    _draw_arrow(draw, (right_x, float(pit_centers[last_top_pit][1]) + 0.30 * pit_height), (right_x, float(pit_centers[first_bottom_pit][1]) - 0.30 * pit_height), fill=theme.arrow_rgb, width=arrow_width)
    for pit_index in range(PITS_PER_ROW, PIT_COUNT - 1):
        start_pit = pit_centers[f"pit_{pit_label(pit_index)}"]
        end_pit = pit_centers[f"pit_{pit_label(pit_index + 1)}"]
        _draw_arrow(draw, (start_pit[0] - 0.36 * pit_width, bottom_y), (end_pit[0] + 0.36 * pit_width, bottom_y), fill=theme.arrow_rgb, width=arrow_width)
    last_bottom_pit = f"pit_{pit_label(PIT_COUNT - 1)}"
    first_top_pit = f"pit_{pit_label(0)}"
    _draw_arrow(draw, (left_x, float(pit_centers[last_bottom_pit][1]) - 0.30 * pit_height), (left_x, float(pit_centers[first_top_pit][1]) + 0.30 * pit_height), fill=theme.arrow_rgb, width=arrow_width)

    option_marker_bboxes: Dict[str, List[float]] = {}
    option_marker_centers: Dict[str, List[float]] = {}
    option_marker_pit_ids: Dict[str, str] = {}
    for option_label, option_index in zip(sample.option_labels, sample.option_pit_indices):
        option_pit_id = f"pit_{pit_label(int(option_index))}"
        option_pit_bbox = pit_bboxes[option_pit_id]
        option_bbox = _pit_badge_bbox(
            option_pit_bbox,
            row=pit_rows[option_pit_id],
            badge_size=max(26.0, float(pit_height) * 0.40),
            side="left",
        )
        draw.ellipse(
            option_bbox,
            fill=tuple(theme.label_fill_rgb) + (250,),
            outline=tuple(theme.tray_border_rgb) + (255,),
            width=max(2, int(round(float(pit_outline_width) * 0.45))),
        )
        _draw_centered_text(
            draw,
            option_bbox,
            str(option_label),
            font=option_font,
            fill=theme.label_text_rgb,
            surface_rgb=theme.label_fill_rgb,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.option_marker.{option_label}",
            role="option_label",
            stroke_width=0,
        )
        option_marker_bboxes[str(option_label)] = [float(value) for value in option_bbox]
        option_marker_centers[str(option_label)] = [
            round(0.5 * (float(option_bbox[0]) + float(option_bbox[2])), 3),
            round(0.5 * (float(option_bbox[1]) + float(option_bbox[3])), 3),
        ]
        option_marker_pit_ids[str(option_label)] = option_pit_id
        entities.append(
            {
                "entity_id": f"landing_option_{option_label}",
                "entity_type": "mancala_landing_option",
                "label": str(option_label),
                "pit_id": option_pit_id,
                "pit_index": int(option_index),
                "center_px": list(option_marker_centers[str(option_label)]),
                "bbox_px": list(option_marker_bboxes[str(option_label)]),
            }
        )

    source_pit_id = f"pit_{pit_label(sample.source_index)}"
    source_bbox = pit_bboxes[source_pit_id]
    source_marker_bbox = _bbox_pad(source_bbox, max(5.0, float(marker_width) * 1.2))
    draw.ellipse(
        source_marker_bbox,
        outline=tuple(theme.source_marker_rgb) + (255,),
        width=max(3, int(marker_width)),
    )
    source_badge_size = max(24.0, float(pit_height) * 0.38)
    source_badge_bbox = _pit_badge_bbox(
        source_bbox,
        row=pit_rows[source_pit_id],
        badge_size=source_badge_size,
        side="right",
    )
    draw.rounded_rectangle(
        source_badge_bbox,
        radius=max(5, int(round(float(source_badge_size) * 0.25))),
        fill=(255, 255, 255, 246),
        outline=tuple(theme.source_marker_rgb) + (255,),
        width=max(1, int(round(float(marker_width) * 0.38))),
    )
    draw_optional_marker_x(
        draw,
        source_badge_bbox,
        enabled=True,
        width=max(3, int(round(float(marker_width) * 0.85))),
        inset_fraction=0.20,
        outer_rgb=(255, 255, 255),
        inner_rgb=theme.source_marker_rgb,
        marker_kind="source_pit_x",
        extra_metadata={"pit_id": source_pit_id},
    )
    marker_metadata: Dict[str, Any] = {
        "source_pit_marker": {
            "pit_id": source_pit_id,
            "bbox_px": list(source_marker_bbox),
            "x_badge_bbox_px": list(source_badge_bbox),
        }
    }
    if sample.target_index is not None:
        target_pit_id = f"pit_{pit_label(int(sample.target_index))}"
        target_bbox = pit_bboxes[target_pit_id]
        target_marker_bbox = _bbox_pad(target_bbox, max(7.0, float(marker_width) * 1.5))
        draw.ellipse(
            target_marker_bbox,
            outline=tuple(theme.target_marker_rgb) + (255,),
            width=max(3, int(marker_width)),
        )
        badge_size = max(24.0, float(pit_height) * 0.38)
        badge_bbox = _pit_badge_bbox(
            target_bbox,
            row=pit_rows[target_pit_id],
            badge_size=badge_size,
            side="right",
        )
        draw.rounded_rectangle(
            badge_bbox,
            radius=max(5, int(round(float(badge_size) * 0.25))),
            fill=(255, 255, 255, 246),
            outline=tuple(theme.target_marker_rgb) + (255,),
            width=max(1, int(round(float(marker_width) * 0.38))),
        )
        _draw_centered_text(
            draw,
            badge_bbox,
            "T",
            font=option_font,
            fill=theme.target_marker_rgb,
            surface_rgb=(255, 255, 255),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.target_marker",
            role="target_marker_label",
            stroke_width=0,
        )
        marker_metadata["target_pit_marker"] = {"pit_id": target_pit_id, "bbox_px": list(target_marker_bbox), "t_badge_bbox_px": list(badge_bbox)}

    render_map = {
        "board_bbox_px": [float(value) for value in board_bbox],
        "tray_bbox_px": [float(value) for value in tray_bbox],
        "pit_bboxes_px": dict(pit_bboxes),
        "pit_centers_px": dict(pit_centers),
        "seed_centers_px": dict(seed_centers),
        "landing_option_marker_bboxes_px": dict(option_marker_bboxes),
        "landing_option_marker_centers_px": dict(option_marker_centers),
        "landing_option_marker_pit_ids": dict(option_marker_pit_ids),
        "layout_jitter": dict(resolved_jitter),
        "marker_metadata": dict(marker_metadata),
        "effective_pit_width_px": int(pit_width),
        "effective_pit_height_px": int(pit_height),
        "effective_seed_diameter_px": int(seed_diameter),
        "effective_pit_outline_width_px": int(pit_outline_width),
        "marker_font": font_role_trace(str(label_font_family), role="readout"),
    }
    return RenderedMancalaScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "mancala_pit_board_style": dict(theme_meta),
            "marker_font": font_role_trace(str(label_font_family), role="readout"),
        },
        background_meta=dict(background_meta),
    )


__all__ = ["render_mancala_scene"]
