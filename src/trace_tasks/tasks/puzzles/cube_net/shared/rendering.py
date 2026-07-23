"""Rendering primitives for cube-net puzzle panels and option cards."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.shared.text_rendering import load_font

from .sampling import resolve_scene_int
from .state import (
    DEFAULTS,
    FACE_IDS,
    NET_COORDS,
    SIDE_OFFSETS,
    FaceOption,
    FaceRelationDataset,
    NetEquivalenceDataset,
    NetEquivalenceOption,
)


def _render_int(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    return int(params.get(str(key), group_default(rendering_defaults, str(key), int(fallback))))


def style_face_colors(style: Any) -> Dict[str, Tuple[int, int, int]]:
    """Map face ids onto the resolved puzzle style's reusable state colors."""

    colors = [
        tuple(int(value) for value in color)
        for color in tuple(style.state_colors)
    ]
    while len(colors) < len(FACE_IDS):
        colors.append(tuple(int(value) for value in style.panel_accent_rgb))
    return {
        face: tuple(colors[index % len(colors)])
        for index, face in enumerate(FACE_IDS)
    }


def _net_rotation_degrees(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Resolve a nonsemantic 90-degree cube-net rotation."""

    allowed = (0, 90, 180, 270)
    explicit = params.get("net_rotation_degrees")
    if explicit is not None:
        value = int(explicit)
        if value not in allowed:
            raise ValueError("net_rotation_degrees must be one of 0, 90, 180, 270")
        return int(value)
    rng = spawn_rng(int(instance_seed), f"{namespace}.net_rotation")
    return int(uniform_choice(rng, allowed))


def _rotated_net_grid_coords(rotation_degrees: int) -> Dict[str, Tuple[int, int]]:
    """Return display-grid coordinates for the cube net after 90-degree turns."""

    min_x = min(coord[0] for coord in NET_COORDS.values())
    max_x = max(coord[0] for coord in NET_COORDS.values())
    min_y = min(coord[1] for coord in NET_COORDS.values())
    max_y = max(coord[1] for coord in NET_COORDS.values())
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    rotation = int(rotation_degrees) % 360
    rotated: Dict[str, Tuple[int, int]] = {}
    for face, (x, y) in NET_COORDS.items():
        nx = int(x - min_x)
        ny = int(y - min_y)
        if rotation == 0:
            rx, ry = nx, ny
        elif rotation == 90:
            rx, ry = int(height - 1 - ny), nx
        elif rotation == 180:
            rx, ry = int(width - 1 - nx), int(height - 1 - ny)
        elif rotation == 270:
            rx, ry = ny, int(width - 1 - nx)
        else:
            raise ValueError("net rotation must be 0, 90, 180, or 270 degrees")
        rotated[str(face)] = (int(rx), int(ry))
    return rotated


def _rotated_side(side: str, rotation_degrees: int) -> str:
    """Map a canonical face side to the displayed side after net rotation."""

    order = ("top", "right", "bottom", "left")
    turns = (int(rotation_degrees) % 360) // 90
    return str(order[(order.index(str(side)) + turns) % len(order)])


def draw_face_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    style: Any,
    font_size: int,
) -> None:
    """Center a face label inside a square or option-card label region."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    draw_centered_text(
        draw,
        text=str(text),
        center=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        font=load_font(int(font_size), bold=True),
        fill=tuple(style.text_rgb),
        stroke_fill=tuple(style.text_stroke_rgb),
        stroke_width=2,
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int,
    dash_px: int = 10,
    gap_px: int = 7,
) -> None:
    """Draw one dashed seam or paper trim segment between two points."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    step = max(1, int(dash_px) + int(gap_px))
    cursor = 0.0
    while cursor < length:
        dash_end = min(length, cursor + max(1, int(dash_px)))
        sx = x0 + (x1 - x0) * (cursor / length)
        sy = y0 + (y1 - y0) * (cursor / length)
        ex = x0 + (x1 - x0) * (dash_end / length)
        ey = y0 + (y1 - y0) * (dash_end / length)
        draw.line([(sx, sy), (ex, ey)], fill=tuple(fill), width=max(1, int(width)))
        cursor += float(step)


def draw_variant_panel_trim(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    scene_variant: str,
    style: Any,
) -> None:
    """Add nonsemantic chrome for paper-model and game-mat scene variants."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in panel_bbox]
    variant = str(scene_variant)
    if variant == "paper_model":
        inner = (x0 + 10, y0 + 10, x1 - 10, y1 - 10)
        segments = [
            ((inner[0], inner[1]), (inner[2], inner[1])),
            ((inner[2], inner[1]), (inner[2], inner[3])),
            ((inner[2], inner[3]), (inner[0], inner[3])),
            ((inner[0], inner[3]), (inner[0], inner[1])),
        ]
        for start, end in segments:
            draw_dashed_line(
                draw,
                start=start,
                end=end,
                fill=tuple(style.panel_accent_rgb),
                width=1,
                dash_px=12,
                gap_px=8,
            )
    elif variant == "game_mat":
        draw.rounded_rectangle(
            (x0 + 6, y0 + 6, x1 - 6, y1 - 6),
            radius=14,
            outline=tuple(style.panel_accent_rgb),
            width=3,
        )
        for cx, cy in (
            (x0 + 20, y0 + 20),
            (x1 - 20, y0 + 20),
            (x0 + 20, y1 - 20),
            (x1 - 20, y1 - 20),
        ):
            draw.ellipse(
                (cx - 4, cy - 4, cx + 4, cy + 4),
                fill=tuple(style.panel_accent_rgb),
            )


def draw_net_fold_seams(
    draw: ImageDraw.ImageDraw,
    *,
    face_bboxes: Mapping[str, Sequence[float]],
    scene_variant: str,
    style: Any,
) -> None:
    """Draw paper-model seams only along net-adjacent face boundaries."""

    if str(scene_variant) != "paper_model":
        return
    seen: set[tuple[str, str]] = set()
    for face, (x, y) in NET_COORDS.items():
        for dx, dy in SIDE_OFFSETS.values():
            neighbor = next(
                (
                    other
                    for other, coord in NET_COORDS.items()
                    if coord == (x + dx, y + dy)
                ),
                None,
            )
            if neighbor is None:
                continue
            key = tuple(sorted((str(face), str(neighbor))))
            if key in seen:
                continue
            seen.add(key)
            ax0, ay0, ax1, ay1 = [float(value) for value in face_bboxes[str(face)]]
            bx0, by0, bx1, by1 = [float(value) for value in face_bboxes[str(neighbor)]]
            if abs(ax1 - bx0) <= 1.0 or abs(bx1 - ax0) <= 1.0:
                seam_x = ax1 if abs(ax1 - bx0) <= 1.0 else ax0
                draw_dashed_line(
                    draw,
                    start=(seam_x, max(ay0, by0) + 4),
                    end=(seam_x, min(ay1, by1) - 4),
                    fill=tuple(style.panel_accent_rgb),
                    width=2,
                    dash_px=8,
                    gap_px=6,
                )
            elif abs(ay1 - by0) <= 1.0 or abs(by1 - ay0) <= 1.0:
                seam_y = ay1 if abs(ay1 - by0) <= 1.0 else ay0
                draw_dashed_line(
                    draw,
                    start=(max(ax0, bx0) + 4, seam_y),
                    end=(min(ax1, bx1) - 4, seam_y),
                    fill=tuple(style.panel_accent_rgb),
                    width=2,
                    dash_px=8,
                    gap_px=6,
                )


def scene_variant_style_metadata(scene_variant: str) -> Dict[str, Any]:
    """Describe why scene-variant chrome does not alter the reasoning contract."""

    variant = str(scene_variant)
    features = {
        "clean_net": ["standard panel borders"],
        "paper_model": ["dashed paper trim", "dashed fold seams"],
        "game_mat": ["accent inset panel frames", "corner pin markers"],
    }.get(variant, ["standard panel borders"])
    return {
        "scene_variant": variant,
        "visual_features": list(features),
        "semantic_policy": "non_semantic_chrome_only_no_layout_or_answer_change",
    }


def draw_net_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    dataset: FaceRelationDataset,
    style: Any,
    font_size: int,
    cell_size_px: int,
    rotation_degrees: int,
) -> Tuple[Dict[str, list[float]], list[float], Dict[str, list[float]]]:
    """Draw the six-face cube net and mark the relation reference face/edge."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in panel_bbox]
    draw_rounded_rect(
        draw,
        (x0, y0, x1, y1),
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )
    rendered_coords = _rotated_net_grid_coords(rotation_degrees)
    max_x = max(coord[0] for coord in rendered_coords.values())
    max_y = max(coord[1] for coord in rendered_coords.values())
    net_w = int(max_x + 1) * int(cell_size_px)
    net_h = int(max_y + 1) * int(cell_size_px)
    origin_x = int(round(0.5 * (x0 + x1 - net_w)))
    origin_y = int(round(y0 + 42 + max(0, (y1 - y0 - 70 - net_h) * 0.5)))
    face_colors = style_face_colors(style)
    face_bboxes: Dict[str, list[float]] = {}
    relation_bboxes: Dict[str, list[float]] = {}
    for face_id in sorted(
        FACE_IDS,
        key=lambda face: (rendered_coords[face][1], rendered_coords[face][0]),
    ):
        gx, gy = rendered_coords[str(face_id)]
        fx0 = origin_x + int(gx) * int(cell_size_px)
        fy0 = origin_y + int(gy) * int(cell_size_px)
        fx1 = fx0 + int(cell_size_px)
        fy1 = fy0 + int(cell_size_px)
        draw.rectangle(
            (fx0, fy0, fx1, fy1),
            fill=tuple(face_colors[str(face_id)]),
            outline=tuple(style.grid_rgb),
            width=2,
        )
        draw_face_label(
            draw,
            text=str(dataset.face_labels[str(face_id)]),
            bbox=(fx0, fy0, fx1, fy1),
            style=style,
            font_size=int(font_size),
        )
        face_bboxes[str(face_id)] = [float(fx0), float(fy0), float(fx1), float(fy1)]
    ref_bbox = face_bboxes[str(dataset.reference_face)]
    if dataset.marked_side is None:
        draw.rectangle(tuple(ref_bbox), outline=tuple(style.mark_rgb), width=6)
        relation_bboxes["marked_face"] = [float(value) for value in ref_bbox]
    if dataset.marked_side is not None:
        rx0, ry0, rx1, ry1 = [float(value) for value in ref_bbox]
        displayed_side = _rotated_side(str(dataset.marked_side), rotation_degrees)
        if displayed_side == "top":
            start, end = (rx0 + 8, ry0 + 2), (rx1 - 8, ry0 + 2)
        elif displayed_side == "bottom":
            start, end = (rx0 + 8, ry1 - 2), (rx1 - 8, ry1 - 2)
        elif displayed_side == "left":
            start, end = (rx0 + 2, ry0 + 8), (rx0 + 2, ry1 - 8)
        else:
            start, end = (rx1 - 2, ry0 + 8), (rx1 - 2, ry1 - 8)
        draw.line([start, end], fill=tuple(style.panel_fill_rgb), width=16)
        draw.line([start, end], fill=tuple(style.mark_rgb), width=10)
        for cx, cy in (start, end):
            radius = 5
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=tuple(style.mark_rgb),
                outline=tuple(style.panel_fill_rgb),
                width=2,
            )
        edge_pad = 12.0
        relation_bboxes["marked_edge"] = [
            float(min(start[0], end[0]) - edge_pad),
            float(min(start[1], end[1]) - edge_pad),
            float(max(start[0], end[0]) + edge_pad),
            float(max(start[1], end[1]) + edge_pad),
        ]
        relation_bboxes["marked_face"] = [float(value) for value in ref_bbox]
    return face_bboxes, [float(x0), float(y0), float(x1), float(y1)], relation_bboxes


def draw_options(
    draw: ImageDraw.ImageDraw,
    *,
    options: Sequence[FaceOption],
    panel_bbox: Sequence[float],
    title: str,
    style: Any,
    columns: int,
) -> Dict[str, list[float]]:
    """Draw labeled face-option cards and return card bboxes by option id."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in panel_bbox]
    draw_rounded_rect(
        draw,
        (x0, y0, x1, y1),
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )
    columns = max(1, int(columns))
    rows = int((len(options) + columns - 1) // columns)
    pad = 20
    gap = 14
    top = y0 + 24
    usable_w = x1 - x0 - 2 * pad - (columns - 1) * gap
    usable_h = y1 - top - pad - (rows - 1) * gap
    card_w = max(60, int(usable_w / columns))
    card_h = max(54, int(usable_h / rows))
    bboxes: Dict[str, list[float]] = {}
    face_colors = style_face_colors(style)
    for index, option in enumerate(options):
        row = int(index // columns)
        col = int(index % columns)
        bx0 = x0 + pad + col * (card_w + gap)
        by0 = top + row * (card_h + gap)
        bx1 = bx0 + card_w
        by1 = by0 + card_h
        draw_rounded_rect(
            draw,
            (bx0, by0, bx1, by1),
            radius=10,
            fill=tuple(style.option_fill_rgb),
            outline=tuple(style.panel_border_rgb),
            width=2,
        )
        draw_rounded_rect(
            draw,
            (bx0 + 8, by0 + 8, bx0 + 34, by0 + 34),
            radius=6,
            fill=tuple(style.option_marker_fill_rgb),
            outline=tuple(style.panel_border_rgb),
            width=1,
        )
        draw_centered_text(
            draw,
            text=str(option.option_label),
            center=(bx0 + 21, by0 + 21),
            font=load_font(15, bold=True),
            fill=tuple(style.text_rgb),
            stroke_fill=tuple(style.text_stroke_rgb),
            stroke_width=1,
        )
        swatch = (bx0 + 44, by0 + 11, bx0 + 80, by0 + 47)
        draw.rectangle(
            swatch,
            fill=tuple(face_colors[str(option.face_id)]),
            outline=tuple(style.grid_rgb),
            width=1,
        )
        draw_face_label(
            draw,
            text=str(option.face_label),
            bbox=(bx0 + 88, by0 + 12, bx1 - 10, by1 - 10),
            style=style,
            font_size=DEFAULTS.option_font_size_px,
        )
        bboxes[f"option_{option.option_label}"] = [
            float(bx0),
            float(by0),
            float(bx1),
            float(by1),
        ]
    return bboxes


def draw_colored_net_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    face_color_names: Mapping[str, str],
    option_label: str | None,
    panel_title: str,
    style: Any,
    cell_size_px: int,
    rotation_degrees: int,
) -> Tuple[list[float], Dict[str, list[float]]]:
    """Draw one colored cube net panel for reference/option comparison."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in panel_bbox]
    draw_rounded_rect(
        draw,
        (x0, y0, x1, y1),
        radius=16,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )
    if option_label:
        badge = (x0 + 14, y0 + 12, x0 + 46, y0 + 44)
        draw_rounded_rect(
            draw,
            badge,
            radius=9,
            fill=tuple(style.option_marker_fill_rgb),
            outline=tuple(style.panel_border_rgb),
            width=1,
        )
        draw_centered_text(
            draw,
            text=str(option_label),
            center=(x0 + 30, y0 + 28),
            font=load_font(16, bold=True),
            fill=tuple(style.text_rgb),
            stroke_fill=tuple(style.text_stroke_rgb),
            stroke_width=1,
        )
    if panel_title:
        draw_centered_text(
            draw,
            text=str(panel_title),
            center=(0.5 * (x0 + x1), y0 + 26),
            font=load_font(15, bold=True),
            fill=tuple(style.text_rgb),
            stroke_fill=tuple(style.text_stroke_rgb),
            stroke_width=1,
        )

    rendered_coords = _rotated_net_grid_coords(rotation_degrees)
    max_x = max(coord[0] for coord in rendered_coords.values())
    max_y = max(coord[1] for coord in rendered_coords.values())
    net_w = int(max_x + 1) * int(cell_size_px)
    net_h = int(max_y + 1) * int(cell_size_px)
    origin_x = int(round(0.5 * (x0 + x1 - net_w)))
    origin_y = int(round(y0 + 50 + max(0, (y1 - y0 - 66 - net_h) * 0.5)))
    face_bboxes: Dict[str, list[float]] = {}
    for face_id in sorted(
        FACE_IDS,
        key=lambda face: (rendered_coords[face][1], rendered_coords[face][0]),
    ):
        gx, gy = rendered_coords[str(face_id)]
        fx0 = origin_x + int(gx) * int(cell_size_px)
        fy0 = origin_y + int(gy) * int(cell_size_px)
        fx1 = fx0 + int(cell_size_px)
        fy1 = fy0 + int(cell_size_px)
        color_name = str(face_color_names[str(face_id)])
        fill = named_color(color_name)
        draw.rectangle(
            (fx0, fy0, fx1, fy1),
            fill=tuple(fill),
            outline=tuple(style.grid_rgb),
            width=2,
        )
        face_bboxes[str(face_id)] = [float(fx0), float(fy0), float(fx1), float(fy1)]
    draw_net_fold_seams(
        draw,
        face_bboxes=face_bboxes,
        scene_variant="paper_model",
        style=style,
    )
    return [float(x0), float(y0), float(x1), float(y1)], face_bboxes


def render_face_relation_scene(
    *,
    dataset: FaceRelationDataset,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Render the cube-net relation layout and return all source bboxes."""

    width = _render_int(params, rendering_defaults, "canvas_width", DEFAULTS.canvas_width)
    height = _render_int(
        params,
        rendering_defaults,
        "face_relation_canvas_height",
        DEFAULTS.face_relation_canvas_height,
    )
    style, style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace="cube_net.face_relation",
    )
    image, background_meta = make_puzzle_scene_background(
        canvas_width=int(width),
        canvas_height=int(height),
        style=style,
    )
    draw = ImageDraw.Draw(image)
    net_panel = (54, 54, 686, int(height) - 54)
    option_panel = (724, 82, int(width) - 54, int(height) - 82)
    net_rotation_degrees = _net_rotation_degrees(
        params=params,
        instance_seed=int(instance_seed),
        namespace="cube_net.face_relation",
    )
    net_bboxes, net_panel_bbox, relation_bboxes = draw_net_panel(
        draw,
        panel_bbox=net_panel,
        dataset=dataset,
        style=style,
        font_size=_render_int(
            params,
            rendering_defaults,
            "face_font_size_px",
            DEFAULTS.face_font_size_px,
        ),
        cell_size_px=_render_int(
            params,
            rendering_defaults,
            "net_cell_size_px",
            DEFAULTS.net_cell_size_px,
        ),
        rotation_degrees=int(net_rotation_degrees),
    )
    option_bboxes = draw_options(
        draw,
        options=dataset.options,
        panel_bbox=option_panel,
        title="Face options",
        style=style,
        columns=2,
    )
    draw_net_fold_seams(
        draw,
        face_bboxes=net_bboxes,
        scene_variant=str(scene_variant),
        style=style,
    )
    draw_variant_panel_trim(
        draw,
        panel_bbox=net_panel_bbox,
        scene_variant=str(scene_variant),
        style=style,
    )
    draw_variant_panel_trim(
        draw,
        panel_bbox=option_panel,
        scene_variant=str(scene_variant),
        style=style,
    )
    return image, {
        "background_style": dict(background_meta),
        "scene_style": dict(style_meta),
        "scene_variant_style": scene_variant_style_metadata(str(scene_variant)),
        "net_rotation_degrees": int(net_rotation_degrees),
        "net_panel_bbox_px": list(net_panel_bbox),
        "face_bboxes_px": dict(net_bboxes),
        "relation_bboxes_px": dict(relation_bboxes),
        "option_panel_bboxes_px": dict(option_bboxes),
    }


def render_equivalent_net_scene(
    *,
    dataset: NetEquivalenceDataset,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Render a reference colored net and four candidate colored nets."""

    width = _render_int(params, rendering_defaults, "canvas_width", DEFAULTS.canvas_width)
    height = _render_int(
        params,
        rendering_defaults,
        "equivalent_net_canvas_height",
        DEFAULTS.equivalent_net_canvas_height,
    )
    style, style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace="cube_net.equivalent_net",
    )
    image, background_meta = make_puzzle_scene_background(
        canvas_width=int(width),
        canvas_height=int(height),
        style=style,
    )
    draw = ImageDraw.Draw(image)
    rng = spawn_rng(int(instance_seed), "cube_net.equivalent_net.display_rotations")
    rotations = (0, 90, 180, 270)
    reference_rotation = int(uniform_choice(rng, rotations))
    option_rotations = {
        str(option.option_label): int(uniform_choice(rng, rotations))
        for option in dataset.options
    }

    reference_panel = (54, 86, 410, int(height) - 92)
    reference_bbox, reference_face_bboxes = draw_colored_net_panel(
        draw,
        panel_bbox=reference_panel,
        face_color_names=dataset.reference_face_color_names,
        option_label=None,
        panel_title="Reference",
        style=style,
        cell_size_px=86,
        rotation_degrees=int(reference_rotation),
    )
    option_layouts = (
        (456, 72, 748, 394),
        (782, 72, 1074, 394),
        (456, 452, 748, 774),
        (782, 452, 1074, 774),
    )
    option_panel_bboxes: Dict[str, list[float]] = {}
    option_face_bboxes: Dict[str, Dict[str, list[float]]] = {}
    for option, panel_bbox in zip(dataset.options, option_layouts):
        panel, face_bboxes = draw_colored_net_panel(
            draw,
            panel_bbox=panel_bbox,
            face_color_names=option.face_color_names,
            option_label=str(option.option_label),
            panel_title="",
            style=style,
            cell_size_px=54,
            rotation_degrees=int(option_rotations[str(option.option_label)]),
        )
        option_panel_bboxes[f"option_{option.option_label}"] = list(panel)
        option_face_bboxes[f"option_{option.option_label}"] = dict(face_bboxes)

    for panel in (reference_bbox, *option_panel_bboxes.values()):
        draw_variant_panel_trim(
            draw,
            panel_bbox=panel,
            scene_variant=str(scene_variant),
            style=style,
        )
    return image, {
        "background_style": dict(background_meta),
        "scene_style": dict(style_meta),
        "scene_variant_style": scene_variant_style_metadata(str(scene_variant)),
        "reference_panel_bbox_px": list(reference_bbox),
        "reference_face_bboxes_px": dict(reference_face_bboxes),
        "reference_net_rotation_degrees": int(reference_rotation),
        "option_panel_bboxes_px": dict(option_panel_bboxes),
        "option_face_bboxes_px": dict(option_face_bboxes),
        "option_net_rotation_degrees": dict(option_rotations),
    }


__all__ = [
    "render_equivalent_net_scene",
    "render_face_relation_scene",
    "scene_variant_style_metadata",
]
