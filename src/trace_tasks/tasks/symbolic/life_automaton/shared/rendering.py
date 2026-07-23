"""Rendering helpers for symbolic Life automaton scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from .....core.visual.noise import apply_post_image_noise
from ....shared.drawing import draw_centered_text, draw_rounded_rect
from ....shared.text_rendering import load_font, temporary_default_font_family
from ...shared.scene_style import (
    DEFAULT_SYMBOLIC_SCENE_STYLE,
    SymbolicSceneStyle,
    draw_symbolic_chrome_by_mode,
    draw_symbolic_grid_cell,
    draw_symbolic_option_card,
    make_symbolic_scene_background,
)

from .defaults import POST_IMAGE_NOISE_DEFAULTS
from .layout import (
    cell_bbox,
    content_metrics,
    grid_bbox,
    inset_bbox,
    marked_cells_bbox,
    option_card_size,
    option_grid_gap,
    source_header_height,
)
from .state import LifeBoardVisual, LifeRenderBundle, LifeRenderParams, LifeSceneSpec, RenderedLifeScene
from .styles import (
    resolve_life_board_visual,
    resolve_life_style,
    style_meta_with_font,
    style_meta_with_life_board,
)


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], alpha_b: float) -> Tuple[int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - alpha)) + (float(color_b[index]) * alpha)))
        for index in range(3)
    )


def _decorate_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    style: SymbolicSceneStyle = DEFAULT_SYMBOLIC_SCENE_STYLE,
    chrome_mode: str = "accent_frame",
) -> None:
    draw_symbolic_chrome_by_mode(
        draw,
        bbox=bbox,
        style=style,
        radius=22,
        border_width=3,
        mode=str(chrome_mode),
    )


def _draw_option_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    label: str,
    fill: Sequence[int],
    style: SymbolicSceneStyle = DEFAULT_SYMBOLIC_SCENE_STYLE,
) -> None:
    draw_symbolic_option_card(
        draw,
        bbox=bbox,
        style=style,
        fill=fill,
        radius=14,
        border_width=2,
    )
    font = load_font(20, bold=True)
    draw_centered_text(
        draw,
        text=str(label),
        center=(bbox[0] + 20, bbox[1] + 20),
        font=font,
        fill=style.text_rgb,
        stroke_fill=style.text_stroke_rgb,
        stroke_width=1,
    )


def _draw_source_grid_label(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Tuple[int, int, int, int],
    grid_top: int,
    label: str,
    render_params: LifeRenderParams,
    style: SymbolicSceneStyle,
    life_visual: LifeBoardVisual,
) -> None:
    """Draw the explicit marker for the source grid without changing its annotation bbox."""

    font = load_font(max(14, int(round(render_params.label_font_size_px * 0.78))), bold=True)
    text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=1)
    text_w = int(text_bbox[2] - text_bbox[0])
    text_h = int(text_bbox[3] - text_bbox[1])
    pad_x = max(8, int(round(render_params.panel_padding_px * 0.28)))
    pad_y = max(4, int(round(render_params.panel_padding_px * 0.16)))
    pill_w = int(text_w + 2 * pad_x)
    pill_h = int(text_h + 2 * pad_y)
    x0 = int((panel_bbox[0] + panel_bbox[2] - pill_w) / 2)
    header_top = int(panel_bbox[1] + max(3, int(round(render_params.panel_padding_px * 0.18))))
    header_bottom = int(grid_top - max(3, int(round(render_params.panel_padding_px * 0.18))))
    header_center_y = int(round((header_top + header_bottom) / 2))
    y0 = int(header_center_y - pill_h / 2)
    pill_bbox = (x0, y0, int(x0 + pill_w), int(y0 + pill_h))
    fill = _blend_rgb(style.panel_fill_rgb, life_visual.accent_rgb, 0.18)
    draw_rounded_rect(
        draw,
        bbox=pill_bbox,
        radius=max(6, int(round(pill_h * 0.32))),
        fill=fill,
        outline=life_visual.edge_rgb,
        width=1,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(int((pill_bbox[0] + pill_bbox[2]) / 2), int((pill_bbox[1] + pill_bbox[3]) / 2)),
        font=font,
        fill=style.text_rgb,
        stroke_fill=style.text_stroke_rgb,
        stroke_width=1,
    )


def draw_life_cell_grid(
    draw: ImageDraw.ImageDraw,
    *,
    grid: Sequence[Sequence[int]],
    left: int,
    top: int,
    cell_size: int,
    gap: int,
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    item_prefix: str,
    target_cells: Sequence[Tuple[int, int]] = tuple(),
    style: SymbolicSceneStyle = DEFAULT_SYMBOLIC_SCENE_STYLE,
    life_visual: LifeBoardVisual,
) -> None:
    """Draw one Life grid and record cell bboxes."""

    target_set = {(int(row), int(col)) for row, col in target_cells}
    rows = len(grid)
    cols = len(grid[0])
    board_box = grid_bbox(left=left, top=top, rows=rows, cols=cols, cell_size=cell_size, gap=gap)
    board_style = str(life_visual.board_style)
    if board_style in {"inset_tiles", "terminal_cells"}:
        draw_rounded_rect(
            draw,
            bbox=board_box,
            radius=max(4, int(cell_size * 0.14)),
            fill=life_visual.grid_rgb,
            outline=life_visual.edge_rgb,
            width=max(1, int(round(cell_size * 0.04))),
        )
    for row in range(rows):
        for col in range(cols):
            bbox = cell_bbox(left=left, top=top, row=row, col=col, cell_size=cell_size, gap=gap)
            state = int(grid[row][col])
            fill = life_visual.alive_rgb if state else life_visual.dead_rgb
            cell_id = f"{item_prefix}_cell_{row}_{col}"
            item_bboxes[cell_id] = bbox
            if board_style == "classic_grid":
                draw.rectangle(
                    bbox,
                    fill=fill,
                    outline=life_visual.grid_rgb,
                    width=max(1, int(round(cell_size * 0.035))),
                )
            elif board_style == "rounded_tiles":
                draw_rounded_rect(
                    draw,
                    bbox=bbox,
                    radius=max(3, int(cell_size * 0.10)),
                    fill=fill,
                    outline=life_visual.grid_rgb,
                    width=max(1, int(round(cell_size * 0.035))),
                )
            elif board_style == "inset_tiles":
                inner = inset_bbox(bbox, max(1, int(round(cell_size * 0.08))))
                draw_rounded_rect(
                    draw,
                    bbox=inner,
                    radius=max(3, int(cell_size * 0.12)),
                    fill=fill,
                    outline=life_visual.edge_rgb,
                    width=1,
                )
            elif board_style == "lab_matrix":
                draw.rectangle(
                    bbox,
                    fill=fill,
                    outline=life_visual.edge_rgb,
                    width=max(1, int(round(cell_size * 0.04))),
                )
                highlight = _blend_rgb(fill, (255, 255, 255), 0.24 if state else 0.42)
                draw.line((bbox[0] + 2, bbox[1] + 2, bbox[2] - 3, bbox[1] + 2), fill=highlight, width=1)
                draw.line((bbox[0] + 2, bbox[1] + 2, bbox[0] + 2, bbox[3] - 3), fill=highlight, width=1)
            elif board_style == "notebook_cells":
                draw.rectangle(bbox, fill=fill, outline=life_visual.grid_rgb, width=1)
                if not state:
                    line_y = int(round((bbox[1] + bbox[3]) / 2.0))
                    draw.line((bbox[0] + 4, line_y, bbox[2] - 4, line_y), fill=life_visual.accent_rgb, width=1)
                else:
                    draw.line((bbox[0] + 3, bbox[1] + 3, bbox[2] - 4, bbox[1] + 3), fill=life_visual.accent_rgb, width=1)
            elif board_style == "terminal_cells":
                draw.rectangle(bbox, fill=life_visual.grid_rgb)
                inner = inset_bbox(bbox, max(1, int(round(cell_size * 0.06))))
                draw.rectangle(inner, fill=fill, outline=life_visual.edge_rgb, width=1)
                if state:
                    glow = _blend_rgb(fill, life_visual.accent_rgb, 0.20)
                    draw.rectangle(inset_bbox(inner, max(2, int(round(cell_size * 0.22)))), fill=glow)
            else:
                draw_symbolic_grid_cell(
                    draw,
                    bbox=bbox,
                    fill=fill,
                    style=style,
                    outline=life_visual.grid_rgb,
                    width=1,
                    selected=False,
                )
            if (row, col) in target_set:
                draw.rectangle(
                    bbox,
                    outline=life_visual.mark_rgb,
                    width=max(3, int(cell_size * 0.10)),
                )


def _draw_life_option_grid(
    draw: ImageDraw.ImageDraw,
    *,
    option_grid: Sequence[Sequence[int]],
    bbox: Tuple[int, int, int, int],
    label: str,
    render_params: LifeRenderParams,
    style: SymbolicSceneStyle,
    life_visual: LifeBoardVisual,
) -> None:
    _draw_option_card(draw, bbox=bbox, label=label, fill=style.option_fill_rgb, style=style)
    rows = len(option_grid)
    cols = len(option_grid[0])
    cell = int(render_params.option_grid_cell_px)
    gap = option_grid_gap(render_params)
    grid_w = cols * cell + max(0, cols - 1) * gap
    grid_h = rows * cell + max(0, rows - 1) * gap
    header_h = max(30, int(round(render_params.label_font_size_px * 1.45)))
    left = int(bbox[0] + (bbox[2] - bbox[0] - grid_w) / 2)
    top = int(bbox[1] + header_h + max(0, (bbox[3] - bbox[1] - header_h - grid_h) / 2))
    dummy: Dict[str, Tuple[int, int, int, int]] = {}
    draw_life_cell_grid(
        draw,
        grid=option_grid,
        left=left,
        top=top,
        cell_size=cell,
        gap=gap,
        item_bboxes=dummy,
        item_prefix="option_preview",
        style=style,
        life_visual=life_visual,
    )


def _draw_life_options(
    draw: ImageDraw.ImageDraw,
    *,
    scene: LifeSceneSpec,
    render_params: LifeRenderParams,
    top: int,
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    style: SymbolicSceneStyle,
    life_visual: LifeBoardVisual,
    left: int | None = None,
) -> Tuple[int, int, int, int]:
    rows = int(scene.rows)
    cols = int(scene.cols)
    card_w, card_h = option_card_size(rows=rows, cols=cols, render_params=render_params)
    gap = int(render_params.option_gap_px)
    count = len(scene.option_specs)
    total_w = count * card_w + max(0, count - 1) * gap
    if left is None:
        left = int((render_params.canvas_width - total_w) // 2)
    else:
        left = int(left)
    for index, option in enumerate(scene.option_specs):
        x0 = int(left + index * (card_w + gap))
        bbox = (x0, int(top), int(x0 + card_w), int(top + card_h))
        _draw_life_option_grid(
            draw,
            option_grid=option.grid,
            bbox=bbox,
            label=str(option.label),
            render_params=render_params,
            style=style,
            life_visual=life_visual,
        )
        item_bboxes[str(option.option_id)] = bbox
    return (left, int(top), int(left + total_w), int(top + card_h))


def render_life_scene(
    *,
    background: Image.Image,
    scene: LifeSceneSpec,
    scene_variant: str,
    render_params: LifeRenderParams,
    style: SymbolicSceneStyle,
    style_meta: Mapping[str, Any],
    life_visual: LifeBoardVisual,
) -> RenderedLifeScene:
    """Render one Life scene after task-owned scene construction."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    item_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    rows, cols = int(scene.rows), int(scene.cols)
    cell = int(render_params.cell_size_px)
    gap = int(render_params.grid_gap_px)
    grid_box = grid_bbox(left=0, top=0, rows=rows, cols=cols, cell_size=cell, gap=gap)
    grid_w = int(grid_box[2] - grid_box[0])
    grid_h = int(grid_box[3] - grid_box[1])
    marker_label = str(scene.source_marker_label).strip()
    metrics = content_metrics(scene=scene, render_params=render_params)
    layout_rng = spawn_rng(instance_seed=int(render_params.layout_seed), namespace="life_automaton_panel_origin")
    safe_margin = max(18, int(round(render_params.panel_padding_px * 0.85)))
    content_w = int(metrics["content_width_px"])
    content_h = int(metrics["content_height_px"])
    panel_w = int(metrics["panel_width_px"])
    panel_h = int(metrics["panel_height_px"])
    min_content_left = int(safe_margin)
    max_content_left = int(render_params.canvas_width - content_w - safe_margin)
    if max_content_left >= min_content_left:
        available_x = int(max_content_left - min_content_left)
        offset_x = layout_rng.randint(0, available_x) if available_x > 0 else 0
        content_left = int(min_content_left + offset_x)
    else:
        available_x = 0
        offset_x = 0
        content_left = max(0, int((render_params.canvas_width - content_w) // 2))
    min_content_top = int(safe_margin)
    max_content_top = int(render_params.canvas_height - content_h - safe_margin)
    if max_content_top >= min_content_top:
        available_y = int(max_content_top - min_content_top)
        offset_y = layout_rng.randint(0, available_y) if available_y > 0 else 0
        content_top = int(min_content_top + offset_y)
    else:
        available_y = 0
        offset_y = 0
        content_top = max(0, int((render_params.canvas_height - content_h) // 2))

    panel_left = int(content_left + max(0, content_w - panel_w) // 2)
    panel_top = int(content_top)
    header_h = source_header_height(render_params, enabled=bool(marker_label))
    grid_left = int(panel_left + int(render_params.panel_padding_px))
    grid_top = int(panel_top + int(render_params.panel_padding_px) + header_h)
    panel_bbox = (
        grid_left - int(render_params.panel_padding_px),
        int(panel_top),
        grid_left + grid_w + int(render_params.panel_padding_px),
        grid_top + grid_h + int(render_params.panel_padding_px),
    )
    _decorate_panel(
        draw,
        bbox=panel_bbox,
        style=style,
        chrome_mode=str(style_meta.get("panel_chrome_mode", "accent_frame")),
    )

    if marker_label:
        _draw_source_grid_label(
            draw,
            panel_bbox=panel_bbox,
            grid_top=grid_top,
            label=marker_label,
            render_params=render_params,
            style=style,
            life_visual=life_visual,
        )
    draw_life_cell_grid(
        draw,
        grid=scene.initial_grid,
        left=grid_left,
        top=grid_top,
        cell_size=cell,
        gap=gap,
        item_bboxes=item_bboxes,
        item_prefix="source",
        target_cells=scene.target_cells,
        style=style,
        life_visual=life_visual,
    )
    item_bboxes["source_grid"] = grid_bbox(left=grid_left, top=grid_top, rows=rows, cols=cols, cell_size=cell, gap=gap)

    if scene.target_cells:
        item_bboxes["marked_line"] = marked_cells_bbox(
            cells=scene.target_cells,
            grid_left=grid_left,
            grid_top=grid_top,
            cell_size=cell,
            gap=gap,
        )

    option_bbox = (0, 0, 0, 0)
    if scene.option_specs:
        option_left = int(content_left + max(0, content_w - int(metrics["option_width_px"])) // 2)
        option_bbox = _draw_life_options(
            draw,
            scene=scene,
            render_params=render_params,
            top=int(panel_bbox[3] + int(metrics["option_vertical_gap_px"])),
            item_bboxes=item_bboxes,
            style=style,
            life_visual=life_visual,
            left=option_left,
        )

    scene_bbox = (
        min(panel_bbox[0], option_bbox[0]) if option_bbox[2] else panel_bbox[0],
        panel_bbox[1],
        max(panel_bbox[2], option_bbox[2]) if option_bbox[2] else panel_bbox[2],
        max(panel_bbox[3], option_bbox[3]) if option_bbox[2] else panel_bbox[3],
    )
    entities = tuple(
        {
            "entity_id": key,
            "bbox_px": list(value),
            "entity_type": "life_automaton_item",
        }
        for key, value in sorted(item_bboxes.items())
    )
    active_grid_key = "source_grid"
    return RenderedLifeScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bboxes=item_bboxes,
        entities=entities,
        layout_jitter={
            "enabled": True,
            "mode": "safe_margin_free_area",
            "canvas_size_px": [int(render_params.canvas_width), int(render_params.canvas_height)],
            "content_size_px": [int(content_w), int(content_h)],
            "panel_size_px": [int(panel_w), int(panel_h)],
            "panel_origin_px": [int(panel_bbox[0]), int(panel_bbox[1])],
            "grid_bbox_px": [int(value) for value in item_bboxes[active_grid_key]],
            "option_bbox_px": [int(value) for value in option_bbox] if option_bbox[2] else [],
            "free_space_px": [
                int(render_params.canvas_width - content_w),
                int(render_params.canvas_height - content_h),
            ],
            "available_offset_px": [int(available_x), int(available_y)],
            "sampled_offset_px": [int(offset_x), int(offset_y)],
            "sampled_fraction": [
                round(float(offset_x) / float(available_x), 6) if available_x > 0 else 0.5,
                round(float(offset_y) / float(available_y), 6) if available_y > 0 else 0.5,
            ],
            "content_origin_px": [int(content_left), int(content_top)],
            "centered_content_origin_px": [
                int((render_params.canvas_width - content_w) // 2),
                int((render_params.canvas_height - content_h) // 2),
            ],
            "dx_dy_from_center_px": [
                int(content_left - int((render_params.canvas_width - content_w) // 2)),
                int(content_top - int((render_params.canvas_height - content_h) // 2)),
            ],
            "safe_margin_px": int(safe_margin),
            "option_vertical_gap_px": int(metrics["option_vertical_gap_px"]),
        },
        style_metadata=dict(style_meta),
    )


def render_life_scene_bundle(
    *,
    scene: LifeSceneSpec,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: LifeRenderParams,
    scene_variant: str,
    instance_seed: int,
    sampling_scope: str,
) -> LifeRenderBundle:
    """Render a Life scene with shared style/noise handling."""

    style, style_meta = resolve_life_style(scene_variant=str(scene_variant), render_params=render_params)
    life_visual = resolve_life_board_visual(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )
    style_meta = style_meta_with_font(style_meta, render_params)
    style_meta = style_meta_with_life_board(style_meta, life_visual=life_visual)
    background, background_meta = make_symbolic_scene_background(
        canvas_width=render_params.canvas_width,
        canvas_height=render_params.canvas_height,
        style=style,
    )
    with temporary_default_font_family(render_params.font_family):
        rendered = render_life_scene(
            background=background,
            scene=scene,
            scene_variant=str(scene_variant),
            render_params=render_params,
            style=style,
            style_meta=style_meta,
            life_visual=life_visual,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return LifeRenderBundle(
        image=image,
        rendered=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        board_style=str(life_visual.board_style),
        board_style_probabilities=dict(life_visual.board_style_probabilities),
        cell_palette_id=str(life_visual.cell_palette_id),
        cell_palette_probabilities=dict(life_visual.cell_palette_probabilities),
    )
