"""Renderer for symbolic agent-automaton scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from .....core.visual.noise import apply_post_image_noise
from ....shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from ....shared.text_rendering import load_font
from ...shared.scene_style import (
    DEFAULT_SYMBOLIC_SCENE_STYLE,
    SymbolicSceneStyle,
    draw_symbolic_chrome_by_mode,
    draw_symbolic_grid_cell,
    draw_symbolic_option_card,
    make_symbolic_scene_background,
)

from .defaults import POST_IMAGE_NOISE_DEFAULTS
from .layout import fit_render_params
from .layout import cell_bbox, content_metrics, grid_bbox, grid_option_card_size, inset_bbox, option_grid_gap, option_vertical_gap, source_header_height
from .rules import AGENT_RGB, DIR_VEC
from .state import AgentRenderBundle, AgentRenderParams, AgentSceneSpec, RenderedAgentScene
from .styles import (
    resolve_agent_style,
    resolve_board_style,
    resolve_render_params,
    style_meta_with_board,
    style_meta_with_font,
)


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], alpha_b: float) -> Tuple[int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - alpha)) + (float(color_b[index]) * alpha)))
        for index in range(3)
    )


def _draw_cell_grid(
    draw: ImageDraw.ImageDraw,
    *,
    grid: Sequence[Sequence[int]],
    left: int,
    top: int,
    cell_size: int,
    gap: int,
    state_colors: Sequence[Sequence[int]],
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    item_prefix: str,
    target_cells: Sequence[Tuple[int, int]] = tuple(),
    draw_labels: bool = False,
    style: SymbolicSceneStyle = DEFAULT_SYMBOLIC_SCENE_STYLE,
    cell_render_style: str = "classic_grid",
) -> None:
    """Draw the source/options grid cells while preserving item bboxes."""

    target_set = {(int(row), int(col)) for row, col in target_cells}
    rows = len(grid)
    cols = len(grid[0])
    font = load_font(max(10, int(cell_size * 0.28)), bold=True)
    for row in range(rows):
        for col in range(cols):
            bbox = cell_bbox(left=left, top=top, row=row, col=col, cell_size=cell_size, gap=gap)
            state = int(grid[row][col])
            fill = tuple(int(value) for value in state_colors[state % len(state_colors)])
            cell_id = f"{item_prefix}_cell_{row}_{col}"
            item_bboxes[cell_id] = bbox
            selected = (row, col) in target_set
            render_style = str(cell_render_style)
            if render_style == "rounded_tiles":
                draw_rounded_rect(
                    draw,
                    bbox=bbox,
                    radius=max(3, int(round(cell_size * 0.10))),
                    fill=fill,
                    outline=style.grid_rgb,
                    width=max(1, int(round(cell_size * 0.035))),
                )
            elif render_style == "inset_cells":
                draw.rectangle(bbox, fill=style.grid_rgb)
                inner = inset_bbox(bbox, max(1, int(round(cell_size * 0.08))))
                draw_rounded_rect(
                    draw,
                    bbox=inner,
                    radius=max(3, int(round(cell_size * 0.10))),
                    fill=fill,
                    outline=style.panel_border_rgb,
                    width=1,
                )
            elif render_style == "lab_matrix":
                draw.rectangle(bbox, fill=fill, outline=style.panel_border_rgb, width=max(1, int(round(cell_size * 0.04))))
                highlight = _blend_rgb(fill, (255, 255, 255), 0.32)
                draw.line((bbox[0] + 2, bbox[1] + 2, bbox[2] - 3, bbox[1] + 2), fill=highlight, width=1)
                draw.line((bbox[0] + 2, bbox[1] + 2, bbox[0] + 2, bbox[3] - 3), fill=highlight, width=1)
            elif render_style == "notebook_cells":
                draw.rectangle(bbox, fill=fill, outline=style.grid_rgb, width=1)
                draw.line(
                    (bbox[0] + 4, bbox[1] + max(4, int(cell_size * 0.28)), bbox[2] - 4, bbox[1] + max(4, int(cell_size * 0.28))),
                    fill=style.notebook_line_rgb,
                    width=1,
                )
            else:
                draw_symbolic_grid_cell(
                    draw,
                    bbox=bbox,
                    fill=fill,
                    style=style,
                    outline=style.grid_rgb,
                    width=1,
                    selected=False,
                )
            if selected:
                draw.rectangle(bbox, outline=style.mark_rgb, width=max(3, int(cell_size * 0.10)))
            if draw_labels:
                draw_centered_text(
                    draw,
                    text=str(state),
                    center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                    font=font,
                    fill=style.text_rgb if state == 0 else style.text_stroke_rgb,
                    stroke_fill=style.text_stroke_rgb if state == 0 else style.text_rgb,
                    stroke_width=1,
                )


def _draw_option_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    label: str,
    fill: Sequence[int],
    style: SymbolicSceneStyle,
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


def _decorate_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    radius: int,
    border_width: int,
    style: SymbolicSceneStyle,
    chrome_mode: str,
) -> None:
    draw_symbolic_chrome_by_mode(
        draw,
        bbox=bbox,
        style=style,
        radius=int(radius),
        border_width=int(border_width),
        mode=str(chrome_mode),
    )


def _draw_source_grid_label(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Tuple[int, int, int, int],
    grid_top: int,
    label: str,
    render_params: AgentRenderParams,
    style: SymbolicSceneStyle,
) -> None:
    """Draw the source-grid marker above the grid without changing the annotation bbox."""

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
    y0 = int(round((header_top + header_bottom - pill_h) / 2))
    pill_bbox = (x0, y0, int(x0 + pill_w), int(y0 + pill_h))
    draw_rounded_rect(
        draw,
        bbox=pill_bbox,
        radius=max(6, int(round(pill_h * 0.32))),
        fill=_blend_rgb(style.panel_fill_rgb, style.panel_accent_rgb, 0.18),
        outline=style.panel_border_rgb,
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


def _draw_agent_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[int],
    direction: int,
    color: Sequence[int] = AGENT_RGB,
    inner_fill: Sequence[int] = (255, 246, 236),
    width: int = 6,
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    radius = 0.34 * min(x1 - x0, y1 - y0)
    dr, dc = DIR_VEC[int(direction)]
    end = (float(cx + dc * radius), float(cy + dr * radius))
    start = (float(cx - dc * radius * 0.55), float(cy - dr * radius * 0.55))
    draw.ellipse(
        (cx - radius * 0.55, cy - radius * 0.55, cx + radius * 0.55, cy + radius * 0.55),
        fill=tuple(int(value) for value in inner_fill),
        outline=tuple(int(value) for value in color),
        width=3,
    )
    draw_arrow(draw, start=start, end=end, fill=color, width=int(width), head_length_px=14, head_width_px=18)


def _draw_agent_pose_options(
    draw: ImageDraw.ImageDraw,
    *,
    scene: AgentSceneSpec,
    render_params: AgentRenderParams,
    top: int,
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    style: SymbolicSceneStyle,
    cell_render_style: str,
    left: int | None = None,
) -> Tuple[int, int, int, int]:
    """Render final-pose option cards with their internal pose markers."""

    card_w = int(render_params.option_card_width_px)
    card_h = int(render_params.option_card_height_px)
    gap = int(render_params.option_gap_px)
    count = len(scene.option_specs)
    total_w = count * card_w + max(0, count - 1) * gap
    if left is None:
        left = int((render_params.canvas_width - total_w) // 2)
    else:
        left = int(left)
    text_font = load_font(int(render_params.small_font_size_px), bold=False)
    for index, option in enumerate(scene.option_specs):
        x0 = int(left + index * (card_w + gap))
        bbox = (x0, int(top), int(x0 + card_w), int(top + card_h))
        _draw_option_card(draw, bbox=bbox, label=str(option.label), fill=style.option_fill_rgb, style=style)
        item_bboxes[str(option.option_id)] = bbox
        marker_side = max(24, min(int(card_w * 0.46), int(card_h * 0.42)))
        marker_left = int(x0 + (card_w - marker_side) / 2)
        marker_top = int(top + max(22, int(card_h * 0.22)))
        marker_bbox = (
            marker_left,
            marker_top,
            int(marker_left + marker_side),
            int(marker_top + marker_side),
        )
        _draw_cell_grid(
            draw,
            grid=((0,),),
            left=int(marker_bbox[0]),
            top=int(marker_bbox[1]),
            cell_size=int(marker_side),
            gap=1,
            state_colors=(style.option_marker_fill_rgb,),
            item_bboxes={},
            item_prefix="agent_option_marker",
            style=style,
            cell_render_style=str(cell_render_style),
        )
        _draw_agent_marker(
            draw,
            bbox=marker_bbox,
            direction=int(option.direction),
            color=style.agent_rgb,
            inner_fill=style.agent_inner_rgb,
            width=max(3, int(render_params.arrow_width_px) - 2),
        )
        draw_centered_text(
            draw,
            text=f"r{int(option.row) + 1}, c{int(option.col) + 1}",
            center=(x0 + card_w / 2, int(top) + card_h - max(14, int(card_h * 0.17))),
            font=text_font,
            fill=style.text_rgb,
            stroke_fill=style.text_stroke_rgb,
            stroke_width=1,
        )
    return (left, int(top), int(left + total_w), int(top + card_h))


def _draw_agent_grid_options(
    draw: ImageDraw.ImageDraw,
    *,
    scene: AgentSceneSpec,
    render_params: AgentRenderParams,
    top: int,
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    style: SymbolicSceneStyle,
    cell_render_style: str,
    left: int | None = None,
) -> Tuple[int, int, int, int]:
    """Render future-grid option cards with full-grid candidates.

    The invariant is that each option card bbox remains the selectable witness
    while the internal grid is purely visual support for that option label.
    """

    card_w, card_h = grid_option_card_size(rows=int(scene.rows), cols=int(scene.cols), render_params=render_params)
    gap = int(render_params.option_gap_px)
    count = len(scene.grid_option_specs)
    total_w = count * card_w + max(0, count - 1) * gap
    if left is None:
        left = int((render_params.canvas_width - total_w) // 2)
    else:
        left = int(left)
    option_cell = int(render_params.option_grid_cell_px)
    grid_gap = option_grid_gap(render_params)
    for index, option in enumerate(scene.grid_option_specs):
        x0 = int(left + index * (card_w + gap))
        card_bbox = (x0, int(top), int(x0 + card_w), int(top + card_h))
        _draw_option_card(draw, bbox=card_bbox, label=str(option.label), fill=style.option_fill_rgb, style=style)
        item_bboxes[str(option.option_id)] = card_bbox
        rows = len(option.grid)
        cols = len(option.grid[0])
        grid_width = int(cols * option_cell + max(0, cols - 1) * grid_gap)
        grid_height = int(rows * option_cell + max(0, rows - 1) * grid_gap)
        grid_left = int(x0 + (card_w - grid_width) / 2)
        header_h = max(30, int(round(render_params.label_font_size_px * 1.45)))
        grid_top = int(top + header_h + max(0, (card_h - header_h - grid_height) / 2))
        _draw_cell_grid(
            draw,
            grid=option.grid,
            left=grid_left,
            top=grid_top,
            cell_size=option_cell,
            gap=grid_gap,
            state_colors=style.state_colors[: scene.state_count],
            item_bboxes={},
            item_prefix=f"{option.option_id}_grid",
            draw_labels=bool(scene.state_count == 3),
            style=style,
            cell_render_style=str(cell_render_style),
        )
    return (left, int(top), int(left + total_w), int(top + card_h))


def render_agent_scene(
    *,
    background: Image.Image,
    scene: AgentSceneSpec,
    scene_variant: str,
    render_params: AgentRenderParams,
    style: SymbolicSceneStyle,
    style_meta: Mapping[str, Any],
    board_style: str,
) -> RenderedAgentScene:
    """Render the agent-automaton source grid and optional pose options."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    item_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    rows, cols = int(scene.rows), int(scene.cols)
    cell = int(render_params.cell_size_px)
    gap = int(render_params.grid_gap_px)
    grid = grid_bbox(left=0, top=0, rows=rows, cols=cols, cell_size=cell, gap=gap)
    grid_w = grid[2] - grid[0]
    grid_h = grid[3] - grid[1]
    marker_label = str(scene.source_marker_label).strip()
    metrics = content_metrics(scene=scene, render_params=render_params)
    layout_rng = spawn_rng(instance_seed=int(render_params.layout_seed), namespace="agent_automaton_panel_origin")
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
        radius=int(render_params.panel_corner_radius_px),
        border_width=int(render_params.panel_border_width_px),
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
        )
    _draw_cell_grid(
        draw,
        grid=scene.initial_grid,
        left=grid_left,
        top=grid_top,
        cell_size=cell,
        gap=gap,
        state_colors=style.state_colors[: scene.state_count],
        item_bboxes=item_bboxes,
        item_prefix="source",
        draw_labels=bool(scene.state_count == 3),
        style=style,
        cell_render_style=str(board_style),
    )
    item_bboxes["source_grid"] = grid_bbox(left=grid_left, top=grid_top, rows=rows, cols=cols, cell_size=cell, gap=gap)
    start_bbox = cell_bbox(left=grid_left, top=grid_top, row=scene.start_row, col=scene.start_col, cell_size=cell, gap=gap)
    item_bboxes["initial_agent"] = start_bbox
    if not scene.option_specs and not scene.grid_option_specs:
        step_font = load_font(max(10, int(cell * 0.24)), bold=True)
        for trace in scene.traces:
            bbox = cell_bbox(left=grid_left, top=grid_top, row=trace.row, col=trace.col, cell_size=cell, gap=gap)
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            radius = max(8, int(cell * 0.19))
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=style.step_fill_rgb,
                outline=style.agent_rgb,
                width=2,
            )
            draw_centered_text(
                draw,
                text=str(int(trace.step)),
                center=(cx, cy),
                font=step_font,
                fill=style.text_rgb,
                stroke_fill=style.text_stroke_rgb,
                stroke_width=1,
            )
    _draw_agent_marker(
        draw,
        bbox=start_bbox,
        direction=scene.start_direction,
        color=style.agent_rgb,
        inner_fill=style.agent_inner_rgb,
        width=int(render_params.arrow_width_px),
    )
    option_bbox = (0, 0, 0, 0)
    if scene.option_specs:
        option_left = int(content_left + max(0, content_w - int(metrics["option_width_px"])) // 2)
        option_bbox = _draw_agent_pose_options(
            draw,
            scene=scene,
            render_params=render_params,
            top=int(panel_bbox[3] + option_vertical_gap(render_params)),
            item_bboxes=item_bboxes,
            style=style,
            cell_render_style=str(board_style),
            left=option_left,
        )
    elif scene.grid_option_specs:
        option_left = int(content_left + max(0, content_w - int(metrics["option_width_px"])) // 2)
        option_bbox = _draw_agent_grid_options(
            draw,
            scene=scene,
            render_params=render_params,
            top=int(panel_bbox[3] + option_vertical_gap(render_params)),
            item_bboxes=item_bboxes,
            style=style,
            cell_render_style=str(board_style),
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
            "entity_type": "automaton_item",
        }
        for key, value in sorted(item_bboxes.items())
    )
    layout_jitter = {
        "enabled": True,
        "layout_seed": int(render_params.layout_seed),
        "canvas_size_px": [int(render_params.canvas_width), int(render_params.canvas_height)],
        "content_size_px": [int(content_w), int(content_h)],
        "panel_size_px": [int(panel_w), int(panel_h)],
        "panel_origin_px": [int(panel_bbox[0]), int(panel_bbox[1])],
        "grid_bbox_px": [int(value) for value in item_bboxes["source_grid"]],
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
        "option_vertical_gap_px": int(option_vertical_gap(render_params)),
    }
    return RenderedAgentScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bboxes=item_bboxes,
        entities=entities,
        layout_jitter=layout_jitter,
        style_metadata=dict(style_meta),
    )


def render_agent_scene_bundle(
    *,
    scene: AgentSceneSpec,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    instance_seed: int,
    sampling_scope: str,
) -> AgentRenderBundle:
    """Resolve reusable render axes and draw one agent-automaton scene."""

    render_params = resolve_render_params(params, render_defaults, instance_seed=int(instance_seed))
    render_params = fit_render_params(scene=scene, render_params=render_params)
    style, style_meta = resolve_agent_style(scene_variant=str(scene_variant), render_params=render_params)
    board_style, board_style_probabilities = resolve_board_style(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )
    style_meta = style_meta_with_font(style_meta, render_params)
    style_meta = style_meta_with_board(
        style_meta,
        board_style=str(board_style),
        board_style_probabilities=board_style_probabilities,
    )
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
    )
    rendered = render_agent_scene(
        background=background,
        scene=scene,
        scene_variant=str(scene_variant),
        render_params=render_params,
        style=style,
        style_meta=style_meta,
        board_style=str(board_style),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return AgentRenderBundle(
        image=image,
        rendered=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        board_style=str(board_style),
        board_style_probabilities={str(key): float(value) for key, value in board_style_probabilities.items()},
    )
