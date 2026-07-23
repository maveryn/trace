"""Rendering helpers for matrix chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from .....core.visual.noise import apply_post_image_noise
from ....shared.bbox_projection import round_bbox as _round_bbox
from ....shared.font_assets import font_asset_version
from ....shared.text_rendering import fit_font_to_box, load_font
from ....shared.text_legibility import draw_text_traced
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    _cell_fill_rgb,
    _column_header_key,
    resolve_render_params,
    _row_header_key,
    _text_rgb_for_fill,
)
from .state import MatrixRenderParams, MatrixRenderResult, RenderedMatrix


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[float, float]:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = _text_bbox(draw, str(text), font)
    xy = (float(x1 + ((x2 - x1 - width) / 2.0)), float(y1 + ((y2 - y1 - height) / 2.0) - 1.0))
    draw_text_traced(draw,
        xy,
        str(text),
        font=font,
        fill=fill,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=stroke_fill if stroke_fill is not None else fill,
     role="readout", required=False,)


def _draw_rotated_text(
    image: Image.Image,
    *,
    box: Sequence[float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    patch_w = max(1, int(x2 - x1))
    patch_h = max(1, int(y2 - y1))
    patch = Image.new("RGBA", (patch_h, patch_w), (255, 255, 255, 0))
    draw = ImageDraw.Draw(patch)
    width, height = _text_bbox(draw, str(text), font)
    draw_text_traced(draw,
        ((patch_h - width) / 2.0, (patch_w - height) / 2.0),
        str(text),
        font=font,
        fill=fill + (255,),
     role="readout", required=False,)
    rotated = patch.rotate(90, expand=True)
    image.alpha_composite(rotated, (x1, y1))


def _render_matrix(
    background: Image.Image,
    *,
    scene_title: str,
    scene_variant: str,
    palette_variant: str,
    header_layout: str,
    grid_style: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    value_min: int,
    value_max: int,
    scene_meta: Mapping[str, Any],
    render_params: MatrixRenderParams,
) -> RenderedMatrix:
    """Draw the complete matrix view and record cell/header projection maps."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    p = render_params
    offset_x = float(p.layout_offset_x_px)
    offset_y = float(p.layout_offset_y_px)
    panel_bbox = (
        float(p.outer_margin_px) + offset_x,
        float(p.outer_margin_px) + offset_y,
        float(p.canvas_width - p.outer_margin_px) + offset_x,
        float(p.canvas_height - p.outer_margin_px) + offset_y,
    )
    draw.rounded_rectangle(panel_bbox, radius=8, fill=p.panel_fill_rgb, outline=p.panel_border_rgb, width=2)
    title_bbox = (
        panel_bbox[0] + p.panel_padding_px,
        panel_bbox[1] + p.panel_padding_px,
        panel_bbox[2] - p.panel_padding_px,
        panel_bbox[1] + p.panel_padding_px + p.title_band_height_px,
    )
    title_font = load_font(p.title_font_size_px, bold=False, font_family=p.font_family)
    _draw_centered_text(draw, box=title_bbox, text=str(scene_title), font=title_font, fill=p.title_rgb)

    row_count = len(row_labels)
    column_count = len(column_labels)
    header_h = p.col_label_height_px
    label_w = p.row_label_width_px
    matrix_left = panel_bbox[0] + p.panel_padding_px + label_w
    matrix_top = title_bbox[3] + header_h
    matrix_right = panel_bbox[2] - p.panel_padding_px - (44 if str(header_layout) == "dual_headers" else 0)
    matrix_bottom = panel_bbox[3] - p.panel_padding_px - p.legend_height_px - (34 if str(header_layout) == "dual_headers" else 0)
    gap = p.cell_gap_px if str(grid_style) == "gapped_tiles" else 0
    cell_w = max(24.0, (float(matrix_right - matrix_left) - (float(gap) * (column_count - 1))) / float(column_count))
    cell_h = max(24.0, (float(matrix_bottom - matrix_top) - (float(gap) * (row_count - 1))) / float(row_count))
    matrix_width = (cell_w * column_count) + (gap * (column_count - 1))
    matrix_height = (cell_h * row_count) + (gap * (row_count - 1))
    matrix_bbox = (matrix_left, matrix_top, matrix_left + matrix_width, matrix_top + matrix_height)

    cell_bbox_map: Dict[str, List[float]] = {}
    row_label_bbox_map: Dict[str, List[float]] = {}
    column_label_bbox_map: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []

    row_axis_box = (panel_bbox[0] + 10, matrix_top, panel_bbox[0] + p.panel_padding_px + 24, matrix_bbox[3])
    col_axis_box = (matrix_left, title_bbox[3], matrix_bbox[2], title_bbox[3] + 28)
    axis_font = load_font(18, bold=False, font_family=p.font_family)
    _draw_centered_text(draw, box=col_axis_box, text=str(scene_meta.get("column_axis_title", "Column")), font=axis_font, fill=p.header_text_rgb)
    _draw_rotated_text(image, box=row_axis_box, text=str(scene_meta.get("row_axis_title", "Row")), font=axis_font, fill=p.header_text_rgb)

    for r, label in enumerate(row_labels):
        y1 = matrix_top + (r * (cell_h + gap))
        bbox = (panel_bbox[0] + p.panel_padding_px, y1, matrix_left - 8, y1 + cell_h)
        row_label_bbox_map[_row_header_key(r)] = _round_bbox(bbox)
        row_font = fit_font_to_box(
            draw,
            text=str(label),
            max_width=bbox[2] - bbox[0],
            max_height=bbox[3] - bbox[1],
            bold=dense_fit_bold(),
            max_size_px=p.header_font_size_px,
            font_family=p.font_family,
        )
        _draw_centered_text(draw, box=bbox, text=str(label), font=row_font, fill=p.header_text_rgb)
        entities.append(
            {
                "entity_id": _row_header_key(r),
                "entity_type": "matrix_row_header",
                "label": str(label),
                "row_index": int(r),
                "bbox_px": _round_bbox(bbox),
            }
        )
        if str(header_layout) == "dual_headers":
            dup_bbox = (matrix_bbox[2] + 8, y1, panel_bbox[2] - p.panel_padding_px, y1 + cell_h)
            _draw_centered_text(draw, box=dup_bbox, text=str(label), font=row_font, fill=p.header_text_rgb)

    for c, label in enumerate(column_labels):
        x1 = matrix_left + (c * (cell_w + gap))
        bbox = (x1, title_bbox[3] + 28, x1 + cell_w, matrix_top - 6)
        column_label_bbox_map[_column_header_key(c)] = _round_bbox(bbox)
        col_font = fit_font_to_box(
            draw,
            text=str(label),
            max_width=bbox[2] - bbox[0],
            max_height=bbox[3] - bbox[1],
            bold=dense_fit_bold(),
            max_size_px=p.header_font_size_px,
            font_family=p.font_family,
        )
        if str(header_layout) == "top_rotated_columns":
            _draw_rotated_text(image, box=bbox, text=str(label), font=col_font, fill=p.header_text_rgb)
        else:
            _draw_centered_text(draw, box=bbox, text=str(label), font=col_font, fill=p.header_text_rgb)
        entities.append(
            {
                "entity_id": _column_header_key(c),
                "entity_type": "matrix_column_header",
                "label": str(label),
                "column_index": int(c),
                "bbox_px": _round_bbox(bbox),
            }
        )
        if str(header_layout) == "dual_headers":
            dup_bbox = (x1, matrix_bbox[3] + 4, x1 + cell_w, matrix_bbox[3] + 32)
            _draw_centered_text(draw, box=dup_bbox, text=str(label), font=col_font, fill=p.header_text_rgb)

    cell_by_id = {str(cell["cell_id"]): dict(cell) for cell in cells}
    for r in range(row_count):
        for c in range(column_count):
            cell_id = f"r{r}_c{c}"
            cell = cell_by_id[cell_id]
            x1 = matrix_left + (c * (cell_w + gap))
            y1 = matrix_top + (r * (cell_h + gap))
            bbox = (x1, y1, x1 + cell_w, y1 + cell_h)
            cell_bbox_map[cell_id] = _round_bbox(bbox)
            active = bool(cell.get("active", False))
            if active:
                fill = _cell_fill_rgb(
                    value=int(cell["value"]),
                    value_min=int(value_min),
                    value_max=int(value_max),
                    palette_variant=str(palette_variant),
                    scene_variant=str(scene_variant),
                )
            else:
                fill = p.inactive_cell_rgb
            outline = p.grid_rgb if str(grid_style) != "minimal_grid" else fill
            draw.rectangle(bbox, fill=fill, outline=outline, width=max(0, int(p.cell_border_width_px)))
            if bool(cell.get("highlighted", False)):
                draw.rectangle(
                    (bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2),
                    outline=p.highlight_rgb,
                    width=5,
                )
            if active:
                text = str(cell.get("display_value", ""))
                cell_font = fit_font_to_box(
                    draw,
                    text=text,
                    max_width=float(cell_w),
                    max_height=float(cell_h),
                    bold=dense_fit_bold(),
                    min_size_px=13,
                    max_size_px=p.cell_font_size_px,
                    fill_ratio=0.72,
                    font_family=p.font_family,
                )
                text_fill = _text_rgb_for_fill(fill)
                _draw_centered_text(
                    draw,
                    box=bbox,
                    text=text,
                    font=cell_font,
                    fill=text_fill,
                    stroke_fill=(255, 255, 255) if text_fill != (255, 255, 255) else (24, 30, 38),
                    stroke_width=dense_stroke_width(),
                )
            entities.append(
                {
                    "entity_id": cell_id,
                    "entity_type": "matrix_cell",
                    "row_index": int(r),
                    "column_index": int(c),
                    "row_label": str(row_labels[r]),
                    "column_label": str(column_labels[c]),
                    "value": None if cell.get("value") is None else int(cell["value"]),
                    "active": bool(active),
                    "highlighted": bool(cell.get("highlighted", False)),
                    "bbox_px": _round_bbox(bbox),
                }
            )

    if str(grid_style) == "heavy_block_lines":
        block_size = int(scene_meta.get("block_size", 3))
        for c in range(block_size, column_count, block_size):
            x = matrix_left + (c * (cell_w + gap)) - (gap / 2.0)
            draw.line((x, matrix_top, x, matrix_bbox[3]), fill=(38, 45, 55), width=4)
        for r in range(block_size, row_count, block_size):
            y = matrix_top + (r * (cell_h + gap)) - (gap / 2.0)
            draw.line((matrix_left, y, matrix_bbox[2], y), fill=(38, 45, 55), width=4)
    draw.rectangle(matrix_bbox, outline=p.panel_border_rgb, width=2)

    legend_bbox = (
        matrix_bbox[0],
        matrix_bbox[3] + 8,
        matrix_bbox[2],
        panel_bbox[3] - p.panel_padding_px,
    )
    legend_font = load_font(p.legend_font_size_px, bold=False, font_family=p.font_family)
    legend_text = "Cell color encodes the printed value; use the printed numbers as the source of truth."
    _draw_centered_text(draw, box=legend_bbox, text=legend_text, font=legend_font, fill=p.legend_text_rgb)
    return RenderedMatrix(
        image=image.convert("RGB"),
        entities=tuple(entities),
        panel_bbox_px=_round_bbox(panel_bbox),
        title_bbox_px=_round_bbox(title_bbox),
        matrix_bbox_px=_round_bbox(matrix_bbox),
        legend_bbox_px=_round_bbox(legend_bbox),
        cell_bbox_map=dict(cell_bbox_map),
        row_label_bbox_map=dict(row_label_bbox_map),
        column_label_bbox_map=dict(column_label_bbox_map),
    )


def render_matrix_scene(
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    palette_variant: str,
    header_layout: str,
    grid_style: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> MatrixRenderResult:
    """Render one matrix scene from neutral sampled data and visual variants."""

    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    render_params = resolve_render_params(render_style_params)
    protected_colors = tuple(
        _cell_fill_rgb(
            value=int(cell["value"]),
            value_min=int(dataset["value_min"]),
            value_max=int(dataset["value_max"]),
            palette_variant=str(palette_variant),
            scene_variant=str(scene_variant),
        )
        for cell in dataset["cells"]
        if bool(cell.get("active", False)) and cell.get("value") is not None
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="matrix",
        render_params=render_params,
        protected_colors=protected_colors,
    )
    rendered_scene = _render_matrix(
        background,
        scene_title=str(dataset["scene_title"]),
        scene_variant=str(scene_variant),
        palette_variant=str(palette_variant),
        header_layout=str(header_layout),
        grid_style=str(grid_style),
        row_labels=list(dataset["row_labels"]),
        column_labels=list(dataset["column_labels"]),
        cells=list(dataset["cells"]),
        value_min=int(dataset["value_min"]),
        value_max=int(dataset["value_max"]),
        scene_meta=dict(dataset["scene_meta"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return MatrixRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
    )


def font_assets_payload(render_params: MatrixRenderParams) -> Dict[str, str]:
    return {
        "asset_version": font_asset_version(),
        "chart_font_family": str(render_params.font_family),
    }


__all__ = ["MatrixRenderResult", "font_assets_payload", "render_matrix_scene"]
