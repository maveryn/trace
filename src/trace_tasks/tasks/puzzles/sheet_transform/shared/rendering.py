"""Rendering helpers for sheet-transform puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.option_layout import (
    centered_option_grid_shape,
    centered_option_row_counts,
)
from trace_tasks.tasks.puzzles.shared.paper_fold_scene import (
    FOLD_RESULT_SUPERSAMPLE_SCALE,
    RenderedPuzzleFoldCutResultScene,
    RenderedPuzzleFoldResultScene,
    render_puzzle_fold_cut_result_scene,
    render_puzzle_fold_result_scene,
)
from trace_tasks.tasks.shared.bbox_projection import round_bbox as _round_bbox
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import OverlayRenderParams
from .state import OVERLAY_MARK_SHAPES, OVERLAY_SCENE_VARIANTS


@dataclass(frozen=True)
class RenderedPuzzleOverlayScene:
    """Rendered transparent-sheet overlay scene plus traced geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    option_choice_bbox_map: Dict[str, List[float]]
    reference_panel_bbox_px: List[float]
    source_sheet_bbox_map: Dict[str, List[float]]


def _draw_paper(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_params: OverlayRenderParams,
) -> None:
    """Draw one paper sheet with a subtle shadow."""

    left, top, right, bottom = [float(value) for value in bbox]
    shadow_offset = 4.0
    draw_rounded_rect(
        draw,
        (
            float(left + shadow_offset),
            float(top + shadow_offset),
            float(right + shadow_offset),
            float(bottom + shadow_offset),
        ),
        radius=int(render_params.paper_corner_radius_px),
        fill=render_params.paper_shadow_rgb,
        outline=render_params.paper_shadow_rgb,
        width=1,
    )
    draw_rounded_rect(
        draw,
        (float(left), float(top), float(right), float(bottom)),
        radius=int(render_params.paper_corner_radius_px),
        fill=render_params.paper_fill_rgb,
        outline=render_params.border_color_rgb,
        width=max(1, int(render_params.border_width_px)),
    )


def _sheet_cell_bbox(
    sheet_bbox: Sequence[float],
    *,
    grid_size: int,
    cell_x: int,
    cell_y: int,
) -> Tuple[float, float, float, float]:
    """Return the hidden cell bbox for one sheet-grid coordinate."""

    left, top, right, bottom = [float(value) for value in sheet_bbox]
    inner_pad = 14.0
    usable_width = float(right - left) - (2.0 * inner_pad)
    usable_height = float(bottom - top) - (2.0 * inner_pad)
    cell_width = float(usable_width / float(grid_size))
    cell_height = float(usable_height / float(grid_size))
    x1 = float(left + inner_pad + (float(cell_x) * cell_width))
    y1 = float(top + inner_pad + (float(cell_y) * cell_height))
    return (
        float(x1),
        float(y1),
        float(x1 + cell_width),
        float(y1 + cell_height),
    )


def _mark_bbox(
    sheet_bbox: Sequence[float],
    *,
    grid_size: int,
    cell_x: int,
    cell_y: int,
) -> Tuple[float, float, float, float]:
    """Return the rendered mark bbox inside one hidden grid cell."""

    cell_bbox = _sheet_cell_bbox(
        sheet_bbox,
        grid_size=int(grid_size),
        cell_x=int(cell_x),
        cell_y=int(cell_y),
    )
    cell_width = float(cell_bbox[2] - cell_bbox[0])
    cell_height = float(cell_bbox[3] - cell_bbox[1])
    inset = float(0.20 * min(cell_width, cell_height))
    return (
        float(cell_bbox[0] + inset),
        float(cell_bbox[1] + inset),
        float(cell_bbox[2] - inset),
        float(cell_bbox[3] - inset),
    )


def _draw_marks(
    draw: ImageDraw.ImageDraw,
    *,
    sheet_bbox: Sequence[float],
    mark_specs: Sequence[Mapping[str, Any]],
    grid_size: int,
    entity_prefix: str,
    entity_type: str,
    base_attrs: Mapping[str, Any],
    render_params: OverlayRenderParams,
) -> List[Dict[str, Any]]:
    """Draw filled marks on one sheet and return traced entities."""

    entities: List[Dict[str, Any]] = []
    mark_shape = str(render_params.mark_shape)
    if mark_shape not in set(OVERLAY_MARK_SHAPES):
        raise ValueError(f"unsupported puzzle overlay mark_shape: {mark_shape}")
    for mark_spec in mark_specs:
        cell_x = int(mark_spec["cell"][0])
        cell_y = int(mark_spec["cell"][1])
        bbox = _mark_bbox(sheet_bbox, grid_size=int(grid_size), cell_x=int(cell_x), cell_y=int(cell_y))
        outline_width = max(1, int(render_params.border_width_px - 1))
        if mark_shape == "circle":
            draw.ellipse(
                bbox,
                fill=render_params.mark_fill_rgb,
                outline=render_params.mark_outline_rgb,
                width=outline_width,
            )
        elif mark_shape == "square":
            draw.rectangle(
                bbox,
                fill=render_params.mark_fill_rgb,
                outline=render_params.mark_outline_rgb,
                width=outline_width,
            )
        elif mark_shape == "rounded_square":
            radius = max(2, int(0.18 * min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))))
            draw_rounded_rect(
                draw,
                bbox,
                radius=radius,
                fill=render_params.mark_fill_rgb,
                outline=render_params.mark_outline_rgb,
                width=outline_width,
            )
        else:
            center_x = float(0.5 * (bbox[0] + bbox[2]))
            center_y = float(0.5 * (bbox[1] + bbox[3]))
            points = [
                (center_x, float(bbox[1])),
                (float(bbox[2]), center_y),
                (center_x, float(bbox[3])),
                (float(bbox[0]), center_y),
            ]
            draw.polygon(points, fill=render_params.mark_fill_rgb)
            draw.line(points + [points[0]], fill=render_params.mark_outline_rgb, width=outline_width, joint="curve")
        attrs = dict(base_attrs)
        attrs.update({"cell": [int(cell_x), int(cell_y)], "mark_shape": str(mark_shape)})
        entities.append(
            {
                "entity_id": f"{str(entity_prefix)}_{str(mark_spec['mark_id'])}",
                "entity_type": str(entity_type),
                "bbox_px": _round_bbox(bbox),
                "attrs": attrs,
            }
        )
    return entities


def _scene_panel_style(
    scene_variant: str,
    *,
    render_params: OverlayRenderParams,
) -> Tuple[Tuple[int, int, int] | None, Tuple[int, int, int] | None]:
    """Resolve outer reference-panel fill and outline styling."""

    if str(scene_variant) == "overlay_outline":
        return None, render_params.border_color_rgb
    if str(scene_variant) == "overlay_strip":
        return render_params.instruction_fill_rgb, None
    return render_params.panel_fill_rgb, render_params.border_color_rgb


def _text_height_px(draw: ImageDraw.ImageDraw, *, text: str, font) -> float:
    """Measure one label height."""

    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    return float(bbox[3] - bbox[1])


def render_puzzle_overlay_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    grid_size: int,
    left_mark_specs: Sequence[Mapping[str, Any]],
    right_mark_specs: Sequence[Mapping[str, Any]],
    option_specs: Sequence[Mapping[str, Any]],
    render_params: OverlayRenderParams,
) -> RenderedPuzzleOverlayScene:
    """Render one transparent-sheet overlay puzzle with labeled image options."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(OVERLAY_SCENE_VARIANTS):
        raise ValueError(f"unsupported puzzle overlay scene_variant: {scene_variant}")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    option_label_font = load_font(int(render_params.option_label_font_size_px))
    combine_symbol_font = load_font(int(render_params.combine_symbol_font_size_px), bold=True)
    option_label_height = _text_height_px(draw, text="A", font=option_label_font)

    scene_left = float(render_params.scene_margin_left_px)
    scene_top = float(render_params.scene_margin_top_px)
    scene_right = float(render_params.canvas_width - render_params.scene_margin_right_px)
    scene_bottom = float(render_params.canvas_height - render_params.scene_margin_bottom_px)

    reference_panel_bbox = [
        round(float(scene_left), 3),
        round(float(scene_top), 3),
        round(float(scene_right), 3),
        round(float(scene_top + int(render_params.reference_panel_height_px)), 3),
    ]
    reference_fill, reference_outline = _scene_panel_style(selected_variant, render_params=render_params)
    if reference_fill is not None or reference_outline is not None:
        draw_rounded_rect(
            draw,
            tuple(float(value) for value in reference_panel_bbox),
            radius=int(render_params.panel_corner_radius_px),
            fill=reference_fill if reference_fill is not None else (255, 255, 255),
            outline=reference_outline if reference_outline is not None else (255, 255, 255),
            width=max(1, int(render_params.border_width_px if reference_outline is not None else 1)),
        )

    # Keep one shared paper scale for both the reference sheets and the answer options.
    paper_size = float(min(render_params.source_paper_size_px, render_params.option_paper_size_px))
    reference_group_width = float((2.0 * paper_size) + float(render_params.source_gap_px))
    group_left = float(scene_left + 0.5 * ((scene_right - scene_left) - reference_group_width))
    group_top = float(reference_panel_bbox[1] + float(render_params.reference_panel_padding_px))
    left_sheet_bbox = (
        float(group_left),
        float(group_top),
        float(group_left + paper_size),
        float(group_top + paper_size),
    )
    right_sheet_bbox = (
        float(left_sheet_bbox[2] + float(render_params.source_gap_px)),
        float(group_top),
        float(left_sheet_bbox[2] + float(render_params.source_gap_px) + paper_size),
        float(group_top + paper_size),
    )
    _draw_paper(draw, bbox=left_sheet_bbox, render_params=render_params)
    _draw_paper(draw, bbox=right_sheet_bbox, render_params=render_params)

    combine_bbox = draw_centered_text(
        draw,
        text="+",
        center=(
            float(0.5 * (left_sheet_bbox[2] + right_sheet_bbox[0])),
            float(0.5 * (left_sheet_bbox[1] + left_sheet_bbox[3])),
        ),
        font=combine_symbol_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
    )

    divider_y = float(reference_panel_bbox[3] + (0.5 * float(render_params.reference_to_options_gap_px)))
    divider_margin = 42.0
    divider_bbox = (
        float(scene_left + divider_margin),
        float(divider_y - 1.5),
        float(scene_right - divider_margin),
        float(divider_y + 1.5),
    )
    draw.line(
        (
            float(divider_bbox[0]),
            float(divider_y),
            float(divider_bbox[2]),
            float(divider_y),
        ),
        fill=render_params.border_color_rgb,
        width=max(2, int(render_params.border_width_px - 1)),
    )

    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "overlay_reference_panel",
            "entity_type": "puzzle_overlay_reference_panel",
            "bbox_px": list(reference_panel_bbox),
            "attrs": {"scene_variant": selected_variant},
        },
        {
            "entity_id": "source_sheet_left",
            "entity_type": "puzzle_overlay_source_sheet",
            "bbox_px": _round_bbox(left_sheet_bbox),
            "attrs": {"sheet_role": "left", "grid_size": int(grid_size)},
        },
        {
            "entity_id": "source_sheet_right",
            "entity_type": "puzzle_overlay_source_sheet",
            "bbox_px": _round_bbox(right_sheet_bbox),
            "attrs": {"sheet_role": "right", "grid_size": int(grid_size)},
        },
        {
            "entity_id": "overlay_operator_plus",
            "entity_type": "puzzle_overlay_operator",
            "bbox_px": list(combine_bbox),
            "attrs": {"operator": "plus"},
        },
        {
            "entity_id": "overlay_section_divider",
            "entity_type": "puzzle_overlay_divider",
            "bbox_px": _round_bbox(divider_bbox),
            "attrs": {"orientation": "horizontal"},
        },
    ]
    entities.extend(
        _draw_marks(
            draw,
            sheet_bbox=left_sheet_bbox,
            mark_specs=left_mark_specs,
            grid_size=int(grid_size),
            entity_prefix="left_sheet_mark",
            entity_type="puzzle_overlay_mark",
            base_attrs={"sheet_role": "left"},
            render_params=render_params,
        )
    )
    entities.extend(
        _draw_marks(
            draw,
            sheet_bbox=right_sheet_bbox,
            mark_specs=right_mark_specs,
            grid_size=int(grid_size),
            entity_prefix="right_sheet_mark",
            entity_type="puzzle_overlay_mark",
            base_attrs={"sheet_role": "right"},
            render_params=render_params,
        )
    )

    option_count = int(len(option_specs))
    option_cols, option_rows = centered_option_grid_shape(int(option_count))
    option_row_counts = centered_option_row_counts(int(option_count), int(option_cols))
    option_paper_size = float(paper_size)
    option_block_height = float(option_paper_size + float(render_params.option_label_gap_px) + option_label_height)
    options_width = float(
        (option_cols * option_paper_size)
        + max(0, option_cols - 1) * float(render_params.option_gap_px)
    )
    options_height = float(
        (option_rows * option_block_height)
        + max(0, option_rows - 1) * float(render_params.option_row_gap_px)
    )
    options_left = float(scene_left + 0.5 * ((scene_right - scene_left) - options_width))
    options_top = float(reference_panel_bbox[3] + float(render_params.reference_to_options_gap_px))
    options_bottom = float(options_top + options_height)
    if float(options_bottom) > float(scene_bottom + 1e-3):
        raise ValueError("overlay options exceed the scene bounds")

    option_choice_bbox_map: Dict[str, List[float]] = {}
    for option_index, option_spec in enumerate(option_specs):
        row_index = int(option_index // option_cols)
        row_option_count = int(option_row_counts[row_index])
        row_base_index = int(sum(option_row_counts[:row_index]))
        col_index = int(option_index - row_base_index)
        row_width = float(
            (row_option_count * option_paper_size)
            + max(0, row_option_count - 1) * float(render_params.option_gap_px)
        )
        row_left = float(options_left + 0.5 * (options_width - row_width))
        option_left = float(row_left + col_index * (option_paper_size + float(render_params.option_gap_px)))
        option_top = float(options_top + row_index * (option_block_height + float(render_params.option_row_gap_px)))
        option_paper_bbox = (
            float(option_left),
            float(option_top),
            float(option_left + option_paper_size),
            float(option_top + option_paper_size),
        )
        option_choice_id = str(option_spec["option_choice_id"])
        option_choice_bbox_map[option_choice_id] = _round_bbox(option_paper_bbox)
        _draw_paper(draw, bbox=option_paper_bbox, render_params=render_params)
        label_bbox = draw_centered_text(
            draw,
            text=str(option_spec["option_label"]),
            center=(
                float(0.5 * (option_paper_bbox[0] + option_paper_bbox[2])),
                float(option_paper_bbox[3] + float(render_params.option_label_gap_px) + (0.5 * option_label_height)),
            ),
            font=option_label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": str(option_choice_id),
                "entity_type": "puzzle_overlay_option_choice",
                "bbox_px": _round_bbox(option_paper_bbox),
                "attrs": {
                    "option_label": str(option_spec["option_label"]),
                    "is_correct": bool(option_spec["is_correct"]),
                    "candidate_kind": str(option_spec["candidate_kind"]),
                },
            }
        )
        entities.append(
            {
                "entity_id": f"{str(option_choice_id)}_label",
                "entity_type": "puzzle_overlay_option_label",
                "bbox_px": list(label_bbox),
                "attrs": {
                    "option_label": str(option_spec["option_label"]),
                    "is_correct_option": bool(option_spec["is_correct"]),
                },
            }
        )
        entities.append(
            {
                "entity_id": f"{str(option_choice_id)}_paper",
                "entity_type": "puzzle_overlay_option_paper",
                "bbox_px": _round_bbox(option_paper_bbox),
                "attrs": {
                    "option_label": str(option_spec["option_label"]),
                    "is_correct_option": bool(option_spec["is_correct"]),
                    "grid_size": int(grid_size),
                },
            }
        )
        entities.extend(
            _draw_marks(
                draw,
                sheet_bbox=option_paper_bbox,
                mark_specs=option_spec["mark_specs"],
                grid_size=int(grid_size),
                entity_prefix=f"{str(option_choice_id)}_mark",
                entity_type="puzzle_overlay_option_mark",
                base_attrs={
                    "option_label": str(option_spec["option_label"]),
                    "is_correct_option": bool(option_spec["is_correct"]),
                },
                render_params=render_params,
            )
        )

    source_sheet_bbox_map = {
        "source_sheet_left": _round_bbox(left_sheet_bbox),
        "source_sheet_right": _round_bbox(right_sheet_bbox),
    }
    return RenderedPuzzleOverlayScene(
        image=image,
        entities=entities,
        scene_bbox_px=[
            round(float(scene_left), 3),
            round(float(scene_top), 3),
            round(float(scene_right), 3),
            round(float(options_bottom), 3),
        ],
        option_choice_bbox_map=option_choice_bbox_map,
        reference_panel_bbox_px=list(reference_panel_bbox),
        source_sheet_bbox_map=dict(source_sheet_bbox_map),
    )


__all__ = [
    "FOLD_RESULT_SUPERSAMPLE_SCALE",
    "RenderedPuzzleFoldCutResultScene",
    "RenderedPuzzleFoldResultScene",
    "RenderedPuzzleOverlayScene",
    "render_puzzle_fold_cut_result_scene",
    "render_puzzle_fold_result_scene",
    "render_puzzle_overlay_scene",
]
