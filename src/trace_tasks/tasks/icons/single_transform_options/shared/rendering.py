"""Rendering helpers for single-transform option icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.labeling import LABEL_POOL_A_L
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from ...shared.icon_assets import render_icon_transformed_rgba
from ...shared.icon_labeled_grid_scene import prepare_two_panel_labeled_grid_scene
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import panel_geometry_to_trace
from ...shared.icon_style import sample_single_icon_tint
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.icon_transform import IDENTITY_TRANSFORM_ID

from .sampling import option_transforms_for_answer, sample_transform_distinct_icon, validate_option_transform_signatures
from .state import SingleTransformOptionsScenePayload


def fit_rgba_inside(image: Image.Image, bbox: Sequence[int | float]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Resize an RGBA image to fit inside a bbox and return its placed bbox."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    max_w = max(1, int(x1 - x0))
    max_h = max(1, int(y1 - y0))
    source = image.convert("RGBA")
    scale = min(1.0, float(max_w) / float(max(1, source.size[0])), float(max_h) / float(max(1, source.size[1])))
    new_w = max(1, int(round(float(source.size[0]) * float(scale))))
    new_h = max(1, int(round(float(source.size[1]) * float(scale))))
    resized = source.resize((int(new_w), int(new_h)), resample=Image.Resampling.LANCZOS)
    px0 = int(x0 + (max_w - new_w) // 2)
    py0 = int(y0 + (max_h - new_h) // 2)
    return resized, (int(px0), int(py0), int(px0 + new_w), int(py0 + new_h))


def draw_centered_fit_text(
    *,
    image: Image.Image,
    text: str,
    bbox: Sequence[int | float],
    font_size_px: int,
    fill_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    """Draw one centered operation cue, shrinking it if necessary."""

    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    max_w = max(1, int(x1 - x0))
    max_h = max(1, int(y1 - y0))
    font_size = max(10, int(font_size_px))
    while font_size > 10:
        font = load_font(int(font_size), bold=True)
        text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
        text_w = int(text_bbox[2] - text_bbox[0])
        text_h = int(text_bbox[3] - text_bbox[1])
        if text_w <= max_w and text_h <= max_h:
            break
        font_size -= 1
    font = load_font(int(font_size), bold=True)
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_w = int(text_bbox[2] - text_bbox[0])
    text_h = int(text_bbox[3] - text_bbox[1])
    px = int(x0 + (max_w - text_w) // 2 - int(text_bbox[0]))
    py = int(y0 + (max_h - text_h) // 2 - int(text_bbox[1]))
    draw_text_traced(
        draw,
        (px, py),
        str(text),
        font=font,
        fill=tuple(int(v) for v in fill_rgb),
        stroke_fill=tuple(int(v) for v in stroke_rgb),
        stroke_width=1,
        role="icon_operation_cue_text",
        required=True,
    )
    return (int(px + text_bbox[0]), int(py + text_bbox[1]), int(px + text_bbox[2]), int(py + text_bbox[3]))


def reference_icon_and_cue_bboxes(
    content_bbox: Sequence[int | float],
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Split the Reference content area into icon and operation cue slots."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in content_bbox]
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    cue_h = max(34, int(round(height * 0.18)))
    gap = 8
    cue_bbox = (int(x0), int(y1 - cue_h), int(x1), int(y1))
    icon_bbox = (int(x0), int(y0), int(x1), int(max(y0 + 1, cue_bbox[1] - gap)))
    square = min(int(icon_bbox[2] - icon_bbox[0]), int(icon_bbox[3] - icon_bbox[1]))
    px0 = int(icon_bbox[0] + (width - square) // 2)
    py0 = int(icon_bbox[1] + (int(icon_bbox[3] - icon_bbox[1]) - square) // 2)
    return (int(px0), int(py0), int(px0 + square), int(py0 + square)), cue_bbox


def sample_and_render_single_transform_scene(
    rng,
    *,
    instance_seed: int,
    option_count: int,
    answer_index: int,
    target_transform_id: str,
    operation_cue: str,
    pool_manifest: str,
    transform_check_size_px: int,
    render_params: Mapping[str, Any],
    reference_transform_id: str = IDENTITY_TRANSFORM_ID,
    option_transform_ids: Sequence[str] | None = None,
) -> Tuple[SingleTransformOptionsScenePayload, Image.Image]:
    """Sample and render one transform-result option scene."""

    labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])
    answer_label = str(labels[int(answer_index)])
    icon_id = sample_transform_distinct_icon(
        rng,
        pool_manifest=str(pool_manifest),
        transform_check_size_px=int(transform_check_size_px),
    )
    if option_transform_ids is None:
        resolved_option_transform_ids = option_transforms_for_answer(
            answer_index=int(answer_index),
            target_transform_id=str(target_transform_id),
        )
    else:
        resolved_option_transform_ids = tuple(str(value) for value in option_transform_ids)
        if len(resolved_option_transform_ids) != int(option_count):
            raise ValueError("explicit option_transform_ids length must match option_count")
    validate_option_transform_signatures(
        icon_id=str(icon_id),
        transform_ids=resolved_option_transform_ids,
        check_size_px=int(transform_check_size_px),
    )

    tint_rgb, sampled_palette_rgb = sample_single_icon_tint(
        rng,
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
            tuple(int(v) for v in render_params["operation_cue_color_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )

    reference_noise_edits, reference_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace="icons.single_transform_options.reference_icon",
        render_params=render_params,
    )
    prepared = prepare_two_panel_labeled_grid_scene(
        scene_labels=labels,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        panel_title_font_size_px=int(render_params["panel_title_font_size_px"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        cell_padding_px=int(render_params["cell_padding_px"]),
        cell_border_rgb=tuple(int(v) for v in render_params["cell_border_rgb"]),
        cell_label_color_rgb=tuple(int(v) for v in render_params["cell_label_color_rgb"]),
        cell_label_stroke_rgb=tuple(int(v) for v in render_params["cell_label_stroke_rgb"]),
        cell_label_stroke_width_px=1,
        cell_label_font_size_px=int(render_params["cell_label_font_size_px"]),
        reference_content_padding_px=int(render_params["reference_content_padding_px"]),
        scene_content_side_padding_px=int(render_params["scene_content_side_padding_px"]),
        scene_content_bottom_padding_px=int(render_params["scene_content_bottom_padding_px"]),
        scene_content_top_offset_px=int(render_params["scene_content_top_offset_px"]),
        reference_square_cell=False,
        scene_square_cells=True,
        reference_title="Reference",
        scene_title="Options",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    image = prepared.image
    reference_icon_slot, cue_slot = reference_icon_and_cue_bboxes(prepared.reference_cell.content_bbox_xyxy)
    reference_sprite = render_icon_transformed_rgba(
        icon_id=str(icon_id),
        size_px=int(render_params["reference_icon_size_px"]),
        tint_rgb=tuple(int(v) for v in tint_rgb),
        transform_id=str(reference_transform_id),
        noise_edits=tuple(reference_noise_edits),
        noise_seed=int(reference_noise_seed),
    )
    reference_fitted, reference_icon_bbox = fit_rgba_inside(reference_sprite, reference_icon_slot)
    image.alpha_composite(reference_fitted, (int(reference_icon_bbox[0]), int(reference_icon_bbox[1])))
    cue_bbox = draw_centered_fit_text(
        image=image,
        text=str(operation_cue),
        bbox=cue_slot,
        font_size_px=int(render_params["operation_cue_font_size_px"]),
        fill_rgb=tuple(int(v) for v in render_params["operation_cue_color_rgb"]),
        stroke_rgb=tuple(int(v) for v in render_params["operation_cue_stroke_rgb"]),
    )

    scene_cells: list[Dict[str, Any]] = []
    for index, (cell, option_transform_id) in enumerate(zip(prepared.scene_cells, resolved_option_transform_ids)):
        option_noise_edits, option_noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"icons.single_transform_options.option_{int(index)}",
            render_params=render_params,
        )
        sprite = render_icon_transformed_rgba(
            icon_id=str(icon_id),
            size_px=int(render_params["scene_icon_size_max_px"]),
            tint_rgb=tuple(int(v) for v in tint_rgb),
            transform_id=str(option_transform_id),
            noise_edits=tuple(option_noise_edits),
            noise_seed=int(option_noise_seed),
        )
        fitted, icon_bbox = fit_rgba_inside(sprite, cell.content_bbox_xyxy)
        image.alpha_composite(fitted, (int(icon_bbox[0]), int(icon_bbox[1])))
        scene_cells.append(
            {
                "panel": "scene",
                "label": str(cell.label),
                "cell_bbox_xyxy": list(cell.cell_bbox_xyxy),
                "content_bbox_xyxy": list(cell.content_bbox_xyxy),
                "icon_bbox_xyxy": [int(value) for value in icon_bbox],
                "icon_id": str(icon_id),
                "transform_id": str(option_transform_id),
                "is_match": bool(int(index) == int(answer_index)),
                "tint_rgb": [int(value) for value in tint_rgb],
                "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(option_noise_edits))],
                "noise_seed": int(option_noise_seed),
                "index": int(index),
            }
        )

    if sum(1 for cell in scene_cells if bool(cell.get("is_match"))) != 1:
        raise ValueError("single-transform option scene must contain exactly one correct option")

    reference_cell = {
        "panel": "reference",
        "cell_bbox_xyxy": list(prepared.reference_cell.cell_bbox_xyxy),
        "content_bbox_xyxy": list(prepared.reference_cell.content_bbox_xyxy),
        "icon_bbox_xyxy": [int(value) for value in reference_icon_bbox],
        "operation_cue_bbox_xyxy": [int(value) for value in cue_bbox],
        "icon_id": str(icon_id),
        "transform_id": str(reference_transform_id),
        "operation_cue": str(operation_cue),
        "target_transform_id": str(target_transform_id),
        "tint_rgb": [int(value) for value in tint_rgb],
        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(reference_noise_edits))],
        "noise_seed": int(reference_noise_seed),
    }

    return (
        SingleTransformOptionsScenePayload(
            object_count=int(option_count),
            cell_labels=tuple(labels),
            target_transform_id=str(target_transform_id),
            operation_cue=str(operation_cue),
            answer_label=str(answer_label),
            icon_id=str(icon_id),
            option_transform_ids=tuple(str(value) for value in resolved_option_transform_ids),
            tint_rgb=tuple(int(value) for value in tint_rgb),
            sampled_palette_rgb=tuple(sampled_palette_rgb),
            panel_geometry=panel_geometry_to_trace(prepared.layout),
            reference_cell=dict(reference_cell),
            scene_cells=tuple(dict(item) for item in scene_cells),
        ),
        image.convert("RGB"),
    )


__all__ = [
    "draw_centered_fit_text",
    "fit_rgba_inside",
    "reference_icon_and_cue_bboxes",
    "sample_and_render_single_transform_scene",
]
