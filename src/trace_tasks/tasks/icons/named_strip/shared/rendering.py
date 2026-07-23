"""Rendering helpers for horizontal named-icon strips."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.icon_grid_scene import resolve_horizontal_row_slots
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import (
    draw_single_panel,
    resolve_single_panel_layout,
    single_panel_geometry_to_trace,
)
from ...shared.procedural_named_icon_field_scene import (
    bbox_from_center_dimensions,
    render_planned_named_icon_sprite,
)
from ...shared.procedural_named_icons import procedural_named_icon_display_name

from .defaults import SCENE_ID
from .state import NamedStripIconPlan, NamedStripScenePayload, RenderedNamedStripIcon


def named_strip_canvas_size(
    *,
    strip_length: int,
    cell_box_width_px: int,
    cell_box_height_px: int,
    render_params: Mapping[str, Any],
) -> Tuple[int, int]:
    """Derive a canvas size that fits the sampled named-strip cell geometry."""

    cell_padding_px = int(render_params["cell_padding_px"])
    panel_padding_px = int(render_params["panel_padding_px"])
    outer_margin_px = int(render_params["outer_margin_px"])
    title_font_size_px = int(render_params["panel_title_font_size_px"])
    title_band_height = max(40, int(round(float(title_font_size_px) * 1.8)))
    content_width = int(strip_length) * int(int(cell_box_width_px) + (2 * cell_padding_px))
    content_height = int(int(cell_box_height_px) + (2 * cell_padding_px))
    panel_width = int(content_width + (2 * panel_padding_px))
    panel_height = int(content_height + title_band_height + panel_padding_px + (panel_padding_px // 2))
    return int(panel_width + (2 * outer_margin_px)), int(panel_height + (2 * outer_margin_px))


def render_named_strip_scene(
    *,
    strip_length: int,
    target_shape_id: str,
    selected_run_indices: Sequence[int],
    plans: Sequence[NamedStripIconPlan],
    render_params: Mapping[str, Any],
    rng,
    sampled_palette_rgb: Sequence[Sequence[int]],
) -> NamedStripScenePayload:
    """Render one single-panel row of boxed procedural named icons."""

    cell_box_width_px = int(
        rng.randint(int(render_params["cell_box_width_min_px"]), int(render_params["cell_box_width_max_px"]))
    )
    cell_box_height_px = int(
        rng.randint(int(render_params["cell_box_height_min_px"]), int(render_params["cell_box_height_max_px"]))
    )
    canvas_width, canvas_height = named_strip_canvas_size(
        strip_length=int(strip_length),
        cell_box_width_px=int(cell_box_width_px),
        cell_box_height_px=int(cell_box_height_px),
        render_params=render_params,
    )
    layout = resolve_single_panel_layout(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    cell_slots = resolve_horizontal_row_slots(
        tuple(int(value) for value in layout.scene_content_xyxy),
        cell_count=int(strip_length),
        cell_padding_px=int(render_params["cell_padding_px"]),
        target_aspect_ratio=float(cell_box_width_px) / float(max(1, cell_box_height_px)),
    )
    draw = ImageDraw.Draw(image)
    rendered_icons: List[RenderedNamedStripIcon] = []
    rendered_cells: List[Dict[str, Any]] = []
    selected_indices = set(int(index) for index in selected_run_indices)
    for cell_index, (plan, cell_bbox) in enumerate(zip(plans, cell_slots)):
        cell_bbox = tuple(int(value) for value in cell_bbox)
        draw.rounded_rectangle(
            cell_bbox,
            radius=max(0, int(render_params["cell_corner_radius_px"])),
            outline=tuple(int(value) for value in render_params["cell_border_rgb"]),
            width=2,
            fill=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        )
        sprite = render_planned_named_icon_sprite(plan)
        inner_bbox = (
            int(cell_bbox[0] + int(render_params["cell_icon_padding_px"])),
            int(cell_bbox[1] + int(render_params["cell_icon_padding_px"])),
            int(cell_bbox[2] - int(render_params["cell_icon_padding_px"])),
            int(cell_bbox[3] - int(render_params["cell_icon_padding_px"])),
        )
        if int(sprite.size[0]) > int(inner_bbox[2] - inner_bbox[0]) or int(sprite.size[1]) > int(inner_bbox[3] - inner_bbox[1]):
            raise ValueError("named icon sprite does not fit row cell")
        center = (
            0.5 * float(inner_bbox[0] + inner_bbox[2]),
            0.5 * float(inner_bbox[1] + inner_bbox[3]),
        )
        paste_bbox = bbox_from_center_dimensions(center, width=int(sprite.size[0]), height=int(sprite.size[1]))
        image.alpha_composite(sprite, (int(paste_bbox[0]), int(paste_bbox[1])))
        instance_id = f"strip_cell_{int(cell_index)}"
        rendered_icons.append(
            RenderedNamedStripIcon(
                instance_id=str(instance_id),
                cell_index=int(cell_index),
                shape_id=str(plan.shape_id),
                shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                bbox_xyxy=tuple(int(value) for value in paste_bbox),
                cell_bbox_xyxy=tuple(int(value) for value in cell_bbox),
                nominal_size_px=int(plan.nominal_size_px),
                tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                fill_style=str(plan.fill_style),
                rotation_degrees=int(plan.rotation_degrees),
                is_target_shape=str(plan.shape_id) == str(target_shape_id),
                is_selected_run_member=int(cell_index) in selected_indices,
                noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                noise_seed=None if plan.noise_seed is None else int(plan.noise_seed),
            )
        )
        rendered_cells.append(
            {
                "entity_kind": "strip_cell",
                "instance_id": f"cell_{int(cell_index)}",
                "cell_index": int(cell_index),
                "cell_bbox_xyxy": [int(value) for value in cell_bbox],
                "shape_id": str(plan.shape_id),
                "shape_name": procedural_named_icon_display_name(str(plan.shape_id)),
            }
        )
    return NamedStripScenePayload(
        image=image.convert("RGB"),
        icons=tuple(rendered_icons),
        cells=tuple(rendered_cells),
        panel_geometry=single_panel_geometry_to_trace(layout),
        cell_box_width_px=int(cell_box_width_px),
        cell_box_height_px=int(cell_box_height_px),
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
    )


def serialize_named_strip_icon(icon: RenderedNamedStripIcon) -> Dict[str, Any]:
    """Serialize one rendered named-strip icon for trace entities."""

    return {
        "entity_kind": "named_icon",
        "instance_id": str(icon.instance_id),
        "cell_index": int(icon.cell_index),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "cell_bbox_xyxy": [int(value) for value in icon.cell_bbox_xyxy],
        "nominal_size_px": int(icon.nominal_size_px),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "rotation_degrees": int(icon.rotation_degrees),
        "is_target_shape": bool(icon.is_target_shape),
        "is_selected_run_member": bool(icon.is_selected_run_member),
        "noise_edits": [dict(value) for value in icon.noise_edits],
        "noise_seed": None if icon.noise_seed is None else int(icon.noise_seed),
    }


__all__ = [
    "SCENE_ID",
    "named_strip_canvas_size",
    "render_named_strip_scene",
    "serialize_named_strip_icon",
]
