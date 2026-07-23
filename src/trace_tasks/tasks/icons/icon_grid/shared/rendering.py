"""Neutral renderer for visible icon-grid scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from ...shared.icon_assets import render_icon_rgba, resolve_icon_pool
from ...shared.icon_grid_scene import BBox
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ...shared.icon_task_rendering import sample_icon_instance_noise

from .state import IconGridFrequencySpec, IconGridScenePayload
from .styles import sample_icon_grid_palette


def _rgb_key(rgb: Tuple[int, int, int]) -> str:
    """Return a stable key for one rendered RGB tint."""

    return ",".join(str(int(channel)) for channel in rgb)


def _grid_shape_for_count(object_count: int) -> tuple[int, int]:
    """Return a compact grid shape with stable row-major positions."""

    count = int(object_count)
    if count <= 6:
        return 2, 3
    if count <= 8:
        return 2, 4
    if count == 9:
        return 3, 3
    if count <= 10:
        return 2, 5
    return 3, 4


def _resolve_icon_grid_bboxes(
    *,
    content_bbox: BBox,
    rows: int,
    cols: int,
    grid_cell_max_size_px: int,
) -> tuple[BBox, tuple[tuple[BBox, ...], ...], int]:
    """Resolve visible grid and cell boxes centered in the content area."""

    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    available_w = max(1, int(x1 - x0))
    available_h = max(1, int(y1 - y0))
    cell_size = min(
        int(grid_cell_max_size_px),
        int(available_w // max(1, int(cols))),
        int(available_h // max(1, int(rows))),
    )
    if int(cell_size) < 48:
        raise ValueError("icon-grid content area is too small for clear cells")
    grid_w = int(cell_size) * int(cols)
    grid_h = int(cell_size) * int(rows)
    gx0 = int(x0 + max(0, (available_w - grid_w) // 2))
    gy0 = int(y0 + max(0, (available_h - grid_h) // 2))
    grid_bbox = (int(gx0), int(gy0), int(gx0 + grid_w), int(gy0 + grid_h))
    cell_rows: List[tuple[BBox, ...]] = []
    for row in range(int(rows)):
        row_boxes: List[BBox] = []
        for col in range(int(cols)):
            cx0 = int(gx0 + int(col) * int(cell_size))
            cy0 = int(gy0 + int(row) * int(cell_size))
            row_boxes.append((int(cx0), int(cy0), int(cx0 + cell_size), int(cy0 + cell_size)))
        cell_rows.append(tuple(row_boxes))
    return tuple(int(value) for value in grid_bbox), tuple(cell_rows), int(cell_size)


def sample_and_render_icon_grid_scene(
    rng,
    *,
    instance_seed: int,
    frequency_spec: IconGridFrequencySpec,
    pool_manifest: str,
    render_params: Mapping[str, Any],
    noise_namespace: str,
) -> tuple[IconGridScenePayload, Image.Image]:
    """Sample and render one visible grid of icons."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < int(frequency_spec.distinct_type_count):
        raise ValueError("icon pool is too small for icon-grid scene")
    sampled_icon_ids = [
        str(icon_id) for icon_id in rng.sample(pool, int(frequency_spec.distinct_type_count))
    ]
    singleton_icon_ids = [str(icon_id) for icon_id in sampled_icon_ids[: int(frequency_spec.singleton_count)]]
    repeated_icon_ids = [str(icon_id) for icon_id in sampled_icon_ids[int(frequency_spec.singleton_count) :]]

    scene_icon_ids = list(singleton_icon_ids)
    type_frequencies: Dict[str, int] = {str(icon_id): 1 for icon_id in singleton_icon_ids}
    for icon_id, multiplicity in zip(repeated_icon_ids, frequency_spec.repeated_type_multiplicities):
        type_frequencies[str(icon_id)] = int(multiplicity)
        scene_icon_ids.extend([str(icon_id)] * int(multiplicity))
    if len(scene_icon_ids) != int(frequency_spec.object_count):
        raise ValueError("icon-grid scene did not realize the requested object count")
    rng.shuffle(scene_icon_ids)

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )

    rows, cols = _grid_shape_for_count(int(frequency_spec.object_count))
    grid_bbox, cell_bboxes, cell_size_px = _resolve_icon_grid_bboxes(
        content_bbox=tuple(int(value) for value in layout.scene_content_xyxy),
        rows=int(rows),
        cols=int(cols),
        grid_cell_max_size_px=int(render_params["grid_cell_max_size_px"]),
    )
    draw = ImageDraw.Draw(image)
    for row in range(int(rows)):
        for col in range(int(cols)):
            cell_bbox = cell_bboxes[int(row)][int(col)]
            fill = (
                tuple(int(value) for value in render_params["alternate_cell_fill_rgb"])
                if (int(row) + int(col)) % 2
                else tuple(int(value) for value in render_params["cell_fill_rgb"])
            )
            draw.rectangle(
                cell_bbox,
                fill=fill,
                outline=tuple(int(value) for value in render_params["grid_line_rgb"]),
                width=max(1, int(render_params["grid_line_width_px"])),
            )
    draw.rectangle(
        grid_bbox,
        outline=tuple(int(value) for value in render_params["grid_line_rgb"]),
        width=max(1, int(render_params["grid_border_width_px"])),
    )

    palette = sample_icon_grid_palette(rng, render_params)
    distinct_color_count = int(
        frequency_spec.distinct_color_count
        if frequency_spec.distinct_color_count is not None
        else frequency_spec.distinct_type_count
    )
    if len(palette) < int(distinct_color_count):
        raise ValueError("icon-grid palette is too small for requested distinct colors")
    sampled_tints = [
        tuple(int(channel) for channel in color)
        for color in rng.sample(list(palette), int(distinct_color_count))
    ]
    if frequency_spec.distinct_color_count is None:
        type_color_groups = {
            str(icon_id): int(type_index)
            for type_index, icon_id in enumerate(sampled_icon_ids)
        }
        per_instance_color_groups: List[int] = []
    else:
        per_instance_color_groups = list(range(int(distinct_color_count)))
        per_instance_color_groups.extend(
            int(rng.randrange(int(distinct_color_count)))
            for _ in range(int(frequency_spec.object_count) - int(distinct_color_count))
        )
        rng.shuffle(per_instance_color_groups)
        type_color_groups = {}

    min_icon_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_icon_size = max(
        min_icon_size,
        min(
            int(render_params["scene_icon_size_max_px"]),
            int(cell_size_px) - (2 * int(render_params["grid_cell_padding_px"])),
        ),
    )
    rotations = tuple(int(value) for value in render_params["rotation_candidates_degrees"])
    type_styles: Dict[str, Dict[str, Any]] = {}
    for type_index, icon_id in enumerate(sampled_icon_ids):
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{noise_namespace}:icon_type_{int(type_index)}",
            render_params=render_params,
        )
        type_styles[str(icon_id)] = {
            "rotation_degrees": int(rng.choice(rotations)) % 360,
            "nominal_size_px": int(rng.randint(int(min_icon_size), int(max_icon_size))),
            "noise_edits": tuple(noise_edits),
            "noise_seed": int(noise_seed),
        }

    scene_instances: List[Dict[str, Any]] = []
    scene_rotations_degrees: List[int] = []
    scene_tint_rgbs: List[Tuple[int, int, int]] = []
    scene_color_keys: List[str] = []
    color_group_indices: List[int] = []
    color_frequencies: Dict[str, int] = {}
    occupied_cells = [(index // int(cols), index % int(cols)) for index in range(int(frequency_spec.object_count))]
    for index, (icon_id, cell) in enumerate(zip(scene_icon_ids, occupied_cells)):
        row, col = int(cell[0]), int(cell[1])
        type_style = type_styles[str(icon_id)]
        color_group_index = (
            int(type_color_groups[str(icon_id)])
            if frequency_spec.distinct_color_count is None
            else int(per_instance_color_groups[int(index)])
        )
        tint_rgb = tuple(int(channel) for channel in sampled_tints[int(color_group_index)])
        color_key = _rgb_key(tint_rgb)
        color_frequencies[str(color_key)] = int(color_frequencies.get(str(color_key), 0)) + 1
        sprite = render_icon_rgba(
            icon_id=str(icon_id),
            size_px=int(type_style["nominal_size_px"]),
            tint_rgb=tint_rgb,
            rotation_degrees=int(type_style["rotation_degrees"]),
            mirror_x=False,
            noise_edits=tuple(type_style["noise_edits"]),
            noise_seed=int(type_style["noise_seed"]),
        )
        cell_bbox = cell_bboxes[int(row)][int(col)]
        cx = 0.5 * float(cell_bbox[0] + cell_bbox[2])
        cy = 0.5 * float(cell_bbox[1] + cell_bbox[3])
        paste_x0 = int(round(cx - (0.5 * float(sprite.size[0]))))
        paste_y0 = int(round(cy - (0.5 * float(sprite.size[1]))))
        icon_bbox = (
            int(paste_x0),
            int(paste_y0),
            int(paste_x0 + sprite.size[0]),
            int(paste_y0 + sprite.size[1]),
        )
        image.alpha_composite(sprite, (int(paste_x0), int(paste_y0)))
        scene_rotations_degrees.append(int(type_style["rotation_degrees"]) % 360)
        scene_tint_rgbs.append(tuple(int(channel) for channel in tint_rgb))
        scene_color_keys.append(str(color_key))
        color_group_indices.append(int(color_group_index))
        scene_instances.append(
            {
                "instance_id": f"grid_r{int(row) + 1}_c{int(col) + 1}",
                "entity_kind": "grid_icon",
                "index": int(index),
                "row_index": int(row),
                "col_index": int(col),
                "row_number": int(row) + 1,
                "column_number": int(col) + 1,
                "icon_id": str(icon_id),
                "panel": "scene",
                "bbox_xyxy": [int(value) for value in icon_bbox],
                "cell_bbox_xyxy": [int(value) for value in cell_bbox],
                "nominal_size_px": int(type_style["nominal_size_px"]),
                "rotation_degrees": int(type_style["rotation_degrees"]) % 360,
                "mirror_x": False,
                "tint_rgb": [int(channel) for channel in tint_rgb],
                "noise_edits": [
                    dict(edit)
                    for edit in serialize_icon_noise_edits(tuple(type_style["noise_edits"]))
                ],
                "noise_seed": int(type_style["noise_seed"]),
                "type_frequency": int(type_frequencies[str(icon_id)]),
                "color_key": str(color_key),
                "color_group_index": int(color_group_index),
            }
        )
    for entity in scene_instances:
        color_key = str(entity.get("color_key", ""))
        entity["color_frequency"] = int(color_frequencies[str(color_key)])

    return (
        IconGridScenePayload(
            object_count=int(frequency_spec.object_count),
            distinct_type_count=int(frequency_spec.distinct_type_count),
            distinct_color_count=int(len(color_frequencies)),
            grid_rows=int(rows),
            grid_cols=int(cols),
            singleton_icon_ids=tuple(str(icon_id) for icon_id in singleton_icon_ids),
            repeated_icon_ids=tuple(str(icon_id) for icon_id in repeated_icon_ids),
            scene_icon_ids=tuple(str(icon_id) for icon_id in scene_icon_ids),
            scene_rotations_degrees=tuple(int(value) for value in scene_rotations_degrees),
            scene_tint_rgbs=tuple(tuple(int(channel) for channel in color) for color in scene_tint_rgbs),
            scene_color_keys=tuple(str(value) for value in scene_color_keys),
            color_group_indices=tuple(int(value) for value in color_group_indices),
            type_frequencies={str(key): int(value) for key, value in type_frequencies.items()},
            color_frequencies={str(key): int(value) for key, value in color_frequencies.items()},
            sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
            grid_bbox_xyxy=tuple(int(value) for value in grid_bbox),
            cell_bboxes_xyxy=tuple(tuple(tuple(int(value) for value in box) for box in row) for row in cell_bboxes),
            panel_geometry=single_panel_geometry_to_trace(layout),
            scene_instances=tuple(dict(entity) for entity in scene_instances),
        ),
        image.convert("RGB"),
    )


__all__ = ["sample_and_render_icon_grid_scene"]
