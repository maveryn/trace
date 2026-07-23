"""Sampling and rendering primitives for icon-cutout scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

from ...shared.icon_assets import render_icon_rgba, resolve_icon_pool
from ...shared.icon_labeled_grid_scene import prepare_two_panel_labeled_grid_scene
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import panel_geometry_to_trace
from ...shared.icon_task_rendering import sample_icon_instance_noise

from .sampling import sample_icon_cutout_option_icon_ids
from .state import FragmentPayload, IconCutoutScenePayload
from .styles import sample_icon_cutout_palette


def alpha_pixel_count(image: Image.Image) -> int:
    """Return the number of visibly nontransparent pixels in an RGBA image."""

    alpha = image.convert("RGBA").getchannel("A")
    return int(sum(1 for value in alpha.getdata() if int(value) > 8))


def apply_window_mask(image: Image.Image, *, style: str) -> Image.Image:
    """Apply the sampled fragment-window shape to one crop."""

    crop = image.convert("RGBA")
    if str(style) == "rectangle":
        return crop

    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    bbox = (0, 0, max(1, int(crop.size[0]) - 1), max(1, int(crop.size[1]) - 1))
    if str(style) == "ellipse":
        draw.ellipse(bbox, fill=255)
    elif str(style) == "rounded":
        radius = max(4, int(round(min(crop.size) * 0.18)))
        draw.rounded_rectangle(bbox, radius=int(radius), fill=255)
    else:
        raise ValueError(f"unsupported fragment window style: {style}")

    existing_alpha = crop.getchannel("A")
    masked_alpha = ImageChops.multiply(existing_alpha, mask)
    crop.putalpha(masked_alpha)
    return crop


def fit_rgba_inside(image: Image.Image, bbox: Sequence[int | float]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Resize an RGBA image to fit inside a bbox and return placed bbox."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    max_w = max(1, int(x1 - x0))
    max_h = max(1, int(y1 - y0))
    source = image.convert("RGBA")
    scale = min(float(max_w) / float(max(1, source.size[0])), float(max_h) / float(max(1, source.size[1])))
    new_w = max(1, int(round(float(source.size[0]) * float(scale))))
    new_h = max(1, int(round(float(source.size[1]) * float(scale))))
    resized = source.resize((int(new_w), int(new_h)), resample=Image.Resampling.LANCZOS)
    px0 = int(x0 + (max_w - new_w) // 2)
    py0 = int(y0 + (max_h - new_h) // 2)
    return resized, (int(px0), int(py0), int(px0 + new_w), int(py0 + new_h))


def sample_icon_fragment(rng, *, sprite: Image.Image, render_params: Mapping[str, Any]) -> FragmentPayload:
    """Sample one visible partial crop from the correct full icon sprite."""

    total_alpha = max(1, alpha_pixel_count(sprite))
    width_min, width_max = [float(value) for value in render_params["fragment_window_width_fraction_range"]]
    height_min, height_max = [float(value) for value in render_params["fragment_window_height_fraction_range"]]
    visible_min, visible_max = [float(value) for value in render_params["fragment_visible_alpha_ratio_range"]]
    density_min = float(render_params["fragment_alpha_density_min"])
    styles = tuple(str(value) for value in render_params["fragment_window_styles"])
    sprite_w, sprite_h = [int(value) for value in sprite.size]

    for _ in range(max(1, int(render_params["fragment_sampling_attempts"]))):
        crop_w = max(10, min(sprite_w - 1, int(round(float(sprite_w) * float(rng.uniform(width_min, width_max))))))
        crop_h = max(10, min(sprite_h - 1, int(round(float(sprite_h) * float(rng.uniform(height_min, height_max))))))
        if crop_w >= sprite_w and crop_h >= sprite_h:
            continue
        x0 = int(rng.randint(0, max(0, sprite_w - crop_w)))
        y0 = int(rng.randint(0, max(0, sprite_h - crop_h)))
        raw_crop = sprite.crop((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
        style = str(rng.choice(styles))
        crop = apply_window_mask(raw_crop, style=str(style))
        crop_alpha = alpha_pixel_count(crop)
        visible_ratio = float(crop_alpha) / float(total_alpha)
        density = float(crop_alpha) / float(max(1, crop.size[0] * crop.size[1]))
        if density <= 0.0:
            continue
        payload = FragmentPayload(
            image=crop,
            crop_xyxy=(int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)),
            window_style=str(style),
            visible_alpha_ratio=float(visible_ratio),
            alpha_density=float(density),
        )
        if float(visible_min) <= float(visible_ratio) <= float(visible_max) and float(density) >= float(density_min):
            return payload

    raise ValueError("failed to sample a visible partial icon fragment")


def draw_fragment_frame(
    *,
    image: Image.Image,
    bbox: Sequence[int | float],
    style: str,
    frame_rgb: Tuple[int, int, int],
    frame_width_px: int,
) -> None:
    """Draw the visible partial-fragment window frame."""

    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    width = max(1, int(frame_width_px))
    frame_bbox = (int(x0), int(y0), int(x1 - 1), int(y1 - 1))
    if str(style) == "ellipse":
        draw.ellipse(frame_bbox, outline=tuple(int(v) for v in frame_rgb), width=int(width))
    elif str(style) == "rounded":
        radius = max(8, int(round(min(max(1, x1 - x0), max(1, y1 - y0)) * 0.10)))
        draw.rounded_rectangle(frame_bbox, radius=int(radius), outline=tuple(int(v) for v in frame_rgb), width=int(width))
    else:
        draw.rectangle(frame_bbox, outline=tuple(int(v) for v in frame_rgb), width=int(width))


def sample_and_render_icon_cutout_scene(
    rng,
    *,
    instance_seed: int,
    render_params: Mapping[str, Any],
    pool_manifest: str,
    labels: Sequence[str],
    matching_index: int,
    noise_namespace: str,
) -> Tuple[IconCutoutScenePayload, Image.Image]:
    """Sample and render one partial-fragment option scene."""

    labels = tuple(str(value) for value in labels)
    object_count = int(len(labels))
    answer_index = int(matching_index)
    if not 0 <= int(answer_index) < int(object_count):
        raise ValueError("matching_index out of range for icon-cutout labels")
    answer_label = str(labels[int(answer_index)])

    pool = tuple(str(icon_id) for icon_id in resolve_icon_pool(str(pool_manifest)))
    if len(pool) < int(object_count):
        raise ValueError("icon-cutout pool resolved too few icons")
    correct_icon_id = str(rng.choice(pool))
    rotation_degrees = int(rng.choice(tuple(int(value) for value in render_params["rotation_candidates_degrees"])))
    option_icon_ids = sample_icon_cutout_option_icon_ids(
        rng,
        pool=pool,
        correct_icon_id=str(correct_icon_id),
        correct_index=int(answer_index),
        object_count=int(object_count),
        signature_size_px=int(render_params["scene_icon_size_max_px"]),
    )

    sampled_palette_rgb = sample_icon_cutout_palette(rng, render_params)
    tint_rgb = tuple(int(channel) for channel in rng.choice(sampled_palette_rgb))
    source_noise_edits, source_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{noise_namespace}:source_icon",
        render_params=render_params,
    )
    source_sprite = render_icon_rgba(
        icon_id=str(correct_icon_id),
        size_px=int(render_params["reference_icon_size_px"]),
        tint_rgb=tuple(int(channel) for channel in tint_rgb),
        rotation_degrees=int(rotation_degrees),
        mirror_x=False,
        noise_edits=tuple(source_noise_edits),
        noise_seed=int(source_noise_seed),
    )
    fragment = sample_icon_fragment(rng, sprite=source_sprite, render_params=render_params)

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
        reference_square_cell=True,
        scene_square_cells=True,
        reference_title="Fragment",
        scene_title="Full icons",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    image = prepared.image
    fragment_image, fragment_bbox = fit_rgba_inside(fragment.image, prepared.reference_cell.content_bbox_xyxy)
    image.alpha_composite(fragment_image, (int(fragment_bbox[0]), int(fragment_bbox[1])))
    draw_fragment_frame(
        image=image,
        bbox=fragment_bbox,
        style=str(fragment.window_style),
        frame_rgb=tuple(int(v) for v in render_params["fragment_frame_rgb"]),
        frame_width_px=int(render_params["fragment_frame_width_px"]),
    )

    reference_cell = {
        "panel": "reference",
        "cell_bbox_xyxy": list(prepared.reference_cell.cell_bbox_xyxy),
        "content_bbox_xyxy": list(prepared.reference_cell.content_bbox_xyxy),
        "fragment_bbox_xyxy": [int(value) for value in fragment_bbox],
        "source_icon_id": str(correct_icon_id),
        "source_icon_size_px": int(render_params["reference_icon_size_px"]),
        "source_icon_crop_xyxy": [int(value) for value in fragment.crop_xyxy],
        "fragment_window_style": str(fragment.window_style),
        "fragment_visible_alpha_ratio": float(fragment.visible_alpha_ratio),
        "fragment_alpha_density": float(fragment.alpha_density),
        "rotation_degrees": int(rotation_degrees),
        "tint_rgb": [int(value) for value in tint_rgb],
        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(source_noise_edits))],
        "noise_seed": int(source_noise_seed),
    }

    scene_cells: List[Dict[str, Any]] = []
    for index, (cell, icon_id) in enumerate(zip(prepared.scene_cells, option_icon_ids)):
        if int(index) == int(answer_index):
            option_noise_edits = tuple(source_noise_edits)
            option_noise_seed = int(source_noise_seed)
        else:
            option_noise_edits, option_noise_seed = sample_icon_instance_noise(
                instance_seed=int(instance_seed),
                namespace=f"{noise_namespace}:option_{int(index)}",
                render_params=render_params,
            )
        sprite = render_icon_rgba(
            icon_id=str(icon_id),
            size_px=int(render_params["scene_icon_size_max_px"]),
            tint_rgb=tuple(int(channel) for channel in tint_rgb),
            rotation_degrees=int(rotation_degrees),
            mirror_x=False,
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
                "is_match": bool(int(index) == int(answer_index)),
                "rotation_degrees": int(rotation_degrees),
                "tint_rgb": [int(value) for value in tint_rgb],
                "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(option_noise_edits))],
                "noise_seed": int(option_noise_seed),
            }
        )

    if sum(1 for cell in scene_cells if bool(cell.get("is_match"))) != 1:
        raise ValueError("icon-cutout scene must contain exactly one matching full-icon option")

    return IconCutoutScenePayload(
        object_count=int(object_count),
        cell_labels=tuple(labels),
        answer_label=str(answer_label),
        correct_icon_id=str(correct_icon_id),
        option_icon_ids=tuple(str(value) for value in option_icon_ids),
        tint_rgb=tuple(int(value) for value in tint_rgb),
        rotation_degrees=int(rotation_degrees),
        fragment_window_style=str(fragment.window_style),
        fragment_visible_alpha_ratio=float(fragment.visible_alpha_ratio),
        fragment_alpha_density=float(fragment.alpha_density),
        fragment_crop_xyxy=tuple(int(value) for value in fragment.crop_xyxy),
        sampled_palette_rgb=tuple(sampled_palette_rgb),
        panel_geometry=panel_geometry_to_trace(prepared.layout),
        reference_cell=dict(reference_cell),
        scene_cells=tuple(dict(item) for item in scene_cells),
    ), image.convert("RGB")


__all__ = [
    "alpha_pixel_count",
    "apply_window_mask",
    "draw_fragment_frame",
    "fit_rgba_inside",
    "sample_and_render_icon_cutout_scene",
    "sample_icon_fragment",
]
