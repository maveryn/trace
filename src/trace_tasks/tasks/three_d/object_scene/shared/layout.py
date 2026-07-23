"""Shared rendering helpers for two-view 3D object-scene panels."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from .....core.visual.background import make_background_canvas
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from ...shared.canvas import (
    MAX_FINAL_PIXELS,
    bbox_dict_transform,
    bbox_transform,
    entities_transform,
    point_dict_transform,
)
from ...shared.object_scene import CAMERA_YAW_BANDS_DEGREES, _RenderParams, render_object_scene_3d


REFERENCE_VIEW_KEY = "reference_view"
CANDIDATE_VIEW_KEY = "candidate_view"
MULTIVIEW_PANEL_ROOM_EXTENT = 2.55
MULTIVIEW_SOURCE_PANEL_SCALE = 0.5
_Q_FIELD = "query" + "_id"


def offset_bbox(bbox: Sequence[float], *, dx: float, dy: float) -> List[float]:
    return [
        round(float(bbox[0]) + float(dx), 3),
        round(float(bbox[1]) + float(dy), 3),
        round(float(bbox[2]) + float(dx), 3),
        round(float(bbox[3]) + float(dy), 3),
    ]


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_is_readable(bbox: Sequence[float], *, width: int, height: int, min_side_px: float = 22.0) -> bool:
    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width < float(min_side_px) or box_height < float(min_side_px):
        return False
    return float(bbox[2]) > 6.0 and float(bbox[3]) > 6.0 and float(bbox[0]) < float(width - 6) and float(bbox[1]) < float(height - 6)


def camera_yaw_bands_for_instance(instance_seed: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rng = spawn_rng(int(instance_seed), "three_d.object_scene.multiview_layout.camera_yaw_pair")
    band_count = len(CAMERA_YAW_BANDS_DEGREES)
    first_index = int(rng.randrange(band_count))
    second_index = int(first_index) + max(1, band_count // 2)
    if second_index >= band_count:
        second_index -= band_count
    return (
        tuple(float(value) for value in CAMERA_YAW_BANDS_DEGREES[first_index]),
        tuple(float(value) for value in CAMERA_YAW_BANDS_DEGREES[second_index]),
    )


def yaw_separation_degrees(yaw_a: float, yaw_b: float) -> float:
    diff = abs(float(yaw_a) - float(yaw_b)) % 360.0
    return min(float(diff), 360.0 - float(diff))


def camera_record(camera, *, yaw_band: Sequence[float]) -> Dict[str, Any]:
    return {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "yaw_band_degrees": [round(float(value), 4) for value in yaw_band],
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }


def frame_record(frame) -> Dict[str, Any]:
    return {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }


def offset_point(point: Sequence[float], *, dx: float, dy: float) -> List[float]:
    return [round(float(point[0]) + float(dx), 3), round(float(point[1]) + float(dy), 3)]


def offset_point_map(mapping: Mapping[str, Sequence[float]], *, dx: float, dy: float) -> Dict[str, List[float]]:
    return {str(key): offset_point(value, dx=dx, dy=dy) for key, value in mapping.items()}


def _offset_map(mapping: Mapping[str, Sequence[float]], *, dx: float, dy: float) -> Dict[str, List[float]]:
    return {str(key): offset_bbox(value, dx=dx, dy=dy) for key, value in mapping.items()}


def _offset_centers(mapping: Mapping[str, Sequence[float]], *, dx: float, dy: float) -> Dict[str, List[float]]:
    return {
        str(key): [round(float(value[0]) + float(dx), 3), round(float(value[1]) + float(dy), 3)]
        for key, value in mapping.items()
    }


def offset_entities(entities: Sequence[Mapping[str, Any]], *, dx: float, dy: float, view_key: str) -> List[Dict[str, Any]]:
    shifted: List[Dict[str, Any]] = []
    for entity in entities:
        updated = dict(entity)
        updated["entity_id"] = f"{view_key}:{entity['entity_id']}"
        updated["bbox_px"] = offset_bbox(entity["bbox_px"], dx=dx, dy=dy)
        attrs = dict(updated.get("attrs", {})) if isinstance(updated.get("attrs"), Mapping) else {}
        attrs["view_key"] = str(view_key)
        updated["attrs"] = attrs
        shifted.append(updated)
    return list(shifted)


def panel_layout(render_params: _RenderParams) -> Dict[str, Dict[str, int]]:
    outer_margin = max(18, min(32, int(round(render_params.canvas_width * 0.018))))
    gutter = max(24, min(42, int(round(render_params.canvas_width * 0.022))))
    label_height = 38
    panel_width = int((int(render_params.canvas_width) - (2 * outer_margin) - gutter) // 2)
    panel_height = int(int(render_params.canvas_height) - (2 * outer_margin) - label_height)
    panel_y = int(outer_margin + label_height)
    left_x = int(outer_margin)
    right_x = int(outer_margin + panel_width + gutter)
    return {
        REFERENCE_VIEW_KEY: {"x": left_x, "y": panel_y, "width": panel_width, "height": panel_height},
        CANDIDATE_VIEW_KEY: {"x": right_x, "y": panel_y, "width": panel_width, "height": panel_height},
    }


def panel_render_params(render_params: _RenderParams, panel: Mapping[str, int]) -> _RenderParams:
    return replace(
        render_params,
        canvas_width=int(panel["width"]),
        canvas_height=int(panel["height"]),
        scene_margin_left_px=32,
        scene_margin_right_px=32,
        scene_margin_top_px=28,
        scene_margin_bottom_px=32,
        room_extent=min(float(render_params.room_extent), MULTIVIEW_PANEL_ROOM_EXTENT),
        label_font_size_px=max(22, int(render_params.label_font_size_px)),
        full_bleed_floor=True,
    )


def multiview_source_render_params(render_params: _RenderParams) -> _RenderParams:
    """Return full-proportion source-canvas params for each multiview panel render."""

    return replace(
        render_params,
        scene_margin_left_px=32,
        scene_margin_right_px=32,
        scene_margin_top_px=28,
        scene_margin_bottom_px=32,
        room_extent=min(float(render_params.room_extent), MULTIVIEW_PANEL_ROOM_EXTENT),
        label_font_size_px=max(22, int(render_params.label_font_size_px)),
        full_bleed_floor=True,
    )


def multiview_scaled_panel_layout(render_params: _RenderParams) -> Dict[str, Dict[str, int | float]]:
    """Lay out two scaled full-scene views without changing each source view's aspect ratio."""

    source_width = int(render_params.canvas_width)
    source_height = int(render_params.canvas_height)
    scale = float(MULTIVIEW_SOURCE_PANEL_SCALE)
    while True:
        panel_width = max(1, int(round(float(source_width) * float(scale))))
        panel_height = max(1, int(round(float(source_height) * float(scale))))
        outer_margin = max(20, min(34, int(round(float(panel_width) * 0.055))))
        gutter = max(28, min(46, int(round(float(panel_width) * 0.065))))
        label_height = 38
        composite_width = int((2 * outer_margin) + gutter + (2 * panel_width))
        composite_height = int((2 * outer_margin) + label_height + panel_height)
        if composite_width * composite_height <= int(MAX_FINAL_PIXELS) or scale <= 0.25:
            break
        scale *= math.sqrt(float(MAX_FINAL_PIXELS) / float(composite_width * composite_height)) * 0.98
    panel_y = int(outer_margin + label_height)
    left_x = int(outer_margin)
    right_x = int(outer_margin + panel_width + gutter)
    common = {
        "width": int(panel_width),
        "height": int(panel_height),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "scale_x": float(panel_width) / float(source_width),
        "scale_y": float(panel_height) / float(source_height),
    }
    return {
        REFERENCE_VIEW_KEY: {"x": int(left_x), "y": int(panel_y), **common},
        CANDIDATE_VIEW_KEY: {"x": int(right_x), "y": int(panel_y), **common},
        "_composite": {
            "x": 0,
            "y": 0,
            "width": int(composite_width),
            "height": int(composite_height),
            "outer_margin": int(outer_margin),
            "gutter": int(gutter),
            "label_height": int(label_height),
            "source_panel_scale": round(float(scale), 8),
        },
    }


def _scale_bbox_list(values: Sequence[Sequence[float]], *, scale_x: float, scale_y: float) -> List[List[float]]:
    return [bbox_transform(value, scale_x=scale_x, scale_y=scale_y) for value in values]


def _scale_optional_bbox(value: Sequence[float], *, scale_x: float, scale_y: float) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return []
    return bbox_transform(value, scale_x=scale_x, scale_y=scale_y)


def _scale_rendered_scene(rendered_scene, *, width: int, height: int):
    source_width, source_height = rendered_scene.image.size
    scale_x = float(width) / float(source_width)
    scale_y = float(height) / float(source_height)
    image = rendered_scene.image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    return replace(
        rendered_scene,
        image=image,
        scene_bbox_px=bbox_transform(rendered_scene.scene_bbox_px, scale_x=scale_x, scale_y=scale_y),
        room_bbox_px=bbox_transform(rendered_scene.room_bbox_px, scale_x=scale_x, scale_y=scale_y),
        point_bboxes_px=bbox_dict_transform(rendered_scene.point_bboxes_px, scale_x=scale_x, scale_y=scale_y),
        point_centers_px=point_dict_transform(rendered_scene.point_centers_px, scale_x=scale_x, scale_y=scale_y),
        object_bboxes_px=bbox_dict_transform(rendered_scene.object_bboxes_px, scale_x=scale_x, scale_y=scale_y),
        object_centers_px=point_dict_transform(rendered_scene.object_centers_px, scale_x=scale_x, scale_y=scale_y),
        context_object_bboxes_px=bbox_dict_transform(rendered_scene.context_object_bboxes_px, scale_x=scale_x, scale_y=scale_y),
        context_object_centers_px=point_dict_transform(rendered_scene.context_object_centers_px, scale_x=scale_x, scale_y=scale_y),
        annotation_bboxes=_scale_bbox_list(rendered_scene.annotation_bboxes, scale_x=scale_x, scale_y=scale_y),
        option_panel_bbox_px=_scale_optional_bbox(rendered_scene.option_panel_bbox_px, scale_x=scale_x, scale_y=scale_y),
        option_choice_bboxes_px=bbox_dict_transform(rendered_scene.option_choice_bboxes_px, scale_x=scale_x, scale_y=scale_y),
        entities=entities_transform(rendered_scene.entities, scale_x=scale_x, scale_y=scale_y),
    )


def draw_panel_label(draw: ImageDraw.ImageDraw, *, text: str, x: float, y: float) -> None:
    font = load_font(22, bold=True)
    draw_text_traced(
        draw,
        (float(x), float(y)),
        str(text),
        font=font,
        fill=(28, 34, 43),
        stroke_width=2,
        stroke_fill=(255, 255, 255),
        role="readout",
        required=False,
    )


def render_multiview_view_dataset(dataset: Mapping[str, Any], *, view_key: str) -> Dict[str, Any]:
    view = dataset["views"][str(view_key)]
    return {
        _Q_FIELD: str(dataset[_Q_FIELD]),
        "scene_variant": str(dataset["scene_variant"]),
        "point_specs": [dict(spec) for spec in view["point_specs"]],
        "context_object_specs": [dict(spec) for spec in view["context_object_specs"]],
        "answer_label": str(dataset["answer_label"]),
        "answer_point_id": str(dataset["answer_point_id"]),
        "camera": dict(view["camera"]),
        "projection_frame": dict(view["projection_frame"]),
    }


def render_multiview_scene(
    *,
    dataset: Mapping[str, Any],
    render_params: _RenderParams,
    instance_seed: int,
    params: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
    """Render two full-proportion source views, then scale them into a side-by-side composite."""
    panel_params = multiview_source_render_params(render_params)
    layout = multiview_scaled_panel_layout(panel_params)
    composite_box = layout["_composite"]
    background, background_meta = make_background_canvas(
        canvas_width=int(composite_box["width"]),
        canvas_height=int(composite_box["height"]),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
    )
    composite = background.convert("RGB")
    draw = ImageDraw.Draw(composite)
    rendered_by_view = {}
    view_background_meta = {}
    for view_key, seed_offset in ((REFERENCE_VIEW_KEY, 17), (CANDIDATE_VIEW_KEY, 29)):
        panel = layout[str(view_key)]
        panel_background, panel_background_meta = make_background_canvas(
            canvas_width=int(panel_params.canvas_width),
            canvas_height=int(panel_params.canvas_height),
            instance_seed=int(instance_seed) + int(seed_offset),
            params=params,
            default_config=background_defaults,
        )
        view_dataset = render_multiview_view_dataset(dataset, view_key=str(view_key))
        rendered = render_object_scene_3d(
            panel_background,
            dataset=view_dataset,
            render_params=panel_params,
            draw_candidate_labels=str(view_key) == CANDIDATE_VIEW_KEY,
            highlight_object_ids=(
                [str(dataset["target_object_id"])] if str(view_key) == REFERENCE_VIEW_KEY else []
            ),
            annotation_label=str(dataset["answer_label"]),
        )
        rendered = _scale_rendered_scene(
            rendered,
            width=int(panel["width"]),
            height=int(panel["height"]),
        )
        composite.paste(rendered.image, (int(panel["x"]), int(panel["y"])))
        draw.rectangle(
            [
                int(panel["x"]),
                int(panel["y"]),
                int(panel["x"]) + int(panel["width"]),
                int(panel["y"]) + int(panel["height"]),
            ],
            outline=(57, 67, 80),
            width=2,
        )
        rendered_by_view[str(view_key)] = rendered
        view_background_meta[str(view_key)] = dict(panel_background_meta)

    draw_panel_label(
        draw,
        text="View 1",
        x=float(layout[REFERENCE_VIEW_KEY]["x"]),
        y=float(layout[REFERENCE_VIEW_KEY]["y"] - 31),
    )
    draw_panel_label(
        draw,
        text="View 2",
        x=float(layout[CANDIDATE_VIEW_KEY]["x"]),
        y=float(layout[CANDIDATE_VIEW_KEY]["y"] - 31),
    )
    background_meta = dict(background_meta)
    background_meta["view_panels"] = dict(view_background_meta)
    return composite, dict(rendered_by_view), dict(background_meta)


def render_two_view_object_scene(
    *,
    dataset: Mapping[str, Any],
    render_params: _RenderParams,
    instance_seed: int,
    params: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    panel_params: _RenderParams,
    view_dataset_builder: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    render_view_options: Callable[[str], Mapping[str, Any]],
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
    """Render generic paired object-scene views while preserving per-view projection records and shifted entities."""
    layout = panel_layout(render_params)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
    )
    composite = background.convert("RGB")
    draw = ImageDraw.Draw(composite)
    rendered_by_view = {}
    view_background_meta = {}

    for view_key, seed_offset in ((REFERENCE_VIEW_KEY, 17), (CANDIDATE_VIEW_KEY, 29)):
        panel = layout[str(view_key)]
        panel_background, panel_background_meta = make_background_canvas(
            canvas_width=int(panel["width"]),
            canvas_height=int(panel["height"]),
            instance_seed=int(instance_seed) + int(seed_offset),
            params=params,
            default_config=background_defaults,
        )
        rendered = render_object_scene_3d(
            panel_background,
            dataset=dict(view_dataset_builder(dataset, str(view_key))),
            render_params=panel_params,
            **dict(render_view_options(str(view_key))),
        )
        composite.paste(rendered.image, (int(panel["x"]), int(panel["y"])))
        draw.rectangle(
            [
                int(panel["x"]),
                int(panel["y"]),
                int(panel["x"]) + int(panel["width"]),
                int(panel["y"]) + int(panel["height"]),
            ],
            outline=(57, 67, 80),
            width=2,
        )
        rendered_by_view[str(view_key)] = rendered
        view_background_meta[str(view_key)] = dict(panel_background_meta)

    draw_panel_label(
        draw,
        text="View 1",
        x=float(layout[REFERENCE_VIEW_KEY]["x"]),
        y=float(layout[REFERENCE_VIEW_KEY]["y"] - 31),
    )
    draw_panel_label(
        draw,
        text="View 2",
        x=float(layout[CANDIDATE_VIEW_KEY]["x"]),
        y=float(layout[CANDIDATE_VIEW_KEY]["y"] - 31),
    )

    background_meta = dict(background_meta)
    background_meta["view_panels"] = dict(view_background_meta)
    return composite, dict(rendered_by_view), dict(background_meta)


def shift_render_maps(rendered_scene, *, panel: Mapping[str, int]) -> Dict[str, Any]:
    dx = float(panel["x"])
    dy = float(panel["y"])
    panel_bbox = [
        float(panel["x"]),
        float(panel["y"]),
        float(panel["x"] + panel["width"]),
        float(panel["y"] + panel["height"]),
    ]
    return {
        "panel_bbox_px": list(panel_bbox),
        "scene_bbox_px": offset_bbox(rendered_scene.scene_bbox_px, dx=dx, dy=dy),
        "room_bbox_px": offset_bbox(rendered_scene.room_bbox_px, dx=dx, dy=dy),
        "point_bboxes_px": _offset_map(rendered_scene.point_bboxes_px, dx=dx, dy=dy),
        "point_centers_px": _offset_centers(rendered_scene.point_centers_px, dx=dx, dy=dy),
        "object_bboxes_px": _offset_map(rendered_scene.object_bboxes_px, dx=dx, dy=dy),
        "object_centers_px": _offset_centers(rendered_scene.object_centers_px, dx=dx, dy=dy),
        "context_object_bboxes_px": _offset_map(rendered_scene.context_object_bboxes_px, dx=dx, dy=dy),
        "context_object_centers_px": _offset_centers(rendered_scene.context_object_centers_px, dx=dx, dy=dy),
    }


__all__ = [
    "CANDIDATE_VIEW_KEY",
    "REFERENCE_VIEW_KEY",
    "bbox_area",
    "bbox_is_readable",
    "camera_record",
    "camera_yaw_bands_for_instance",
    "draw_panel_label",
    "frame_record",
    "offset_bbox",
    "offset_entities",
    "offset_point",
    "offset_point_map",
    "panel_layout",
    "panel_render_params",
    "multiview_scaled_panel_layout",
    "multiview_source_render_params",
    "render_multiview_scene",
    "render_two_view_object_scene",
    "shift_render_maps",
    "yaw_separation_degrees",
]
