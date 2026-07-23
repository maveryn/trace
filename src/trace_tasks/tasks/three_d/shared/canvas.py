"""Canvas presets and coordinate transforms for three_d renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image

from trace_tasks.core.seed import spawn_rng


MAX_FINAL_PIXELS = 1_280_000


CANONICAL_CANVAS_PRESETS: Mapping[str, Tuple[int, int]] = {
    "landscape": (1200, 800),
    "portrait": (800, 1200),
    "square": (960, 960),
}


@dataclass(frozen=True)
class ThreeDCanvasSpec:
    """Resolved source canvas for one three_d scene render."""

    preset_id: str
    canvas_width: int
    canvas_height: int
    policy: str

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "scene_canvas_preset": str(self.preset_id),
            "scene_canvas_width": int(self.canvas_width),
            "scene_canvas_height": int(self.canvas_height),
            "scene_canvas_policy": str(self.policy),
        }


@dataclass(frozen=True)
class ImageScaleSpec:
    """Affine transform from source image coordinates into output coordinates."""

    source_width: int
    source_height: int
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    @property
    def changed(self) -> bool:
        return (
            int(self.source_width) != int(self.output_width)
            or int(self.source_height) != int(self.output_height)
            or abs(float(self.offset_x)) > 1e-9
            or abs(float(self.offset_y)) > 1e-9
        )

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "output_width": int(self.output_width),
            "output_height": int(self.output_height),
            "scale_x": round(float(self.scale_x), 8),
            "scale_y": round(float(self.scale_y), 8),
            "offset_x": round(float(self.offset_x), 3),
            "offset_y": round(float(self.offset_y), 3),
        }


def _as_weights(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {str(key): 1.0 for key in CANONICAL_CANVAS_PRESETS}
    weights: Dict[str, float] = {}
    for key, weight in value.items():
        preset_id = str(key)
        if preset_id not in CANONICAL_CANVAS_PRESETS:
            continue
        numeric = float(weight)
        if numeric > 0.0:
            weights[preset_id] = numeric
    return weights or {str(key): 1.0 for key in CANONICAL_CANVAS_PRESETS}


def _select_weighted_preset(*, weights: Mapping[str, float], instance_seed: int, namespace: str) -> str:
    ordered = [(str(key), float(weights[key])) for key in sorted(weights)]
    total = sum(weight for _key, weight in ordered)
    if total <= 0.0:
        raise ValueError("three_d canvas preset weights must contain at least one positive value")
    rng = spawn_rng(int(instance_seed), str(namespace))
    cursor = float(rng.random()) * float(total)
    running = 0.0
    for preset_id, weight in ordered:
        running += float(weight)
        if cursor <= running:
            return str(preset_id)
    return str(ordered[-1][0])


def resolve_three_d_canvas_spec(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int = 0,
    namespace: str = "three_d.canvas",
    fallback_width: int = 1200,
    fallback_height: int = 800,
) -> ThreeDCanvasSpec:
    """Resolve the source canvas for one canonical three_d scene/panel render."""

    explicit_width = params.get("canvas_width")
    explicit_height = params.get("canvas_height")
    if explicit_width is not None or explicit_height is not None:
        width = int(explicit_width if explicit_width is not None else fallback_width)
        height = int(explicit_height if explicit_height is not None else fallback_height)
        if width <= 0 or height <= 0:
            raise ValueError("explicit three_d canvas dimensions must be positive")
        return ThreeDCanvasSpec(
            preset_id="explicit",
            canvas_width=int(width),
            canvas_height=int(height),
            policy="explicit_dimensions",
        )

    merged = dict(render_defaults)
    preset_value = params.get("canvas_preset", params.get("scene_canvas_preset", params.get("canvas_size_preset")))
    if preset_value is not None:
        preset_id = str(preset_value)
        if preset_id not in CANONICAL_CANVAS_PRESETS:
            raise ValueError(f"unsupported three_d canvas preset: {preset_id}")
        width, height = CANONICAL_CANVAS_PRESETS[preset_id]
        return ThreeDCanvasSpec(
            preset_id=str(preset_id),
            canvas_width=int(width),
            canvas_height=int(height),
            policy="explicit_preset",
        )

    preset_weights = _as_weights(merged.get("canvas_preset_weights", merged.get("canvas_size_preset_weights")))
    preset_id = _select_weighted_preset(
        weights=preset_weights,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    width, height = CANONICAL_CANVAS_PRESETS[preset_id]
    return ThreeDCanvasSpec(
        preset_id=str(preset_id),
        canvas_width=int(width),
        canvas_height=int(height),
        policy="sampled_canonical_preset",
    )


def bbox_transform(
    bbox: Sequence[float],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[float]:
    return [
        round(float(bbox[0]) * float(scale_x) + float(offset_x), 3),
        round(float(bbox[1]) * float(scale_y) + float(offset_y), 3),
        round(float(bbox[2]) * float(scale_x) + float(offset_x), 3),
        round(float(bbox[3]) * float(scale_y) + float(offset_y), 3),
    ]


def point_transform(
    point: Sequence[float],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[float]:
    return [
        round(float(point[0]) * float(scale_x) + float(offset_x), 3),
        round(float(point[1]) * float(scale_y) + float(offset_y), 3),
    ]


def bbox_dict_transform(
    values: Mapping[str, Sequence[float]],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Dict[str, List[float]]:
    return {
        str(key): bbox_transform(value, scale_x=scale_x, scale_y=scale_y, offset_x=offset_x, offset_y=offset_y)
        for key, value in values.items()
    }


def point_dict_transform(
    values: Mapping[str, Sequence[float]],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Dict[str, List[float]]:
    return {
        str(key): point_transform(value, scale_x=scale_x, scale_y=scale_y, offset_x=offset_x, offset_y=offset_y)
        for key, value in values.items()
    }


def entities_transform(
    entities: Sequence[Mapping[str, Any]],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for entity in entities:
        updated: MutableMapping[str, Any] = dict(entity)
        bbox = updated.get("bbox_px")
        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) >= 4:
            updated["bbox_px"] = bbox_transform(
                bbox,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        transformed.append(dict(updated))
    return transformed


def resize_image_to_fit_pixel_cap(
    image: Image.Image,
    *,
    max_pixels: int = MAX_FINAL_PIXELS,
) -> Tuple[Image.Image, ImageScaleSpec]:
    """Downsample an image only when needed to satisfy the final pixel cap."""

    source_width, source_height = image.size
    source_pixels = int(source_width) * int(source_height)
    if source_pixels <= int(max_pixels):
        return image, ImageScaleSpec(
            source_width=int(source_width),
            source_height=int(source_height),
            output_width=int(source_width),
            output_height=int(source_height),
            scale_x=1.0,
            scale_y=1.0,
        )
    scale = math.sqrt(float(max_pixels) / float(source_pixels))
    output_width = max(1, int(math.floor(float(source_width) * float(scale))))
    output_height = max(1, int(math.floor(float(source_height) * float(scale))))
    while int(output_width) * int(output_height) > int(max_pixels):
        if output_width >= output_height:
            output_width -= 1
        else:
            output_height -= 1
    resized = image.resize((int(output_width), int(output_height)), Image.Resampling.LANCZOS)
    return resized, ImageScaleSpec(
        source_width=int(source_width),
        source_height=int(source_height),
        output_width=int(output_width),
        output_height=int(output_height),
        scale_x=float(output_width) / float(source_width),
        scale_y=float(output_height) / float(source_height),
    )


def expand_canvas_size_to_pixel_cap(
    width: int,
    height: int,
    *,
    max_pixels: int = MAX_FINAL_PIXELS,
) -> Tuple[int, int]:
    """Scale a composite canvas up without exceeding the final pixel cap."""

    source_width = int(width)
    source_height = int(height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("canvas dimensions must be positive")
    source_pixels = int(source_width) * int(source_height)
    if source_pixels >= int(max_pixels):
        image = Image.new("RGB", (int(source_width), int(source_height)))
        resized, _scale = resize_image_to_fit_pixel_cap(image, max_pixels=int(max_pixels))
        return int(resized.width), int(resized.height)

    scale = math.sqrt(float(max_pixels) / float(source_pixels))
    output_width = max(1, int(math.floor(float(source_width) * float(scale))))
    output_height = max(1, int(math.floor(float(source_height) * float(scale))))
    while int(output_width) * int(output_height) > int(max_pixels):
        if output_width >= output_height:
            output_width -= 1
        else:
            output_height -= 1
    return int(output_width), int(output_height)


def render_params_canvas_metadata(render_params: Any) -> Dict[str, Any]:
    metadata = {
        "scene_canvas_preset": str(getattr(render_params, "canvas_preset", "unknown")),
        "scene_canvas_width": int(getattr(render_params, "canvas_width")),
        "scene_canvas_height": int(getattr(render_params, "canvas_height")),
        "scene_canvas_policy": str(getattr(render_params, "canvas_policy", "unknown")),
    }
    tone_id = getattr(render_params, "background_tone_id", None)
    if tone_id is not None:
        metadata["background_tone_id"] = str(tone_id)
        metadata["background_tone_rgb"] = list(getattr(render_params, "background_tone_rgb", getattr(render_params, "floor_rgb", ())))
        metadata["background_floor_rgb"] = list(getattr(render_params, "floor_rgb", ()))
        metadata["background_grid_rgb"] = list(getattr(render_params, "grid_rgb", ()))
        metadata["background_edge_rgb"] = list(getattr(render_params, "edge_rgb", ()))
        metadata["background_surface_accent_rgb"] = list(getattr(render_params, "surface_accent_rgb", ()))
    belt_style_id = getattr(render_params, "conveyor_belt_style_id", None)
    if belt_style_id is not None:
        metadata["conveyor_belt_style_id"] = str(belt_style_id)
        metadata["conveyor_belt_fill_rgb"] = list(getattr(render_params, "conveyor_belt_fill_rgb", ()))
        metadata["conveyor_belt_outline_rgb"] = list(getattr(render_params, "conveyor_belt_outline_rgb", ()))
        metadata["conveyor_belt_arrow_rgb"] = list(getattr(render_params, "conveyor_belt_arrow_rgb", ()))
    return metadata


def final_canvas_metadata(image: Image.Image) -> Dict[str, Any]:
    return {
        "final_canvas_width": int(image.width),
        "final_canvas_height": int(image.height),
        "final_canvas_pixels": int(image.width) * int(image.height),
        "max_final_pixels": int(MAX_FINAL_PIXELS),
    }


__all__ = [
    "CANONICAL_CANVAS_PRESETS",
    "MAX_FINAL_PIXELS",
    "ImageScaleSpec",
    "ThreeDCanvasSpec",
    "bbox_dict_transform",
    "bbox_transform",
    "entities_transform",
    "expand_canvas_size_to_pixel_cap",
    "final_canvas_metadata",
    "point_dict_transform",
    "point_transform",
    "render_params_canvas_metadata",
    "resize_image_to_fit_pixel_cap",
    "resolve_three_d_canvas_spec",
]
