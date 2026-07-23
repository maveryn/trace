"""Mixed-object canvas sampling and rendering for illustration tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .object_library import (
    BBox,
    RGB,
    STYLE_IDS,
    aspect_ratio_for_object,
    choose_object_colors,
    family_for_object,
)
from .object_catalog import variant_ids_with_tag
from .object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
    serialize_rendered_illustration_object,
)
from .object_variants import RENDERER_STYLE_VECTOR
from .person_rendering import sample_person_gender


BACKGROUND_IDS: Tuple[str, ...] = variant_ids_with_tag("mixed_background")
SKY_OBJECT_TYPES: Tuple[str, ...] = variant_ids_with_tag("mixed_sky")
WATER_OBJECT_TYPES: Tuple[str, ...] = variant_ids_with_tag("mixed_water")
ROADLIKE_OBJECT_TYPES: Tuple[str, ...] = variant_ids_with_tag("mixed_roadlike")


@dataclass(frozen=True)
class ObjectPlacementSpec:
    """A selected object type and resolved canvas footprint."""

    object_id: str
    object_type: str
    bbox_xyxy: BBox
    primary_color_rgb: RGB
    accent_color_rgb: RGB
    style_id: str


@dataclass(frozen=True)
class RenderedMixedObjectScene:
    """Rendered mixed-object scene plus trace metadata."""

    image: Image.Image
    objects: Tuple[Any, ...]
    placements: Tuple[ObjectPlacementSpec, ...]
    background_id: str
    canvas_width: int
    canvas_height: int
    content_bbox: BBox
    render_scale: int
    background_layout: Mapping[str, Any]


def _rgb(value: Sequence[int]) -> RGB:
    return (int(value[0]), int(value[1]), int(value[2]))


def _bbox_overlap_area(a: BBox, b: BBox) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_area(box: BBox) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _inflate(box: BBox, margin: float) -> BBox:
    m = float(margin)
    return (float(box[0]) - m, float(box[1]) - m, float(box[2]) + m, float(box[3]) + m)


def _fits(existing: Sequence[BBox], candidate: BBox, *, min_gap_px: float, max_overlap_fraction: float) -> bool:
    expanded = _inflate(candidate, float(min_gap_px))
    for other in existing:
        overlap = _bbox_overlap_area(expanded, _inflate(other, float(min_gap_px)))
        if overlap <= 0.0:
            continue
        denom = max(1.0, min(_bbox_area(candidate), _bbox_area(other)))
        if float(overlap) / float(denom) > float(max_overlap_fraction):
            return False
    return True


def _jitter_rgb(rng, color: RGB, amount: int = 8) -> List[int]:
    return [max(0, min(255, int(value) + int(rng.randint(-amount, amount)))) for value in color]


def sample_background_layout(rng, *, background_id: str, canvas_width: int, canvas_height: int) -> Dict[str, Any]:
    """Sample explicit mixed-canvas background geometry used by rendering and placement."""

    sid = str(background_id)
    height = float(canvas_height)
    width = float(canvas_width)
    if sid == "meadow":
        horizon = float(rng.uniform(0.56, 0.68)) * height
        return {
            "layout_id": str(rng.choice(("low_meadow", "wide_meadow", "high_meadow"))),
            "ground_y": round(horizon, 3),
            "horizon_y": round(horizon, 3),
            "sky_rgb": _jitter_rgb(rng, (238, 247, 255), 7),
            "ground_rgb": _jitter_rgb(rng, (219, 235, 205), 9),
            "horizon_rgb": _jitter_rgb(rng, (188, 211, 180), 8),
        }
    if sid == "sky_ground":
        horizon = float(rng.uniform(0.63, 0.74)) * height
        return {
            "layout_id": str(rng.choice(("high_sky", "broad_ground", "low_sky"))),
            "ground_y": round(horizon, 3),
            "horizon_y": round(horizon, 3),
            "sky_rgb": _jitter_rgb(rng, (232, 244, 255), 8),
            "ground_rgb": _jitter_rgb(rng, (229, 223, 205), 10),
        }
    if sid == "tabletop":
        table_y = float(rng.uniform(0.62, 0.74)) * height
        return {
            "layout_id": str(rng.choice(("near_table", "mid_table", "deep_table"))),
            "ground_y": round(table_y, 3),
            "table_y": round(table_y, 3),
            "wall_rgb": _jitter_rgb(rng, (244, 246, 250), 7),
            "table_rgb": _jitter_rgb(rng, (225, 206, 176), 11),
            "edge_rgb": _jitter_rgb(rng, (192, 168, 137), 9),
        }
    if sid == "paper":
        gap = float(rng.uniform(36.0, 56.0))
        return {
            "layout_id": str(rng.choice(("wide_rule", "narrow_rule", "offset_rule"))),
            "ground_y": round(0.56 * height, 3),
            "line_gap": round(gap, 3),
            "line_offset": round(float(rng.uniform(0.2, 0.9)) * gap, 3),
            "paper_rgb": _jitter_rgb(rng, (251, 249, 240), 5),
            "line_rgb": _jitter_rgb(rng, (235, 231, 216), 7),
        }
    if sid == "shelf":
        upper = float(rng.uniform(0.30, 0.40)) * height
        lower = float(rng.uniform(0.58, 0.70)) * height
        return {
            "layout_id": str(rng.choice(("two_shelves", "high_shelves", "low_shelves"))),
            "ground_y": round(upper, 3),
            "shelf_levels": [round(upper, 3), round(lower, 3)],
            "wall_rgb": _jitter_rgb(rng, (246, 246, 241), 6),
            "shelf_rgb": _jitter_rgb(rng, (210, 188, 158), 10),
        }
    return {
        "layout_id": str(rng.choice(("warm_studio", "cool_studio", "plain_studio"))),
        "ground_y": round(0.54 * height, 3),
        "background_rgb": _jitter_rgb(rng, (247, 248, 251), 7),
        "width_px": round(width, 3),
    }


def _layout_float(layout: Mapping[str, Any] | None, key: str, fallback: float) -> float:
    if layout is None:
        return float(fallback)
    try:
        return float(layout.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _layout_rgb(layout: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    value = layout.get(key, fallback)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return _rgb(value)
    return _rgb(fallback)


def _draw_background(
    draw: ImageDraw.ImageDraw,
    *,
    background_id: str,
    width: int,
    height: int,
    scale: int,
    background_layout: Mapping[str, Any],
) -> None:
    sid = str(background_id)
    w = int(width) * int(scale)
    h = int(height) * int(scale)
    if sid == "meadow":
        horizon = int(round(_layout_float(background_layout, "horizon_y", 0.64 * float(height)) * int(scale)))
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "sky_rgb", (238, 247, 255)))
        draw.rectangle((0, horizon, w, h), fill=_layout_rgb(background_layout, "ground_rgb", (219, 235, 205)))
        draw.line((0, horizon, w, horizon), fill=_layout_rgb(background_layout, "horizon_rgb", (188, 211, 180)), width=max(1, 2 * int(scale)))
    elif sid == "sky_ground":
        horizon = int(round(_layout_float(background_layout, "horizon_y", 0.70 * float(height)) * int(scale)))
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "sky_rgb", (232, 244, 255)))
        draw.rectangle((0, horizon, w, h), fill=_layout_rgb(background_layout, "ground_rgb", (229, 223, 205)))
    elif sid == "tabletop":
        table_y = int(round(_layout_float(background_layout, "table_y", 0.70 * float(height)) * int(scale)))
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "wall_rgb", (244, 246, 250)))
        draw.rectangle((0, table_y, w, h), fill=_layout_rgb(background_layout, "table_rgb", (225, 206, 176)))
        draw.line((0, table_y, w, table_y), fill=_layout_rgb(background_layout, "edge_rgb", (192, 168, 137)), width=max(1, 2 * int(scale)))
    elif sid == "paper":
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "paper_rgb", (251, 249, 240)))
        gap = max(18, int(round(_layout_float(background_layout, "line_gap", 44.0) * int(scale))))
        offset = int(round(_layout_float(background_layout, "line_offset", 44.0) * int(scale)))
        for y in range(offset, h, gap):
            draw.line((0, y, w, y), fill=_layout_rgb(background_layout, "line_rgb", (235, 231, 216)), width=max(1, int(scale)))
    elif sid == "shelf":
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "wall_rgb", (246, 246, 241)))
        levels = background_layout.get("shelf_levels", (0.35 * float(height), 0.66 * float(height)))
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            levels = (0.35 * float(height), 0.66 * float(height))
        for level in levels:
            y = int(round(float(level) * int(scale)))
            draw.rectangle((0, y, w, y + 8 * int(scale)), fill=_layout_rgb(background_layout, "shelf_rgb", (210, 188, 158)))
    else:
        draw.rectangle((0, 0, w, h), fill=_layout_rgb(background_layout, "background_rgb", (247, 248, 251)))


def choose_background_id(rng, background_weights: Mapping[str, float], *, object_types: Sequence[str] = ()) -> str:
    """Choose a background, lightly favoring natural contexts for outdoor objects."""

    weights = {bg: max(0.0, float(background_weights.get(bg, 0.0))) for bg in BACKGROUND_IDS}
    if not any(value > 0.0 for value in weights.values()):
        weights = {bg: 1.0 for bg in BACKGROUND_IDS}
    object_type_set = {str(value) for value in object_types}
    if object_type_set.intersection(SKY_OBJECT_TYPES):
        for bg in ("meadow", "sky_ground"):
            weights[bg] = max(weights.get(bg, 0.0), 1.0) * 1.8
        for bg in ("paper", "tabletop", "shelf"):
            weights[bg] = weights.get(bg, 0.0) * 0.08
    if object_type_set.intersection(WATER_OBJECT_TYPES):
        for bg in ("meadow", "sky_ground"):
            weights[bg] = max(weights.get(bg, 0.0), 1.0) * 1.35
        for bg in ("paper", "tabletop", "shelf"):
            weights[bg] = weights.get(bg, 0.0) * 0.35
    if object_type_set.intersection(ROADLIKE_OBJECT_TYPES):
        for bg in ("meadow", "sky_ground", "studio"):
            weights[bg] = max(weights.get(bg, 0.0), 0.7) * 1.15
    total = sum(float(value) for value in weights.values())
    threshold = float(rng.random()) * float(total)
    running = 0.0
    for bg in BACKGROUND_IDS:
        running += float(weights.get(bg, 0.0))
        if running >= threshold:
            return str(bg)
    return str(BACKGROUND_IDS[-1])


def _background_ground_y(background_id: str, canvas_height: int, background_layout: Mapping[str, Any] | None = None) -> float:
    if background_layout is not None and "ground_y" in background_layout:
        return _layout_float(background_layout, "ground_y", 0.56 * float(canvas_height))
    sid = str(background_id)
    if sid == "meadow":
        return 0.64 * float(canvas_height)
    if sid == "sky_ground":
        return 0.70 * float(canvas_height)
    if sid == "tabletop":
        return 0.70 * float(canvas_height)
    if sid == "shelf":
        return 0.35 * float(canvas_height)
    return 0.56 * float(canvas_height)


def _placement_y_range(
    *,
    object_type: str,
    background_id: str | None,
    background_layout: Mapping[str, Any] | None = None,
    content_bbox: BBox,
    canvas_height: int,
    object_height: float,
) -> Tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in content_bbox]
    del x0, x1
    if not background_id:
        return y0, y1 - float(object_height)
    ground_y = _background_ground_y(str(background_id), int(canvas_height), background_layout)
    if str(object_type) in SKY_OBJECT_TYPES:
        high = min(y1 - float(object_height), max(y0 + 1.0, ground_y - float(object_height) - 26.0))
        return y0, high
    if str(background_id) in {"meadow", "sky_ground", "tabletop"}:
        low = max(y0, ground_y + 8.0)
        return low, y1 - float(object_height)
    if str(background_id) == "shelf":
        levels = tuple(float(value) for value in (background_layout or {}).get("shelf_levels", (0.35 * float(canvas_height), 0.66 * float(canvas_height))))
        shelf_y = levels[-1] if levels else 0.66 * float(canvas_height)
        low = max(y0, shelf_y + 8.0)
        return low, y1 - float(object_height)
    if str(object_type) in WATER_OBJECT_TYPES or str(object_type) in ROADLIKE_OBJECT_TYPES:
        low = max(y0, ground_y)
        return low, y1 - float(object_height)
    return y0, y1 - float(object_height)


def resolve_content_bbox(*, canvas_width: int, canvas_height: int, margin_px: int) -> BBox:
    """Return the drawable region for the mixed-object scene."""

    margin = int(margin_px)
    return (float(margin), float(margin), float(int(canvas_width) - margin), float(int(canvas_height) - margin))


def sample_placements(
    *,
    object_types: Sequence[str],
    rng,
    canvas_width: int,
    canvas_height: int,
    content_bbox: BBox,
    object_size_min_px: int,
    object_size_max_px: int,
    min_gap_px: int,
    max_overlap_fraction: float,
    placement_max_attempts: int,
    style_weights: Mapping[str, float],
    background_id: str | None = None,
    background_layout: Mapping[str, Any] | None = None,
) -> Tuple[ObjectPlacementSpec, ...]:
    """Resolve deterministic non-overlapping object placements."""

    if not object_types:
        raise ValueError("mixed object scene requires at least one object")
    x0, y0, x1, y1 = [float(v) for v in content_bbox]
    existing: List[BBox] = []
    placements: List[ObjectPlacementSpec] = []
    style_ids = tuple(style for style in STYLE_IDS if float(style_weights.get(str(style), 0.0)) > 0.0)
    style_pool = style_ids if style_ids else STYLE_IDS
    placement_layout_id = str(rng.choice(("free_scatter", "loose_grid", "two_clusters", "diagonal_band")))
    cluster_centers = tuple((float(rng.uniform(0.24, 0.76)), float(rng.uniform(0.24, 0.76))) for _ in range(2))
    if isinstance(background_layout, dict):
        background_layout["placement_layout_id"] = placement_layout_id
        if placement_layout_id == "two_clusters":
            background_layout["placement_cluster_centers_norm"] = [
                [round(float(x), 3), round(float(y), 3)] for x, y in cluster_centers
            ]
    grid_cols = max(1, int(round(len(object_types) ** 0.5)))
    grid_rows = max(1, int((len(object_types) + grid_cols - 1) // grid_cols))
    for index, object_type in enumerate(object_types):
        placed = False
        aspect = max(0.35, float(aspect_ratio_for_object(str(object_type))))
        for attempt in range(max(1, int(placement_max_attempts))):
            shrink = 0.94 ** max(0, attempt // 40)
            base_h = int(round(float(rng.randint(int(object_size_min_px), int(object_size_max_px))) * float(shrink)))
            h = max(36.0, float(base_h))
            w = max(36.0, h * aspect)
            if w > (x1 - x0) or h > (y1 - y0):
                continue
            px = float(rng.uniform(x0, x1 - w))
            py_min, py_max = _placement_y_range(
                object_type=str(object_type),
                background_id=background_id,
                background_layout=background_layout,
                content_bbox=content_bbox,
                canvas_height=int(canvas_height),
                object_height=float(h),
            )
            if py_max < py_min:
                continue
            if attempt >= 60 or placement_layout_id == "free_scatter":
                px = float(rng.uniform(x0, x1 - w))
                py = float(rng.uniform(float(py_min), float(py_max)))
            elif placement_layout_id == "loose_grid":
                col = index % grid_cols
                row = index // grid_cols
                cell_w = max(1.0, (x1 - x0 - w) / float(max(1, grid_cols)))
                cell_h = max(1.0, (float(py_max) - float(py_min)) / float(max(1, grid_rows)))
                px = x0 + col * cell_w + float(rng.uniform(0.12, 0.72)) * cell_w
                py = float(py_min) + row * cell_h + float(rng.uniform(0.12, 0.72)) * cell_h
                px = max(x0, min(x1 - w, px))
                py = max(float(py_min), min(float(py_max), py))
            elif placement_layout_id == "two_clusters":
                center_x, center_y = rng.choice(cluster_centers)
                px = x0 + center_x * max(1.0, x1 - x0 - w) + float(rng.uniform(-88.0, 88.0))
                py = float(py_min) + center_y * max(1.0, float(py_max) - float(py_min)) + float(rng.uniform(-62.0, 62.0))
                px = max(x0, min(x1 - w, px))
                py = max(float(py_min), min(float(py_max), py))
            else:
                px = float(rng.uniform(x0, x1 - w))
                t = (px - x0) / max(1.0, x1 - x0 - w)
                py = float(py_min) + (0.20 + 0.60 * t) * max(1.0, float(py_max) - float(py_min)) + float(rng.uniform(-48.0, 48.0))
                py = max(float(py_min), min(float(py_max), py))
            candidate = (px, py, px + w, py + h)
            if not _fits(existing, candidate, min_gap_px=float(min_gap_px), max_overlap_fraction=float(max_overlap_fraction)):
                continue
            primary, accent = choose_object_colors(rng, str(object_type))
            style_id = str(rng.choice(style_pool))
            placements.append(
                ObjectPlacementSpec(
                    object_id=f"obj_{index:02d}",
                    object_type=str(object_type),
                    bbox_xyxy=tuple(float(v) for v in candidate),
                    primary_color_rgb=primary,
                    accent_color_rgb=accent,
                    style_id=style_id,
                )
            )
            existing.append(candidate)
            placed = True
            break
        if not placed:
            raise ValueError("could not place all illustration objects without overlap")
    return tuple(placements)


def render_mixed_object_scene(
    *,
    placements: Sequence[ObjectPlacementSpec],
    rng,
    canvas_width: int,
    canvas_height: int,
    background_weights: Mapping[str, float],
    render_scale: int,
    content_bbox: BBox,
    background_id: str | None = None,
    background_layout: Mapping[str, Any] | None = None,
) -> RenderedMixedObjectScene:
    """Render one mixed object scene and return object/part metadata."""

    scale = max(1, int(render_scale))
    width = int(canvas_width)
    height = int(canvas_height)
    if background_id is None:
        background_pool = tuple(bg for bg in BACKGROUND_IDS if float(background_weights.get(str(bg), 0.0)) > 0.0)
        background_id = str(rng.choice(background_pool if background_pool else BACKGROUND_IDS))
    else:
        background_id = str(background_id)
        if background_id not in set(BACKGROUND_IDS):
            raise ValueError(f"unsupported illustration background_id: {background_id}")
    if background_layout is None:
        background_layout = sample_background_layout(rng, background_id=background_id, canvas_width=width, canvas_height=height)
    else:
        background_layout = dict(background_layout)
    image = Image.new("RGB", (width * scale, height * scale), (247, 248, 251))
    draw = ImageDraw.Draw(image)
    _draw_background(draw, background_id=background_id, width=width, height=height, scale=scale, background_layout=background_layout)
    rendered_objects = []
    for placement in sorted(placements, key=lambda item: (float(item.bbox_xyxy[1]), float(item.bbox_xyxy[0]), str(item.object_id))):
        visual_attributes: dict[str, Any] = {
            "primary_color_rgb": placement.primary_color_rgb,
            "accent_color_rgb": placement.accent_color_rgb,
            "style_id": str(placement.style_id),
        }
        gender_id = sample_person_gender(rng) if family_for_object(str(placement.object_type)) == "person" else None
        if gender_id is not None:
            visual_attributes["gender_id"] = gender_id
        rendered = render_illustration_object(
            IllustrationObjectSpec(
                object_id=str(placement.object_id),
                object_type=str(placement.object_type),
                bbox_xyxy=placement.bbox_xyxy,
                visual_attributes=visual_attributes,
                source_entity_type="illustration_object",
            ),
            RenderContext(
                renderer_style=RENDERER_STYLE_VECTOR,
                draw=draw,
                render_scale=scale,
            ),
        )
        rendered_objects.append(rendered)
    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return RenderedMixedObjectScene(
        image=image,
        objects=tuple(rendered_objects),
        placements=tuple(placements),
        background_id=background_id,
        canvas_width=width,
        canvas_height=height,
        content_bbox=tuple(float(v) for v in content_bbox),
        render_scale=scale,
        background_layout=dict(background_layout),
    )


def scene_entities(scene: RenderedMixedObjectScene) -> List[Dict[str, Any]]:
    """Return scene IR entities for rendered objects and parts."""

    entities: List[Dict[str, Any]] = []
    for rendered in scene.objects:
        serialized = serialize_rendered_illustration_object(rendered)
        entities.append(
            {
                "entity_id": serialized["object_id"],
                "entity_type": "illustration_object",
                "object_type": serialized["object_type"],
                "family": serialized["family"],
                "bbox": serialized["bbox"],
                "attributes": serialized["attributes"],
                "object_record": serialized["object_record"],
            }
        )
        for part in serialized["parts"]:
            entities.append(
                {
                    "entity_id": part["part_id"],
                    "entity_type": "illustration_part",
                    "part_kind": part["part_kind"],
                    "parent_object_id": serialized["object_id"],
                    "bbox": part["bbox"],
                    "attributes": part["attributes"],
                }
            )
    return entities


__all__ = [
    "BACKGROUND_IDS",
    "ObjectPlacementSpec",
    "ROADLIKE_OBJECT_TYPES",
    "RenderedMixedObjectScene",
    "SKY_OBJECT_TYPES",
    "WATER_OBJECT_TYPES",
    "choose_background_id",
    "render_mixed_object_scene",
    "resolve_content_bbox",
    "sample_background_layout",
    "sample_placements",
    "scene_entities",
]
