"""Shared illustration object render specs and renderer dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .object_library import choose_object_colors, draw_illustration_object, serialize_object
from .object_registry import family_for_object_type, make_object_record, public_name_for_object_type
from .object_schema import BBox, json_safe
from .style_registry import style_outline_params
from .object_variants import (
    RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
    RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
    RENDERER_STYLE_VECTOR,
    object_variant_ids,
    variant_visual_metadata,
)
from .pixel_world_objects import (
    draw_pixel_animal,
    draw_pixel_autumn_overlay,
    draw_pixel_barrel,
    draw_pixel_basket,
    draw_pixel_barn,
    draw_pixel_bed,
    draw_pixel_bench,
    draw_pixel_boulder,
    draw_pixel_bottle,
    draw_pixel_bowl,
    draw_pixel_archway,
    draw_pixel_brazier,
    draw_pixel_broken_wall,
    draw_pixel_bridge,
    draw_pixel_candle,
    draw_pixel_castle,
    draw_pixel_cave_entrance,
    draw_pixel_cart,
    draw_pixel_cemetery_gate,
    draw_pixel_chair,
    draw_pixel_chest,
    draw_pixel_chicken_coop,
    draw_pixel_church,
    draw_pixel_counter,
    draw_pixel_crop_row,
    draw_pixel_crate,
    draw_pixel_crystal_cluster,
    draw_pixel_dead_tree,
    draw_pixel_farm_gate,
    draw_pixel_fence,
    draw_pixel_fireplace,
    draw_pixel_floor_switch,
    draw_pixel_flower_patch,
    draw_pixel_fountain,
    draw_pixel_gazebo,
    draw_pixel_grave_marker,
    draw_pixel_hay_bale,
    draw_pixel_house,
    draw_pixel_iron_fence,
    draw_pixel_jar,
    draw_pixel_lamp_post,
    draw_pixel_ladder,
    draw_pixel_magic_circle,
    draw_pixel_market_stall,
    draw_pixel_mine_cart,
    draw_pixel_mug,
    draw_pixel_notice_board,
    draw_pixel_ore_vein,
    draw_pixel_pond,
    draw_pixel_plate,
    draw_pixel_person,
    draw_pixel_pot,
    draw_pixel_produce_bin,
    draw_pixel_rail_track,
    draw_pixel_rock,
    draw_pixel_room_divider,
    draw_pixel_rubble,
    draw_pixel_rug,
    draw_pixel_sack,
    draw_pixel_scarecrow,
    draw_pixel_sealed_door,
    draw_pixel_shelf,
    draw_pixel_sign,
    draw_pixel_stairs,
    draw_pixel_stalagmite,
    draw_pixel_statue,
    draw_pixel_stone_column,
    draw_pixel_stool,
    draw_pixel_table,
    draw_pixel_torch,
    draw_pixel_tree,
    draw_pixel_trough,
    draw_pixel_vegetable_patch,
    draw_pixel_wagon,
    draw_pixel_well,
    draw_pixel_windmill,
    draw_pixel_woodpile,
    draw_pixel_wood_support,
    draw_pixel_winter_overlay,
)


RGB = Tuple[int, int, int]
TileBox = Tuple[int, int, int, int]
TileProjector = Callable[[TileBox, int], tuple[float, float]]

PIXEL_RPG_SHARED_OBJECT_TYPES: tuple[str, ...] = (
    "barrel",
    "barn",
    "basket",
    "bed",
    "bench",
    "boulder",
    "archway",
    "brazier",
    "broken_wall",
    "bridge",
    "candle",
    "cart",
    "castle",
    "cave_entrance",
    "cemetery_gate",
    "chair",
    "chicken_coop",
    "chest",
    "church",
    "coop",
    "counter",
    "crate",
    "crystal_cluster",
    "crop_row",
    "dead_tree",
    "domestic_animal",
    "farm_gate",
    "fence",
    "fireplace",
    "floor_switch",
    "flower",
    "fountain",
    "gazebo",
    "grave_marker",
    "hay_bale",
    "house",
    "inn",
    "iron_fence",
    "jar",
    "lamp_post",
    "ladder",
    "magic_circle",
    "market_stall",
    "mine_cart",
    "notice_board",
    "ore_vein",
    "plate",
    "person",
    "pond",
    "pot",
    "produce_bin",
    "rail_track",
    "rock",
    "rubble",
    "room_divider",
    "rug",
    "sack",
    "scarecrow",
    "sealed_door",
    "shelf",
    "shop",
    "sign",
    "stairs",
    "stalagmite",
    "statue",
    "stone_column",
    "stool",
    "table",
    "tower",
    "torch",
    "tree",
    "trough",
    "vegetable_patch",
    "wagon",
    "well",
    "windmill",
    "woodpile",
    "wood_support",
)

_ISO_NATIVE_PIXEL_OBJECT_TYPES = {
    "archway",
    "barrel",
    "bed",
    "bench",
    "basket",
    "brazier",
    "broken_wall",
    "chair",
    "chest",
    "counter",
    "crate",
    "crop_row",
    "fireplace",
    "floor_switch",
    "hay_bale",
    "magic_circle",
    "produce_bin",
    "rubble",
    "room_divider",
    "rug",
    "sealed_door",
    "shelf",
    "stone_column",
    "stool",
    "table",
    "trough",
}
_ISO_FLAT_PIXEL_OBJECT_TYPES = {
    "bridge",
    "cemetery_gate",
    "farm_gate",
    "fence",
    "iron_fence",
    "ladder",
    "ore_vein",
    "pond",
    "rail_track",
    "stairs",
    "vegetable_patch",
}


@dataclass(frozen=True)
class IllustrationObjectSpec:
    """Renderer-neutral request to draw one reusable illustration object."""

    object_id: str
    object_type: str
    public_name: str = ""
    bbox_xyxy: BBox | None = None
    tile_xywh: TileBox | None = None
    level: int = 0
    variant_id: str = ""
    renderer_id: str = ""
    renderer_variant_id: str = ""
    semantic_attributes: Mapping[str, Any] = field(default_factory=dict)
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    parts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    role: str = "distractor"
    source_entity_type: str = "illustration_object"


@dataclass(frozen=True)
class RenderContext:
    """Renderer-specific drawing state for one object dispatch call."""

    renderer_style: str
    draw: ImageDraw.ImageDraw | None = None
    image: Image.Image | None = None
    render_scale: int = 1
    style_id: str = "flat_vector"
    primary_color_rgb: RGB = (72, 115, 166)
    accent_color_rgb: RGB = (245, 190, 88)
    project_tile_center: TileProjector | None = None


@dataclass(frozen=True)
class RenderedIllustrationObject:
    """Trace-ready result returned by shared object rendering."""

    object_id: str
    object_type: str
    public_name: str
    bbox_xyxy: BBox | None
    semantic_attributes: Mapping[str, Any]
    visual_attributes: Mapping[str, Any]
    parts: Tuple[Mapping[str, Any], ...]
    object_record: Mapping[str, Any]
    support_items: Tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": str(self.object_id),
            "object_type": str(self.object_type),
            "public_name": str(self.public_name),
            "bbox": [round(float(v), 3) for v in self.bbox_xyxy] if self.bbox_xyxy is not None else None,
            "semantic_attributes": json_safe(self.semantic_attributes),
            "visual_attributes": json_safe(self.visual_attributes),
            "parts": [json_safe(part) for part in self.parts],
            "object_record": json_safe(self.object_record),
            "support_items": [json_safe(item) for item in self.support_items],
        }


def render_illustration_object(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> RenderedIllustrationObject:
    """Draw one object through the renderer selected by ``context``."""

    renderer_style = str(context.renderer_style)
    if renderer_style == RENDERER_STYLE_VECTOR:
        return _render_vector_object(spec, context)
    if renderer_style == RENDERER_STYLE_TOP_DOWN_PIXEL_RPG:
        return _render_top_down_pixel_object(spec, context)
    if renderer_style == RENDERER_STYLE_ISOMETRIC_PIXEL_RPG:
        return _render_isometric_pixel_object(spec, context)
    raise ValueError(f"unsupported illustration renderer style: {renderer_style}")


def object_record_for_spec(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> dict[str, Any]:
    """Build a normalized object record for one reusable object spec."""

    object_type = _record_object_type(spec)
    public_name = _public_name(spec)
    visual_attributes = _resolved_visual_attributes(spec, context)
    semantic_attributes = _resolved_semantic_attributes(spec)
    record = make_object_record(
        object_id=str(spec.object_id),
        object_type=object_type,
        public_name=public_name,
        family=_record_family(object_type, spec),
        bbox_xyxy=spec.bbox_xyxy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        role=str(spec.role),
        source_entity_type=str(spec.source_entity_type),
        parts=tuple(dict(part) for part in spec.parts),
    )
    return record.as_dict()


def vector_object_record_for_spec(
    spec: IllustrationObjectSpec,
    *,
    render_scale: int = 1,
    style_id: str = "flat_vector",
) -> dict[str, Any]:
    """Build a vector-style object record without drawing the object."""

    return object_record_for_spec(
        spec,
        RenderContext(
            renderer_style=RENDERER_STYLE_VECTOR,
            render_scale=int(render_scale),
            style_id=str(style_id),
        ),
    )


def make_vector_scene_object_record(
    *,
    object_id: str,
    object_type: str,
    bbox_xyxy: BBox,
    semantic_attributes: Mapping[str, Any],
    visual_attributes: Mapping[str, Any] | None = None,
    role: str = "distractor",
    source_entity_type: str,
    render_scale: int = 1,
    style_id: str = "flat_vector",
) -> dict[str, Any]:
    """Build a vector object record from scene-local entity fields."""

    return vector_object_record_for_spec(
        IllustrationObjectSpec(
            object_id=str(object_id),
            object_type=str(object_type),
            bbox_xyxy=bbox_xyxy,
            semantic_attributes=semantic_attributes,
            visual_attributes=visual_attributes or {},
            role=str(role),
            source_entity_type=str(source_entity_type),
        ),
        render_scale=int(render_scale),
        style_id=str(style_id),
    )


def render_vector_scene_object(
    draw: ImageDraw.ImageDraw,
    *,
    object_id: str,
    object_type: str,
    bbox_xyxy: BBox,
    renderer_id: str,
    renderer_variant_id: str,
    semantic_attributes: Mapping[str, Any],
    visual_attributes: Mapping[str, Any] | None = None,
    role: str = "distractor",
    source_entity_type: str,
    render_scale: int = 1,
    style_id: str = "flat_vector",
) -> RenderedIllustrationObject:
    """Render a vector object from scene-local entity fields."""

    return render_illustration_object(
        IllustrationObjectSpec(
            object_id=str(object_id),
            object_type=str(object_type),
            bbox_xyxy=bbox_xyxy,
            renderer_id=str(renderer_id),
            renderer_variant_id=str(renderer_variant_id),
            semantic_attributes=semantic_attributes,
            visual_attributes=visual_attributes or {},
            role=str(role),
            source_entity_type=str(source_entity_type),
        ),
        RenderContext(
            renderer_style=RENDERER_STYLE_VECTOR,
            draw=draw,
            render_scale=int(render_scale),
            style_id=str(style_id),
        ),
    )


def serialize_rendered_illustration_object(rendered: RenderedIllustrationObject) -> dict[str, Any]:
    """Return the legacy serialized object payload for a shared rendered object."""

    record = json_safe(rendered.object_record)
    visual = dict(rendered.visual_attributes)
    semantic = dict(rendered.semantic_attributes)
    attributes: dict[str, Any] = {
        "object_type": str(rendered.object_type),
        "family": str(record.get("family", "")),
    }
    attributes.update(semantic)
    for key in (
        "gender_id",
        "object_variant_id",
        "object_variant_public_name",
        "object_variant_queryable",
        "person_variant_id",
        "renderer_id",
        "renderer_style",
        "renderer_variant_id",
        "shadow_enabled",
        "shadow_kind",
        "shadow_policy",
        "tree_style",
    ):
        if key in visual:
            attributes[key] = visual[key]
    return {
        "object_id": str(rendered.object_id),
        "object_type": str(rendered.object_type),
        "family": str(record.get("family", "")),
        "bbox": [round(float(v), 3) for v in rendered.bbox_xyxy] if rendered.bbox_xyxy is not None else None,
        "primary_color_rgb": json_safe(visual.get("primary_color_rgb")),
        "accent_color_rgb": json_safe(visual.get("accent_color_rgb")),
        "style_id": str(visual.get("style_id", "")),
        "parts": [json_safe(part) for part in rendered.parts],
        "attributes": json_safe(attributes),
        "object_record": record,
    }


def _render_vector_object(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> RenderedIllustrationObject:
    if context.draw is None:
        raise ValueError("vector object rendering requires RenderContext.draw")
    if spec.bbox_xyxy is None:
        raise ValueError("vector object rendering requires IllustrationObjectSpec.bbox_xyxy")
    from .vector_object_renderers import render_registered_vector_object

    registered = render_registered_vector_object(spec, context)
    if registered is not None:
        return _rendered_from_vector_result(spec, context, registered)
    object_type = str(spec.object_type)
    primary = _rgb(spec.visual_attributes.get("primary_color_rgb", context.primary_color_rgb))
    accent = _rgb(spec.visual_attributes.get("accent_color_rgb", context.accent_color_rgb))
    rendered = draw_illustration_object(
        context.draw,
        object_id=str(spec.object_id),
        object_type=object_type,
        bbox_xyxy=spec.bbox_xyxy,
        primary_color_rgb=primary,
        accent_color_rgb=accent,
        style_id=str(spec.visual_attributes.get("style_id", context.style_id)),
        render_scale=int(context.render_scale),
        gender_id=spec.visual_attributes.get("gender_id"),
        object_variant_id=str(spec.variant_id) if spec.variant_id else None,
    )
    payload = serialize_object(rendered)
    record = dict(payload["object_record"])
    semantic_attributes = dict(record["semantic_attributes"])
    semantic_attributes.update(_resolved_semantic_attributes(spec))
    visual_attributes = dict(record["visual_attributes"])
    visual_attributes.update(_resolved_visual_attributes(spec, context))
    record["semantic_attributes"] = json_safe(semantic_attributes)
    record["visual_attributes"] = json_safe(visual_attributes)
    record["role"] = str(spec.role)
    record["source_entity_type"] = str(spec.source_entity_type)
    if spec.public_name:
        record["public_name"] = str(spec.public_name)
    return RenderedIllustrationObject(
        object_id=str(spec.object_id),
        object_type=object_type,
        public_name=str(record["public_name"]),
        bbox_xyxy=tuple(float(v) for v in payload["bbox"]),  # type: ignore[assignment]
        semantic_attributes=dict(record["semantic_attributes"]),
        visual_attributes=dict(record["visual_attributes"]),
        parts=tuple(payload["parts"]),
        object_record=record,
    )


def _rendered_from_vector_result(
    spec: IllustrationObjectSpec,
    context: RenderContext,
    result: Any,
) -> RenderedIllustrationObject:
    object_type = _record_object_type(spec)
    public_name = _public_name(spec)
    bbox_xyxy = tuple(float(value) for value in result.bbox_xyxy)
    semantic_attributes = _resolved_semantic_attributes(spec)
    visual_attributes = _resolved_visual_attributes(spec, context)
    visual_attributes.update({str(key): value for key, value in dict(result.visual_attributes).items()})
    _apply_shadow_policy_visual_attributes(visual_attributes, context)
    parts = tuple(dict(part) for part in result.parts)
    record = make_object_record(
        object_id=str(spec.object_id),
        object_type=object_type,
        public_name=public_name,
        family=_record_family(object_type, spec),
        bbox_xyxy=bbox_xyxy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        role=str(spec.role),
        source_entity_type=str(spec.source_entity_type),
        parts=parts,
    ).as_dict()
    return RenderedIllustrationObject(
        object_id=str(spec.object_id),
        object_type=object_type,
        public_name=public_name,
        bbox_xyxy=bbox_xyxy,  # type: ignore[arg-type]
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        parts=parts,
        object_record=record,
        support_items=tuple(dict(item) for item in result.support_items),
    )


def _render_top_down_pixel_object(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> RenderedIllustrationObject:
    if context.draw is None:
        raise ValueError("top-down pixel object rendering requires RenderContext.draw")
    tile_xywh = _required_tile(spec)
    _draw_pixel_object_on_tile(context.draw, tile_xywh, spec)
    return _rendered_from_record(spec, context)


def _render_isometric_pixel_object(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> RenderedIllustrationObject:
    if context.image is None:
        raise ValueError("isometric pixel object rendering requires RenderContext.image")
    if context.project_tile_center is None:
        raise ValueError("isometric pixel object rendering requires RenderContext.project_tile_center")
    tile_xywh = _required_tile(spec)
    center = context.project_tile_center(tile_xywh, int(spec.level))
    image = context.image
    draw = ImageDraw.Draw(image, "RGBA")
    kind = _pixel_kind(spec)
    if _draw_native_isometric_pixel_object(draw, center=center, tile_xywh=tile_xywh, spec=spec, kind=kind):
        return _rendered_from_record(spec, context)
    sprite_tile_w, sprite_tile_h = _pixel_sprite_tile_size(kind, tile_xywh, spec)
    sprite_w = sprite_tile_w * 16
    sprite_h = sprite_tile_h * 16
    sprite = Image.new("RGBA", (sprite_w, sprite_h), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    if kind == "domestic_animal":
        animal_type = _animal_type(spec)
        draw_pixel_animal(
            sprite_draw,
            (0, 0, sprite_tile_w, sprite_tile_h),
            animal_type=animal_type,
            facing=str(spec.visual_attributes.get("facing", "right")),
            body_rgb=_optional_rgb(spec.visual_attributes.get("body_rgb")),
            accent_rgb=_optional_rgb(spec.visual_attributes.get("accent_rgb")),
            spot_rgb=_optional_rgb(spec.visual_attributes.get("spot_rgb")),
        )
    else:
        _draw_pixel_object_on_tile(sprite_draw, (0, 0, sprite_tile_w, sprite_tile_h), spec, isometric_view=True)
    x0 = int(round(float(center[0]) - sprite_w * 0.5))
    if kind in _ISO_FLAT_PIXEL_OBJECT_TYPES:
        y0 = int(round(float(center[1]) - sprite_h * 0.5))
    else:
        y0 = int(round(float(center[1]) + 6 - sprite_h))
    image.alpha_composite(sprite, dest=(x0, y0))
    return _rendered_from_record(spec, context)


def _draw_native_isometric_pixel_object(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
    kind: str,
) -> bool:
    if kind not in _ISO_NATIVE_PIXEL_OBJECT_TYPES:
        return False
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    if kind == "hay_bale":
        top = [(cx - 9, cy - 5), (cx, cy - 10), (cx + 9, cy - 5), (cx, cy)]
        left = [(cx - 9, cy - 5), (cx, cy), (cx, cy + 7), (cx - 9, cy + 2)]
        right = [(cx + 9, cy - 5), (cx, cy), (cx, cy + 7), (cx + 9, cy + 2)]
        draw.polygon(left, fill=(191, 143, 57), outline=(120, 87, 41))
        draw.polygon(right, fill=(218, 173, 70), outline=(120, 87, 41))
        draw.polygon(top, fill=(238, 198, 88), outline=(120, 87, 41))
        draw.line((cx - 5, cy - 6, cx + 5, cy - 1), fill=(245, 218, 112))
        draw.line((cx - 6, cy + 1, cx + 5, cy + 6), fill=(154, 110, 46))
        draw.line((cx, cy - 9, cx, cy + 7), fill=(117, 85, 39))
        return True
    if kind == "crate":
        top = [(cx - 9, cy - 6), (cx, cy - 11), (cx + 9, cy - 6), (cx, cy - 1)]
        left = [(cx - 9, cy - 6), (cx, cy - 1), (cx, cy + 8), (cx - 9, cy + 3)]
        right = [(cx + 9, cy - 6), (cx, cy - 1), (cx, cy + 8), (cx + 9, cy + 3)]
        draw.polygon(left, fill=(139, 82, 43), outline=(82, 52, 33))
        draw.polygon(right, fill=(174, 104, 52), outline=(82, 52, 33))
        draw.polygon(top, fill=(195, 126, 65), outline=(82, 52, 33))
        draw.line((cx - 7, cy - 5, cx, cy + 7), fill=(96, 60, 36))
        draw.line((cx + 7, cy - 5, cx, cy + 7), fill=(96, 60, 36))
        draw.line((cx, cy - 10, cx, cy + 8), fill=(110, 67, 38))
        return True
    if kind == "rug":
        _, _, w, h = tile_xywh
        cloth_rgb = _rgb(spec.visual_attributes.get("cloth_rgb", (168, 74, 82)))
        trim_rgb = _rgb(spec.visual_attributes.get("trim_rgb", (230, 190, 96)))
        half_x = max(20, int(round((max(1, int(w)) + max(1, int(h))) * 8.0)))
        half_y = max(8, int(round((max(1, int(w)) + max(1, int(h))) * 4.0)))
        diamond = [(cx, cy - half_y), (cx + half_x, cy), (cx, cy + half_y), (cx - half_x, cy)]
        inner = [(cx, cy - half_y + 3), (cx + half_x - 7, cy), (cx, cy + half_y - 3), (cx - half_x + 7, cy)]
        draw.polygon(diamond, fill=cloth_rgb, outline=_shade_rgb(cloth_rgb, -48))
        draw.line((diamond[0][0], diamond[0][1], diamond[1][0], diamond[1][1]), fill=trim_rgb)
        draw.line((diamond[2][0], diamond[2][1], diamond[3][0], diamond[3][1]), fill=trim_rgb)
        draw.polygon(inner, outline=_shade_rgb(trim_rgb, -44))
        return True
    if kind == "basket":
        _draw_iso_basket(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "barrel":
        _draw_iso_barrel(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "bench":
        _draw_iso_bench(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "table":
        _draw_iso_table(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "chair":
        _draw_iso_chair(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "stool":
        _draw_iso_stool(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "bed":
        _draw_iso_bed(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "chest":
        _draw_iso_chest(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "counter":
        _draw_iso_counter(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "fireplace":
        _draw_iso_fireplace(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "stone_column":
        _draw_iso_stone_column(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "archway":
        _draw_iso_archway(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "sealed_door":
        _draw_iso_sealed_door(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "brazier":
        _draw_iso_brazier(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "floor_switch":
        _draw_iso_floor_switch(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "broken_wall":
        _draw_iso_broken_wall(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "rubble":
        _draw_iso_rubble(draw, cx=cx, cy=cy, spec=spec)
        return True
    if kind == "magic_circle":
        _draw_iso_magic_circle(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind == "room_divider":
        _draw_iso_room_divider(draw, cx=cx, cy=cy, tile_xywh=tile_xywh, spec=spec)
        return True
    if kind in {"shelf", "produce_bin"}:
        _, _, w, h = tile_xywh
        width_units = max(1, int(w)) + max(0, int(h) - 1) * 0.55
        half_x = max(12, int(round(width_units * 10.0)))
        half_y = max(5, int(round((max(1, int(w)) + max(1, int(h))) * 2.6)))
        body_h = 12 if kind in {"counter", "shelf"} else 9
        wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (139, 82, 43)))
        wood_dark = _shade_rgb(wood_rgb, -48)
        wood_light = _shade_rgb(wood_rgb, 32)
        top_rgb = _rgb(spec.visual_attributes.get("top_rgb", _shade_rgb(wood_rgb, 28)))
        top = [(cx - half_x, cy - half_y), (cx, cy - half_y - 7), (cx + half_x, cy - half_y), (cx, cy + half_y)]
        left = [(cx - half_x, cy - half_y), (cx, cy + half_y), (cx, cy + half_y + body_h), (cx - half_x, cy - half_y + body_h)]
        right = [(cx + half_x, cy - half_y), (cx, cy + half_y), (cx, cy + half_y + body_h), (cx + half_x, cy - half_y + body_h)]
        if kind == "produce_bin":
            goods_type = str(spec.visual_attributes.get("goods_type", "fruit"))
            draw.polygon(left, fill=wood_rgb, outline=wood_dark)
            draw.polygon(right, fill=_shade_rgb(wood_rgb, 12), outline=wood_dark)
            draw.polygon(top, fill=(92, 63, 37), outline=wood_dark)
            palette = {
                "vegetable": ((64, 150, 72), (98, 183, 82), (46, 116, 62)),
                "grain": ((220, 178, 70), (238, 204, 96), (170, 121, 50)),
                "fruit": ((213, 67, 55), (225, 140, 43), (235, 190, 70)),
            }.get(goods_type, ((213, 67, 55), (225, 140, 43), (235, 190, 70)))
            for index, px in enumerate(range(cx - half_x + 8, cx + half_x - 5, 6)):
                color = palette[index % len(palette)]
                py = cy - half_y + 1 + (index % 2)
                if goods_type == "grain":
                    draw.line((px, py - 4, px, py + 3), fill=_shade_rgb(color, -42))
                    draw.point((px - 1, py - 2), fill=color)
                    draw.point((px + 1, py - 1), fill=_shade_rgb(color, 24))
                else:
                    draw.ellipse((px - 2, py - 4, px + 3, py + 1), fill=color, outline=_shade_rgb(color, -44))
            return True
        draw.polygon(left, fill=wood_rgb, outline=wood_dark)
        draw.polygon(right, fill=_shade_rgb(wood_rgb, 16), outline=wood_dark)
        draw.polygon(top, fill=top_rgb, outline=_shade_rgb(top_rgb, -42))
        draw.line((cx - half_x + 4, cy - half_y + 2, cx + half_x - 4, cy - half_y + 2), fill=wood_light)
        goods_type = str(spec.visual_attributes.get("goods_type", "mixed"))
        back_top = cy - half_y - 13
        draw.polygon(
            [(cx - half_x + 2, cy - half_y - 1), (cx - 2, back_top), (cx + half_x - 2, cy - half_y - 1), (cx + half_x - 2, cy - half_y + 5), (cx, cy + 1), (cx - half_x + 2, cy - half_y + 5)],
            fill=_shade_rgb(wood_rgb, -5),
            outline=wood_dark,
        )
        for index, px in enumerate(range(cx - half_x + 8, cx + half_x - 5, 9)):
            if goods_type == "jars" or (goods_type == "mixed" and index % 3 == 0):
                color = ((86, 153, 184), (200, 93, 78), (235, 190, 75))[index % 3]
                draw.rectangle((px, cy - half_y - 8, px + 3, cy - half_y - 3), fill=color, outline=_shade_rgb(color, -45))
            elif goods_type == "produce" or (goods_type == "mixed" and index % 3 == 1):
                color = ((215, 67, 55), (95, 169, 77), (224, 139, 45))[index % 3]
                draw.ellipse((px - 1, cy - half_y - 7, px + 4, cy - half_y - 2), fill=color, outline=_shade_rgb(color, -42))
            else:
                color = ((76, 117, 176), (183, 75, 116), (95, 145, 94))[index % 3]
                draw.rectangle((px - 1, cy - half_y - 9, px + 3, cy - half_y - 3), fill=color, outline=_shade_rgb(color, -42))
        return True
    if kind == "crop_row":
        x, y, w, h = tile_xywh
        crop_rgb = _rgb(spec.visual_attributes.get("crop_rgb", (210, 171, 58)))
        soil_rgb = _rgb(spec.visual_attributes.get("soil_rgb", (130, 87, 48)))
        crop_style = str(spec.visual_attributes.get("crop_style", "wheat"))
        soil_dark = _shade_rgb(soil_rgb, -34)
        soil_light = _shade_rgb(soil_rgb, 20)
        crop_dark = _shade_rgb(crop_rgb, -38)
        crop_light = _shade_rgb(crop_rgb, 34)
        center_x = float(x) + (float(w) - 1.0) * 0.5
        center_y = float(y) + (float(h) - 1.0) * 0.5
        for row in range(max(1, int(h))):
            for col in range(max(1, int(w))):
                local_x = float(x + col)
                local_y = float(y + row)
                tile_cx = int(round(float(center[0]) + ((local_x - center_x) - (local_y - center_y)) * 16.0))
                tile_cy = int(round(float(center[1]) + ((local_x - center_x) + (local_y - center_y)) * 8.0))
                draw.line((tile_cx - 9, tile_cy + 2, tile_cx, tile_cy + 6, tile_cx + 9, tile_cy + 2), fill=soil_dark)
                draw.line((tile_cx - 6, tile_cy + 1, tile_cx, tile_cy + 4, tile_cx + 6, tile_cy + 1), fill=soil_light)
                for offset_x in (-4, 2):
                    px = tile_cx + offset_x
                    py = tile_cy + 2
                    if crop_style == "leafy":
                        draw.line((px, py - 1, px, py + 5), fill=crop_dark)
                        draw.point((px - 2, py + 1), fill=crop_rgb)
                        draw.point((px - 1, py), fill=crop_light)
                        draw.point((px + 1, py), fill=crop_rgb)
                    elif crop_style == "flowering":
                        leaf = (45, 128, 62)
                        draw.line((px, py, px, py + 5), fill=_shade_rgb(leaf, -28))
                        draw.point((px - 1, py + 2), fill=leaf)
                        draw.point((px + 1, py + 3), fill=leaf)
                        draw.point((px, py - 1), fill=crop_light)
                        draw.point((px - 1, py), fill=crop_rgb)
                        draw.point((px + 1, py), fill=crop_rgb)
                    else:
                        draw.line((px, py - 2, px, py + 5), fill=crop_dark)
                        draw.point((px - 1, py - 1), fill=crop_rgb)
                        draw.point((px, py - 2), fill=crop_light)
                        draw.point((px + 1, py - 1), fill=crop_rgb)
        return True
    if kind == "trough":
        water_rgb = _rgb(spec.visual_attributes.get("water_rgb", (78, 151, 180)))
        wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (124, 83, 48)))
        wood_dark = _shade_rgb(wood_rgb, -42)
        top = [(cx - 15, cy - 5), (cx, cy - 10), (cx + 15, cy - 5), (cx, cy + 1)]
        body = [(cx - 15, cy - 5), (cx, cy + 1), (cx + 15, cy - 5), (cx + 12, cy + 2), (cx, cy + 8), (cx - 12, cy + 2)]
        draw.polygon(body, fill=wood_rgb, outline=wood_dark)
        draw.polygon(top, fill=water_rgb, outline=(43, 91, 114))
        draw.line((cx - 8, cy - 5, cx + 6, cy - 5), fill=(151, 211, 225))
        return True
    return False


def _iso_extent(tile_xywh: TileBox, *, min_half_x: int, min_half_y: int) -> tuple[int, int]:
    _, _, width, height = tile_xywh
    footprint = max(1, int(width)) + max(1, int(height))
    return (
        max(int(min_half_x), int(round(float(footprint) * 7.8))),
        max(int(min_half_y), int(round(float(footprint) * 3.7))),
    )


def _iso_diamond(cx: int, cy: int, half_x: int, half_y: int) -> list[tuple[int, int]]:
    return [(cx - half_x, cy), (cx, cy - half_y), (cx + half_x, cy), (cx, cy + half_y)]


def _draw_iso_basket(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    wicker_rgb = _rgb(spec.visual_attributes.get("wicker_rgb", (180, 121, 62)))
    wicker_dark = _shade_rgb(wicker_rgb, -48)
    wicker_light = _shade_rgb(wicker_rgb, 36)
    inside_rgb = _shade_rgb(wicker_rgb, -70)
    rim = _iso_diamond(cx, cy - 3, 13, 6)
    lower = _iso_diamond(cx, cy + 8, 9, 4)
    left = [rim[0], rim[3], lower[3], lower[0]]
    right = [rim[2], rim[3], lower[3], lower[2]]
    front = [lower[0], lower[3], lower[2], rim[3]]
    draw.polygon(left, fill=_shade_rgb(wicker_rgb, -12), outline=wicker_dark)
    draw.polygon(right, fill=wicker_rgb, outline=wicker_dark)
    draw.polygon(front, fill=_shade_rgb(wicker_rgb, 8), outline=wicker_dark)
    draw.polygon(rim, fill=inside_rgb, outline=wicker_dark)
    draw.line((rim[0][0] + 2, rim[0][1], rim[1][0], rim[1][1] - 1, rim[2][0] - 2, rim[2][1]), fill=wicker_light, width=2)
    draw.line((rim[0][0] + 2, rim[0][1] + 2, rim[3][0], rim[3][1] + 1, rim[2][0] - 2, rim[2][1] + 2), fill=_shade_rgb(wicker_rgb, 20))
    for start, end in (
        ((cx - 7, cy + 0), (cx - 4, cy + 10)),
        ((cx - 1, cy - 1), (cx - 1, cy + 12)),
        ((cx + 7, cy + 0), (cx + 4, cy + 10)),
    ):
        draw.line((*start, *end), fill=_shade_rgb(wicker_rgb, -28))
    for yy in (cy + 3, cy + 7):
        draw.line((cx - 8, yy, cx, yy + 3, cx + 8, yy), fill=wicker_light)


def _draw_iso_barrel(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (139, 82, 43)))
    wood_dark = _shade_rgb(wood_rgb, -50)
    wood_light = _shade_rgb(wood_rgb, 38)
    band_rgb = (91, 78, 62)
    body = [(cx - 8, cy - 5), (cx + 8, cy - 5), (cx + 7, cy + 8), (cx - 7, cy + 8)]
    draw.polygon(body, fill=wood_rgb, outline=wood_dark)
    draw.ellipse((cx - 9, cy - 10, cx + 9, cy - 1), fill=_shade_rgb(wood_rgb, 20), outline=wood_dark)
    draw.arc((cx - 7, cy - 8, cx + 7, cy - 2), 190, 350, fill=wood_light)
    draw.line((cx - 7, cy - 2, cx + 7, cy - 2), fill=band_rgb)
    draw.line((cx - 7, cy + 5, cx + 7, cy + 5), fill=band_rgb)
    for px in (cx - 4, cx, cx + 4):
        draw.line((px, cy - 4, px, cy + 7), fill=_shade_rgb(wood_rgb, -22))


def _draw_iso_bench(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (138, 83, 47)))
    wood_dark = _shade_rgb(wood_rgb, -48)
    wood_light = _shade_rgb(wood_rgb, 34)
    _, _, width, height = tile_xywh
    long_axis = max(1, int(width)) >= max(1, int(height))
    half_x = 25 if long_axis else 14
    half_y = 8 if long_axis else 15
    seat_cy = cy
    seat = _iso_diamond(cx, seat_cy, half_x, half_y)
    front_left = [seat[0], seat[3], (seat[3][0], seat[3][1] + 5), (seat[0][0], seat[0][1] + 5)]
    front_right = [seat[2], seat[3], (seat[3][0], seat[3][1] + 5), (seat[2][0], seat[2][1] + 5)]
    for leg_x, leg_y, lean in (
        (seat[0][0] + 7, seat[0][1] + 1, -2),
        (seat[2][0] - 7, seat[2][1] + 1, 2),
        (seat[3][0] - 8, seat[3][1] - 1, -1),
        (seat[3][0] + 8, seat[3][1] - 1, 1),
    ):
        draw.line((leg_x, leg_y, leg_x + lean, leg_y + 11), fill=(49, 44, 38), width=2)
        draw.point((leg_x + lean, leg_y + 5), fill=(93, 81, 63))
    draw.polygon(front_left, fill=_shade_rgb(wood_rgb, -14), outline=wood_dark)
    draw.polygon(front_right, fill=_shade_rgb(wood_rgb, -2), outline=wood_dark)
    draw.polygon(seat, fill=wood_rgb, outline=wood_dark)
    draw.line((seat[0][0] + 5, seat[0][1], seat[2][0] - 5, seat[2][1]), fill=wood_light)
    draw.line((seat[0][0] + 4, seat[0][1] + 3, seat[3][0], seat[3][1] + 2, seat[2][0] - 4, seat[2][1] + 3), fill=_shade_rgb(wood_rgb, -22))
    for offset in (-10, 0, 10) if long_axis else (-5, 5):
        draw.line((cx + offset, seat_cy - half_y + 2, cx + offset, seat_cy + half_y - 2), fill=_shade_rgb(wood_rgb, -26))


def _draw_iso_table(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (139, 82, 43)))
    shape = str(spec.visual_attributes.get("table_shape", "square"))
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=18 if shape != "long" else 28, min_half_y=9)
    if shape == "round":
        half_x = min(18, half_x)
        half_y = min(9, half_y)
    top_y = cy - 7
    wood_dark = _shade_rgb(wood_rgb, -48)
    wood_light = _shade_rgb(wood_rgb, 36)
    leg_rgb = _shade_rgb(wood_rgb, -16)
    if shape == "round":
        for leg_x, leg_y in ((cx - 8, top_y), (cx + 8, top_y), (cx - 4, top_y + half_y - 1), (cx + 4, top_y + half_y - 1)):
            draw.line((leg_x, leg_y, leg_x, cy + 8), fill=wood_dark, width=2)
            draw.point((leg_x + 1, leg_y + 2), fill=leg_rgb)
        draw.ellipse((cx - half_x, top_y - half_y, cx + half_x, top_y + half_y), fill=wood_rgb, outline=wood_dark)
        draw.arc((cx - half_x + 3, top_y - half_y + 2, cx + half_x - 3, top_y + half_y - 1), 190, 345, fill=wood_light)
        return
    top = _iso_diamond(cx, top_y, half_x, half_y)
    front_left = [top[0], top[3], (top[3][0], top[3][1] + 5), (top[0][0], top[0][1] + 5)]
    front_right = [top[2], top[3], (top[3][0], top[3][1] + 5), (top[2][0], top[2][1] + 5)]
    for leg_x, leg_y in (
        (cx - half_x + 6, top_y + 2),
        (cx + half_x - 6, top_y + 2),
        (cx - 5, top_y + half_y - 1),
        (cx + 5, top_y + half_y - 1),
    ):
        draw.line((leg_x, leg_y, leg_x, cy + 11), fill=wood_dark, width=2)
        draw.point((leg_x + 1, leg_y + 3), fill=leg_rgb)
    draw.polygon(front_left, fill=_shade_rgb(wood_rgb, -14), outline=wood_dark)
    draw.polygon(front_right, fill=_shade_rgb(wood_rgb, -3), outline=wood_dark)
    draw.polygon(top, fill=wood_rgb, outline=wood_dark)
    draw.line((cx - half_x + 5, top_y, cx + half_x - 5, top_y), fill=wood_light)
    if shape == "long":
        for offset in (-8, 0, 8):
            draw.line((cx + offset, top_y - half_y + 2, cx + offset, top_y + half_y - 2), fill=_shade_rgb(wood_rgb, -25))


def _draw_iso_chair(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (126, 75, 42)))
    cushion_rgb = _rgb(spec.visual_attributes.get("cushion_rgb", _shade_rgb(wood_rgb, 28)))
    wood_dark = _shade_rgb(wood_rgb, -46)
    wood_light = _shade_rgb(wood_rgb, 36)
    cushion_dark = _shade_rgb(cushion_rgb, -42)
    cushion_light = _shade_rgb(cushion_rgb, 38)
    facing = str(spec.visual_attributes.get("facing", "down"))
    seat = _iso_diamond(cx, cy - 1, 10, 5)
    front_left = [seat[0], seat[3], (seat[3][0], seat[3][1] + 3), (seat[0][0], seat[0][1] + 3)]
    front_right = [seat[2], seat[3], (seat[3][0], seat[3][1] + 3), (seat[2][0], seat[2][1] + 3)]
    for leg_x, leg_y, lean in ((cx - 7, cy, -1), (cx + 7, cy, 1), (cx - 3, cy + 4, -1), (cx + 3, cy + 4, 1)):
        draw.line((leg_x, leg_y, leg_x + lean, leg_y + 9), fill=wood_dark, width=2)
        draw.point((leg_x, leg_y + 2), fill=wood_light)

    if facing == "left":
        posts = ((cx - 9, cy - 2, cx - 16, cy - 11), (cx - 1, cy - 7, cx - 8, cy - 16))
        back = [(cx - 11, cy - 1), (cx, cy - 6), (cx - 1, cy - 14), (cx - 10, cy - 18), (cx - 17, cy - 12)]
    elif facing == "right":
        posts = ((cx + 9, cy - 2, cx + 16, cy - 11), (cx + 1, cy - 7, cx + 8, cy - 16))
        back = [(cx + 11, cy - 1), (cx, cy - 6), (cx + 1, cy - 14), (cx + 10, cy - 18), (cx + 17, cy - 12)]
    else:
        posts = ((cx - 9, cy + 1, cx - 9, cy - 16), (cx + 9, cy + 1, cx + 9, cy - 16))
        back = [(cx - 11, cy), (cx, cy + 5), (cx + 11, cy), (cx + 11, cy - 12), (cx, cy - 17), (cx - 11, cy - 12)]
    for line in posts:
        draw.line(line, fill=wood_dark, width=2)
    draw.polygon(back, fill=wood_rgb, outline=wood_dark)
    if len(back) >= 3:
        draw.line((back[0][0] + 3, back[0][1] - 1, back[2][0] - 3, back[2][1] - 1), fill=wood_light)
    draw.polygon(front_left, fill=_shade_rgb(cushion_rgb, -18), outline=cushion_dark)
    draw.polygon(front_right, fill=_shade_rgb(cushion_rgb, -5), outline=cushion_dark)
    draw.polygon(seat, fill=cushion_rgb, outline=cushion_dark)
    draw.line((cx - 7, cy - 1, cx, cy - 5, cx + 7, cy - 1), fill=cushion_light)
    draw.line((cx - 8, cy + 1, cx, cy + 4, cx + 8, cy + 1), fill=_shade_rgb(cushion_rgb, -20))


def _draw_iso_stool(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (126, 75, 42)))
    cushion_rgb = _rgb(spec.visual_attributes.get("cushion_rgb", (151, 83, 58)))
    wood_dark = _shade_rgb(wood_rgb, -46)
    wood_light = _shade_rgb(wood_rgb, 34)
    cushion_dark = _shade_rgb(cushion_rgb, -42)
    cushion_light = _shade_rgb(cushion_rgb, 38)
    for leg_x, leg_y, lean in ((cx - 7, cy - 1, -1), (cx + 7, cy - 1, 1), (cx, cy + 1, 0)):
        draw.line((leg_x, leg_y, leg_x + lean, leg_y + 9), fill=wood_dark, width=2)
        draw.point((leg_x, leg_y + 1), fill=wood_light)
    draw.line((cx - 6, cy + 7, cx + 6, cy + 7), fill=wood_rgb)
    draw.ellipse((cx - 11, cy - 8, cx + 11, cy + 2), fill=cushion_dark, outline=cushion_dark)
    draw.ellipse((cx - 10, cy - 10, cx + 10, cy), fill=cushion_rgb, outline=cushion_dark)
    draw.arc((cx - 7, cy - 8, cx + 7, cy - 2), 190, 350, fill=cushion_light)
    draw.line((cx - 7, cy - 1, cx + 7, cy - 1), fill=_shade_rgb(cushion_rgb, -20))


def _draw_iso_bed(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (118, 72, 42)))
    blanket_rgb = _rgb(spec.visual_attributes.get("blanket_rgb", (97, 132, 173)))
    pillow_rgb = _rgb(spec.visual_attributes.get("pillow_rgb", (232, 222, 188)))
    is_double = str(spec.visual_attributes.get("bed_size", "single")) == "double"
    half_x = 34 if is_double else 27
    half_y = 16 if is_double else 13
    wood_dark = _shade_rgb(wood_rgb, -48)
    wood_light = _shade_rgb(wood_rgb, 34)
    blanket_dark = _shade_rgb(blanket_rgb, -44)
    pillow_dark = _shade_rgb(pillow_rgb, -44)
    top_y = cy - 4
    frame = _iso_diamond(cx, top_y, half_x, half_y)
    left = [frame[0], frame[3], (frame[3][0], frame[3][1] + 9), (frame[0][0], frame[0][1] + 9)]
    right = [frame[2], frame[3], (frame[3][0], frame[3][1] + 9), (frame[2][0], frame[2][1] + 9)]
    foot = [
        (frame[0][0] + 2, frame[3][1] - 1),
        frame[3],
        (frame[2][0] - 2, frame[3][1] - 1),
        (frame[2][0] - 2, frame[3][1] + 8),
        (frame[3][0], frame[3][1] + 12),
        (frame[0][0] + 2, frame[3][1] + 8),
    ]
    mattress = _iso_diamond(cx, top_y - 2, half_x - 5, half_y - 4)
    blanket = [
        (cx - half_x + 8, top_y + 1),
        (cx - 2, top_y - half_y + 6),
        (cx + half_x - 7, top_y + 1),
        (cx, top_y + half_y - 4),
    ]
    pillow_left = [
        (cx - half_x + 7, top_y - 2),
        (cx - half_x + 18, top_y - half_y + 4),
        (cx - half_x + 29, top_y - half_y + 7),
        (cx - half_x + 15, top_y + 2),
    ]
    pillow_right = [
        (cx - 3, top_y - half_y + 4),
        (cx + 9, top_y - half_y - 1),
        (cx + 21, top_y + 1),
        (cx + 8, top_y + 5),
    ]
    draw.polygon(left, fill=_shade_rgb(wood_rgb, -12), outline=wood_dark)
    draw.polygon(right, fill=_shade_rgb(wood_rgb, 4), outline=wood_dark)
    for px, py in ((frame[0][0] + 4, frame[0][1] + 1), (frame[2][0] - 4, frame[2][1] + 1), (frame[3][0] - 9, frame[3][1] + 2), (frame[3][0] + 9, frame[3][1] + 2)):
        draw.line((px, py, px, py + 12), fill=wood_dark, width=2)
    draw.polygon(foot, fill=wood_rgb, outline=wood_dark)
    draw.polygon(frame, fill=_shade_rgb(wood_rgb, 10), outline=wood_dark)
    draw.polygon(mattress, fill=(231, 218, 187), outline=_shade_rgb(pillow_rgb, -48))
    draw.polygon(blanket, fill=blanket_rgb, outline=blanket_dark)
    draw.polygon(pillow_left, fill=pillow_rgb, outline=pillow_dark)
    if is_double:
        draw.polygon(pillow_right, fill=_shade_rgb(pillow_rgb, 8), outline=pillow_dark)
    draw.line((blanket[0][0] + 3, blanket[0][1] + 1, blanket[2][0] - 4, blanket[2][1] + 1), fill=_shade_rgb(blanket_rgb, 36))
    draw.line((blanket[0][0] + 4, blanket[0][1] + 4, blanket[3][0] - 3, blanket[3][1] - 2), fill=_shade_rgb(blanket_rgb, -24))
    if is_double:
        draw.line((cx, top_y - half_y + 5, cx, top_y + half_y - 3), fill=_shade_rgb(blanket_rgb, -34))


def _draw_iso_chest(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (139, 82, 43)))
    metal_rgb = _rgb(spec.visual_attributes.get("metal_rgb", (189, 160, 80)))
    wood_dark = _shade_rgb(wood_rgb, -50)
    wood_light = _shade_rgb(wood_rgb, 38)
    metal_dark = _shade_rgb(metal_rgb, -50)
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=20, min_half_y=8)
    top_y = cy - 2
    body_top = _iso_diamond(cx, top_y, half_x, half_y)
    interior = _iso_diamond(cx, top_y - 2, half_x - 4, max(4, half_y - 3))
    left = [body_top[0], body_top[3], (body_top[3][0], body_top[3][1] + 12), (body_top[0][0], body_top[0][1] + 8)]
    right = [body_top[2], body_top[3], (body_top[3][0], body_top[3][1] + 12), (body_top[2][0], body_top[2][1] + 8)]
    front = [(body_top[0][0] + 2, body_top[3][1] - 1), body_top[3], (body_top[2][0] - 2, body_top[3][1] - 1), (body_top[2][0] - 2, body_top[3][1] + 10), (body_top[3][0], body_top[3][1] + 14), (body_top[0][0] + 2, body_top[3][1] + 10)]

    draw.polygon(left, fill=_shade_rgb(wood_rgb, -8), outline=wood_dark)
    draw.polygon(right, fill=_shade_rgb(wood_rgb, 10), outline=wood_dark)
    draw.polygon(front, fill=wood_rgb, outline=wood_dark)
    draw.polygon(body_top, fill=_shade_rgb(wood_rgb, 12), outline=wood_dark)
    draw.polygon(interior, fill=(48, 35, 27), outline=wood_dark)
    rear_lid = [
        (body_top[0][0] + 4, body_top[0][1] + 1),
        (body_top[1][0], body_top[1][1] - 3),
        (body_top[2][0] - 4, body_top[2][1] + 1),
        (body_top[2][0] - 7, body_top[2][1] + 5),
        (body_top[1][0], body_top[1][1] + 3),
        (body_top[0][0] + 7, body_top[0][1] + 5),
    ]
    draw.polygon(rear_lid, fill=_shade_rgb(wood_rgb, 24), outline=wood_dark)
    draw.line((rear_lid[0][0] + 4, rear_lid[0][1] + 1, rear_lid[2][0] - 4, rear_lid[2][1] + 1), fill=wood_light)
    draw.line((body_top[0][0] + 2, body_top[0][1] + 3, body_top[3][0], body_top[3][1] + 10, body_top[2][0] - 2, body_top[2][1] + 3), fill=wood_dark)
    for offset in (-9, 9):
        draw.line((cx + offset, top_y - half_y + 1, cx + offset // 2, top_y + half_y + 11), fill=metal_rgb, width=2)
    draw.rectangle((cx - 3, top_y + half_y + 4, cx + 3, top_y + half_y + 10), fill=metal_rgb, outline=metal_dark)
    draw.point((cx, top_y + half_y + 7), fill=_shade_rgb(metal_rgb, 45))


def _draw_iso_counter(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (139, 82, 43)))
    top_rgb = _rgb(spec.visual_attributes.get("top_rgb", _shade_rgb(wood_rgb, 32)))
    wood_dark = _shade_rgb(wood_rgb, -48)
    _, _, w, h = tile_xywh
    half_x = max(28, int(round((max(1, int(w)) + max(1, int(h)) * 0.4) * 10.0)))
    half_y = max(10, int(round((max(1, int(w)) + max(1, int(h))) * 2.9)))
    top_y = cy - 8
    top = _iso_diamond(cx, top_y, half_x, half_y)
    left = [top[0], top[3], (top[3][0], top[3][1] + 17), (top[0][0], top[0][1] + 13)]
    right = [top[2], top[3], (top[3][0], top[3][1] + 17), (top[2][0], top[2][1] + 13)]
    draw.polygon(left, fill=_shade_rgb(wood_rgb, -12), outline=wood_dark)
    draw.polygon(right, fill=_shade_rgb(wood_rgb, 5), outline=wood_dark)
    draw.polygon(top, fill=top_rgb, outline=_shade_rgb(top_rgb, -45))
    draw.line((top[0][0] + 6, top[0][1] + 1, top[2][0] - 6, top[2][1] + 1), fill=_shade_rgb(top_rgb, 34))
    for offset in (-17, 0, 17):
        draw.line((cx + offset, top_y + 2, cx + offset // 2, top_y + half_y + 15), fill=_shade_rgb(wood_rgb, -28))
    draw.rectangle((cx + half_x - 17, top_y - 5, cx + half_x - 8, top_y + 2), fill=(225, 201, 125), outline=(103, 75, 43))
    draw.point((cx + half_x - 12, top_y - 3), fill=(174, 56, 51))
    draw.rectangle((cx - half_x + 9, top_y - 4, cx - half_x + 17, top_y + 1), fill=(92, 153, 178), outline=(50, 91, 112))


def _draw_iso_fireplace(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    stone_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (116, 109, 96)))
    flame_rgb = _rgb(spec.visual_attributes.get("flame_rgb", (238, 126, 45)))
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=18, min_half_y=6)
    stone_dark = _shade_rgb(stone_rgb, -48)
    wall_y = cy - half_y
    height = 26
    draw.polygon(
        [(cx - half_x, wall_y), (cx + half_x, wall_y), (cx + half_x, wall_y - height), (cx - half_x, wall_y - height)],
        fill=stone_rgb,
        outline=stone_dark,
    )
    draw.rectangle((cx - half_x + 4, wall_y - height, cx + half_x - 4, wall_y - height + 5), fill=_shade_rgb(stone_rgb, 34), outline=stone_dark)
    mouth = (cx - 8, wall_y - 16, cx + 8, wall_y - 3)
    draw.rectangle(mouth, fill=(45, 37, 33), outline=(31, 28, 26))
    draw.polygon([(cx - half_x, wall_y), (cx, wall_y + half_y), (cx + half_x, wall_y), (cx, wall_y - half_y)], fill=_shade_rgb(stone_rgb, -10), outline=stone_dark)
    if str(spec.visual_attributes.get("fire_state", "lit")) == "lit":
        draw.polygon([(cx, wall_y - 14), (cx - 5, wall_y - 6), (cx - 1, wall_y - 2), (cx + 4, wall_y - 5)], fill=flame_rgb, outline=_shade_rgb(flame_rgb, -46))
        draw.polygon([(cx, wall_y - 10), (cx - 2, wall_y - 5), (cx + 1, wall_y - 3), (cx + 2, wall_y - 7)], fill=(255, 226, 84))


def _draw_iso_stone_column(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    stone_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (126, 123, 114)))
    dark = _shade_rgb(stone_rgb, -58)
    mid = _shade_rgb(stone_rgb, -18)
    light = _shade_rgb(stone_rgb, 38)
    top_y = cy - 26
    base_y = cy + 5
    base = _iso_diamond(cx, base_y, 13, 6)
    shaft = [
        (cx - 7, top_y + 4),
        (cx - 2, top_y + 7),
        (cx - 2, base_y + 2),
        (cx - 7, base_y - 1),
    ]
    shaft_r = [
        (cx + 7, top_y + 4),
        (cx + 2, top_y + 7),
        (cx + 2, base_y + 2),
        (cx + 7, base_y - 1),
    ]
    draw.polygon(base, fill=_shade_rgb(stone_rgb, -8), outline=dark)
    draw.polygon(shaft, fill=mid, outline=dark)
    draw.polygon(shaft_r, fill=stone_rgb, outline=dark)
    draw.rectangle((cx - 7, top_y + 5, cx + 7, base_y), fill=stone_rgb, outline=dark)
    draw.ellipse((cx - 9, top_y - 2, cx + 9, top_y + 8), fill=_shade_rgb(stone_rgb, 12), outline=dark)
    draw.line((cx - 4, top_y + 8, cx - 4, base_y - 1), fill=light)
    draw.line((cx + 5, top_y + 8, cx + 5, base_y - 1), fill=_shade_rgb(stone_rgb, -34))
    draw.line((cx - 7, base_y - 1, cx + 7, base_y - 1), fill=dark)


def _draw_iso_archway(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    stone_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (120, 118, 110)))
    shadow_rgb = _rgb(spec.visual_attributes.get("shadow_rgb", (37, 36, 39)))
    dark = _shade_rgb(stone_rgb, -58)
    light = _shade_rgb(stone_rgb, 36)
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=23, min_half_y=8)
    wall_y = cy - half_y
    height = 31
    left_col = [(cx - half_x, wall_y), (cx - half_x + 7, wall_y - 4), (cx - half_x + 7, wall_y - height + 9), (cx - half_x, wall_y - height + 5)]
    right_col = [(cx + half_x, wall_y), (cx + half_x - 7, wall_y - 4), (cx + half_x - 7, wall_y - height + 9), (cx + half_x, wall_y - height + 5)]
    top = [(cx - half_x + 5, wall_y - height + 8), (cx, wall_y - height - 5), (cx + half_x - 5, wall_y - height + 8), (cx + half_x - 10, wall_y - height + 14), (cx, wall_y - height + 3), (cx - half_x + 10, wall_y - height + 14)]
    opening = [(cx - half_x + 9, wall_y - 3), (cx, wall_y - 8), (cx + half_x - 9, wall_y - 3), (cx + half_x - 9, wall_y - height + 11), (cx, wall_y - height + 2), (cx - half_x + 9, wall_y - height + 11)]
    draw.polygon(left_col, fill=_shade_rgb(stone_rgb, -14), outline=dark)
    draw.polygon(right_col, fill=stone_rgb, outline=dark)
    draw.polygon(top, fill=_shade_rgb(stone_rgb, 10), outline=dark)
    draw.polygon(opening, fill=shadow_rgb, outline=_shade_rgb(shadow_rgb, -18))
    draw.line((cx - half_x + 10, wall_y - 2, cx + half_x - 10, wall_y - 2), fill=_shade_rgb(shadow_rgb, -24))
    for offset in (-15, -5, 7):
        draw.line((cx + offset, wall_y - height + 12, cx + offset + 5, wall_y - height + 17), fill=_shade_rgb(stone_rgb, -30))
    draw.line((cx - half_x + 8, wall_y - height + 12, cx, wall_y - height + 2, cx + half_x - 8, wall_y - height + 12), fill=light)
    draw.line((cx - half_x + 3, wall_y - 3, cx - half_x + 3, wall_y - height + 7), fill=_shade_rgb(stone_rgb, -34))
    draw.line((cx + half_x - 4, wall_y - 3, cx + half_x - 4, wall_y - height + 7), fill=_shade_rgb(stone_rgb, -34))


def _draw_iso_sealed_door(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (134, 87, 45)))
    seal_rgb = _rgb(spec.visual_attributes.get("seal_rgb", (94, 74, 55)))
    dark = _shade_rgb(wood_rgb, -56)
    light = _shade_rgb(wood_rgb, 34)
    iron = (58, 56, 52)
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=21, min_half_y=7)
    orientation = str(spec.visual_attributes.get("door_orientation", spec.visual_attributes.get("orientation", "horizontal")))
    if orientation == "vertical":
        top_y = cy - half_y - 16
        bottom_y = cy + half_y + 13
        face = [
            (cx - 13, top_y + 9),
            (cx, top_y + 1),
            (cx + 13, top_y + 9),
            (cx + 13, bottom_y - 9),
            (cx, bottom_y),
            (cx - 13, bottom_y - 9),
        ]
        right = [
            (cx + 13, top_y + 9),
            (cx + 18, top_y + 12),
            (cx + 18, bottom_y - 7),
            (cx + 13, bottom_y - 9),
        ]
        draw.polygon(right, fill=_shade_rgb(wood_rgb, -22), outline=dark)
        draw.polygon(face, fill=wood_rgb, outline=dark)
        for offset in (-6, 0, 6):
            draw.line((cx + offset, top_y + 7, cx + offset, bottom_y - 6), fill=_shade_rgb(wood_rgb, -26))
        draw.line((cx - 9, top_y + 13, cx + 10, top_y + 13), fill=iron, width=2)
        draw.line((cx - 10, bottom_y - 12, cx + 9, bottom_y - 12), fill=iron, width=2)
        draw.line((cx, top_y + 5, cx, bottom_y - 5), fill=_shade_rgb(seal_rgb, -42), width=2)
        draw.rectangle((cx - 4, cy - 1, cx + 4, cy + 7), fill=seal_rgb, outline=_shade_rgb(seal_rgb, -42))
        draw.line((cx - 8, top_y + 7, cx + 2, top_y + 2, cx + 11, top_y + 8), fill=light)
    else:
        top_y = cy - half_y - 13
        bottom_y = cy + half_y + 13
        face = [
            (cx - half_x + 5, top_y + 7),
            (cx + half_x - 5, top_y + 7),
            (cx + half_x - 9, bottom_y - 4),
            (cx - half_x + 9, bottom_y - 4),
        ]
        top_lip = [
            (cx - half_x + 5, top_y + 7),
            (cx - half_x + 11, top_y + 2),
            (cx + half_x - 11, top_y + 2),
            (cx + half_x - 5, top_y + 7),
        ]
        right = [
            (cx + half_x - 5, top_y + 7),
            (cx + half_x + 1, top_y + 11),
            (cx + half_x - 3, bottom_y),
            (cx + half_x - 9, bottom_y - 4),
        ]
        left = [
            (cx - half_x + 5, top_y + 7),
            (cx - half_x - 1, top_y + 11),
            (cx - half_x + 3, bottom_y),
            (cx - half_x + 9, bottom_y - 4),
        ]
        draw.polygon(left, fill=_shade_rgb(wood_rgb, -18), outline=dark)
        draw.polygon(right, fill=_shade_rgb(wood_rgb, -24), outline=dark)
        draw.polygon(face, fill=wood_rgb, outline=dark)
        draw.polygon(top_lip, fill=_shade_rgb(wood_rgb, 10), outline=dark)
        for offset in (-14, -5, 5, 14):
            draw.line((cx + offset, top_y + 8, cx + offset - 2, bottom_y - 5), fill=_shade_rgb(wood_rgb, -28))
        draw.line((cx, top_y + 7, cx, bottom_y - 5), fill=_shade_rgb(wood_rgb, -38), width=2)
        draw.line((cx - half_x + 8, top_y + 12, cx + half_x - 8, top_y + 12), fill=iron, width=2)
        draw.line((cx - half_x + 10, bottom_y - 10, cx + half_x - 10, bottom_y - 10), fill=iron, width=2)
        draw.rectangle((cx - 5, cy, cx + 5, cy + 8), fill=seal_rgb, outline=_shade_rgb(seal_rgb, -42))
        draw.point((cx + 7, cy + 4), fill=light)


def _draw_iso_brazier(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    metal_rgb = _rgb(spec.visual_attributes.get("metal_rgb", (93, 91, 88)))
    flame_rgb = _rgb(spec.visual_attributes.get("flame_rgb", (238, 126, 45)))
    dark = _shade_rgb(metal_rgb, -48)
    light = _shade_rgb(metal_rgb, 38)
    draw.polygon(_iso_diamond(cx, cy + 6, 9, 4), fill=(49, 48, 46), outline=None)
    draw.line((cx - 2, cy - 2, cx - 3, cy + 8), fill=dark, width=2)
    draw.line((cx + 2, cy - 2, cx + 3, cy + 8), fill=dark, width=2)
    bowl = _iso_diamond(cx, cy - 4, 13, 6)
    left = [bowl[0], bowl[3], (bowl[3][0], bowl[3][1] + 5), (bowl[0][0], bowl[0][1] + 3)]
    right = [bowl[2], bowl[3], (bowl[3][0], bowl[3][1] + 5), (bowl[2][0], bowl[2][1] + 3)]
    draw.polygon(left, fill=_shade_rgb(metal_rgb, -8), outline=dark)
    draw.polygon(right, fill=metal_rgb, outline=dark)
    draw.polygon(bowl, fill=_shade_rgb(metal_rgb, 10), outline=dark)
    draw.line((bowl[0][0] + 3, bowl[0][1] + 1, bowl[2][0] - 3, bowl[2][1] + 1), fill=light)
    if str(spec.visual_attributes.get("fire_state", "lit")) == "lit":
        draw.polygon([(cx, cy - 19), (cx - 5, cy - 9), (cx - 1, cy - 4), (cx + 5, cy - 9)], fill=flame_rgb, outline=_shade_rgb(flame_rgb, -48))
        draw.polygon([(cx, cy - 14), (cx - 2, cy - 8), (cx + 1, cy - 5), (cx + 2, cy - 10)], fill=(255, 226, 84))


def _draw_iso_floor_switch(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    plate_rgb = _rgb(spec.visual_attributes.get("plate_rgb", (142, 126, 88)))
    dark = _shade_rgb(plate_rgb, -56)
    light = _shade_rgb(plate_rgb, 36)
    pressed = str(spec.visual_attributes.get("switch_state", "raised")) == "pressed"
    half_x = 13
    half_y = 7
    if pressed:
        base = _iso_diamond(cx, cy + 1, half_x, half_y)
        inner = _iso_diamond(cx, cy + 1, 6, 3)
        draw.polygon(base, fill=_shade_rgb(plate_rgb, -24), outline=dark)
        draw.polygon(inner, fill=_shade_rgb(plate_rgb, -4), outline=_shade_rgb(plate_rgb, -36))
        draw.line((base[0][0] + 4, base[0][1], base[2][0] - 4, base[2][1]), fill=_shade_rgb(plate_rgb, 18))
    else:
        bottom = _iso_diamond(cx, cy + 4, half_x, half_y)
        top = _iso_diamond(cx, cy - 2, half_x - 1, half_y - 1)
        left = [bottom[0], bottom[3], top[3], top[0]]
        right = [bottom[2], bottom[3], top[3], top[2]]
        draw.polygon(left, fill=_shade_rgb(plate_rgb, -24), outline=dark)
        draw.polygon(right, fill=_shade_rgb(plate_rgb, -12), outline=dark)
        draw.polygon(top, fill=plate_rgb, outline=dark)
        draw.polygon(_iso_diamond(cx, cy - 2, 6, 3), fill=_shade_rgb(plate_rgb, 18), outline=_shade_rgb(plate_rgb, -24))
        draw.line((top[0][0] + 3, top[0][1], top[1][0], top[1][1] + 1, top[2][0] - 3, top[2][1]), fill=light)


def _draw_iso_broken_wall(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    stone_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (132, 132, 124)))
    dark = _shade_rgb(stone_rgb, -58)
    light = _shade_rgb(stone_rgb, 34)
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=24, min_half_y=7)
    top_y = cy - half_y - 2
    height = 21
    top = [(cx - half_x + 4, top_y - height), (cx, top_y - height - 7), (cx + half_x - 4, top_y - height), (cx, top_y - height + 5)]
    front = [(cx - half_x, top_y), (cx + half_x, top_y), (cx + half_x - 5, top_y - height), (cx - half_x + 5, top_y - height)]
    side = [(cx + half_x, top_y), (cx + half_x - 5, top_y - height), (cx, top_y - height + 5), (cx, top_y + half_y + 1)]
    draw.polygon(front, fill=stone_rgb, outline=dark)
    draw.polygon(side, fill=_shade_rgb(stone_rgb, -12), outline=dark)
    draw.polygon(top, fill=_shade_rgb(stone_rgb, 13), outline=dark)
    mortar_y = top_y - 8
    draw.line((cx - half_x + 3, mortar_y, cx + half_x - 5, mortar_y), fill=_shade_rgb(stone_rgb, -30))
    for offset in (-15, -5, 6):
        draw.line((cx + offset, top_y - 2, cx + offset + 6, top_y - 8), fill=_shade_rgb(stone_rgb, -30))
    crack_x = cx - 4
    draw.line((crack_x, top_y - height + 2, crack_x + 4, top_y - 10, crack_x + 1, top_y - 5, crack_x + 8, top_y - 1), fill=(45, 44, 42))
    if str(spec.visual_attributes.get("break_style", "cracked")) == "gap":
        draw.polygon([(cx + 4, top_y - height + 2), (cx + 16, top_y - 14), (cx + 14, top_y - 1), (cx + 4, top_y + 1)], fill=(45, 44, 42), outline=dark)
        draw.polygon([(cx - 12, top_y - height + 5), (cx - 5, top_y - 12), (cx - 8, top_y - 2), (cx - 17, top_y - 1)], fill=_shade_rgb(stone_rgb, -20), outline=dark)
    draw.line((front[0][0] + 4, front[0][1] - 3, front[1][0] - 5, front[1][1] - 3), fill=light)


def _draw_iso_rubble(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    spec: IllustrationObjectSpec,
) -> None:
    stone_rgb = _rgb(spec.visual_attributes.get("stone_rgb", (116, 113, 105)))
    dark = _shade_rgb(stone_rgb, -58)
    light = _shade_rgb(stone_rgb, 36)
    draw.polygon(_iso_diamond(cx, cy + 4, 13, 6), fill=(55, 53, 50), outline=None)
    stones = (
        ((cx - 11, cy + 2), 6, 4, _shade_rgb(stone_rgb, -10)),
        ((cx - 2, cy - 3), 8, 5, stone_rgb),
        ((cx + 8, cy + 1), 6, 4, _shade_rgb(stone_rgb, -28)),
        ((cx - 5, cy - 8), 5, 3, _shade_rgb(stone_rgb, 8)),
    )
    for (sx, sy), hx, hy, color in stones:
        top = _iso_diamond(sx, sy, hx, hy)
        front = [(top[0][0] + 1, top[3][1]), top[3], (top[2][0] - 1, top[3][1]), (top[2][0] - 2, top[3][1] + 4), (top[3][0], top[3][1] + 6), (top[0][0] + 2, top[3][1] + 4)]
        draw.polygon(front, fill=_shade_rgb(color, -16), outline=dark)
        draw.polygon(top, fill=color, outline=dark)
    draw.point((cx - 5, cy - 8), fill=light)


def _draw_iso_magic_circle(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    rune_rgb = _rgb(spec.visual_attributes.get("rune_rgb", (92, 214, 232)))
    glow_rgb = _rgb(spec.visual_attributes.get("glow_rgb", (84, 79, 186)))
    half_x, half_y = _iso_extent(tile_xywh, min_half_x=24, min_half_y=10)
    outer = _iso_diamond(cx, cy, half_x, half_y)
    inner = _iso_diamond(cx, cy, max(9, half_x - 8), max(5, half_y - 4))
    rune_dark = _shade_rgb(rune_rgb, -56)
    draw.polygon(_iso_diamond(cx, cy, half_x + 3, half_y + 2), outline=_shade_rgb(glow_rgb, -28))
    draw.polygon(outer, outline=_shade_rgb(glow_rgb, 8))
    draw.line((outer[0][0] + 4, outer[0][1], outer[2][0] - 4, outer[2][1]), fill=_shade_rgb(glow_rgb, 20))
    draw.line((outer[1][0], outer[1][1] + 2, outer[3][0], outer[3][1] - 2), fill=_shade_rgb(glow_rgb, 20))
    draw.polygon(inner, outline=rune_rgb)
    top, right, bottom, left = inner
    draw.line((top[0], top[1], right[0] - 3, right[1] - 1), fill=rune_dark)
    draw.line((right[0] - 3, right[1] - 1, left[0] + 3, left[1] - 1), fill=rune_dark)
    draw.line((left[0] + 3, left[1] - 1, top[0], top[1]), fill=rune_dark)
    for px, py in (top, right, bottom, left):
        draw.rectangle((px - 1, py - 1, px + 1, py + 1), fill=rune_rgb)


def _draw_iso_room_divider(
    draw: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> None:
    wood_rgb = _rgb(spec.visual_attributes.get("wood_rgb", (121, 76, 43)))
    cloth_rgb = _rgb(spec.visual_attributes.get("cloth_rgb", (170, 94, 82)))
    wood_dark = _shade_rgb(wood_rgb, -48)
    cloth_dark = _shade_rgb(cloth_rgb, -42)
    style = str(spec.visual_attributes.get("divider_style", "screen"))
    _, _, width, _ = tile_xywh
    panel_count = max(3, min(4, int(width)))
    start = -(panel_count - 1) * 7
    for index in range(panel_count):
        panel_cx = cx + start + index * 14
        base_y = cy - 4 + (index % 2) * 2
        panel = [(panel_cx - 6, base_y), (panel_cx, base_y - 3), (panel_cx + 6, base_y), (panel_cx + 6, base_y - 24), (panel_cx, base_y - 27), (panel_cx - 6, base_y - 24)]
        fill = _shade_rgb(cloth_rgb, 12) if style == "curtain" and index % 2 else cloth_rgb
        draw.polygon(panel, fill=fill, outline=wood_dark)
        draw.line((panel_cx - 4, base_y - 20, panel_cx + 4, base_y - 22), fill=_shade_rgb(fill, 34))
        draw.line((panel_cx - 4, base_y - 4, panel_cx + 4, base_y - 6), fill=cloth_dark)
    draw.line((cx + start - 6, cy - 4, cx - start + 6, cy - 4), fill=wood_dark)


def _pixel_sprite_tile_size(
    kind: str,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
) -> tuple[int, int]:
    _, _, width, height = tile_xywh
    tile_w = max(1, int(width))
    tile_h = max(1, int(height))
    if kind == "domestic_animal" and _animal_type(spec) == "cow":
        tile_w = max(tile_w, 2)
    if kind in {"dead_tree", "lamp_post", "scarecrow", "tree"}:
        tile_h = max(tile_h, 2)
    return tile_w, tile_h


def _draw_pixel_object_on_tile(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
    *,
    isometric_view: bool = False,
) -> None:
    kind = _pixel_kind(spec)
    visual = spec.visual_attributes
    if kind == "tree":
        draw_pixel_tree(
            draw,
            tile_xywh,
            style=str(visual.get("tree_style", spec.variant_id or "oak")),
            leaf_rgb=_rgb(visual.get("leaf_rgb", (38, 144, 78))),
            fruit_rgb=_rgb(visual.get("fruit_rgb", (220, 65, 61))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target=str(visual.get("tree_style", spec.variant_id or "tree")))
    elif kind == "flower":
        draw_pixel_flower_patch(
            draw,
            tile_xywh,
            flower_rgb=_rgb(visual.get("flower_rgb", (236, 82, 111))),
            leaf_rgb=_rgb(visual.get("leaf_rgb", (40, 128, 62))),
        )
    elif kind == "crop_row":
        draw_pixel_crop_row(
            draw,
            tile_xywh,
            style=str(visual.get("crop_style", "wheat")),
            crop_rgb=_rgb(visual.get("crop_rgb", (210, 171, 58))),
            soil_rgb=_rgb(visual.get("soil_rgb", (130, 87, 48))),
        )
    elif kind == "vegetable_patch":
        draw_pixel_vegetable_patch(
            draw,
            tile_xywh,
            style=str(visual.get("vegetable_style", "carrot")),
            vegetable_rgb=_optional_rgb(visual.get("vegetable_rgb")),
            leaf_rgb=_rgb(visual.get("leaf_rgb", (45, 139, 67))),
            soil_rgb=_rgb(visual.get("soil_rgb", (130, 87, 48))),
        )
    elif kind == "person":
        draw_pixel_person(
            draw,
            tile_xywh,
            skin_rgb=_rgb(visual.get("skin_rgb", (225, 171, 109))),
            shirt_rgb=_rgb(visual.get("shirt_rgb", (55, 116, 190))),
            pants_rgb=_rgb(visual.get("pants_rgb", (48, 72, 105))),
            hair_rgb=_rgb(visual.get("hair_rgb", (82, 50, 33))),
            gender_id=str(visual.get("gender_id", "male")),
            facing=str(visual.get("facing", "down")),
            person_variant_id=str(visual.get("person_variant_id", spec.variant_id or "adult")),
        )
    elif kind == "domestic_animal":
        draw_pixel_animal(
            draw,
            tile_xywh,
            animal_type=_animal_type(spec),
            facing=str(visual.get("facing", "right")),
            body_rgb=_optional_rgb(visual.get("body_rgb")),
            accent_rgb=_optional_rgb(visual.get("accent_rgb")),
            spot_rgb=_optional_rgb(visual.get("spot_rgb")),
        )
    elif kind == "barrel":
        draw_pixel_barrel(
            draw,
            tile_xywh,
            barrel_rgb=_rgb(visual.get("barrel_rgb", (151, 86, 45))),
            band_rgb=_rgb(visual.get("band_rgb", (82, 59, 42))),
        )
    elif kind == "barn":
        draw_pixel_barn(
            draw,
            tile_xywh,
            body_rgb=_rgb(visual.get("body_rgb", (181, 71, 59))),
            roof_rgb=_rgb(visual.get("roof_rgb", (126, 50, 50))),
            door_state=str(visual.get("building_door_state", visual.get("door_state", "closed"))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "bench":
        draw_pixel_bench(
            draw,
            tile_xywh,
            orientation=str(visual.get("orientation", "horizontal")),
            wood_rgb=_rgb(visual.get("wood_rgb", (138, 83, 47))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="bench")
    elif kind == "bridge":
        draw_pixel_bridge(draw, tile_xywh, orientation=str(visual.get("orientation", "horizontal")))
    elif kind == "cart":
        draw_pixel_cart(
            draw,
            tile_xywh,
            facing=str(visual.get("facing", "right")),
            body_rgb=_rgb(visual.get("body_rgb", (151, 88, 45))),
        )
    elif kind == "castle":
        draw_pixel_castle(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", visual.get("body_rgb", (145, 148, 144)))),
            door_state=str(visual.get("building_door_state", visual.get("door_state", "closed"))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "cemetery_gate":
        draw_pixel_cemetery_gate(draw, tile_xywh, orientation=str(visual.get("orientation", "horizontal")))
    elif kind in {"chicken_coop", "coop"}:
        draw_pixel_chicken_coop(
            draw,
            tile_xywh,
            body_rgb=_rgb(visual.get("body_rgb", (188, 126, 70))),
            roof_rgb=_rgb(visual.get("roof_rgb", (137, 62, 50))),
            door_state=str(visual.get("building_door_state", visual.get("door_state", "closed"))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "church":
        draw_pixel_church(
            draw,
            tile_xywh,
            body_rgb=_rgb(visual.get("body_rgb", (211, 190, 150))),
            roof_rgb=_rgb(visual.get("roof_rgb", (77, 113, 166))),
            door_state=str(visual.get("building_door_state", visual.get("door_state", "closed"))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "crate":
        draw_pixel_crate(draw, tile_xywh)
    elif kind == "counter":
        draw_pixel_counter(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (151, 91, 49))),
            top_rgb=_rgb(visual.get("top_rgb", (194, 139, 78))),
        )
    elif kind == "shelf":
        draw_pixel_shelf(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (135, 82, 45))),
            goods_type=str(visual.get("goods_type", "mixed")),
        )
    elif kind == "produce_bin":
        draw_pixel_produce_bin(
            draw,
            tile_xywh,
            goods_type=str(visual.get("goods_type", "fruit")),
            wood_rgb=_rgb(visual.get("wood_rgb", (139, 82, 43))),
            produce_rgb=_optional_rgb(visual.get("produce_rgb")),
        )
    elif kind == "sack":
        draw_pixel_sack(
            draw,
            tile_xywh,
            cloth_rgb=_rgb(visual.get("cloth_rgb", (191, 158, 95))),
        )
    elif kind == "jar":
        draw_pixel_jar(
            draw,
            tile_xywh,
            glass_rgb=_rgb(visual.get("glass_rgb", (92, 155, 178))),
        )
    elif kind == "pot":
        draw_pixel_pot(
            draw,
            tile_xywh,
            clay_rgb=_rgb(visual.get("clay_rgb", (174, 94, 58))),
        )
    elif kind == "basket":
        draw_pixel_basket(
            draw,
            tile_xywh,
            wicker_rgb=_rgb(visual.get("wicker_rgb", (180, 121, 62))),
        )
    elif kind == "rug":
        draw_pixel_rug(
            draw,
            tile_xywh,
            cloth_rgb=_rgb(visual.get("cloth_rgb", (168, 74, 82))),
            trim_rgb=_rgb(visual.get("trim_rgb", (230, 190, 96))),
        )
    elif kind == "chest":
        draw_pixel_chest(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (139, 82, 43))),
            metal_rgb=_rgb(visual.get("metal_rgb", (189, 160, 80))),
        )
    elif kind == "table":
        draw_pixel_table(
            draw,
            tile_xywh,
            table_shape=str(visual.get("table_shape", "square")),
            wood_rgb=_rgb(visual.get("wood_rgb", (139, 82, 43))),
        )
    elif kind == "chair":
        draw_pixel_chair(
            draw,
            tile_xywh,
            facing=str(visual.get("facing", "down")),
            wood_rgb=_rgb(visual.get("wood_rgb", (126, 75, 42))),
            cushion_rgb=_optional_rgb(visual.get("cushion_rgb")),
        )
    elif kind == "stool":
        draw_pixel_stool(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (126, 75, 42))),
            cushion_rgb=_rgb(visual.get("cushion_rgb", (151, 83, 58))),
        )
    elif kind == "bed":
        draw_pixel_bed(
            draw,
            tile_xywh,
            bed_size=str(visual.get("bed_size", "single")),
            wood_rgb=_rgb(visual.get("wood_rgb", (118, 72, 42))),
            blanket_rgb=_rgb(visual.get("blanket_rgb", (97, 132, 173))),
            pillow_rgb=_rgb(visual.get("pillow_rgb", (232, 222, 188))),
        )
    elif kind == "fireplace":
        draw_pixel_fireplace(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (116, 109, 96))),
            fire_state=str(visual.get("fire_state", "lit")),
            flame_rgb=_rgb(visual.get("flame_rgb", (238, 126, 45))),
        )
    elif kind == "room_divider":
        draw_pixel_room_divider(
            draw,
            tile_xywh,
            divider_style=str(visual.get("divider_style", "screen")),
            wood_rgb=_rgb(visual.get("wood_rgb", (121, 76, 43))),
            cloth_rgb=_rgb(visual.get("cloth_rgb", (170, 94, 82))),
        )
    elif kind == "mug":
        draw_pixel_mug(
            draw,
            tile_xywh,
            ceramic_rgb=_rgb(visual.get("ceramic_rgb", (218, 205, 165))),
            drink_rgb=_rgb(visual.get("drink_rgb", (125, 77, 42))),
        )
    elif kind == "bottle":
        draw_pixel_bottle(
            draw,
            tile_xywh,
            glass_rgb=_rgb(visual.get("glass_rgb", (57, 130, 91))),
        )
    elif kind == "plate":
        draw_pixel_plate(
            draw,
            tile_xywh,
            ceramic_rgb=_rgb(visual.get("ceramic_rgb", (224, 218, 198))),
            food_rgb=_optional_rgb(visual.get("food_rgb")),
        )
    elif kind == "bowl":
        draw_pixel_bowl(
            draw,
            tile_xywh,
            ceramic_rgb=_rgb(visual.get("ceramic_rgb", (207, 188, 148))),
            contents_rgb=_rgb(visual.get("contents_rgb", (190, 119, 62))),
        )
    elif kind == "candle":
        draw_pixel_candle(
            draw,
            tile_xywh,
            wax_rgb=_rgb(visual.get("wax_rgb", (238, 222, 171))),
            flame_state=str(visual.get("flame_state", "lit")),
            flame_rgb=_rgb(visual.get("flame_rgb", (243, 153, 55))),
        )
    elif kind == "dead_tree":
        draw_pixel_dead_tree(
            draw,
            tile_xywh,
            bark_rgb=_rgb(visual.get("bark_rgb", (82, 63, 50))),
        )
    elif kind == "farm_gate":
        draw_pixel_farm_gate(draw, tile_xywh, orientation=str(visual.get("orientation", "horizontal")))
    elif kind == "fence":
        draw_pixel_fence(draw, tile_xywh, orientation=str(visual.get("orientation", "horizontal")))
    elif kind == "fountain":
        draw_pixel_fountain(draw, tile_xywh)
    elif kind == "grave_marker":
        draw_pixel_grave_marker(
            draw,
            tile_xywh,
            style=str(visual.get("marker_style", visual.get("grave_marker_style", "rounded"))),
            stone_rgb=_rgb(visual.get("stone_rgb", (158, 164, 158))),
            mound_rgb=_rgb(visual.get("mound_rgb", (104, 86, 56))),
            flower_rgb=_optional_rgb(visual.get("flower_rgb")),
        )
    elif kind == "hay_bale":
        draw_pixel_hay_bale(draw, tile_xywh)
    elif kind in {"house", "inn", "shop", "tower"}:
        draw_pixel_house(
            draw,
            tile_xywh,
            roof_rgb=_rgb(visual.get("roof_rgb", (179, 72, 55))),
            body_rgb=_rgb(visual.get("body_rgb", (211, 176, 119))),
            variant=kind,
            roof_style=str(visual.get("building_roof_style", visual.get("roof_style", "shingle"))),
            wall_style=str(visual.get("building_wall_style", visual.get("wall_style", "stucco"))),
            door_state=str(visual.get("building_door_state", visual.get("door_state", "closed"))),
            draw_ground_shadow=False,
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "iron_fence":
        draw_pixel_iron_fence(draw, tile_xywh, orientation=str(visual.get("orientation", "horizontal")))
    elif kind == "lamp_post":
        draw_pixel_lamp_post(
            draw,
            tile_xywh,
            glow_rgb=_rgb(visual.get("glow_rgb", (248, 210, 104))),
            metal_rgb=_rgb(visual.get("metal_rgb", (61, 70, 75))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="lamp_post")
    elif kind == "market_stall":
        draw_pixel_market_stall(
            draw,
            tile_xywh,
            canopy_rgb=_rgb(visual.get("canopy_rgb", (194, 72, 66))),
            wood_rgb=_rgb(visual.get("wood_rgb", (129, 80, 45))),
            goods_type=str(visual.get("goods_type", "fruit")),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="market_stall")
    elif kind == "notice_board":
        draw_pixel_notice_board(
            draw,
            tile_xywh,
            board_rgb=_rgb(visual.get("board_rgb", (177, 119, 62))),
            paper_rgb=_rgb(visual.get("paper_rgb", (236, 214, 154))),
        )
    elif kind == "wagon":
        draw_pixel_wagon(
            draw,
            tile_xywh,
            facing=str(visual.get("facing", "right")),
            body_rgb=_rgb(visual.get("body_rgb", (143, 84, 43))),
            cover_rgb=_optional_rgb(visual.get("cover_rgb")),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="wagon")
    elif kind == "well":
        draw_pixel_well(draw, tile_xywh)
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "windmill":
        draw_pixel_windmill(
            draw,
            tile_xywh,
            body_rgb=_rgb(visual.get("body_rgb", (213, 184, 137))),
            roof_rgb=_rgb(visual.get("roof_rgb", (68, 116, 171))),
            blade_pose=str(visual.get("blade_pose", "plus")),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="building")
    elif kind == "gazebo":
        draw_pixel_gazebo(
            draw,
            tile_xywh,
            roof_rgb=_rgb(visual.get("roof_rgb", (154, 82, 74))),
            wood_rgb=_rgb(visual.get("wood_rgb", (123, 83, 48))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="gazebo")
    elif kind == "cave_entrance":
        draw_pixel_cave_entrance(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (92, 92, 86))),
            shadow_rgb=_rgb(visual.get("shadow_rgb", (31, 31, 33))),
        )
    elif kind == "boulder":
        draw_pixel_boulder(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (113, 112, 104))),
        )
    elif kind == "ore_vein":
        draw_pixel_ore_vein(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (103, 101, 94))),
            ore_rgb=_rgb(visual.get("ore_rgb", (218, 171, 71))),
        )
    elif kind == "crystal_cluster":
        draw_pixel_crystal_cluster(
            draw,
            tile_xywh,
            crystal_rgb=_rgb(visual.get("crystal_rgb", (111, 189, 213))),
        )
    elif kind == "stalagmite":
        draw_pixel_stalagmite(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (126, 122, 111))),
        )
    elif kind == "torch":
        draw_pixel_torch(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (112, 72, 42))),
            flame_rgb=_rgb(visual.get("flame_rgb", (244, 153, 45))),
        )
    elif kind == "stone_column":
        draw_pixel_stone_column(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (126, 123, 114))),
        )
    elif kind == "archway":
        draw_pixel_archway(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (120, 118, 110))),
            shadow_rgb=_rgb(visual.get("shadow_rgb", (37, 36, 39))),
        )
    elif kind == "sealed_door":
        draw_pixel_sealed_door(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (134, 87, 45))),
            seal_rgb=_rgb(visual.get("seal_rgb", (94, 74, 55))),
            orientation=str(visual.get("door_orientation", visual.get("orientation", "horizontal"))),
        )
    elif kind == "floor_switch":
        draw_pixel_floor_switch(
            draw,
            tile_xywh,
            plate_rgb=_rgb(visual.get("plate_rgb", (142, 126, 88))),
            switch_state=str(visual.get("switch_state", "raised")),
        )
    elif kind == "brazier":
        draw_pixel_brazier(
            draw,
            tile_xywh,
            metal_rgb=_rgb(visual.get("metal_rgb", (93, 91, 88))),
            flame_rgb=_rgb(visual.get("flame_rgb", (238, 126, 45))),
            fire_state=str(visual.get("fire_state", "lit")),
        )
    elif kind == "broken_wall":
        draw_pixel_broken_wall(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (132, 132, 124))),
            break_style=str(visual.get("break_style", "cracked")),
        )
    elif kind == "rubble":
        draw_pixel_rubble(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (116, 113, 105))),
        )
    elif kind == "magic_circle":
        draw_pixel_magic_circle(
            draw,
            tile_xywh,
            rune_rgb=_rgb(visual.get("rune_rgb", (92, 214, 232))),
            glow_rgb=_rgb(visual.get("glow_rgb", (84, 79, 186))),
        )
    elif kind == "ladder":
        draw_pixel_ladder(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (128, 86, 45))),
            orientation=str(visual.get("orientation", "vertical")),
        )
    elif kind == "mine_cart":
        draw_pixel_mine_cart(
            draw,
            tile_xywh,
            body_rgb=_rgb(visual.get("body_rgb", (93, 93, 92))),
            ore_rgb=_rgb(visual.get("ore_rgb", (198, 145, 58))),
            orientation=str(visual.get("orientation", "horizontal")),
        )
    elif kind == "rail_track":
        draw_pixel_rail_track(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (114, 75, 42))),
            rail_rgb=_rgb(visual.get("rail_rgb", (73, 76, 75))),
            track_shape=str(visual.get("track_shape", "horizontal")),
        )
    elif kind == "wood_support":
        draw_pixel_wood_support(
            draw,
            tile_xywh,
            wood_rgb=_rgb(visual.get("wood_rgb", (128, 78, 43))),
        )
    elif kind == "stairs":
        draw_pixel_stairs(
            draw,
            tile_xywh,
            stone_rgb=_rgb(visual.get("stone_rgb", (112, 112, 106))),
            stair_direction=str(visual.get("stair_direction", "down")),
        )
    elif kind == "rock":
        draw_pixel_rock(draw, tile_xywh)
    elif kind == "scarecrow":
        draw_pixel_scarecrow(
            draw,
            tile_xywh,
            shirt_rgb=_rgb(visual.get("shirt_rgb", (182, 84, 60))),
            hat_rgb=_rgb(visual.get("hat_rgb", (154, 103, 50))),
        )
    elif kind == "sign":
        draw_pixel_sign(draw, tile_xywh)
    elif kind == "statue":
        draw_pixel_statue(
            draw,
            tile_xywh,
            material_rgb=_rgb(visual.get("material_rgb", (145, 151, 148))),
        )
    elif kind == "trough":
        draw_pixel_trough(draw, tile_xywh)
    elif kind == "woodpile":
        draw_pixel_woodpile(
            draw,
            tile_xywh,
            log_rgb=_rgb(visual.get("log_rgb", (135, 82, 45))),
            stack_variant=str(visual.get("stack_variant", "low")),
        )
    elif kind == "pond":
        draw_pixel_pond(
            draw,
            tile_xywh,
            shape=str(visual.get("pond_shape", "round")),
            water_rgb=_rgb(visual.get("water_rgb", (55, 133, 188))),
            rim_rgb=_rgb(visual.get("rim_rgb", (74, 129, 74))),
        )
        _draw_pixel_theme_overlay(draw, tile_xywh, spec, target="pond")
    else:
        raise ValueError(f"unsupported shared pixel object type: {spec.object_type}")


def _draw_pixel_theme_overlay(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    spec: IllustrationObjectSpec,
    *,
    target: str,
) -> None:
    visual = spec.visual_attributes
    theme_id = str(visual.get("theme_id", ""))
    if theme_id == "winter":
        draw_pixel_winter_overlay(
            draw,
            tile_xywh,
            target=str(target),
            snow_rgb=_rgb(visual.get("snow_rgb", (239, 246, 248))),
            shadow_rgb=_rgb(visual.get("snow_shadow_rgb", (178, 202, 215))),
            coverage=float(visual.get("snow_coverage", 0.5)),
            style=str(visual.get("snow_style", "patchy")),
        )
        return
    if theme_id == "autumn":
        draw_pixel_autumn_overlay(
            draw,
            tile_xywh,
            target=str(target),
            leaf_rgb=_rgb(visual.get("leaf_overlay_rgb", visual.get("leaf_rgb", (177, 126, 55)))),
            shadow_rgb=_rgb(visual.get("leaf_shadow_rgb", (104, 82, 47))),
            accent_rgb=_rgb(visual.get("leaf_accent_rgb", (154, 78, 51))),
            coverage=float(visual.get("leaf_coverage", 0.25)),
            style=str(visual.get("leaf_style", "scattered")),
        )
        return


def _rendered_from_record(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> RenderedIllustrationObject:
    record = object_record_for_spec(spec, context)
    return RenderedIllustrationObject(
        object_id=str(spec.object_id),
        object_type=str(record["object_type"]),
        public_name=str(record["public_name"]),
        bbox_xyxy=spec.bbox_xyxy,
        semantic_attributes=dict(record["semantic_attributes"]),
        visual_attributes=dict(record["visual_attributes"]),
        parts=(),
        object_record=record,
        support_items=(),
    )


def _resolved_semantic_attributes(spec: IllustrationObjectSpec) -> dict[str, Any]:
    attrs = dict(spec.semantic_attributes)
    kind = _pixel_kind(spec)
    if kind == "domestic_animal":
        attrs.setdefault("animal_type", _animal_type(spec))
    return attrs


def _resolved_visual_attributes(
    spec: IllustrationObjectSpec,
    context: RenderContext,
) -> dict[str, Any]:
    attrs = {str(key): value for key, value in spec.visual_attributes.items() if str(key) != "object_record"}
    variant_object_type = _variant_object_type(spec)
    if spec.variant_id and object_variant_ids(variant_object_type):
        attrs.update(variant_visual_metadata(variant_object_type, spec.variant_id, str(context.renderer_style)))
    if spec.renderer_id:
        attrs.setdefault("renderer_id", str(spec.renderer_id))
    if spec.renderer_variant_id:
        attrs.setdefault("renderer_variant_id", str(spec.renderer_variant_id))
    attrs.setdefault("renderer_style", str(context.renderer_style))
    if str(context.renderer_style) == RENDERER_STYLE_VECTOR:
        attrs.setdefault("style_id", str(context.style_id))
    _apply_shadow_policy_visual_attributes(attrs, context)
    if _pixel_kind(spec) == "domestic_animal":
        attrs.setdefault("animal_type", _animal_type(spec))
    return attrs


def _apply_shadow_policy_visual_attributes(attrs: dict[str, Any], context: RenderContext) -> None:
    renderer_style = str(context.renderer_style)
    if renderer_style == RENDERER_STYLE_VECTOR:
        style_id = str(attrs.get("style_id", context.style_id))
        attrs.setdefault("style_id", style_id)
        _, _, shadow_enabled = style_outline_params(style_id)
        attrs["shadow_policy"] = "vector_style_decorative"
        attrs["shadow_enabled"] = bool(shadow_enabled)
        attrs["shadow_kind"] = "decorative_oval" if shadow_enabled else "none"
        return
    attrs["shadow_policy"] = "none"
    attrs["shadow_enabled"] = False
    attrs["shadow_kind"] = "none"


def _required_tile(spec: IllustrationObjectSpec) -> TileBox:
    if spec.tile_xywh is None:
        raise ValueError("pixel object rendering requires IllustrationObjectSpec.tile_xywh")
    values = tuple(int(v) for v in spec.tile_xywh)
    if len(values) != 4:
        raise ValueError("tile_xywh must contain exactly four values")
    return values  # type: ignore[return-value]


def _pixel_kind(spec: IllustrationObjectSpec) -> str:
    object_type = str(spec.object_type)
    if object_type in {"animal", "domestic_animal"}:
        return "domestic_animal"
    if object_type in {"person", "pedestrian_with_bag", "worker"}:
        return "person"
    if object_type == "tree":
        return "tree"
    if object_type == "flower":
        return "flower"
    if object_type == "crop_row":
        return "crop_row"
    if object_type in {"bench", "lamp_post", "market_stall", "wagon", "gazebo", "pond"}:
        return object_type
    return object_type


def _record_object_type(spec: IllustrationObjectSpec) -> str:
    kind = _pixel_kind(spec)
    if kind == "domestic_animal":
        return "domestic_animal"
    if kind == "crop_row":
        return "crop_row"
    if kind == "person":
        return str(spec.object_type)
    return str(spec.object_type)


def _record_family(object_type: str, spec: IllustrationObjectSpec) -> str:
    if object_type == "domestic_animal":
        return "animal"
    if object_type == "crop_row":
        return "plant"
    try:
        return family_for_object_type(object_type)
    except Exception:
        return str(spec.semantic_attributes.get("family", "object"))


def _variant_object_type(spec: IllustrationObjectSpec) -> str:
    if _pixel_kind(spec) == "person":
        return "person"
    return str(spec.object_type)


def _public_name(spec: IllustrationObjectSpec) -> str:
    if spec.public_name:
        return str(spec.public_name)
    if _pixel_kind(spec) == "domestic_animal":
        return _animal_type(spec)
    if _pixel_kind(spec) == "crop_row":
        return "crop row"
    try:
        return public_name_for_object_type(str(spec.object_type))
    except Exception:
        return str(spec.object_type).replace("_", " ")


def _animal_type(spec: IllustrationObjectSpec) -> str:
    value = spec.semantic_attributes.get("animal_type", spec.visual_attributes.get("animal_type", spec.public_name))
    return str(value or "sheep")


def _rgb(value: Any) -> RGB:
    if value is None:
        raise ValueError("RGB value cannot be None")
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError("RGB value must contain exactly three channels")
    return values  # type: ignore[return-value]


def _optional_rgb(value: Any) -> RGB | None:
    if value is None:
        return None
    return _rgb(value)


def _shade_rgb(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def sample_vector_object_colors(rng: Any, object_type: str) -> tuple[RGB, RGB]:
    """Compatibility helper for specs that need shared vector colors."""

    return choose_object_colors(rng, str(object_type))


__all__ = [
    "IllustrationObjectSpec",
    "PIXEL_RPG_SHARED_OBJECT_TYPES",
    "RGB",
    "RenderContext",
    "RenderedIllustrationObject",
    "TileBox",
    "make_vector_scene_object_record",
    "object_record_for_spec",
    "render_illustration_object",
    "render_vector_scene_object",
    "sample_vector_object_colors",
    "serialize_rendered_illustration_object",
    "vector_object_record_for_spec",
]
