"""Shared three_d object render specs and renderer dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from PIL import ImageDraw

from . import object_scene_rendering as scene_rendering
from .object_schema import BBox, ThreeDObjectRecord, json_safe
from .object_variants import RENDERER_STYLE_PROJECTED_3D, variant_visual_metadata
from .scene_schema import ThreeDPlacementSpec


RGB = Tuple[int, int, int]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class ThreeDObjectSpec:
    """Renderer-neutral request to draw one reusable three_d object."""

    object_id: str
    object_type: str
    shape_type: str = ""
    public_name: str = ""
    canonical_id: str = ""
    family: str = "object"
    variant_id: str = ""
    renderer_id: str = "object_scene_shape"
    renderer_variant_id: str = ""
    world_xyz: Point3D | None = None
    base_xyz: Point3D | None = None
    dimensions_xyz: Point3D | None = None
    screen_xy: Point2D | None = None
    semantic_attributes: Mapping[str, Any] = field(default_factory=dict)
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    parts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    role: str = "distractor"
    source_entity_type: str = "three_d_object"
    render_spec: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        spec: Mapping[str, Any],
        *,
        object_type_key: str = "shape_type",
        default_renderer_id: str = "object_scene_shape",
        role: str | None = None,
        source_entity_type: str = "three_d_object",
    ) -> "ThreeDObjectSpec":
        """Build a shared object spec from a scene-local mapping."""

        object_type = str(spec.get(object_type_key, spec.get("object_type", spec.get("shape_type", ""))))
        shape_type = str(spec.get("shape_type", object_type))
        public_name = str(spec.get("prompt_name", spec.get("object_name", object_type.replace("_", " "))))
        return cls(
            object_id=str(spec.get("object_id", "")),
            object_type=object_type,
            shape_type=shape_type,
            public_name=public_name,
            canonical_id=str(spec.get("canonical_id", spec.get("profile_id", object_type))),
            family=str(spec.get("family", spec.get("resource_kind", "object"))),
            variant_id=str(spec.get("variant_id", spec.get("object_variant_id", ""))),
            renderer_id=str(spec.get("renderer_id", default_renderer_id)),
            renderer_variant_id=str(spec.get("renderer_variant_id", "")),
            world_xyz=_point3(spec.get("world_xyz")),
            base_xyz=_point3(spec.get("base_xyz")),
            dimensions_xyz=_point3(spec.get("dimensions_xyz")),
            screen_xy=_point2(spec.get("screen_xy")),
            semantic_attributes=_semantic_attributes_from_mapping(spec),
            visual_attributes=_visual_attributes_from_mapping(spec),
            parts=tuple(dict(part) for part in spec.get("parts", ())),
            role=str(role if role is not None else spec.get("object_role", "distractor")),
            source_entity_type=str(source_entity_type),
            render_spec=dict(spec),
        )

    @classmethod
    def from_placement(
        cls,
        placement: ThreeDPlacementSpec | Mapping[str, Any],
        *,
        object_type_key: str = "object_type",
        default_renderer_id: str = "object_scene_shape",
        role: str | None = None,
        source_entity_type: str = "three_d_object",
    ) -> "ThreeDObjectSpec":
        """Build an object spec from a normalized placement without a registry profile."""

        resolved = (
            placement
            if isinstance(placement, ThreeDPlacementSpec)
            else ThreeDPlacementSpec.from_mapping(
                placement,
                object_type_key=object_type_key,
                role=role,
                source_entity_type=source_entity_type,
            )
        )
        mapping = resolved.as_render_mapping()
        return cls.from_mapping(
            mapping,
            object_type_key=object_type_key,
            default_renderer_id=str(mapping.get("renderer_id", default_renderer_id)),
            role=role if role is not None else resolved.role,
            source_entity_type=source_entity_type,
        )

    @classmethod
    def from_profile_and_placement(
        cls,
        profile: Any,
        placement: ThreeDPlacementSpec | Mapping[str, Any],
        *,
        object_type_key: str = "object_type",
        role: str | None = None,
        source_entity_type: str = "three_d_object",
    ) -> "ThreeDObjectSpec":
        """Build an object spec from centralized profile identity plus scene placement."""

        resolved = (
            placement
            if isinstance(placement, ThreeDPlacementSpec)
            else ThreeDPlacementSpec.from_mapping(
                placement,
                object_type_key=object_type_key,
                role=role,
                source_entity_type=source_entity_type,
            )
        )
        mapping = resolved.as_render_mapping(profile=profile)
        object_type = str(mapping.get("object_type", getattr(profile, "object_type", "")))
        shape_type = str(mapping.get("shape_type", object_type))
        semantic_attributes = {
            **_semantic_attributes_from_mapping(mapping),
            **dict(resolved.semantic_attributes),
        }
        visual_attributes = {
            **_visual_attributes_from_mapping(mapping),
            **dict(resolved.visual_attributes),
        }
        return cls(
            object_id=str(mapping.get("object_id", resolved.object_id)),
            object_type=object_type,
            shape_type=shape_type,
            public_name=str(mapping.get("prompt_name", getattr(profile, "display_name", object_type.replace("_", " ")))),
            canonical_id=str(mapping.get("canonical_id", getattr(profile, "canonical_id", object_type))),
            family=str(mapping.get("family", getattr(profile, "resource_kind", "object"))),
            variant_id=str(mapping.get("variant_id", mapping.get("object_variant_id", resolved.variant_id))),
            renderer_id=str(mapping.get("renderer_id", getattr(profile, "renderer", "object_scene_shape"))),
            renderer_variant_id=str(mapping.get("renderer_variant_id", resolved.renderer_variant_id)),
            world_xyz=_point3(mapping.get("world_xyz")),
            base_xyz=_point3(mapping.get("base_xyz")),
            dimensions_xyz=_point3(mapping.get("dimensions_xyz")),
            screen_xy=_point2(mapping.get("screen_xy")),
            semantic_attributes=semantic_attributes,
            visual_attributes=visual_attributes,
            parts=tuple(dict(part) for part in mapping.get("parts", resolved.parts)),
            role=str(role if role is not None else resolved.role),
            source_entity_type=str(source_entity_type),
            render_spec=dict(mapping),
        )


@dataclass(frozen=True)
class ThreeDRenderContext:
    """Renderer-specific drawing state for one object dispatch call."""

    draw: ImageDraw.ImageDraw
    camera: Any
    frame: Any
    render_params: Any | None = None
    fill_rgb: RGB | None = None
    renderer_style: str = RENDERER_STYLE_PROJECTED_3D
    scene_variant: str = ""
    floor_rgb: RGB = (232, 239, 242)


@dataclass(frozen=True)
class RenderedThreeDObject:
    """Trace-ready result returned by shared object rendering."""

    object_id: str
    object_type: str
    shape_type: str
    public_name: str
    bbox_xyxy: BBox
    center_xy: Point2D | None
    semantic_attributes: Mapping[str, Any]
    visual_attributes: Mapping[str, Any]
    parts: Tuple[Mapping[str, Any], ...]
    object_record: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": str(self.object_id),
            "object_type": str(self.object_type),
            "shape_type": str(self.shape_type),
            "public_name": str(self.public_name),
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "center_xy": [round(float(value), 3) for value in self.center_xy] if self.center_xy is not None else None,
            "semantic_attributes": json_safe(self.semantic_attributes),
            "visual_attributes": json_safe(self.visual_attributes),
            "parts": [json_safe(part) for part in self.parts],
            "object_record": json_safe(self.object_record),
        }


def render_three_d_object(
    spec: ThreeDObjectSpec | Mapping[str, Any],
    context: ThreeDRenderContext,
) -> RenderedThreeDObject:
    """Draw one three_d object through the renderer selected by ``context``."""

    resolved_spec = spec if isinstance(spec, ThreeDObjectSpec) else ThreeDObjectSpec.from_mapping(spec)
    renderer_style = str(context.renderer_style)
    if renderer_style != RENDERER_STYLE_PROJECTED_3D:
        raise ValueError(f"unsupported three_d renderer style: {renderer_style}")
    draw_spec = _draw_spec_for_object(resolved_spec)
    bbox = _draw_projected_object_by_renderer(resolved_spec, context, draw_spec)
    visual_attributes = _resolved_visual_attributes(resolved_spec, context)
    semantic_attributes = _resolved_semantic_attributes(resolved_spec)
    center_xy = resolved_spec.screen_xy
    object_record = record_for_three_d_object(
        resolved_spec,
        bbox_xyxy=bbox,
        center_xy=center_xy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
    )
    return RenderedThreeDObject(
        object_id=str(resolved_spec.object_id),
        object_type=str(resolved_spec.object_type),
        shape_type=str(resolved_spec.shape_type or resolved_spec.object_type),
        public_name=_public_name(resolved_spec),
        bbox_xyxy=_bbox_tuple(bbox),
        center_xy=center_xy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        parts=tuple(dict(part) for part in resolved_spec.parts),
        object_record=object_record,
    )


def rendered_three_d_object_from_bbox(
    spec: ThreeDObjectSpec | Mapping[str, Any],
    context: ThreeDRenderContext,
    *,
    bbox_xyxy: Sequence[float],
) -> RenderedThreeDObject:
    """Build a normalized rendered object payload for a scene-local draw call."""

    resolved_spec = spec if isinstance(spec, ThreeDObjectSpec) else ThreeDObjectSpec.from_mapping(spec)
    bbox = _bbox_tuple(bbox_xyxy)
    visual_attributes = _resolved_visual_attributes(resolved_spec, context)
    semantic_attributes = _resolved_semantic_attributes(resolved_spec)
    center_xy = resolved_spec.screen_xy
    object_record = record_for_three_d_object(
        resolved_spec,
        bbox_xyxy=bbox,
        center_xy=center_xy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
    )
    return RenderedThreeDObject(
        object_id=str(resolved_spec.object_id),
        object_type=str(resolved_spec.object_type),
        shape_type=str(resolved_spec.shape_type or resolved_spec.object_type),
        public_name=_public_name(resolved_spec),
        bbox_xyxy=bbox,
        center_xy=center_xy,
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        parts=tuple(dict(part) for part in resolved_spec.parts),
        object_record=object_record,
    )


def record_for_three_d_object(
    spec: ThreeDObjectSpec,
    *,
    bbox_xyxy: Sequence[float] | None = None,
    center_xy: Sequence[float] | None = None,
    semantic_attributes: Mapping[str, Any] | None = None,
    visual_attributes: Mapping[str, Any] | None = None,
    role: str | None = None,
    source_entity_type: str | None = None,
) -> dict[str, Any]:
    """Build a normalized object record for one reusable three_d object spec."""

    record = ThreeDObjectRecord(
        object_id=str(spec.object_id),
        object_type=str(spec.object_type),
        canonical_id=str(spec.canonical_id or spec.object_type),
        public_name=_public_name(spec),
        family=str(spec.family or "object"),
        bbox_xyxy=_bbox_tuple(bbox_xyxy) if bbox_xyxy is not None else None,
        center_xy=_point2(center_xy),
        world_xyz=spec.world_xyz,
        base_xyz=spec.base_xyz,
        dimensions_xyz=spec.dimensions_xyz,
        semantic_attributes=dict(semantic_attributes if semantic_attributes is not None else _resolved_semantic_attributes(spec)),
        visual_attributes=dict(visual_attributes if visual_attributes is not None else spec.visual_attributes),
        role=str(role if role is not None else spec.role),
        source_entity_type=str(source_entity_type if source_entity_type is not None else spec.source_entity_type),
        parts=tuple(dict(part) for part in spec.parts),
    )
    return record.as_dict()


def _draw_projected_object_by_renderer(
    spec: ThreeDObjectSpec,
    context: ThreeDRenderContext,
    draw_spec: Mapping[str, Any],
) -> BBox:
    renderer_id = str(spec.renderer_id or "object_scene_shape")
    if renderer_id == "object_scene_shape":
        return _draw_projected_object_shape(
            context.draw,
            draw_spec,
            shape_type=str(spec.shape_type or spec.object_type),
            camera=context.camera,
            frame=context.frame,
            fill=_rgb(context.fill_rgb),
            floor_rgb=_rgb(context.floor_rgb),
        )
    if renderer_id == "room_wall_object":
        return _draw_room_wall_object(context, draw_spec)
    if renderer_id == "room_floor_object":
        return _draw_room_floor_object(context, draw_spec)
    if renderer_id == "street_object":
        return _draw_street_object(context, draw_spec)
    if renderer_id == "warehouse_object":
        return _draw_warehouse_object(context, draw_spec)
    raise ValueError(f"unsupported three_d object renderer_id: {renderer_id}")


def _draw_room_wall_object(context: ThreeDRenderContext, spec: Mapping[str, Any]) -> BBox:
    from .room_wall_object_rendering import _draw_wall_object

    return _bbox_tuple(_draw_wall_object(context.draw, spec, camera=context.camera, frame=context.frame))


def _draw_room_floor_object(context: ThreeDRenderContext, spec: Mapping[str, Any]) -> BBox:
    from .room_floor_object_rendering import _draw_floor_object

    return _bbox_tuple(_draw_floor_object(context.draw, spec, camera=context.camera, frame=context.frame))


def _draw_street_object(context: ThreeDRenderContext, spec: Mapping[str, Any]) -> BBox:
    from .street_object_rendering import _draw_candidate_object, _draw_context_object

    object_role = str(spec.get("object_role", ""))
    if bool(spec.get("is_answer_candidate", False)) or object_role in {"street_candidate", "street_reference"}:
        bbox = _draw_candidate_object(context.draw, spec, camera=context.camera, frame=context.frame)
    else:
        bbox = _draw_context_object(context.draw, spec, camera=context.camera, frame=context.frame)
    return _bbox_tuple(bbox)


def _draw_warehouse_object(context: ThreeDRenderContext, spec: Mapping[str, Any]) -> BBox:
    from .object_resources import (
        WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB,
        WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE,
    )
    from .warehouse_object_rendering import _draw_warehouse_object as _draw_native_warehouse_object
    from .warehouse_object_rendering import _fill_for_object

    if str(spec.get("object_type")) == WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE:
        fill = _rgb(context.fill_rgb, WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB)
        return _bbox_tuple(scene_rendering._draw_sphere_object(context.draw, spec, camera=context.camera, frame=context.frame, fill=fill))
    fill = _rgb(context.fill_rgb, _fill_for_object(spec, scene_variant=str(context.scene_variant)))
    return _bbox_tuple(_draw_native_warehouse_object(context.draw, spec, camera=context.camera, frame=context.frame, fill=fill))


def _draw_projected_object_shape(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    shape_type: str,
    camera: Any,
    frame: Any,
    fill: RGB,
    floor_rgb: RGB,
) -> BBox:
    shape_type = str(shape_type)
    if shape_type == "sphere":
        shape_bbox = scene_rendering._draw_sphere_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cylinder":
        shape_bbox = scene_rendering._draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cone":
        shape_bbox = scene_rendering._draw_cone_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "arrow":
        shape_bbox = scene_rendering._draw_footprint_prism_object(
            draw,
            spec,
            camera=camera,
            frame=frame,
            fill=fill,
            footprint_xy=scene_rendering._arrow_footprint_points(),
        )
    elif shape_type == "sword":
        shape_bbox = scene_rendering._draw_sword_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "shield":
        shape_bbox = scene_rendering._draw_shield_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "diamond":
        shape_bbox = scene_rendering._draw_diamond_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "heart":
        shape_bbox = scene_rendering._draw_heart_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "key":
        shape_bbox = scene_rendering._draw_key_object(draw, spec, camera=camera, frame=frame, fill=fill, floor_rgb=floor_rgb)
    elif shape_type == "crown":
        shape_bbox = scene_rendering._draw_crown_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "anchor":
        shape_bbox = scene_rendering._draw_anchor_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "horseshoe":
        shape_bbox = scene_rendering._draw_horseshoe_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "hammer":
        shape_bbox = scene_rendering._draw_hammer_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "gear":
        shape_bbox = scene_rendering._draw_footprint_prism_object(
            draw,
            spec,
            camera=camera,
            frame=frame,
            fill=fill,
            footprint_xy=scene_rendering._gear_footprint_points(),
        )
    elif shape_type == "bell":
        shape_bbox = scene_rendering._draw_bell_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "trophy":
        shape_bbox = scene_rendering._draw_trophy_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "open_book":
        shape_bbox = scene_rendering._draw_open_book_object(draw, spec, camera=camera, frame=frame)
    elif shape_type == "mushroom":
        shape_bbox = scene_rendering._draw_mushroom_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "lantern":
        shape_bbox = scene_rendering._draw_lantern_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "candle":
        shape_bbox = scene_rendering._draw_candle_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "goblet":
        shape_bbox = scene_rendering._draw_goblet_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "mail_envelope":
        shape_bbox = scene_rendering._draw_mail_envelope_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "compass":
        shape_bbox = scene_rendering._draw_compass_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "flask":
        shape_bbox = scene_rendering._draw_flask_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "clock":
        shape_bbox = scene_rendering._draw_clock_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "apple":
        shape_bbox = scene_rendering._draw_apple_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "carrot":
        shape_bbox = scene_rendering._draw_carrot_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "fish":
        shape_bbox = scene_rendering._draw_fish_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "leaf":
        shape_bbox = scene_rendering._draw_leaf_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "glove":
        shape_bbox = scene_rendering._draw_glove_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "hat":
        shape_bbox = scene_rendering._draw_hat_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "helmet":
        shape_bbox = scene_rendering._draw_helmet_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cup":
        shape_bbox = scene_rendering._draw_cup_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bottle":
        shape_bbox = scene_rendering._draw_bottle_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "umbrella":
        shape_bbox = scene_rendering._draw_umbrella_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "calculator":
        shape_bbox = scene_rendering._draw_calculator_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "dice":
        shape_bbox = scene_rendering._draw_dice_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "kite":
        shape_bbox = scene_rendering._draw_kite_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cactus":
        shape_bbox = scene_rendering._draw_cactus_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "drum":
        shape_bbox = scene_rendering._draw_drum_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "ruler":
        shape_bbox = scene_rendering._draw_ruler_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "remote_control":
        shape_bbox = scene_rendering._draw_remote_control_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "plug":
        shape_bbox = scene_rendering._draw_plug_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "torus":
        shape_bbox = scene_rendering._draw_torus_object(draw, spec, camera=camera, frame=frame, fill=fill, floor_rgb=floor_rgb)
    elif shape_type == "pyramid":
        shape_bbox = scene_rendering._draw_pyramid_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "wedge":
        shape_bbox = scene_rendering._draw_wedge_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "star_prism":
        shape_bbox = scene_rendering._draw_footprint_prism_object(
            draw,
            spec,
            camera=camera,
            frame=frame,
            fill=fill,
            footprint_xy=scene_rendering._star_footprint_points(),
        )
    elif shape_type == "hexagonal_prism":
        shape_bbox = scene_rendering._draw_footprint_prism_object(
            draw,
            spec,
            camera=camera,
            frame=frame,
            fill=fill,
            footprint_xy=scene_rendering._hexagon_footprint_points(),
        )
    elif shape_type == "half_cylinder":
        shape_bbox = scene_rendering._draw_half_cylinder_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"pen", "pencil", "highlighter", "marker", "sharpie"}:
        shape_bbox = scene_rendering._draw_pencil_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "thumb_pin":
        shape_bbox = scene_rendering._draw_thumb_pin_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"card", "bookmark", "sachet", "packet", "small_box", "towel", "adhesive_pad", "furniture_slider"}:
        shape_bbox = scene_rendering._draw_flat_rect_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"ticket", "tag"}:
        shape_bbox = scene_rendering._draw_ticket_tag_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "puzzle_piece":
        shape_bbox = scene_rendering._draw_puzzle_piece_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "candy_disc":
        shape_bbox = scene_rendering._draw_candy_disc_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cd":
        shape_bbox = scene_rendering._draw_cd_object(draw, spec, camera=camera, frame=frame, floor_rgb=floor_rgb)
    elif shape_type == "berry":
        shape_bbox = scene_rendering._draw_berry_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "marble":
        shape_bbox = scene_rendering._draw_marble_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bead":
        shape_bbox = scene_rendering._draw_bead_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "dot":
        shape_bbox = scene_rendering._draw_dot_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "button":
        shape_bbox = scene_rendering._draw_button_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "plate":
        shape_bbox = scene_rendering._draw_plate_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "screw":
        shape_bbox = scene_rendering._draw_screw_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"bolt", "socket_adapter"}:
        shape_bbox = scene_rendering._draw_bolt_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "washer":
        shape_bbox = scene_rendering._draw_washer_object(draw, spec, camera=camera, frame=frame, fill=fill, floor_rgb=floor_rgb)
    elif shape_type == "paper_clip":
        shape_bbox = scene_rendering._draw_paper_clip_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "u_bolt":
        shape_bbox = scene_rendering._draw_u_bolt_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "nail":
        shape_bbox = scene_rendering._draw_nail_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "rod":
        shape_bbox = scene_rendering._draw_rod_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"stick", "cinnamon_stick"}:
        shape_bbox = scene_rendering._draw_stick_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "tube":
        shape_bbox = scene_rendering._draw_tube_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "clip":
        shape_bbox = scene_rendering._draw_clip_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "socket":
        shape_bbox = scene_rendering._draw_socket_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "magnet":
        shape_bbox = scene_rendering._draw_magnet_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "heater":
        shape_bbox = scene_rendering._draw_heater_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"flower", "bay_leaf", "clove", "cardamom", "pepper"}:
        shape_bbox = scene_rendering._draw_flower_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"plant_pot", "flower_pot", "pot"}:
        shape_bbox = scene_rendering._draw_plant_pot_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "glass":
        shape_bbox = scene_rendering._draw_glass_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "jar":
        shape_bbox = scene_rendering._draw_jar_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "can":
        shape_bbox = scene_rendering._draw_can_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "lid":
        shape_bbox = scene_rendering._draw_lid_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"pillow", "cushion"}:
        shape_bbox = scene_rendering._draw_pillow_cushion_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "stool":
        shape_bbox = scene_rendering._draw_stool_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"drawer", "locker_box"}:
        shape_bbox = scene_rendering._draw_drawer_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cap":
        shape_bbox = scene_rendering._draw_cap_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bucket":
        shape_bbox = scene_rendering._draw_bucket_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "tray":
        shape_bbox = scene_rendering._draw_tray_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "coaster":
        shape_bbox = scene_rendering._draw_coaster_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "rose":
        shape_bbox = scene_rendering._draw_rose_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "banana":
        shape_bbox = scene_rendering._draw_banana_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "tomato":
        shape_bbox = scene_rendering._draw_tomato_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"peanut", "almond"}:
        shape_bbox = scene_rendering._draw_peanut_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "coffee_bean":
        shape_bbox = scene_rendering._draw_coffee_bean_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "hook":
        shape_bbox = scene_rendering._draw_hook_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bracket":
        shape_bbox = scene_rendering._draw_bracket_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "battery":
        shape_bbox = scene_rendering._draw_battery_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "tape_roll":
        shape_bbox = scene_rendering._draw_tape_roll_object(draw, spec, camera=camera, frame=frame, fill=fill, floor_rgb=floor_rgb)
    elif shape_type == "bag":
        shape_bbox = scene_rendering._draw_bag_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "hex_nut":
        shape_bbox = scene_rendering._draw_hex_nut_object(draw, spec, camera=camera, frame=frame, fill=fill, floor_rgb=floor_rgb)
    elif shape_type == "fork":
        shape_bbox = scene_rendering._draw_fork_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "spoon":
        shape_bbox = scene_rendering._draw_spoon_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "knife":
        shape_bbox = scene_rendering._draw_knife_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bowl":
        shape_bbox = scene_rendering._draw_bowl_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "basket":
        shape_bbox = scene_rendering._draw_basket_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "dumbbell":
        shape_bbox = scene_rendering._draw_dumbbell_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "chess_piece":
        shape_bbox = scene_rendering._draw_chess_piece_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "hanger":
        shape_bbox = scene_rendering._draw_hanger_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "light_bulb":
        shape_bbox = scene_rendering._draw_light_bulb_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "egg":
        shape_bbox = scene_rendering._draw_egg_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "chili":
        shape_bbox = scene_rendering._draw_chili_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "paint_brush":
        shape_bbox = scene_rendering._draw_paint_brush_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "paint_roller":
        shape_bbox = scene_rendering._draw_paint_roller_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "arch":
        shape_bbox = scene_rendering._draw_arch_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"table", "mini_table"}:
        shape_bbox = scene_rendering._draw_table_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "shelf":
        shape_bbox = scene_rendering._draw_shelf_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "open_box":
        shape_bbox = scene_rendering._draw_open_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "refrigerator":
        shape_bbox = scene_rendering._draw_refrigerator_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "washing_machine":
        shape_bbox = scene_rendering._draw_washing_machine_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "vending_machine":
        shape_bbox = scene_rendering._draw_vending_machine_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "trash_bin":
        shape_bbox = scene_rendering._draw_trash_bin_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "bench":
        shape_bbox = scene_rendering._draw_bench_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "piano":
        shape_bbox = scene_rendering._draw_piano_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "locker":
        shape_bbox = scene_rendering._draw_locker_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "cabinet":
        shape_bbox = scene_rendering._draw_cabinet_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "sofa":
        shape_bbox = scene_rendering._draw_sofa_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type == "barrel":
        shape_bbox = scene_rendering._draw_barrel_object(draw, spec, camera=camera, frame=frame, fill=fill)
    elif shape_type in {"chair", "mini_chair"}:
        shape_bbox = scene_rendering._draw_chair_object(draw, spec, camera=camera, frame=frame, fill=fill)
    else:
        shape_bbox = scene_rendering._draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    return _bbox_tuple(shape_bbox)


def _draw_spec_for_object(spec: ThreeDObjectSpec) -> dict[str, Any]:
    draw_spec = dict(spec.render_spec)
    draw_spec.setdefault("object_id", str(spec.object_id))
    draw_spec.setdefault("shape_type", str(spec.shape_type or spec.object_type))
    if spec.world_xyz is not None:
        draw_spec["world_xyz"] = tuple(float(value) for value in spec.world_xyz)
    if spec.base_xyz is not None:
        draw_spec["base_xyz"] = tuple(float(value) for value in spec.base_xyz)
    if spec.dimensions_xyz is not None:
        draw_spec["dimensions_xyz"] = tuple(float(value) for value in spec.dimensions_xyz)
    if spec.screen_xy is not None:
        draw_spec["screen_xy"] = tuple(float(value) for value in spec.screen_xy)
    return draw_spec


def _semantic_attributes_from_mapping(spec: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "shape_type",
        "object_name",
        "prompt_name",
        "nameable_for_prompt",
        "object_role",
        "is_answer_candidate",
        "is_countable_object",
        "matches_query",
        "count_role",
        "point_label",
        "resource_kind",
        "support_type",
        "mounting",
        "is_wall_mounted",
        "wall",
        "support_object_id",
        "support_surface_type",
        "adjacent_wall",
        "wall_axis_interval",
        "wall_gap",
        "road_arm",
        "lane_id",
        "travel_direction_vector_xy",
        "is_in_forward_path_corridor",
        "is_first_reached_object",
        "forward_distance_from_robot",
        "lateral_offset_from_robot",
        "rack_id",
        "rack_color_name",
        "rack_color_label",
        "shelf_level",
        "shelf_level_index",
        "color_name",
        "prompt_color_name",
    )
    return {key: json_safe(spec[key]) for key in keys if key in spec}


def _visual_attributes_from_mapping(spec: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "fill_rgb",
        "dimensions_xyz",
        "dimension_scale",
        "renderer_id",
        "renderer_variant_id",
        "render_style",
        "color_id",
        "material_id",
        "scene_variant",
        "building_style",
        "building_style_name",
        "orientation_axis",
        "orientation_deg",
        "scenery_variant",
        "picture_content",
        "shelf_style",
        "shelf_levels",
        "shelf_frame_rgb",
        "shelf_height_scale",
        "robot_design",
        "robot_base_rgb",
        "robot_accent_rgb",
    )
    return {key: json_safe(spec[key]) for key in keys if key in spec}


def _resolved_semantic_attributes(spec: ThreeDObjectSpec) -> dict[str, Any]:
    semantic = dict(spec.semantic_attributes)
    semantic.setdefault("shape_type", str(spec.shape_type or spec.object_type))
    return semantic


def _resolved_visual_attributes(spec: ThreeDObjectSpec, context: ThreeDRenderContext) -> dict[str, Any]:
    visual = dict(spec.visual_attributes)
    visual.update(
        variant_visual_metadata(
            str(spec.object_type),
            variant_id=str(spec.variant_id),
            renderer_id=str(spec.renderer_id),
            renderer_style=str(context.renderer_style),
            renderer_variant_id=str(spec.renderer_variant_id),
        )
    )
    if context.fill_rgb is not None:
        visual["fill_rgb"] = [int(channel) for channel in _rgb(context.fill_rgb)]
    if spec.dimensions_xyz is not None:
        visual["dimensions_xyz"] = [round(float(value), 3) for value in spec.dimensions_xyz]
    if context.scene_variant:
        visual["scene_variant"] = str(context.scene_variant)
    return visual


def _public_name(spec: ThreeDObjectSpec) -> str:
    return str(spec.public_name or spec.object_type.replace("_", " "))


def _rgb(value: Any, fallback: RGB = (72, 115, 166)) -> RGB:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("#") and len(value) == 7:
            try:
                return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
            except ValueError:
                return tuple(int(channel) for channel in fallback)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        channels = [int(round(float(channel))) for channel in value]
        if len(channels) >= 3:
            return (
                max(0, min(255, channels[0])),
                max(0, min(255, channels[1])),
                max(0, min(255, channels[2])),
            )
    return tuple(int(channel) for channel in fallback)


def _bbox_tuple(value: Sequence[float]) -> BBox:
    values = [float(item) for item in value]
    if len(values) != 4:
        raise ValueError("bbox must contain exactly four values")
    return (values[0], values[1], values[2], values[3])


def _point2(value: Any) -> Point2D | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) != 2:
        raise ValueError("2D point must contain exactly two values")
    return (values[0], values[1])


def _point3(value: Any) -> Point3D | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) != 3:
        raise ValueError("3D point must contain exactly three values")
    return (values[0], values[1], values[2])


__all__ = [
    "RGB",
    "RenderedThreeDObject",
    "ThreeDObjectSpec",
    "ThreeDRenderContext",
    "record_for_three_d_object",
    "rendered_three_d_object_from_bbox",
    "render_three_d_object",
]
