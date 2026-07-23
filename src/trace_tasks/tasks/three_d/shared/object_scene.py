"""Shared object-scene assembly for synthetic three_d spatial tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ...shared.color_distance import coerce_rgb as _rgb
from ...shared.text_rendering import load_font
from .task_support import float_value as _float_value
from .task_support import int_value as _int_value
from .visual_styles import resolve_conveyor_belt_style, resolve_three_d_surface_tone
from .color_variation import resolve_three_d_object_fill_rgb
from .camera_projection import (
    CameraSpec as _CameraSpec,
    ProjectionFrame as _ProjectionFrame,
    build_projection_frame as _build_projection_frame,
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    dedupe_line_points as _dedupe_line_points,
    distance as _distance,
    grid_values_for_range as _grid_values_for_range,
    min_pairwise as _min_pairwise,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_normalized as _project_normalized,
    project_screen as _project_screen,
    project_xy as _project_xy,
    sample_camera as _sample_camera,
    screen_to_floor_xy as _screen_to_floor_xy,
    screen_to_normalized as _screen_to_normalized,
    stage_reference_points as _stage_reference_points,
    vec_cross as _vec_cross,
    vec_dot as _vec_dot,
    vec_norm as _vec_norm,
    vec_sub as _vec_sub,
)
from .canvas import (
    bbox_dict_transform,
    bbox_transform,
    entities_transform,
    final_canvas_metadata,
    point_dict_transform,
    render_params_canvas_metadata,
    resize_image_to_fit_pixel_cap,
    resolve_three_d_canvas_spec,
)
from .object_rendering import ThreeDObjectSpec, ThreeDRenderContext, render_three_d_object
from .scene_schema import ThreeDPlacementSpec
from .object_scene_rendering import _bbox_union, _draw_line, _draw_option_label
from ..object_scene.shared.rendering import draw_object_scene_room
from .option_panel import append_text_option_panel, empty_option_panel_metadata
from .projected_object_geometry import (
    _bbox_intersection_area,
    _object_reference_points,
    _object_screen_bbox,
    bbox_intersection_area,
    object_reference_points,
    object_screen_bbox,
)
from .object_resources import (
    OBJECT_SCENE_ID,
    OBJECT_SCENE_CONTEXT_DIMENSIONS,
    OBJECT_SCENE_CONTEXT_SHAPE_TYPES,
    OBJECT_SCENE_NAME_BY_SHAPE_TYPE,
    OBJECT_SCENE_NAMED_CANDIDATE_SHAPE_TYPES,
    OBJECT_SCENE_SHAPE_TYPES,
    OBJECT_SCENE_SMALL_DIMENSIONS,
    OBJECT_SCENE_SMALL_SHAPE_TYPES,
    object_profile_by_id,
    object_profile_or_none,
)

SCENE_ID = "object_scene"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("floor_grid_room", "tabletop_room", "studio_platform")
POINT_LABELS: Tuple[str, ...] = tuple("ABCDEFGH")
SMALL_OBJECT_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_SMALL_SHAPE_TYPES
NAMED_SMALL_OBJECT_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_NAMED_CANDIDATE_SHAPE_TYPES
LARGE_CONTEXT_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_CONTEXT_SHAPE_TYPES
SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_SHAPE_TYPES
OBJECT_NAME_BY_SHAPE_TYPE: Dict[str, str] = dict(OBJECT_SCENE_NAME_BY_SHAPE_TYPE)
NAMEABLE_CONTEXT_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_CONTEXT_SHAPE_TYPES
POINT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (224, 71, 61),
    (59, 122, 221),
    (56, 166, 103),
    (153, 82, 205),
    (232, 154, 44),
    (42, 170, 188),
    (213, 78, 139),
    (119, 150, 58),
    (237, 103, 55),
    (83, 104, 216),
    (48, 178, 150),
    (177, 93, 67),
)
POINT_COLOR_BY_LABEL: Dict[str, Tuple[int, int, int]] = {
    str(label): tuple(int(channel) for channel in POINT_COLORS[int(index)])
    for index, label in enumerate(POINT_LABELS)
    if int(index) < len(POINT_COLORS)
}
CONTEXT_OBJECT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (150, 105, 72),
    (91, 128, 159),
    (109, 143, 88),
    (157, 101, 139),
    (185, 137, 61),
    (87, 151, 149),
    (164, 91, 74),
    (111, 119, 153),
    (130, 142, 67),
    (177, 111, 111),
    (119, 100, 158),
    (102, 139, 121),
)
CAMERA_YAW_BANDS_DEGREES: Tuple[Tuple[float, float], ...] = (
    (-145.0, -108.0),
    (-82.0, -48.0),
    (-42.0, -20.0),
    (20.0, 42.0),
    (48.0, 82.0),
    (108.0, 145.0),
)


def _object_name(shape_type: str) -> str:
    return str(OBJECT_NAME_BY_SHAPE_TYPE.get(str(shape_type), str(shape_type).replace("_", " ")))


def _nameable_for_prompt(shape_type: str, *, object_role: str) -> bool:
    if str(object_role) == "context":
        return str(shape_type) in set(NAMEABLE_CONTEXT_SHAPE_TYPES)
    return str(shape_type) in set(NAMED_SMALL_OBJECT_SHAPE_TYPES)


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    room_extent: float
    room_height: float
    grid_step: float
    marker_radius_px: int
    label_font_size_px: int
    line_width_px: int
    floor_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    full_bleed_floor: bool
    full_bleed_floor_extent_multiplier: float
    background_tone_id: str = "custom"
    background_tone_rgb: Tuple[int, int, int] = (232, 239, 242)
    surface_accent_rgb: Tuple[int, int, int] = (214, 221, 219)
    conveyor_belt_style_id: str | None = None
    conveyor_belt_fill_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_fill_alt_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_fill_secondary_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_outline_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_outline_secondary_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_rail_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_arrow_rgb: Tuple[int, int, int] | None = None
    conveyor_belt_shadow_rgb: Tuple[int, int, int] | None = None
    canvas_preset: str = "explicit"
    canvas_policy: str = "explicit_dimensions"


ObjectSceneRenderParams = _RenderParams


@dataclass(frozen=True)
class _RenderedScene:
    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    point_bboxes_px: Dict[str, List[float]]
    point_centers_px: Dict[str, List[float]]
    object_bboxes_px: Dict[str, List[float]]
    object_centers_px: Dict[str, List[float]]
    context_object_bboxes_px: Dict[str, List[float]]
    context_object_centers_px: Dict[str, List[float]]
    room_bbox_px: List[float]
    annotation_bboxes: List[List[float]]
    annotation_entity_ids: List[str]
    option_panel_bbox_px: List[float]
    option_choice_bboxes_px: Dict[str, List[float]]
    option_choices: List[Dict[str, Any]]
    option_panel_height_px: int






def _bool_value(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(str(key), bool(default))
    if isinstance(value, str):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int = 0,
    namespace: str = "three_d.object_scene.canvas",
) -> _RenderParams:
    merged = dict(render_defaults)
    merged.update(dict(params))
    canvas = resolve_three_d_canvas_spec(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        fallback_width=_int_value(merged, "canvas_width", 1200),
        fallback_height=_int_value(merged, "canvas_height", 800),
    )
    tone = resolve_three_d_surface_tone(
        params=params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.surface_tone",
    )
    belt_style = (
        resolve_conveyor_belt_style(
            params=params,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.conveyor_belt_style",
        )
        if _bool_value(merged, "enable_conveyor_belt_styles", False)
        else None
    )
    return _RenderParams(
        canvas_width=int(canvas.canvas_width),
        canvas_height=int(canvas.canvas_height),
        canvas_preset=str(canvas.preset_id),
        canvas_policy=str(canvas.policy),
        scene_margin_left_px=_int_value(merged, "scene_margin_left_px", 70),
        scene_margin_right_px=_int_value(merged, "scene_margin_right_px", 70),
        scene_margin_top_px=_int_value(merged, "scene_margin_top_px", 54),
        scene_margin_bottom_px=_int_value(merged, "scene_margin_bottom_px", 64),
        room_extent=_float_value(merged, "room_extent", 3.2),
        room_height=_float_value(merged, "room_height", 3.0),
        grid_step=_float_value(merged, "grid_step", 0.8),
        marker_radius_px=_int_value(merged, "marker_radius_px", 22),
        label_font_size_px=_int_value(merged, "label_font_size_px", 24),
        line_width_px=_int_value(merged, "line_width_px", 2),
        floor_rgb=tuple(int(value) for value in tone.floor_rgb),
        grid_rgb=tuple(int(value) for value in tone.grid_rgb),
        edge_rgb=tuple(int(value) for value in tone.edge_rgb),
        text_rgb=tuple(int(value) for value in tone.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in tone.text_stroke_rgb),
        full_bleed_floor=_bool_value(merged, "full_bleed_floor", False),
        full_bleed_floor_extent_multiplier=_float_value(merged, "full_bleed_floor_extent_multiplier", 3.0),
        background_tone_id=str(tone.tone_id),
        background_tone_rgb=tuple(int(value) for value in tone.floor_rgb),
        surface_accent_rgb=tuple(int(value) for value in tone.surface_accent_rgb),
        conveyor_belt_style_id=str(belt_style.style_id) if belt_style is not None else None,
        conveyor_belt_fill_rgb=tuple(int(value) for value in belt_style.fill_rgb) if belt_style is not None else None,
        conveyor_belt_fill_alt_rgb=tuple(int(value) for value in belt_style.fill_alt_rgb) if belt_style is not None else None,
        conveyor_belt_fill_secondary_rgb=tuple(int(value) for value in belt_style.fill_secondary_rgb) if belt_style is not None else None,
        conveyor_belt_outline_rgb=tuple(int(value) for value in belt_style.outline_rgb) if belt_style is not None else None,
        conveyor_belt_outline_secondary_rgb=tuple(int(value) for value in belt_style.outline_secondary_rgb) if belt_style is not None else None,
        conveyor_belt_rail_rgb=tuple(int(value) for value in belt_style.rail_rgb) if belt_style is not None else None,
        conveyor_belt_arrow_rgb=tuple(int(value) for value in belt_style.arrow_rgb) if belt_style is not None else None,
        conveyor_belt_shadow_rgb=tuple(int(value) for value in belt_style.shadow_rgb) if belt_style is not None else None,
    )


def resolve_object_scene_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int = 0,
    namespace: str = "three_d.object_scene.canvas",
) -> ObjectSceneRenderParams:
    """Resolve shared object-scene render parameters for scene-family renderers."""

    return _resolve_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def _camera_yaw_band_for_instance(instance_seed: int) -> Tuple[float, float]:
    rng = spawn_rng(int(instance_seed), "three_d.object_scene.camera_yaw_band")
    yaw_band = uniform_choice(rng, CAMERA_YAW_BANDS_DEGREES, sort_keys=True)
    return tuple(float(value) for value in yaw_band)


def _base_shape_dimensions(shape_type: str, *, object_role: str = "candidate") -> Tuple[float, float, float]:
    small_dimensions = OBJECT_SCENE_SMALL_DIMENSIONS
    context_dimensions = OBJECT_SCENE_CONTEXT_DIMENSIONS
    dimensions = context_dimensions if str(object_role) == "context" else small_dimensions
    fallback = context_dimensions.get(str(shape_type)) or small_dimensions.get(str(shape_type)) or (0.52, 0.52, 0.52)
    return tuple(float(value) for value in dimensions.get(str(shape_type), fallback))


def _sample_shape_dimensions(
    shape_type: str,
    *,
    object_role: str,
    rng,
) -> Tuple[Tuple[float, float, float], float]:
    base_width, base_depth, base_height = _base_shape_dimensions(str(shape_type), object_role=str(object_role))
    if str(object_role) == "context":
        scale = float(rng.uniform(0.96, 1.20))
    else:
        scale = float(rng.uniform(0.86, 1.16))
    return (
        (
            round(float(base_width * scale), 4),
            round(float(base_depth * scale), 4),
            round(float(base_height * scale), 4),
        ),
        round(float(scale), 4),
    )


def _make_object_spec(
    *,
    object_id: str,
    shape_type: str,
    object_role: str,
    xy: Tuple[float, float],
    dimensions_xyz: Tuple[float, float, float],
    dimension_scale: float,
    label: str | None = None,
) -> Dict[str, Any]:
    width, depth, height = (float(value) for value in dimensions_xyz)
    footprint = 0.5 * math.sqrt(float(width) * float(width) + float(depth) * float(depth))
    profile_role = "spatial_context_shape" if str(object_role) == "context" else "spatial_small_shape"
    profile = object_profile_or_none(
        source_scene=OBJECT_SCENE_ID,
        role=profile_role,
        object_type=str(shape_type),
    )
    object_name = str(profile.display_name) if profile is not None else _object_name(str(shape_type))
    spec = {
        "object_id": str(object_id),
        "object_type": str(shape_type),
        "shape_type": str(shape_type),
        "object_name": str(object_name),
        "prompt_name": str(object_name),
        "nameable_for_prompt": bool(_nameable_for_prompt(str(shape_type), object_role=str(object_role))),
        "object_role": str(object_role),
        "is_answer_candidate": bool(label),
        "dimension_scale": round(float(dimension_scale), 4),
        "world_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(height * 0.5), 4)],
        "base_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
        "dimensions_xyz": [round(float(width), 4), round(float(depth), 4), round(float(height), 4)],
        "footprint_radius": round(float(footprint), 4),
    }
    if profile is not None:
        spec.update(
            {
                "profile_id": str(profile.profile_id),
                "canonical_id": str(profile.canonical_id),
                "resource_kind": str(profile.resource_kind),
                "family": str(profile.resource_kind),
                "renderer_id": str(profile.renderer),
            }
        )
    if label is not None:
        spec.update(
            {
                "point_id": f"object_{label}",
                "point_label": str(label),
                "object_label": str(label),
            }
        )
    return spec


def make_object_spec(
    *,
    object_id: str,
    shape_type: str,
    object_role: str,
    xy: Tuple[float, float],
    dimensions_xyz: Tuple[float, float, float],
    dimension_scale: float,
    label: str | None = None,
) -> Dict[str, Any]:
    """Build a canonical object-scene spec for shared 3D renderers."""

    return _make_object_spec(
        object_id=str(object_id),
        shape_type=str(shape_type),
        object_role=str(object_role),
        xy=xy,
        dimensions_xyz=dimensions_xyz,
        dimension_scale=float(dimension_scale),
        label=label,
    )


def _sample_scene_object_specs(
    *,
    rng,
    candidate_count: int,
    context_object_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_shape_types = list(NAMED_SMALL_OBJECT_SHAPE_TYPES)
    context_shape_types = list(LARGE_CONTEXT_SHAPE_TYPES)
    rng.shuffle(candidate_shape_types)
    rng.shuffle(context_shape_types)
    candidate_shape_types = candidate_shape_types[: int(candidate_count)]
    context_shape_types = context_shape_types[: int(context_object_count)]
    labels = list(POINT_LABELS[: int(candidate_count)])
    rng.shuffle(labels)
    context_slots = [(-1.85, 0.0), (1.85, 0.0), (0.0, 1.82), (-1.72, 1.72), (1.72, 1.72), (0.0, -1.78)]
    candidate_slots = [
        (x, y)
        for x in (-2.55, -1.28, 0.0, 1.28, 2.55)
        for y in (-2.55, -1.28, 0.0, 1.28, 2.55)
    ]
    rng.shuffle(context_slots)
    rng.shuffle(candidate_slots)
    placed: List[Dict[str, Any]] = []
    context_specs: List[Dict[str, Any]] = []
    candidate_specs: List[Dict[str, Any]] = []

    def place_object(shape_type: str, *, object_role: str, object_id: str, label: str | None, slots: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        dimensions_xyz, dimension_scale = _sample_shape_dimensions(str(shape_type), object_role=str(object_role), rng=rng)
        width, depth, _height = (float(value) for value in dimensions_xyz)
        footprint = 0.5 * math.sqrt(float(width) * float(width) + float(depth) * float(depth))
        jitter = 0.10 if str(object_role) == "context" else 0.16
        for slot_x, slot_y in slots:
            candidate_xy = (
                float(slot_x + rng.uniform(-jitter, jitter)),
                float(slot_y + rng.uniform(-jitter, jitter)),
            )
            if all(
                math.hypot(candidate_xy[0] - float(item["world_xyz"][0]), candidate_xy[1] - float(item["world_xyz"][1]))
                >= float(footprint + float(item["footprint_radius"]) + 0.10)
                for item in placed
            ):
                return _make_object_spec(
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    object_role=str(object_role),
                    xy=candidate_xy,
                    dimensions_xyz=dimensions_xyz,
                    dimension_scale=float(dimension_scale),
                    label=label,
                )
        raise ValueError(f"could not place {object_role} 3D object: {shape_type}")

    for index, shape_type in enumerate(context_shape_types):
        spec = place_object(
            str(shape_type),
            object_role="context",
            object_id=f"context_{index}_{shape_type}",
            label=None,
            slots=context_slots,
        )
        context_specs.append(spec)
        placed.append(spec)

    for index, shape_type in enumerate(candidate_shape_types):
        label = str(labels[index])
        spec = place_object(
            str(shape_type),
            object_role="candidate",
            object_id=f"object_{label}",
            label=label,
            slots=candidate_slots,
        )
        candidate_specs.append(spec)
        placed.append(spec)

    if len(candidate_specs) < int(candidate_count):
        raise ValueError("could not sample enough small candidate 3D objects")
    if len(context_specs) < int(context_object_count):
        raise ValueError("could not sample enough large context 3D objects")
    return list(candidate_specs), list(context_specs)


def _camera_from_dataset(dataset: Mapping[str, Any]) -> _CameraSpec:
    raw = dataset["camera"]
    return _CameraSpec(
        camera_position=tuple(float(value) for value in raw["camera_position"]),
        target=tuple(float(value) for value in raw["target"]),
        right=tuple(float(value) for value in raw["right"]),
        up=tuple(float(value) for value in raw["up"]),
        forward=tuple(float(value) for value in raw["forward"]),
        yaw_degrees=float(raw["yaw_degrees"]),
        pitch_degrees=float(raw["pitch_degrees"]),
        distance=float(raw["distance"]),
    )


def _frame_from_dataset(dataset: Mapping[str, Any]) -> _ProjectionFrame:
    raw = dataset["projection_frame"]
    return _ProjectionFrame(
        scale=float(raw["scale"]),
        center_x=float(raw["center_x"]),
        center_y=float(raw["center_y"]),
        normalized_center_u=float(raw["normalized_center_u"]),
        normalized_center_v=float(raw["normalized_center_v"]),
    )


def render_object_scene_3d(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: _RenderParams,
    draw_candidate_labels: bool = True,
    highlight_object_ids: Sequence[str] = (),
    annotation_label: str | None = None,
    compute_single_annotation: bool = True,
    option_choices: Sequence[Mapping[str, Any]] = (),
) -> _RenderedScene:
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    camera = _camera_from_dataset(dataset)
    frame = _frame_from_dataset(dataset)
    scene_variant = str(dataset.get("scene_variant", "floor_grid_room"))
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    room_bbox, entities = draw_object_scene_room(
        draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        scene_variant=scene_variant,
    )

    point_specs = [dict(spec) for spec in dataset["point_specs"]]
    context_object_specs = [dict(spec) for spec in dataset.get("context_object_specs", [])]
    all_specs = [*point_specs, *context_object_specs]

    def draw_order_key(item: Mapping[str, Any]) -> float:
        return float(item["camera_distance"]) + float(item.get("render_order_bias", 0.0))

    point_bboxes: Dict[str, List[float]] = {}
    point_centers: Dict[str, List[float]] = {}
    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    context_object_bboxes: Dict[str, List[float]] = {}
    context_object_centers: Dict[str, List[float]] = {}
    draw_scene_candidate_labels = bool(draw_candidate_labels) and not bool(option_choices)
    for spec in sorted(all_specs, key=draw_order_key, reverse=True):
        label = str(spec.get("point_label", ""))
        shape_type = str(spec["shape_type"])
        x, y = float(spec["screen_xy"][0]), float(spec["screen_xy"][1])
        if spec.get("fill_rgb") is not None:
            fallback_color = POINT_COLOR_BY_LABEL.get(str(label), POINT_COLORS[0])
            color = _rgb(spec.get("fill_rgb"), fallback_color)
        elif bool(spec.get("is_answer_candidate", False)):
            base_color = POINT_COLOR_BY_LABEL.get(str(label), POINT_COLORS[0])
            color = resolve_three_d_object_fill_rgb(
                spec,
                base_rgb=base_color,
                salt=f"{scene_variant}.candidate",
                variation_strength=0.10,
            )
        else:
            color = resolve_three_d_object_fill_rgb(
                spec,
                palette=CONTEXT_OBJECT_COLORS,
                salt=f"{scene_variant}.context",
                variation_strength=0.26,
            )
        object_role = (
            "candidate"
            if bool(spec.get("is_answer_candidate", False))
            else "countable"
            if bool(spec.get("is_countable_object", False))
            else "context"
        )
        object_render_spec = dict(spec)
        object_render_spec["fill_rgb"] = [int(channel) for channel in color]
        object_render_spec["scene_variant"] = str(scene_variant)
        profile = None
        if str(object_render_spec.get("profile_id", "")):
            try:
                profile = object_profile_by_id(str(object_render_spec["profile_id"]))
            except KeyError:
                profile = None
        placement = ThreeDPlacementSpec.from_mapping(
            object_render_spec,
            object_type_key="shape_type",
            role=object_role,
            source_entity_type="three_d_object_scene_object",
        )
        shared_spec = (
            ThreeDObjectSpec.from_profile_and_placement(
                profile,
                placement,
                object_type_key="shape_type",
                role=object_role,
                source_entity_type="three_d_object_scene_object",
            )
            if profile is not None
            else ThreeDObjectSpec.from_placement(
                placement,
                object_type_key="shape_type",
                default_renderer_id="object_scene_shape",
                role=object_role,
                source_entity_type="three_d_object_scene_object",
            )
        )
        rendered_object = render_three_d_object(
            shared_spec,
            ThreeDRenderContext(
                draw=draw,
                camera=camera,
                frame=frame,
                render_params=render_params,
                fill_rgb=color,
                scene_variant=str(scene_variant),
                floor_rgb=render_params.floor_rgb,
            ),
        )
        shape_bbox = list(rendered_object.bbox_xyxy)
        bbox = list(shape_bbox)
        if bool(spec.get("is_answer_candidate", False)):
            if bool(draw_scene_candidate_labels):
                label_center = (x, y)
                if shape_type == "torus":
                    label_center = (x, float(shape_bbox[1]) + 0.34 * (float(shape_bbox[3]) - float(shape_bbox[1])))
                label_bbox = _draw_option_label(draw, label=label, center=label_center, font=label_font)
                bbox = _bbox_union(shape_bbox, label_bbox)
            point_bboxes[label] = list(bbox)
            point_centers[label] = [round(float(x), 3), round(float(y), 3)]
        else:
            context_object_bboxes[str(spec["object_id"])] = list(bbox)
            context_object_centers[str(spec["object_id"])] = [round(float(x), 3), round(float(y), 3)]
        object_bboxes[str(spec["object_id"])] = list(bbox)
        object_centers[str(spec["object_id"])] = [round(float(x), 3), round(float(y), 3)]
        entities.append(
            {
                "entity_id": str(spec["object_id"]),
                "entity_type": (
                    "three_d_candidate_object"
                    if bool(spec.get("is_answer_candidate", False))
                    else "three_d_countable_object"
                    if bool(spec.get("is_countable_object", False))
                    else "three_d_context_object"
                ),
                "bbox_px": list(bbox),
                "attrs": {
                    "point_label": str(label) if label else None,
                    "object_label": str(label) if label else None,
                    "shape_type": str(shape_type),
                    "object_name": str(spec.get("object_name", _object_name(shape_type))),
                    "prompt_name": str(spec.get("prompt_name", _object_name(shape_type))),
                    "nameable_for_prompt": bool(spec.get("nameable_for_prompt", False)),
                    "object_role": str(spec.get("object_role", "candidate")),
                    "is_answer_candidate": bool(spec.get("is_answer_candidate", False)),
                    "is_countable_object": bool(spec.get("is_countable_object", False)),
                    "matches_query": bool(spec.get("matches_query", False)),
                    "count_role": str(spec.get("count_role", "")) or None,
                    "fill_rgb": [int(channel) for channel in color],
                    "world_xyz": list(spec["world_xyz"]),
                    "base_xyz": list(spec["base_xyz"]),
                    "dimensions_xyz": list(spec["dimensions_xyz"]),
                    "dimension_scale": float(spec.get("dimension_scale", 1.0)),
                    "orientation_deg": float(spec.get("orientation_deg", 0.0)),
                    "screen_xy": [round(float(x), 3), round(float(y), 3)],
                    "camera_xyz": list(spec["camera_xyz"]),
                    "camera_distance": float(spec["camera_distance"]),
                    "scene_variant": str(scene_variant),
                    "object_record": dict(rendered_object.object_record),
                },
            }
        )

    for object_id in tuple(str(item) for item in highlight_object_ids):
        if object_id not in object_bboxes:
            continue
        raw_bbox = object_bboxes[str(object_id)]
        highlight_bbox = [
            round(max(0.0, float(raw_bbox[0]) - 8.0), 3),
            round(max(0.0, float(raw_bbox[1]) - 8.0), 3),
            round(min(float(render_params.canvas_width), float(raw_bbox[2]) + 8.0), 3),
            round(min(float(render_params.canvas_height), float(raw_bbox[3]) + 8.0), 3),
        ]
        draw.rectangle(highlight_bbox, outline=(24, 25, 28), width=8)
        draw.rectangle(highlight_bbox, outline=(220, 42, 45), width=4)
        entities.append(
            {
                "entity_id": f"red_reference_box_{object_id}",
                "entity_type": "red_reference_box",
                "bbox_px": list(highlight_bbox),
                "attrs": {
                    "target_object_id": str(object_id),
                    "scene_variant": str(scene_variant),
                },
            }
        )

    annotation_bboxes: List[List[float]] = []
    annotation_entity_ids: List[str] = []
    if bool(compute_single_annotation):
        answer_label = str(annotation_label if annotation_label is not None else dataset["answer_label"])
        annotation_bboxes = [list(point_bboxes[answer_label])]
        annotation_entity_ids = [str(dataset["answer_point_id"])]
    all_bboxes = [list(room_bbox)] + [list(bbox) for bbox in object_bboxes.values()]
    scene_bbox = [
        round(float(min(bbox[0] for bbox in all_bboxes)), 3),
        round(float(min(bbox[1] for bbox in all_bboxes)), 3),
        round(float(max(bbox[2] for bbox in all_bboxes)), 3),
        round(float(max(bbox[3] for bbox in all_bboxes)), 3),
    ]
    option_metadata = empty_option_panel_metadata()
    if option_choices:
        image, option_metadata, option_entities = append_text_option_panel(
            image,
            option_choices=option_choices,
            font_size_px=int(render_params.label_font_size_px),
            text_rgb=render_params.text_rgb,
            stroke_rgb=render_params.text_stroke_rgb,
        )
        entities.extend(option_entities)
    image, image_scale = resize_image_to_fit_pixel_cap(image)
    if image_scale.changed:
        scale_x = float(image_scale.scale_x)
        scale_y = float(image_scale.scale_y)
        scene_bbox = bbox_transform(scene_bbox, scale_x=scale_x, scale_y=scale_y)
        point_bboxes = bbox_dict_transform(point_bboxes, scale_x=scale_x, scale_y=scale_y)
        point_centers = point_dict_transform(point_centers, scale_x=scale_x, scale_y=scale_y)
        object_bboxes = bbox_dict_transform(object_bboxes, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(object_centers, scale_x=scale_x, scale_y=scale_y)
        context_object_bboxes = bbox_dict_transform(context_object_bboxes, scale_x=scale_x, scale_y=scale_y)
        context_object_centers = point_dict_transform(context_object_centers, scale_x=scale_x, scale_y=scale_y)
        room_bbox = bbox_transform(room_bbox, scale_x=scale_x, scale_y=scale_y)
        annotation_bboxes = [bbox_transform(bbox, scale_x=scale_x, scale_y=scale_y) for bbox in annotation_bboxes]
        option_metadata = dict(option_metadata)
        if option_metadata.get("option_panel_bbox_px"):
            option_metadata["option_panel_bbox_px"] = bbox_transform(
                option_metadata["option_panel_bbox_px"],
                scale_x=scale_x,
                scale_y=scale_y,
            )
        option_metadata["option_choice_bboxes_px"] = bbox_dict_transform(
            option_metadata.get("option_choice_bboxes_px", {}),
            scale_x=scale_x,
            scale_y=scale_y,
        )
        option_metadata["option_panel_height_px"] = int(round(float(option_metadata.get("option_panel_height_px", 0)) * scale_y))
        entities = entities_transform(entities, scale_x=scale_x, scale_y=scale_y)
    return _RenderedScene(
        image=image,
        entities=list(entities),
        scene_bbox_px=list(scene_bbox),
        point_bboxes_px=dict(point_bboxes),
        point_centers_px=dict(point_centers),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        context_object_bboxes_px=dict(context_object_bboxes),
        context_object_centers_px=dict(context_object_centers),
        room_bbox_px=list(room_bbox),
        annotation_bboxes=list(annotation_bboxes),
        annotation_entity_ids=list(annotation_entity_ids),
        option_panel_bbox_px=list(option_metadata["option_panel_bbox_px"]),
        option_choice_bboxes_px={str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()},
        option_choices=[dict(choice) for choice in option_metadata["option_choices"]],
        option_panel_height_px=int(option_metadata["option_panel_height_px"]),
    )

__all__ = [
    "CAMERA_YAW_BANDS_DEGREES",
    "CONTEXT_OBJECT_COLORS",
    "LARGE_CONTEXT_SHAPE_TYPES",
    "NAMEABLE_CONTEXT_SHAPE_TYPES",
    "NAMED_SMALL_OBJECT_SHAPE_TYPES",
    "OBJECT_NAME_BY_SHAPE_TYPE",
    "ObjectSceneRenderParams",
    "POINT_COLORS",
    "POINT_LABELS",
    "SCENE_ID",
    "SHAPE_TYPES",
    "SMALL_OBJECT_SHAPE_TYPES",
    "SUPPORTED_SCENE_VARIANTS",
    "_RenderedScene",
    "_RenderParams",
    "_base_shape_dimensions",
    "_bbox_intersection_area",
    "_bool_value",
    "_camera_from_dataset",
    "_camera_yaw_band_for_instance",
    "_frame_from_dataset",
    "_make_object_spec",
    "_nameable_for_prompt",
    "_object_name",
    "_object_reference_points",
    "_object_screen_bbox",
    "_resolve_render_params",
    "_sample_scene_object_specs",
    "_sample_shape_dimensions",
    "bbox_intersection_area",
    "make_object_spec",
    "object_reference_points",
    "object_screen_bbox",
    "render_object_scene_3d",
    "resolve_object_scene_render_params",
]
