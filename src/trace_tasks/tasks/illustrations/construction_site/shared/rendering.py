"""Rendering helpers for construction-site illustrations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ....shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from ...shared.object_library import BBox, RGB, STYLE_IDS
from ...shared.object_rendering import IllustrationObjectSpec, RenderContext, render_illustration_object, render_vector_scene_object
from ...shared.object_variants import RENDERER_STYLE_VECTOR
from ...shared.person_rendering import sample_person_gender
from ...shared.render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from ...shared.style_registry import style_outline_params

from .labels import (
    CONSTRUCTION_COLOR_RGB,
    construction_equipment_display_name,
    construction_material_display_name,
)
from .layout import (
    build_construction_zones,
    construction_zone_lookup,
    place_construction_box,
    sample_construction_layout,
)
from .state import (
    CONSTRUCTION_SETTING_IDS,
    CONSTRUCTION_ZONE_TYPES,
    ConstructionDecor,
    ConstructionEquipment,
    ConstructionEquipmentSpec,
    ConstructionMaterial,
    ConstructionMaterialSpec,
    ConstructionWorker,
    ConstructionWorkerSpec,
    ConstructionZone,
    RenderedConstructionSiteScene,
    safe_json_value,
)


def _jitter_rgb(rng, color: RGB, amount: int = 10) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(rng.randint(-int(amount), int(amount))))) for channel in color)  # type: ignore[return-value]


def _choose_weighted(rng, weights: Mapping[str, float], support: Sequence[str]) -> str:
    choices = [(str(value), max(0.0, float(weights.get(str(value), 0.0)))) for value in support]
    total = sum(weight for _value, weight in choices)
    if total <= 0.0:
        return str(rng.choice(tuple(support)))
    threshold = float(rng.random()) * total
    running = 0.0
    for value, weight in choices:
        running += weight
        if running >= threshold:
            return str(value)
    return str(choices[-1][0])


def _style_outline(style_id: str) -> Tuple[RGB, int, bool]:
    outline, width, shadow = style_outline_params(str(style_id))
    return tuple(outline or (74, 78, 87)), int(width), bool(shadow)


def _role_priority(role: str) -> int:
    return 0 if str(role) == "target" else 1


def _rect(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    *,
    fill: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
    radius: float = 0.0,
) -> None:
    box = _scale_bbox(bbox, scale)
    line_width = max(1, int(width) * int(scale))
    if float(radius) > 0:
        draw.rounded_rectangle(
            box,
            radius=max(1, int(round(float(radius) * int(scale)))),
            fill=tuple(fill),
            outline=tuple(outline) if outline else None,
            width=line_width,
        )
    else:
        draw.rectangle(box, fill=tuple(fill), outline=tuple(outline) if outline else None, width=line_width)


def _poly(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
) -> None:
    draw.polygon(_scale_points(points, scale), fill=tuple(fill))
    if outline:
        draw.line(_scale_points([*points, points[0]], scale), fill=tuple(outline), width=max(1, int(width) * int(scale)), joint="curve")


def _line(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, width: int, scale: int) -> None:
    draw.line(_scale_points(points, scale), fill=tuple(fill), width=max(1, int(width) * int(scale)), joint="curve")


def _font(size: int, scale: int, *, font_family: str | None = None) -> ImageFont.ImageFont:
    return load_font(max(8, int(size) * int(scale)), bold=True, font_family=font_family)


def _sample_zone_label_font_trace(*, instance_seed: int | None, params: Mapping[str, Any] | None) -> Dict[str, Any]:
    font_family = sample_font_family(
        role="readout",
        instance_seed=0 if instance_seed is None else int(instance_seed),
        namespace="illustrations.construction_site.zone_labels",
        params=params,
        explicit_key="construction_zone_label_font_family",
        weights_key="construction_zone_label_font_family_weights",
    )
    record = get_font_family_record(str(font_family))
    return {
        **record.to_trace(),
        "font_asset_version": font_asset_version(),
        "pool": "global_approved_font_pool",
        "role": "construction_zone_label",
        "consistent_scope": "construction_site_zone_labels",
    }


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: Tuple[float, float],
    *,
    font: ImageFont.ImageFont,
    fill: RGB,
    scale: int,
    stroke_fill: RGB | None = None,
) -> None:
    cx, cy = center
    try:
        text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(scale) if stroke_fill else 0))
        text_w = float(text_bbox[2] - text_bbox[0])
        text_h = float(text_bbox[3] - text_bbox[1])
    except Exception:  # pragma: no cover
        text_w, text_h = draw.textsize(str(text), font=font)
    draw_text_traced(
        draw,
        (int(round(float(cx) * int(scale) - text_w / 2.0)), int(round(float(cy) * int(scale) - text_h / 2.0))),
        str(text),
        font=font,
        fill=tuple(fill),
        stroke_width=max(0, int(scale) if stroke_fill else 0),
        stroke_fill=tuple(stroke_fill) if stroke_fill else None,
        role="readout",
        required=False,
    )


def _draw_background(
    draw: ImageDraw.ImageDraw,
    *,
    rng,
    setting_id: str,
    layout: Mapping[str, Any],
    zones: Sequence[ConstructionZone],
    zone_label_font_family: str,
    show_zone_labels: bool,
    width: int,
    height: int,
    scale: int,
    style_id: str,
) -> List[ConstructionDecor]:
    """Draw site context while preserving zone readability."""

    decor: List[ConstructionDecor] = []
    sky = _jitter_rgb(rng, (196, 219, 228), 8)
    ground = _jitter_rgb(rng, (180, 168, 138), 10)
    far = _jitter_rgb(rng, (150, 162, 153), 10)
    s = int(scale)
    horizon = float(rng.uniform(178.0, 232.0))
    draw.rectangle((0, 0, int(width) * s, int(height) * s), fill=tuple(sky))
    draw.rectangle(_scale_bbox((0.0, horizon, float(width), float(height)), s), fill=tuple(ground))
    draw.rectangle(_scale_bbox((0.0, horizon, float(width), horizon + 38.0), s), fill=tuple(far))

    x = 0.0
    building_idx = 0
    while x < float(width):
        bw = float(rng.uniform(72.0, 132.0))
        if x + bw > float(width):
            bw = float(width) - x
        if bw < 36.0:
            break
        bh = float(rng.uniform(82.0, 168.0))
        bbox = (x, horizon - bh, x + bw, horizon + 8.0)
        _rect(draw, bbox, fill=_jitter_rgb(rng, (126, 137, 145), 10), outline=None, width=1, scale=s)
        decor.append(ConstructionDecor(f"decor_building_{building_idx}", "background_building", bbox, {"setting_id": str(setting_id)}))
        x += bw + float(rng.uniform(14.0, 34.0))
        building_idx += 1
    _line(draw, [(0.0, horizon + 52.0), (float(width), horizon + 52.0)], fill=(118, 105, 86), width=4, scale=s)
    for post_x in range(20, int(width), 78):
        _rect(draw, (float(post_x), horizon + 26.0, float(post_x + 8), horizon + 84.0), fill=(126, 105, 78), outline=None, width=1, scale=s)

    outline, outline_w, _shadow = _style_outline(str(style_id))
    if str(setting_id) in {"scaffold_site", "urban_build"} or str(layout.get("layout_id")) == "scaffold_front":
        scaffold_box = (74.0, 236.0, 548.0, 528.0)
        for level in range(4):
            y = scaffold_box[1] + 58.0 * level
            _line(draw, [(scaffold_box[0], y), (scaffold_box[2], y)], fill=(94, 101, 109), width=3, scale=s)
        for col in range(6):
            x0 = scaffold_box[0] + 92.0 * col
            _line(draw, [(x0, scaffold_box[1]), (x0, scaffold_box[3])], fill=(94, 101, 109), width=3, scale=s)
        decor.append(ConstructionDecor("decor_scaffold", "scaffold", scaffold_box, {"setting_id": str(setting_id)}))
    if str(setting_id) in {"urban_build", "foundation_yard", "scaffold_site"}:
        crane_w = min(292.0, max(220.0, float(width) * 0.34))
        crane_scale = crane_w / 292.0
        base_x_max = max(42.0, float(width) - 42.0 - crane_w)
        base_x_min = max(58.0, min(base_x_max, float(width) * 0.58))
        base_x = float(rng.uniform(base_x_min, base_x_max))
        crane_box = (base_x, 72.0, base_x + crane_w, 292.0)

        def sx(offset: float) -> float:
            return base_x + float(offset) * crane_scale

        _line(draw, [(sx(24.0), 286.0), (sx(24.0), 92.0)], fill=(192, 139, 45), width=10, scale=s)
        _line(draw, [(sx(0.0), 108.0), (sx(292.0), 108.0)], fill=(206, 157, 51), width=8, scale=s)
        _line(draw, [(sx(24.0), 92.0), (sx(292.0), 108.0)], fill=(166, 119, 40), width=4, scale=s)
        _line(draw, [(sx(238.0), 108.0), (sx(238.0), 178.0)], fill=(66, 68, 74), width=3, scale=s)
        _rect(draw, (sx(220.0), 174.0, sx(254.0), 206.0), fill=(112, 99, 82), outline=outline, width=outline_w, scale=s, radius=2)
        decor.append(ConstructionDecor("decor_crane", "tower_crane", crane_box, {"setting_id": str(setting_id)}))
    if str(setting_id) == "roadwork" or str(layout.get("layout_id")) == "diagonal_road":
        road_poly = [(0.0, 752.0), (float(width), 628.0), (float(width), 734.0), (0.0, 860.0)]
        _poly(draw, road_poly, fill=(92, 96, 98), outline=None, width=1, scale=s)
        _line(draw, [(0.0, 802.0), (float(width), 680.0)], fill=(240, 203, 82), width=5, scale=s)

    label_font = _font(17, s, font_family=str(zone_label_font_family))
    for zone in zones:
        _rect(draw, zone.bbox_xyxy, fill=zone.fill_rgb, outline=zone.outline_rgb, width=2, scale=s, radius=18)
        if bool(show_zone_labels):
            cx = (float(zone.bbox_xyxy[0]) + float(zone.bbox_xyxy[2])) / 2.0
            _draw_centered_text(draw, zone.label, (cx, float(zone.bbox_xyxy[1]) + 23.0), font=label_font, fill=(49, 54, 61), stroke_fill=(255, 255, 245), scale=s)
    return decor


def render_construction_site_scene(
    *,
    rng,
    worker_specs: Sequence[ConstructionWorkerSpec],
    material_specs: Sequence[ConstructionMaterialSpec],
    equipment_specs: Sequence[ConstructionEquipmentSpec],
    canvas_width: int,
    canvas_height: int,
    render_scale: int,
    setting_weights: Mapping[str, float],
    style_weights: Mapping[str, float],
    instance_seed: int | None = None,
    font_params: Mapping[str, Any] | None = None,
    show_zone_labels: bool = True,
) -> RenderedConstructionSiteScene:
    """Render a construction-site illustration from semantic specs."""

    width = int(canvas_width)
    height = int(canvas_height)
    scale = int(render_scale)
    setting_id = _choose_weighted(rng, setting_weights, CONSTRUCTION_SETTING_IDS)
    style_id = _choose_weighted(rng, style_weights, STYLE_IDS)
    layout = sample_construction_layout(rng, width=width, height=height, setting_id=setting_id)
    zone_label_font_trace = _sample_zone_label_font_trace(instance_seed=instance_seed, params=font_params)
    layout["zone_label_font"] = dict(zone_label_font_trace)
    layout["show_zone_labels"] = bool(show_zone_labels)
    zones = build_construction_zones(layout)
    zone_by_id = construction_zone_lookup(zones)
    image = Image.new("RGB", (width * scale, height * scale), (244, 239, 224))
    draw = ImageDraw.Draw(image)
    decor = _draw_background(
        draw,
        rng=rng,
        setting_id=setting_id,
        layout=layout,
        zones=zones,
        zone_label_font_family=str(zone_label_font_trace["font_family"]),
        show_zone_labels=bool(show_zone_labels),
        width=width,
        height=height,
        scale=scale,
        style_id=style_id,
    )

    occupied: List[BBox] = []
    protected_occupied: List[BBox] = []
    materials: List[ConstructionMaterial] = []
    ordered_material_specs = sorted(enumerate(material_specs), key=lambda item: (_role_priority(str(item[1].role)), int(item[0])))
    for index, spec in ordered_material_specs:
        zone = zone_by_id[str(rng.choice(CONSTRUCTION_ZONE_TYPES))]
        box_w = float(rng.uniform(86.0, 138.0))
        box_h = float(rng.uniform(58.0, 94.0))
        is_target = str(spec.role) == "target"
        bbox = place_construction_box(
            rng,
            zone.bbox_xyxy,
            width=box_w,
            height=box_h,
            occupied=occupied,
            protected=protected_occupied,
            allow_overlap_fallback=not is_target,
            gap=8.0,
        )
        if is_target:
            protected_occupied.append(bbox)
        material = ConstructionMaterial(
            material_id=f"material_{index}",
            material_type=str(spec.material_type),
            material_label=construction_material_display_name(str(spec.material_type)),
            bbox_xyxy=bbox,
            style_id=str(style_id),
            role=str(spec.role),
            attributes=dict(spec.attributes),
        )
        materials.append(material)
    rendered_material_records: Dict[str, Mapping[str, Any]] = {}
    for material in sorted(materials, key=lambda item: (float(item.bbox_xyxy[1]), float(item.bbox_xyxy[0]))):
        rendered = render_vector_scene_object(
            draw,
            object_id=str(material.material_id),
            object_type="construction_material",
            bbox_xyxy=material.bbox_xyxy,
            renderer_id="construction_material",
            renderer_variant_id=str(material.material_type),
            semantic_attributes={
                "material_type": str(material.material_type),
                "material_label": str(material.material_label),
                **dict(material.attributes),
            },
            visual_attributes={"style_id": str(material.style_id)},
            role=str(material.role),
            source_entity_type="construction_material",
            render_scale=scale,
            style_id=str(style_id),
        )
        rendered_material_records[str(material.material_id)] = rendered.object_record
    materials = [replace(material, object_record=rendered_material_records.get(str(material.material_id))) for material in materials]

    equipment_items: List[ConstructionEquipment] = []
    ordered_equipment_specs = sorted(enumerate(equipment_specs), key=lambda item: (_role_priority(str(item[1].role)), int(item[0])))
    for index, spec in ordered_equipment_specs:
        zone_id = str(spec.zone_id or rng.choice(CONSTRUCTION_ZONE_TYPES))
        zone = zone_by_id[zone_id]
        box_w = float(rng.uniform(132.0, 188.0))
        box_h = float(rng.uniform(94.0, 134.0))
        is_target = str(spec.role) == "target"
        bbox = place_construction_box(
            rng,
            zone.bbox_xyxy,
            width=box_w,
            height=box_h,
            occupied=occupied,
            protected=protected_occupied,
            allow_overlap_fallback=not is_target,
            gap=12.0,
        )
        if is_target:
            protected_occupied.append(bbox)
        equipment = ConstructionEquipment(
            equipment_id=f"equipment_{index}",
            equipment_type=str(spec.equipment_type),
            equipment_label=construction_equipment_display_name(str(spec.equipment_type)),
            zone_id=zone_id,
            bbox_xyxy=bbox,
            style_id=str(style_id),
            role=str(spec.role),
            attributes=dict(spec.attributes),
        )
        equipment_items.append(equipment)
    rendered_equipment_records: Dict[str, Mapping[str, Any]] = {}
    for equipment in sorted(equipment_items, key=lambda item: (float(item.bbox_xyxy[1]), float(item.bbox_xyxy[0]))):
        rendered = render_vector_scene_object(
            draw,
            object_id=str(equipment.equipment_id),
            object_type="construction_equipment",
            bbox_xyxy=equipment.bbox_xyxy,
            renderer_id="construction_equipment",
            renderer_variant_id=str(equipment.equipment_type),
            semantic_attributes={
                "equipment_type": str(equipment.equipment_type),
                "equipment_label": str(equipment.equipment_label),
                "zone_id": str(equipment.zone_id),
                **dict(equipment.attributes),
            },
            visual_attributes={"style_id": str(equipment.style_id)},
            role=str(equipment.role),
            source_entity_type="construction_equipment",
            render_scale=scale,
            style_id=str(style_id),
        )
        rendered_equipment_records[str(equipment.equipment_id)] = rendered.object_record
    equipment_items = [replace(equipment, object_record=rendered_equipment_records.get(str(equipment.equipment_id))) for equipment in equipment_items]

    workers: List[ConstructionWorker] = []
    ordered_worker_specs = sorted(enumerate(worker_specs), key=lambda item: (_role_priority(str(item[1].role)), int(item[0])))
    for index, spec in ordered_worker_specs:
        zone = zone_by_id[str(rng.choice(CONSTRUCTION_ZONE_TYPES))]
        box_w = float(rng.uniform(44.0, 58.0))
        box_h = float(rng.uniform(94.0, 116.0))
        is_target = str(spec.role) == "target"
        bbox = place_construction_box(
            rng,
            zone.bbox_xyxy,
            width=box_w,
            height=box_h,
            occupied=occupied,
            protected=protected_occupied,
            allow_overlap_fallback=not is_target,
            gap=4.0,
        )
        if is_target:
            protected_occupied.append(bbox)
        worker = ConstructionWorker(
            worker_id=f"worker_{index}",
            hard_hat_color=str(spec.hard_hat_color),
            vest_color=str(spec.vest_color),
            tool_type=str(spec.tool_type) if spec.tool_type else None,
            bbox_xyxy=bbox,
            style_id=str(style_id),
            gender_id=sample_person_gender(rng),
            role=str(spec.role),
            attributes=dict(spec.attributes),
        )
        workers.append(worker)
    rendered_worker_records: Dict[str, Mapping[str, Any]] = {}
    for worker in sorted(workers, key=lambda item: (float(item.bbox_xyxy[1]), float(item.bbox_xyxy[0]))):
        hat_rgb = CONSTRUCTION_COLOR_RGB.get(str(worker.hard_hat_color), (238, 194, 64))
        vest_rgb = CONSTRUCTION_COLOR_RGB.get(str(worker.vest_color), (232, 126, 54))
        rendered = render_illustration_object(
            IllustrationObjectSpec(
                object_id=str(worker.worker_id),
                object_type="worker",
                bbox_xyxy=worker.bbox_xyxy,
                renderer_id="construction_worker",
                renderer_variant_id="standing",
                semantic_attributes={
                    "hard_hat_color": str(worker.hard_hat_color),
                    "vest_color": str(worker.vest_color),
                    "tool_type": str(worker.tool_type) if worker.tool_type else None,
                    **dict(worker.attributes),
                },
                visual_attributes={
                    "primary_color_rgb": [84, 118, 154],
                    "accent_color_rgb": [int(v) for v in vest_rgb],
                    "skin_color_rgb": [178, 126, 83],
                    "hard_hat_color_rgb": [int(v) for v in hat_rgb],
                    "vest_color_rgb": [int(v) for v in vest_rgb],
                    "style_id": str(worker.style_id),
                    "gender_id": str(worker.gender_id),
                },
                role=str(worker.role),
                source_entity_type="construction_worker",
            ),
            RenderContext(
                renderer_style=RENDERER_STYLE_VECTOR,
                draw=draw,
                render_scale=scale,
                style_id=str(style_id),
            ),
        )
        rendered_worker_records[str(worker.worker_id)] = rendered.object_record
    workers = [replace(worker, object_record=rendered_worker_records.get(str(worker.worker_id))) for worker in workers]

    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return RenderedConstructionSiteScene(
        image=image,
        setting_id=str(setting_id),
        zones=tuple(zones),
        workers=tuple(workers),
        materials=tuple(materials),
        equipment=tuple(equipment_items),
        decor=tuple(decor),
        canvas_width=width,
        canvas_height=height,
        render_scale=scale,
        style_id=str(style_id),
        layout=safe_json_value(layout),
    )


__all__ = ["render_construction_site_scene"]
