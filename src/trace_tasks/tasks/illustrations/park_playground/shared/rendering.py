"""Park and playground illustration scene with activity-labeled people."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from ...shared.object_catalog import label_map_for_tag, plural_name_map_for_tag, variant_ids_with_tag
from ...shared.object_library import BBox, RGB, STYLE_IDS
from ...shared.object_rendering import IllustrationObjectSpec, RenderContext, make_vector_scene_object_record, render_illustration_object, render_vector_scene_object
from ...shared.object_variants import RENDERER_STYLE_VECTOR
from ...shared.person_rendering import sample_person_gender


PARK_SETTING_IDS: Tuple[str, ...] = variant_ids_with_tag("park_setting")
PARK_PERSON_ACTIVITIES: Tuple[str, ...] = variant_ids_with_tag("park_person_activity")
PARK_EQUIPMENT_TYPES: Tuple[str, ...] = variant_ids_with_tag("park_equipment")
PARK_ZONE_TYPES: Tuple[str, ...] = variant_ids_with_tag("park_zone")
PARK_PERSON_ACTIVITY_LABELS: Dict[str, str] = label_map_for_tag("park_person_activity")
PARK_EQUIPMENT_LABELS: Dict[str, str] = plural_name_map_for_tag("park_equipment")
PARK_ZONE_LABELS: Dict[str, str] = label_map_for_tag("park_zone")
MIN_EQUIPMENT_BBOX_SIDE_PX = 24.5


@dataclass(frozen=True)
class ParkPersonSpec:
    """Requested semantic activity for one rendered person."""

    activity: str
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParkEquipmentSpec:
    """Requested semantic equipment type for one rendered playground item."""

    equipment_type: str
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParkPerson:
    """One rendered person with activity metadata."""

    person_id: str
    activity: str
    activity_label: str
    bbox_xyxy: BBox
    primary_color_rgb: RGB
    accent_color_rgb: RGB
    skin_color_rgb: RGB
    style_id: str
    gender_id: str
    role: str
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ParkDecor:
    """Non-query park decor or support object."""

    decor_id: str
    decor_type: str
    bbox_xyxy: BBox
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RenderedParkPlaygroundScene:
    """Rendered park scene plus verifier-ready metadata."""

    image: Image.Image
    setting_id: str
    persons: Tuple[ParkPerson, ...]
    decor: Tuple[ParkDecor, ...]
    canvas_width: int
    canvas_height: int
    render_scale: int
    style_id: str
    layout: Mapping[str, Any]


def park_activity_display_name(activity: str) -> str:
    """Return prompt-facing activity text."""

    return PARK_PERSON_ACTIVITY_LABELS.get(str(activity), str(activity).replace("_", " "))


def park_equipment_display_name(equipment_type: str) -> str:
    """Return prompt-facing plural equipment text."""

    return PARK_EQUIPMENT_LABELS.get(str(equipment_type), str(equipment_type).replace("_", " ") + "s")


def park_zone_display_name(zone: str) -> str:
    """Return prompt-facing park-zone text."""

    return PARK_ZONE_LABELS.get(str(zone), str(zone).replace("_", " ") + " area")


def _decor_object_type(decor_type: str) -> str:
    if str(decor_type) in set(PARK_EQUIPMENT_TYPES):
        return "playground_equipment"
    if str(decor_type) in {"playground_sand", "picnic_area", "garden_area"}:
        return "zone"
    if str(decor_type) in {"walking_path", "pond"}:
        return "environment_feature"
    return str(decor_type)


def _park_equipment_visual_attributes(rng, equipment_type: str, style_id: str) -> Dict[str, Any]:
    outline = _jitter_rgb(rng, (61, 71, 82), amount=6)
    if str(equipment_type) == "slide":
        primary = _jitter_rgb(rng, (216, 96, 82), amount=8)
        accent = _jitter_rgb(rng, (229, 168, 67), amount=8)
    elif str(equipment_type) == "swing_set":
        primary = _jitter_rgb(rng, (83, 139, 190), amount=8)
        accent = _jitter_rgb(rng, (83, 139, 190), amount=8)
    elif str(equipment_type) == "seesaw":
        primary = _jitter_rgb(rng, (218, 111, 82), amount=8)
        accent = _jitter_rgb(rng, (142, 105, 77), amount=8)
    else:
        primary = outline
        accent = _jitter_rgb(rng, (83, 139, 190), amount=8)
    return {
        "primary_color_rgb": [int(v) for v in primary],
        "accent_color_rgb": [int(v) for v in accent],
        "outline_color_rgb": [int(v) for v in outline],
        "style_id": str(style_id),
    }


def _bench_visual_attributes(rng, *, style_id: str, wood_base: RGB = (146, 92, 58), outline_base: RGB = (75, 56, 43)) -> Dict[str, Any]:
    wood = _jitter_rgb(rng, wood_base, amount=10)
    outline = _jitter_rgb(rng, outline_base, amount=6)
    return {
        "wood_color_rgb": [int(v) for v in wood],
        "outline_color_rgb": [int(v) for v in outline],
        "style_id": str(style_id),
    }




def _jitter_rgb(rng, color: RGB, amount: int = 12) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(rng.randint(-int(amount), int(amount))))) for channel in color)  # type: ignore[return-value]


def _choose_weighted(rng, weights: Mapping[str, float], support: Sequence[str]) -> str:
    choices = [(str(value), max(0.0, float(weights.get(str(value), 0.0)))) for value in support]
    total = sum(weight for _value, weight in choices)
    if total <= 0.0:
        return str(rng.choice(tuple(support)))
    threshold = float(rng.random()) * float(total)
    running = 0.0
    for value, weight in choices:
        running += float(weight)
        if running >= threshold:
            return str(value)
    return str(choices[-1][0])


def _rect(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int, radius: float = 0.0) -> None:
    box = _scale_bbox(bbox, scale)
    if radius > 0:
        draw.rounded_rectangle(box, radius=max(1, int(round(float(radius) * int(scale)))), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))
    else:
        draw.rectangle(box, fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _ellipse(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.ellipse(_scale_bbox(bbox, scale), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _poly(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.polygon(_scale_points(points, scale), fill=tuple(fill))
    if outline:
        draw.line(_scale_points([*points, points[0]], scale), fill=tuple(outline), width=max(1, int(width) * int(scale)), joint="curve")


def _line(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, width: int, scale: int) -> None:
    draw.line(_scale_points(points, scale), fill=tuple(fill), width=max(1, int(width) * int(scale)), joint="curve")


def _bbox_union(boxes: Iterable[BBox]) -> BBox:
    values = [tuple(float(v) for v in box) for box in boxes]
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(v[0] for v in values), min(v[1] for v in values), max(v[2] for v in values), max(v[3] for v in values))


def _expanded_intersects(a: BBox, b: BBox, gap: float) -> bool:
    return not (
        float(a[2]) + float(gap) <= float(b[0])
        or float(b[2]) + float(gap) <= float(a[0])
        or float(a[3]) + float(gap) <= float(b[1])
        or float(b[3]) + float(gap) <= float(a[1])
    )


def _clamp_bbox_to_canvas(box: BBox, *, width: int, height: int, margin: float = 20.0) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in box)
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    box_w = min(box_w, max(1.0, float(width) - 2.0 * float(margin)))
    box_h = min(box_h, max(1.0, float(height) - 2.0 * float(margin)))
    max_x0 = max(float(margin), float(width) - float(margin) - box_w)
    max_y0 = max(float(margin), float(height) - float(margin) - box_h)
    clamped_x0 = max(float(margin), min(float(x0), max_x0))
    clamped_y0 = max(float(margin), min(float(y0), max_y0))
    return (
        round(float(clamped_x0), 3),
        round(float(clamped_y0), 3),
        round(float(clamped_x0 + box_w), 3),
        round(float(clamped_y0 + box_h), 3),
    )


def _expand_bbox_to_min_side(box: BBox, *, min_side: float, bounds: BBox) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, by0, bx1, by1 = (float(v) for v in bounds)
    target_w = min(max(float(min_side), x1 - x0), max(1.0, bx1 - bx0))
    target_h = min(max(float(min_side), y1 - y0), max(1.0, by1 - by0))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    nx0 = cx - target_w / 2.0
    ny0 = cy - target_h / 2.0
    nx0 = max(bx0, min(nx0, bx1 - target_w))
    ny0 = max(by0, min(ny0, by1 - target_h))
    return (
        round(float(nx0), 3),
        round(float(ny0), 3),
        round(float(nx0 + target_w), 3),
        round(float(ny0 + target_h), 3),
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 3)
    return value


def _sample_layout(rng, *, width: int, height: int, setting_id: str, required_zones: Sequence[str] = ()) -> Dict[str, Any]:
    """Choose scene regions while preserving required semantic zone visibility."""

    layout_id = str(rng.choice(("curved_walk", "diagonal_walk", "playground_left", "pond_corner")))
    if str(setting_id) == "pond_playground":
        layout_id = str(rng.choice(("pond_corner", "curved_walk", "diagonal_walk")))
    path_y = float(rng.uniform(610.0, 710.0))
    path_amp = float(rng.uniform(34.0, 68.0))
    path_phase = float(rng.uniform(0.0, math.pi))
    if layout_id == "diagonal_walk":
        path_points = [
            (-40.0, float(height) * 0.82),
            (float(width) * 0.22, float(height) * 0.72),
            (float(width) * 0.52, float(height) * 0.66),
            (float(width) + 40.0, float(height) * 0.55),
        ]
    else:
        path_points = [
            (-40.0, path_y),
            (float(width) * 0.22, path_y - path_amp * 0.40),
            (float(width) * 0.50, path_y + path_amp * math.sin(path_phase)),
            (float(width) * 0.78, path_y - path_amp * 0.55),
            (float(width) + 40.0, path_y + path_amp * 0.20),
        ]
    if layout_id == "playground_left":
        playground_box = (60.0, 305.0, 510.0, 605.0)
        picnic_box = (760.0, 560.0, 1120.0, 760.0)
    elif layout_id == "pond_corner":
        playground_box = (650.0, 300.0, 1165.0, 600.0)
        picnic_box = (90.0, 560.0, 430.0, 760.0)
    else:
        playground_box = (float(rng.uniform(90.0, 170.0)), float(rng.uniform(295.0, 340.0)), float(rng.uniform(560.0, 670.0)), float(rng.uniform(575.0, 635.0)))
        picnic_box = (float(rng.uniform(760.0, 875.0)), float(rng.uniform(545.0, 610.0)), float(rng.uniform(1110.0, 1210.0)), float(rng.uniform(735.0, 810.0)))
    pond_box: BBox | None = None
    if str(setting_id) == "pond_playground" or layout_id == "pond_corner":
        pond_box = (float(rng.uniform(60.0, 150.0)), float(rng.uniform(320.0, 420.0)), float(rng.uniform(360.0, 500.0)), float(rng.uniform(500.0, 625.0)))
    if "pond" in {str(value) for value in required_zones} and pond_box is None:
        pond_box = (float(rng.uniform(62.0, 150.0)), float(rng.uniform(322.0, 410.0)), float(rng.uniform(360.0, 492.0)), float(rng.uniform(500.0, 620.0)))
    if layout_id == "playground_left":
        garden_box = (780.0, 300.0, 1188.0, 500.0)
    elif layout_id == "pond_corner":
        garden_box = (705.0, 625.0, 1150.0, 825.0)
    else:
        garden_box = (float(rng.uniform(520.0, 645.0)), float(rng.uniform(275.0, 335.0)), float(rng.uniform(930.0, 1060.0)), float(rng.uniform(455.0, 545.0)))
    playground_box = _clamp_bbox_to_canvas(playground_box, width=int(width), height=int(height), margin=24.0)
    picnic_box = _clamp_bbox_to_canvas(picnic_box, width=int(width), height=int(height), margin=24.0)
    garden_box = _clamp_bbox_to_canvas(garden_box, width=int(width), height=int(height), margin=24.0)
    if pond_box is not None:
        pond_box = _clamp_bbox_to_canvas(pond_box, width=int(width), height=int(height), margin=24.0)
    return {
        "layout_id": layout_id,
        "path_points": [[round(float(x), 3), round(float(y), 3)] for x, y in path_points],
        "path_width": round(float(rng.uniform(52.0, 72.0)), 3),
        "playground_bbox": [round(float(v), 3) for v in playground_box],
        "picnic_bbox": [round(float(v), 3) for v in picnic_box],
        "garden_bbox": [round(float(v), 3) for v in garden_box],
        "pond_bbox": [round(float(v), 3) for v in pond_box] if pond_box else None,
    }


def _draw_background(draw: ImageDraw.ImageDraw, *, rng, setting_id: str, layout: Mapping[str, Any], width: int, height: int, scale: int) -> List[ParkDecor]:
    """Draw stable named park areas before foreground objects are placed."""

    decor: List[ParkDecor] = []
    sky = _jitter_rgb(rng, (206, 229, 239), amount=10)
    grass = _jitter_rgb(rng, (128, 179, 113), amount=12)
    far_grass = _jitter_rgb(rng, (151, 196, 127), amount=12)
    s = int(scale)
    horizon = float(rng.uniform(190.0, 238.0))
    draw.rectangle((0, 0, int(width) * s, int(height) * s), fill=tuple(sky))
    draw.rectangle(_scale_bbox((0.0, horizon, float(width), float(height)), s), fill=tuple(grass))
    draw.rectangle(_scale_bbox((0.0, horizon, float(width), horizon + 42.0), s), fill=tuple(far_grass))
    for index in range(int(rng.randint(2, 5))):
        cx = float(rng.uniform(90.0, float(width) - 90.0))
        cy = float(rng.uniform(44.0, 135.0))
        cloud_w = float(rng.uniform(80.0, 140.0))
        cloud_h = float(rng.uniform(30.0, 48.0))
        for offset in (-0.28, 0.0, 0.26):
            _ellipse(draw, (cx + offset * cloud_w, cy, cx + offset * cloud_w + cloud_w * 0.48, cy + cloud_h), fill=(244, 248, 250), outline=None, width=1, scale=s)
    path_points = [(float(x), float(y)) for x, y in layout["path_points"]]
    path_width = int(round(float(layout["path_width"])))
    _line(draw, path_points, fill=_jitter_rgb(rng, (198, 182, 151), amount=10), width=path_width, scale=s)
    _line(draw, path_points, fill=_jitter_rgb(rng, (169, 151, 122), amount=8), width=max(3, int(path_width * 0.10)), scale=s)
    path_bbox = _bbox_union([(x - path_width * 0.5, y - path_width * 0.5, x + path_width * 0.5, y + path_width * 0.5) for x, y in path_points])
    path_bbox = _clamp_bbox_to_canvas(path_bbox, width=int(width), height=int(height), margin=0.0)
    decor.append(ParkDecor("decor_path", "walking_path", tuple(round(float(v), 3) for v in path_bbox), {"points": layout["path_points"], "width": float(path_width)}))
    playground = tuple(float(v) for v in layout["playground_bbox"])
    _ellipse(draw, playground, fill=_jitter_rgb(rng, (219, 186, 124), amount=12), outline=_jitter_rgb(rng, (139, 114, 84), amount=8), width=3, scale=s)
    decor.append(ParkDecor("decor_playground_zone", "playground_sand", tuple(round(float(v), 3) for v in playground), {}))
    picnic = tuple(float(v) for v in layout["picnic_bbox"])
    _rect(draw, picnic, fill=_jitter_rgb(rng, (132, 171, 118), amount=8), outline=_jitter_rgb(rng, (93, 128, 88), amount=7), width=2, scale=s, radius=18)
    blanket = (picnic[0] + 0.18 * (picnic[2] - picnic[0]), picnic[1] + 0.32 * (picnic[3] - picnic[1]), picnic[0] + 0.72 * (picnic[2] - picnic[0]), picnic[1] + 0.76 * (picnic[3] - picnic[1]))
    _rect(draw, blanket, fill=_jitter_rgb(rng, rng.choice(((205, 91, 86), (89, 138, 190), (219, 181, 84))), amount=8), outline=(92, 82, 76), width=2, scale=s, radius=8)
    decor.append(ParkDecor("decor_picnic_zone", "picnic_area", tuple(round(float(v), 3) for v in picnic), {"blanket_bbox": [round(float(v), 3) for v in blanket]}))
    garden = tuple(float(v) for v in layout["garden_bbox"])
    _ellipse(draw, garden, fill=_jitter_rgb(rng, (112, 157, 93), amount=8), outline=_jitter_rgb(rng, (69, 115, 66), amount=7), width=2, scale=s)
    for flower_index in range(int(rng.randint(10, 16))):
        px = float(rng.uniform(garden[0] + 16.0, garden[2] - 24.0))
        py = float(rng.uniform(garden[1] + 18.0, garden[3] - 28.0))
        _ellipse(
            draw,
            (px, py, px + 13.0, py + 11.0),
            fill=_jitter_rgb(rng, rng.choice(((219, 82, 117), (238, 197, 77), (146, 97, 177), (242, 134, 82))), amount=5),
            outline=None,
            width=1,
            scale=s,
        )
    decor.append(ParkDecor("decor_garden_zone", "garden_area", tuple(round(float(v), 3) for v in garden), {}))
    pond_raw = layout.get("pond_bbox")
    if isinstance(pond_raw, Sequence) and pond_raw:
        pond = tuple(float(v) for v in pond_raw)
        _ellipse(draw, pond, fill=_jitter_rgb(rng, (92, 169, 205), amount=10), outline=_jitter_rgb(rng, (60, 124, 159), amount=8), width=3, scale=s)
        for lily_index in range(int(rng.randint(2, 5))):
            lx = float(rng.uniform(pond[0] + 35.0, pond[2] - 45.0))
            ly = float(rng.uniform(pond[1] + 32.0, pond[3] - 36.0))
            leaf = (lx, ly, lx + 34.0, ly + 20.0)
            _ellipse(draw, leaf, fill=_jitter_rgb(rng, (73, 151, 91), amount=8), outline=(48, 105, 68), width=1, scale=s)
            decor.append(ParkDecor(f"decor_lily_{lily_index}", "lily_pad", tuple(round(float(v), 3) for v in leaf), {}))
        decor.append(ParkDecor("decor_pond", "pond", tuple(round(float(v), 3) for v in pond), {}))
    return decor


def _equipment_boxes(rng, *, playground: BBox, count: int) -> Tuple[BBox, ...]:
    columns = 3 if int(count) >= 5 else 2
    rows = max(1, (int(count) + columns - 1) // columns)
    gap_x = 18.0
    gap_y = 12.0
    inner = (playground[0] + 28.0, playground[1] + 34.0, playground[2] - 24.0, playground[3] - 18.0)
    cell_w = max(86.0, (inner[2] - inner[0] - float(columns - 1) * gap_x) / float(columns))
    cell_h = max(74.0, (inner[3] - inner[1] - float(rows - 1) * gap_y) / float(rows))
    boxes: List[BBox] = []
    for index in range(int(count)):
        col = index % columns
        row = index // columns
        x0 = inner[0] + float(col) * (cell_w + gap_x) + float(rng.uniform(-7.0, 7.0))
        y0 = inner[1] + float(row) * (cell_h + gap_y) + float(rng.uniform(-5.0, 6.0))
        ew = min(cell_w - 6.0, float(rng.uniform(94.0, 138.0)))
        eh = min(cell_h - 4.0, float(rng.uniform(74.0, 116.0)))
        box = (round(x0, 3), round(y0, 3), round(min(inner[2], x0 + ew), 3), round(min(inner[3], y0 + eh), 3))
        boxes.append(_expand_bbox_to_min_side(box, min_side=MIN_EQUIPMENT_BBOX_SIDE_PX, bounds=inner))
    return tuple(boxes)


def _draw_equipment_and_fixtures(
    draw: ImageDraw.ImageDraw,
    *,
    rng,
    layout: Mapping[str, Any],
    width: int,
    height: int,
    scale: int,
    style_id: str,
    equipment_specs: Sequence[ParkEquipmentSpec] | None = None,
) -> List[ParkDecor]:
    """Draw playground equipment first so people can attach to it."""

    decor: List[ParkDecor] = []
    playground = tuple(float(v) for v in layout["playground_bbox"])
    if equipment_specs is None:
        equipment_types = [str(value) for value in rng.sample(PARK_EQUIPMENT_TYPES, k=int(rng.randint(2, 4)))]
        resolved_specs = tuple(ParkEquipmentSpec(equipment_type=str(value), role="decor") for value in equipment_types)
    else:
        resolved_specs = tuple(equipment_specs)
    boxes = _equipment_boxes(rng, playground=playground, count=len(resolved_specs))
    ordered = list(zip(resolved_specs, boxes))
    rng.shuffle(ordered)
    for index, (equipment_spec, box) in enumerate(ordered):
        equipment_type = str(equipment_spec.equipment_type)
        visual_attributes = _park_equipment_visual_attributes(rng, equipment_type, str(style_id))
        semantic_attributes = {
            "decor_type": equipment_type,
            "equipment_type": equipment_type,
            "equipment_label": park_equipment_display_name(equipment_type),
            "zone": "playground",
            **dict(equipment_spec.attributes),
        }
        rendered = render_vector_scene_object(
            draw,
            object_id=f"equipment_{index:02d}",
            object_type="playground_equipment",
            bbox_xyxy=box,
            renderer_id="park_equipment",
            renderer_variant_id=equipment_type,
            semantic_attributes=semantic_attributes,
            visual_attributes=visual_attributes,
            role=str(equipment_spec.role),
            source_entity_type="park_decor",
            render_scale=scale,
            style_id=str(style_id),
        )
        decor.append(
            ParkDecor(
                f"equipment_{index:02d}",
                equipment_type,
                tuple(round(float(v), 3) for v in box),
                {"zone": "playground", "role": str(equipment_spec.role), **dict(equipment_spec.attributes)},
                object_record=rendered.object_record,
            )
        )
    tree_count = int(rng.randint(4, 8))
    for index in range(tree_count):
        h = float(rng.uniform(100.0, 165.0))
        w = h * float(rng.uniform(0.65, 0.90))
        x = float(rng.uniform(28.0, max(28.0, float(width) - w - 28.0)))
        y = float(rng.uniform(245.0, max(245.0, float(height) - h - 28.0)))
        box = (x, y, x + w, y + h)
        box = _clamp_bbox_to_canvas(box, width=int(width), height=int(height), margin=18.0)
        x, y, x1, y1 = [float(v) for v in box]
        w = x1 - x
        h = y1 - y
        trunk = (x + 0.43 * w, y + 0.52 * h, x + 0.58 * w, y + 0.96 * h)
        _rect(draw, trunk, fill=_jitter_rgb(rng, (117, 78, 49), amount=8), outline=(80, 60, 43), width=1, scale=scale, radius=4)
        for lx, ly in ((0.28, 0.20), (0.52, 0.12), (0.68, 0.30), (0.38, 0.39), (0.58, 0.46)):
            leaf = (x + (lx - 0.20) * w, y + (ly - 0.15) * h, x + (lx + 0.20) * w, y + (ly + 0.18) * h)
            _ellipse(draw, leaf, fill=_jitter_rgb(rng, (71, 139, 80), amount=12), outline=(51, 103, 60), width=1, scale=scale)
        decor.append(ParkDecor(f"decor_tree_{index}", "tree", tuple(round(float(v), 3) for v in box), {}))
    fixture_count = int(rng.randint(3, 6))
    for index in range(fixture_count):
        fixture_type = str(rng.choice(("bench", "lamp_post", "flower_bed")))
        x = float(rng.uniform(56.0, float(width) - 170.0))
        y = float(rng.uniform(625.0, float(height) - 88.0))
        if fixture_type == "bench":
            box = (x, y, x + 130.0, y + 52.0)
            box = _clamp_bbox_to_canvas(box, width=int(width), height=int(height), margin=20.0)
            rendered = render_vector_scene_object(
                draw,
                object_id=f"decor_fixture_{index}",
                object_type="bench",
                bbox_xyxy=box,
                renderer_id="fixture_bench",
                renderer_variant_id="bench",
                semantic_attributes={"decor_type": fixture_type},
                visual_attributes=_bench_visual_attributes(rng, style_id=str(style_id)),
                role="distractor",
                source_entity_type="park_decor",
                render_scale=scale,
                style_id=str(style_id),
            )
            object_record = rendered.object_record
        elif fixture_type == "lamp_post":
            box = (x, y - 110.0, x + 38.0, y + 20.0)
            box = _clamp_bbox_to_canvas(box, width=int(width), height=int(height), margin=18.0)
            x0, y0, x1, y1 = [float(v) for v in box]
            cx = 0.5 * (x0 + x1)
            _line(draw, [(cx, y0 + 34.0), (cx, y1 - 2.0)], fill=(74, 82, 91), width=5, scale=scale)
            _ellipse(draw, (cx - 15.0, y0, cx + 15.0, y0 + 34.0), fill=(246, 220, 122), outline=(74, 82, 91), width=2, scale=scale)
            object_record = None
        else:
            box = (x, y, x + 148.0, y + 48.0)
            box = _clamp_bbox_to_canvas(box, width=int(width), height=int(height), margin=20.0)
            x, y, x1, y1 = [float(v) for v in box]
            _ellipse(draw, box, fill=_jitter_rgb(rng, (93, 148, 81), amount=8), outline=(65, 111, 65), width=1, scale=scale)
            for petal in range(8):
                px = x + float(rng.uniform(16.0, max(17.0, x1 - x - 16.0)))
                py = y + float(rng.uniform(10.0, max(11.0, y1 - y - 14.0)))
                _ellipse(draw, (px, py, px + 12.0, py + 10.0), fill=_jitter_rgb(rng, rng.choice(((221, 85, 114), (238, 197, 77), (145, 97, 177))), amount=5), outline=None, width=1, scale=scale)
            object_record = None
        decor.append(ParkDecor(f"decor_fixture_{index}", fixture_type, tuple(round(float(v), 3) for v in box), {}, object_record=object_record))
    return decor


def _path_point(rng, layout: Mapping[str, Any]) -> Tuple[float, float]:
    points = [(float(x), float(y)) for x, y in layout["path_points"]]
    segment = int(rng.randint(0, max(0, len(points) - 2)))
    t = float(rng.uniform(0.05, 0.95))
    x0, y0 = points[segment]
    x1, y1 = points[segment + 1]
    return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))


def _zone_bbox(layout: Mapping[str, Any], zone: str) -> BBox | None:
    if str(zone) == "playground":
        return tuple(float(v) for v in layout["playground_bbox"])  # type: ignore[return-value]
    if str(zone) == "picnic":
        return tuple(float(v) for v in layout["picnic_bbox"])  # type: ignore[return-value]
    if str(zone) == "garden":
        return tuple(float(v) for v in layout["garden_bbox"])  # type: ignore[return-value]
    return None


def _zone_for_point(layout: Mapping[str, Any], point: Tuple[float, float]) -> str:
    px, py = float(point[0]), float(point[1])
    for zone in PARK_ZONE_TYPES:
        box = _zone_bbox(layout, str(zone))
        if box and box[0] <= px <= box[2] and box[1] <= py <= box[3]:
            return str(zone)
    return "open_lawn"


def _equipment_user_box(
    rng,
    *,
    equipment_decor: Sequence[ParkDecor],
    equipment_type: str,
    activity: str,
    width: int,
    height: int,
) -> BBox | None:
    """Return a person box aligned to an existing equipment item."""

    matches = tuple(
        item
        for item in equipment_decor
        if str(item.decor_id).startswith("equipment_") and str(item.decor_type) == str(equipment_type)
    )
    if not matches:
        return None
    item = rng.choice(matches)
    ex0, ey0, ex1, ey1 = (float(v) for v in item.bbox_xyxy)
    ew = max(1.0, ex1 - ex0)
    eh = max(1.0, ey1 - ey0)
    if str(equipment_type) == "slide":
        person_h = float(rng.uniform(68.0, 88.0))
        person_w = person_h * float(rng.uniform(0.50, 0.62))
        cx = ex0 + ew * float(rng.uniform(0.42, 0.78))
        bottom = ey0 + eh * float(rng.uniform(0.66, 0.94))
    elif str(equipment_type) == "swing_set":
        person_h = float(rng.uniform(60.0, 78.0)) if str(activity) == "sitting" else float(rng.uniform(68.0, 86.0))
        person_w = person_h * float(rng.uniform(0.60, 0.76))
        seat_fraction = float(rng.choice((0.36, 0.62)))
        cx = ex0 + ew * seat_fraction + float(rng.uniform(-0.05, 0.05)) * ew
        bottom = ey0 + eh * float(rng.uniform(0.78, 0.96))
    elif str(equipment_type) == "seesaw":
        person_h = float(rng.uniform(60.0, 78.0)) if str(activity) == "sitting" else float(rng.uniform(68.0, 86.0))
        person_w = person_h * float(rng.uniform(0.60, 0.78))
        end_fraction = float(rng.choice((0.24, 0.76)))
        cx = ex0 + ew * end_fraction + float(rng.uniform(-0.04, 0.04)) * ew
        bottom = ey0 + eh * float(rng.uniform(0.70, 0.88))
    else:
        person_h = float(rng.uniform(70.0, 90.0))
        person_w = person_h * float(rng.uniform(0.48, 0.62))
        cx = ex0 + ew * float(rng.uniform(0.30, 0.70))
        bottom = ey0 + eh * float(rng.uniform(0.58, 0.88))
    x0 = max(30.0, min(float(width) - person_w - 30.0, cx - person_w * 0.5))
    y1 = max(320.0, min(float(height) - 20.0, bottom))
    return (round(x0, 3), round(y1 - person_h, 3), round(x0 + person_w, 3), round(y1, 3))


def _candidate_person_box(
    rng,
    *,
    activity: str,
    layout: Mapping[str, Any],
    width: int,
    height: int,
    zone: str | None = None,
    equipment_decor: Sequence[ParkDecor] = (),
    equipment_type: str | None = None,
) -> BBox:
    """Sample one activity-aware person box from zones or equipment."""

    if equipment_type is not None:
        equipment_box = _equipment_user_box(
            rng,
            equipment_decor=equipment_decor,
            equipment_type=str(equipment_type),
            activity=str(activity),
            width=int(width),
            height=int(height),
        )
        if equipment_box is not None:
            return equipment_box
    zone_box = _zone_bbox(layout, str(zone)) if zone else None
    if zone_box is not None:
        person_h = float(rng.uniform(76.0, 104.0))
        if str(activity) == "sitting":
            person_h = float(rng.uniform(66.0, 86.0))
        person_w = person_h * float(rng.uniform(0.52, 0.74))
        cx_low = min(zone_box[2] - 24.0, zone_box[0] + max(34.0, 0.55 * person_w))
        cx_high = max(cx_low, zone_box[2] - max(34.0, 0.55 * person_w))
        bottom_low = min(zone_box[3] - 8.0, zone_box[1] + person_h + 12.0)
        bottom_high = max(bottom_low, zone_box[3] - 8.0)
        cx = float(rng.uniform(cx_low, cx_high))
        bottom = float(rng.uniform(bottom_low, bottom_high))
    elif str(activity) == "sitting":
        person_h = float(rng.uniform(68.0, 88.0))
        person_w = person_h * float(rng.uniform(0.68, 0.84))
        if float(rng.random()) < 0.55:
            picnic = tuple(float(v) for v in layout["picnic_bbox"])
            cx = float(rng.uniform(picnic[0] + 40.0, picnic[2] - 40.0))
            bottom = float(rng.uniform(picnic[1] + 110.0, min(picnic[3] + 28.0, float(height) - 40.0)))
        else:
            cx = float(rng.uniform(110.0, float(width) - 110.0))
            bottom = float(rng.uniform(float(height) * 0.69, float(height) - 46.0))
    elif str(activity) == "walking":
        person_h = float(rng.uniform(88.0, 112.0))
        person_w = person_h * float(rng.uniform(0.48, 0.60))
        cx, cy = _path_point(rng, layout)
        cx += float(rng.uniform(-34.0, 34.0))
        bottom = cy + float(rng.uniform(28.0, 48.0))
    elif str(activity) == "playing_ball":
        person_h = float(rng.uniform(84.0, 108.0))
        person_w = person_h * float(rng.uniform(0.50, 0.62))
        if float(rng.random()) < 0.65:
            playground = tuple(float(v) for v in layout["playground_bbox"])
            cx = float(rng.uniform(playground[0] + 38.0, playground[2] - 38.0))
            bottom = float(rng.uniform(playground[1] + 150.0, playground[3] + 42.0))
        else:
            cx = float(rng.uniform(110.0, float(width) - 110.0))
            bottom = float(rng.uniform(560.0, float(height) - 50.0))
    else:
        person_h = float(rng.uniform(88.0, 114.0))
        person_w = person_h * float(rng.uniform(0.48, 0.58))
        cx = float(rng.uniform(90.0, float(width) - 90.0))
        bottom = float(rng.uniform(500.0, float(height) - 44.0))
    x0 = max(30.0, min(float(width) - person_w - 30.0, cx - person_w * 0.5))
    y1 = max(360.0, min(float(height) - 20.0, bottom))
    return (round(x0, 3), round(y1 - person_h, 3), round(x0 + person_w, 3), round(y1, 3))


def _place_persons(
    rng,
    *,
    specs: Sequence[ParkPersonSpec],
    layout: Mapping[str, Any],
    width: int,
    height: int,
    equipment_decor: Sequence[ParkDecor] = (),
) -> Tuple[Tuple[ParkPersonSpec, BBox, str], ...]:
    """Place all requested people while avoiding severe visual overlap."""

    placed: List[Tuple[ParkPersonSpec, BBox, str]] = []
    existing: List[BBox] = []
    ordered = list(specs)
    rng.shuffle(ordered)
    for index, spec in enumerate(ordered):
        box: BBox | None = None
        last_candidate: BBox | None = None
        requested_zone = spec.attributes.get("zone")
        equipment_type = spec.attributes.get("using_equipment_type")
        zone = "playground" if equipment_type is not None else (str(requested_zone) if requested_zone is not None else None)
        for attempt in range(220):
            candidate = _candidate_person_box(
                rng,
                activity=str(spec.activity),
                layout=layout,
                width=width,
                height=height,
                zone=zone,
                equipment_decor=equipment_decor,
                equipment_type=str(equipment_type) if equipment_type is not None else None,
            )
            candidate = _clamp_bbox_to_canvas(candidate, width=int(width), height=int(height))
            last_candidate = candidate
            gap = 12.0 if int(attempt) < 130 else 5.0 if int(attempt) < 190 else 0.0
            if all(not _expanded_intersects(candidate, other, gap) for other in existing):
                box = candidate
                break
        if box is None:
            if last_candidate is not None:
                box = last_candidate
            else:
                col = index % 6
                row = index // 6
                px = 88.0 + col * 190.0
                py = 500.0 + row * 160.0
                box = _clamp_bbox_to_canvas((px, py, px + 56.0, py + 96.0), width=int(width), height=int(height))
        existing.append(tuple(float(v) for v in box))
        center = (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))
        resolved_zone = zone if zone in set(PARK_ZONE_TYPES) else _zone_for_point(layout, center)
        placed.append((spec, tuple(float(v) for v in box), str(resolved_zone)))
    return tuple(placed)


def _skin_color(rng) -> RGB:
    return tuple(int(v) for v in rng.choice(((200, 145, 101), (169, 111, 82), (219, 164, 113), (136, 89, 68), (232, 181, 130))))  # type: ignore[return-value]


def _clothes_color(rng) -> RGB:
    return tuple(int(v) for v in rng.choice(((74, 122, 180), (194, 83, 85), (220, 169, 75), (94, 151, 124), (130, 104, 160), (83, 154, 177))))  # type: ignore[return-value]


def _draw_activity_support(
    draw: ImageDraw.ImageDraw,
    *,
    rng,
    person_id: str,
    activity: str,
    bbox: BBox,
    width: int,
    height: int,
    scale: int,
    style_id: str,
) -> Tuple[ParkDecor, ...]:
    """Draw small activity-linked support props while preserving person-owned decor ids."""

    x0, y0, x1, y1 = bbox
    decor: List[ParkDecor] = []
    if str(activity) == "sitting":
        bench = (x0 - 24.0, y0 + 0.47 * (y1 - y0), x1 + 34.0, y0 + 0.86 * (y1 - y0))
        bench = _clamp_bbox_to_canvas(bench, width=int(width), height=int(height), margin=12.0)
        rendered = render_vector_scene_object(
            draw,
            object_id=f"{person_id}_bench",
            object_type="bench",
            bbox_xyxy=bench,
            renderer_id="fixture_bench",
            renderer_variant_id="bench",
            semantic_attributes={"decor_type": "activity_bench", "supports_person_id": str(person_id)},
            visual_attributes=_bench_visual_attributes(rng, style_id=str(style_id)),
            role="distractor",
            source_entity_type="park_decor",
            render_scale=scale,
            style_id=str(style_id),
        )
        decor.append(ParkDecor(f"{person_id}_bench", "activity_bench", tuple(round(float(v), 3) for v in bench), {"supports_person_id": str(person_id)}, object_record=rendered.object_record))
    if str(activity) == "playing_ball":
        ball_size = max(18.0, 0.20 * (y1 - y0))
        side = -1.0 if float(rng.random()) < 0.5 else 1.0
        bx = x0 - ball_size * 0.9 if side < 0 else x1 + ball_size * 0.25
        by = y1 - ball_size * float(rng.uniform(0.75, 1.05))
        ball = (bx, by, bx + ball_size, by + ball_size)
        ball = _clamp_bbox_to_canvas(ball, width=int(width), height=int(height), margin=12.0)
        _ellipse(draw, ball, fill=_jitter_rgb(rng, (236, 128, 71), amount=8), outline=(65, 71, 80), width=2, scale=scale)
        _line(draw, [(ball[0] + 0.18 * ball_size, ball[1] + 0.50 * ball_size), (ball[2] - 0.18 * ball_size, ball[1] + 0.50 * ball_size)], fill=(65, 71, 80), width=1, scale=scale)
        decor.append(ParkDecor(f"{person_id}_ball", "activity_ball", tuple(round(float(v), 3) for v in ball), {"supports_person_id": str(person_id)}))
    return tuple(decor)


def render_park_playground_scene(
    *,
    rng,
    person_specs: Sequence[ParkPersonSpec],
    equipment_specs: Sequence[ParkEquipmentSpec] | None = None,
    required_zones: Sequence[str] = (),
    canvas_width: int = 1280,
    canvas_height: int = 900,
    render_scale: int = 2,
    setting_weights: Mapping[str, float] | None = None,
    style_weights: Mapping[str, float] | None = None,
) -> RenderedParkPlaygroundScene:
    """Render one synthetic park/playground scene from semantic person specs."""

    width = int(canvas_width)
    height = int(canvas_height)
    scale = max(1, int(render_scale))
    if not person_specs:
        raise ValueError("park playground scene needs at least one person")
    setting_id = _choose_weighted(rng, setting_weights or {setting: 1.0 for setting in PARK_SETTING_IDS}, PARK_SETTING_IDS)
    style_id = _choose_weighted(rng, style_weights or {style: 1.0 for style in STYLE_IDS}, STYLE_IDS)
    layout = _sample_layout(rng, width=width, height=height, setting_id=str(setting_id), required_zones=tuple(required_zones))
    image = Image.new("RGB", (width * scale, height * scale), (232, 238, 221))
    draw = ImageDraw.Draw(image)
    decor: List[ParkDecor] = []
    decor.extend(_draw_background(draw, rng=rng, setting_id=str(setting_id), layout=layout, width=width, height=height, scale=scale))
    decor.extend(_draw_equipment_and_fixtures(draw, rng=rng, layout=layout, width=width, height=height, scale=scale, style_id=str(style_id), equipment_specs=equipment_specs))
    equipment_decor = tuple(item for item in decor if str(item.decor_id).startswith("equipment_"))
    placed = _place_persons(rng, specs=person_specs, layout=layout, width=width, height=height, equipment_decor=equipment_decor)
    persons: List[ParkPerson] = []
    placed_sorted = sorted(placed, key=lambda item: (float(item[1][3]), float(item[1][0])))
    for index, (spec, bbox, _zone) in enumerate(placed_sorted):
        if bool(spec.attributes.get("suppress_activity_support", False)):
            continue
        person_id = f"person_{index:02d}"
        decor.extend(
            _draw_activity_support(
                draw,
                rng=rng,
                person_id=person_id,
                activity=str(spec.activity),
                bbox=bbox,
                width=int(width),
                height=int(height),
                scale=scale,
                style_id=str(style_id),
            )
        )
    for index, (spec, bbox, zone) in enumerate(placed_sorted):
        person_id = f"person_{index:02d}"
        primary = _clothes_color(rng)
        accent = _clothes_color(rng)
        skin = _skin_color(rng)
        gender_id = sample_person_gender(rng)
        semantic_attributes = {
            "activity": str(spec.activity),
            "activity_label": park_activity_display_name(str(spec.activity)),
            **dict(spec.attributes),
            "zone": str(zone),
        }
        visual_attributes = {
            "primary_color_rgb": [int(v) for v in primary],
            "accent_color_rgb": [int(v) for v in accent],
            "skin_color_rgb": [int(v) for v in skin],
            "style_id": str(style_id),
            "gender_id": str(gender_id),
        }
        rendered = render_illustration_object(
            IllustrationObjectSpec(
                object_id=person_id,
                object_type="person",
                bbox_xyxy=bbox,
                renderer_id="park_person",
                renderer_variant_id=str(spec.activity),
                semantic_attributes=semantic_attributes,
                visual_attributes=visual_attributes,
                role=str(spec.role),
                source_entity_type="park_person",
            ),
            RenderContext(
                renderer_style=RENDERER_STYLE_VECTOR,
                draw=draw,
                render_scale=scale,
                style_id=str(style_id),
            ),
        )
        persons.append(
            ParkPerson(
                person_id=person_id,
                activity=str(spec.activity),
                activity_label=park_activity_display_name(str(spec.activity)),
                bbox_xyxy=tuple(round(float(v), 3) for v in bbox),
                primary_color_rgb=tuple(int(v) for v in primary),
                accent_color_rgb=tuple(int(v) for v in accent),
                skin_color_rgb=tuple(int(v) for v in skin),
                style_id=str(style_id),
                gender_id=str(gender_id),
                role=str(spec.role),
                attributes={**dict(spec.attributes), "zone": str(zone)},
                object_record=rendered.object_record,
            )
        )
    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return RenderedParkPlaygroundScene(
        image=image,
        setting_id=str(setting_id),
        persons=tuple(persons),
        decor=tuple(decor),
        canvas_width=width,
        canvas_height=height,
        render_scale=scale,
        style_id=str(style_id),
        layout=dict(layout),
    )


def park_person_bbox_map(scene: RenderedParkPlaygroundScene) -> Dict[str, List[float]]:
    """Return person bboxes keyed by person id."""

    return {str(person.person_id): [round(float(v), 3) for v in person.bbox_xyxy] for person in scene.persons}


def park_decor_bbox_map(scene: RenderedParkPlaygroundScene) -> Dict[str, List[float]]:
    """Return decor bboxes keyed by decor id."""

    return {str(item.decor_id): [round(float(v), 3) for v in item.bbox_xyxy] for item in scene.decor}


def sort_park_bboxes(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> List[List[float]]:
    """Return bboxes sorted top-to-bottom then left-to-right for stable annotation."""

    boxes = [list(float(v) for v in bbox_map[str(item_id)]) for item_id in ids]
    boxes.sort(key=lambda box: (round(float(box[1]), 3), round(float(box[0]), 3), round(float(box[3]), 3), round(float(box[2]), 3)))
    return [[round(float(v), 3) for v in box] for box in boxes]


def park_scene_entities(scene: RenderedParkPlaygroundScene) -> List[Dict[str, Any]]:
    """Return generic entity records for the scene trace."""

    entities: List[Dict[str, Any]] = []
    for person in scene.persons:
        object_record = (
            dict(person.object_record)
            if person.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(person.person_id),
                object_type="person",
                bbox_xyxy=person.bbox_xyxy,
                semantic_attributes={
                    "activity": str(person.activity),
                    "activity_label": str(person.activity_label),
                    **dict(person.attributes),
                },
                visual_attributes={
                    "primary_color_rgb": [int(v) for v in person.primary_color_rgb],
                    "accent_color_rgb": [int(v) for v in person.accent_color_rgb],
                    "skin_color_rgb": [int(v) for v in person.skin_color_rgb],
                    "style_id": str(person.style_id),
                    "gender_id": str(person.gender_id),
                },
                role=str(person.role),
                source_entity_type="park_person",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(person.person_id),
                "entity_type": "park_person",
                "activity": str(person.activity),
                "activity_label": str(person.activity_label),
                "bbox": [round(float(v), 3) for v in person.bbox_xyxy],
                "role": str(person.role),
                "attributes": _safe_json(person.attributes),
                "object_record": object_record,
            }
        )
    for item in scene.decor:
        object_type = _decor_object_type(str(item.decor_type))
        semantic_attributes = {
            "decor_type": str(item.decor_type),
            **dict(item.attributes),
        }
        if object_type == "playground_equipment":
            semantic_attributes["equipment_type"] = str(item.decor_type)
            semantic_attributes["equipment_label"] = park_equipment_display_name(str(item.decor_type))
        object_record = (
            dict(item.object_record)
            if item.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(item.decor_id),
                object_type=object_type,
                bbox_xyxy=item.bbox_xyxy,
                semantic_attributes=semantic_attributes,
                role=str(item.attributes.get("role", "distractor")),
                source_entity_type="park_decor",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(item.decor_id),
                "entity_type": "park_decor",
                "decor_type": str(item.decor_type),
                "bbox": [round(float(v), 3) for v in item.bbox_xyxy],
                "attributes": _safe_json(item.attributes),
                "object_record": object_record,
            }
        )
    return entities


def serialize_park_scene(scene: RenderedParkPlaygroundScene) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Serialize park scene records and person bbox map."""

    person_records = [
        {
            "person_id": str(person.person_id),
            "activity": str(person.activity),
            "activity_label": str(person.activity_label),
            "bbox": [round(float(v), 3) for v in person.bbox_xyxy],
            "primary_color_rgb": [int(v) for v in person.primary_color_rgb],
            "accent_color_rgb": [int(v) for v in person.accent_color_rgb],
            "skin_color_rgb": [int(v) for v in person.skin_color_rgb],
            "style_id": str(person.style_id),
            "gender_id": str(person.gender_id),
            "role": str(person.role),
            "attributes": _safe_json(person.attributes),
        }
        for person in scene.persons
    ]
    decor_records = [
        {
            "decor_id": str(item.decor_id),
            "decor_type": str(item.decor_type),
            "bbox": [round(float(v), 3) for v in item.bbox_xyxy],
            "attributes": _safe_json(item.attributes),
        }
        for item in scene.decor
    ]
    return (
        [
            {
                "setting_id": str(scene.setting_id),
                "layout": _safe_json(scene.layout),
                "persons": person_records,
                "decor": decor_records,
            }
        ],
        park_person_bbox_map(scene),
    )


__all__ = [
    "PARK_EQUIPMENT_LABELS",
    "PARK_EQUIPMENT_TYPES",
    "PARK_PERSON_ACTIVITIES",
    "PARK_PERSON_ACTIVITY_LABELS",
    "PARK_SETTING_IDS",
    "PARK_ZONE_LABELS",
    "PARK_ZONE_TYPES",
    "ParkEquipmentSpec",
    "ParkPersonSpec",
    "RenderedParkPlaygroundScene",
    "park_activity_display_name",
    "park_decor_bbox_map",
    "park_equipment_display_name",
    "park_person_bbox_map",
    "park_scene_entities",
    "park_zone_display_name",
    "render_park_playground_scene",
    "serialize_park_scene",
    "sort_park_bboxes",
]
