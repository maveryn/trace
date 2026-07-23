"""Layout and placement helpers for construction-site illustrations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...shared.object_library import BBox
from ...shared.bounds import clamp_box_size_to_region

from .labels import construction_zone_display_name
from .state import CONSTRUCTION_ZONE_TYPES, ConstructionZone


def _expanded_intersects(a: BBox, b: BBox, gap: float) -> bool:
    return not (
        float(a[2]) + float(gap) <= float(b[0])
        or float(b[2]) + float(gap) <= float(a[0])
        or float(a[3]) + float(gap) <= float(b[1])
        or float(b[3]) + float(gap) <= float(a[1])
    )


def _canvas_profile_kind(*, width: int, height: int) -> str:
    if int(height) > int(width):
        return "portrait"
    if abs(int(width) - int(height)) <= max(12, int(0.04 * max(int(width), int(height)))):
        return "square"
    return "landscape"


def _profile_zone_bboxes(layout_id: str, *, width: int, height: int) -> Dict[str, BBox]:
    """Return profile-aware construction zone boxes before small jitter."""

    w = float(width)
    h = float(height)
    profile = _canvas_profile_kind(width=int(width), height=int(height))
    margin_x = max(38.0, min(72.0, 0.058 * w))
    bottom = h - max(34.0, min(54.0, 0.045 * h))
    gap = max(22.0, min(34.0, 0.028 * w))

    if profile == "portrait":
        top = max(250.0, 0.23 * h)
        row_h = max(172.0, (bottom - top - 2.0 * gap) / 3.0)
        row_h = min(row_h, (bottom - top - 2.0 * gap) / 3.0)
        boxes = {}
        for index, zone_id in enumerate(CONSTRUCTION_ZONE_TYPES):
            y0 = top + float(index) * (row_h + gap)
            boxes[str(zone_id)] = (margin_x, y0, w - margin_x, y0 + row_h)
        return boxes

    if profile == "square":
        top = max(240.0, 0.245 * h)
        row_h = (bottom - top - gap) / 2.0
        col_w = (w - 2.0 * margin_x - gap) / 2.0
        return {
            "excavation_zone": (margin_x, top, margin_x + col_w, top + row_h),
            "loading_zone": (margin_x + col_w + gap, top, w - margin_x, top + row_h),
            "roadwork_zone": (margin_x, top + row_h + gap, w - margin_x, bottom),
        }

    top = max(232.0, 0.30 * h)
    if str(layout_id) == "vertical_yard":
        col_w = (w - 2.0 * margin_x - 2.0 * gap) / 3.0
        return {
            "excavation_zone": (margin_x, top, margin_x + col_w, bottom),
            "loading_zone": (margin_x + col_w + gap, top - 4.0, margin_x + 2.0 * col_w + gap, bottom),
            "roadwork_zone": (margin_x + 2.0 * (col_w + gap), top, w - margin_x, bottom),
        }
    if str(layout_id) == "diagonal_road":
        top_h = max(250.0, 0.38 * h)
        return {
            "excavation_zone": (margin_x, top, margin_x + 0.36 * w, top + top_h),
            "loading_zone": (w - margin_x - 0.40 * w, top - 2.0, w - margin_x, top + top_h),
            "roadwork_zone": (margin_x + 0.12 * w, top + top_h + gap, w - margin_x - 0.06 * w, bottom),
        }
    if str(layout_id) == "scaffold_front":
        zone_h = max(225.0, 0.31 * h)
        y0 = bottom - zone_h
        col_w = (w - 2.0 * margin_x - 2.0 * gap) / 3.0
        return {
            "excavation_zone": (margin_x, y0, margin_x + col_w, bottom),
            "loading_zone": (margin_x + col_w + gap, y0, margin_x + 2.0 * col_w + gap, bottom),
            "roadwork_zone": (margin_x + 2.0 * (col_w + gap), y0, w - margin_x, bottom),
        }

    top_h = max(250.0, 0.39 * h)
    return {
        "excavation_zone": (margin_x, top, margin_x + 0.38 * w, top + top_h),
        "loading_zone": (margin_x + 0.42 * w, top - 2.0, w - margin_x, top + top_h),
        "roadwork_zone": (margin_x + 0.08 * w, top + top_h + gap, w - margin_x - 0.02 * w, bottom),
    }


def sample_construction_layout(rng, *, width: int, height: int, setting_id: str) -> Dict[str, Any]:
    """Sample stable profile-aware work-zone geometry shared by construction tasks."""

    layout_id = str(rng.choice(("horizontal_yard", "vertical_yard", "diagonal_road", "scaffold_front")))
    if str(setting_id) == "roadwork":
        layout_id = str(rng.choice(("diagonal_road", "horizontal_yard", "vertical_yard")))
    elif str(setting_id) == "scaffold_site":
        layout_id = str(rng.choice(("scaffold_front", "horizontal_yard", "vertical_yard")))

    zones = _profile_zone_bboxes(layout_id, width=int(width), height=int(height))
    jittered: Dict[str, List[float]] = {}
    for zone_id, box in zones.items():
        dx = float(rng.uniform(-10.0, 10.0))
        dy = float(rng.uniform(-8.0, 8.0))
        box_w = float(box[2]) - float(box[0])
        box_h = float(box[3]) - float(box[1])
        min_x0 = max(34.0, float(width) * 0.035)
        min_y0 = max(204.0, float(height) * 0.19)
        max_x0 = max(min_x0, float(width) - min_x0 - box_w)
        max_y0 = max(min_y0, float(height) - 32.0 - box_h)
        x0 = max(min_x0, min(float(box[0]) + dx, max_x0))
        y0 = max(min_y0, min(float(box[1]) + dy, max_y0))
        jittered[zone_id] = [
            round(x0, 3),
            round(y0, 3),
            round(x0 + box_w, 3),
            round(y0 + box_h, 3),
        ]
    return {
        "layout_id": layout_id,
        "canvas_profile_kind": _canvas_profile_kind(width=int(width), height=int(height)),
        "zone_bboxes": jittered,
    }


def build_construction_zones(layout: Mapping[str, Any]) -> Tuple[ConstructionZone, ...]:
    """Build visible labeled zones from sampled construction layout."""

    zone_bboxes = layout.get("zone_bboxes", {})
    fills = {
        "excavation_zone": (218, 191, 136),
        "loading_zone": (195, 208, 217),
        "roadwork_zone": (207, 197, 176),
    }
    outlines = {
        "excavation_zone": (140, 102, 54),
        "loading_zone": (86, 113, 133),
        "roadwork_zone": (126, 115, 94),
    }
    zones: List[ConstructionZone] = []
    for zone_id in CONSTRUCTION_ZONE_TYPES:
        raw = zone_bboxes.get(zone_id)
        if not isinstance(raw, Sequence) or len(raw) != 4:
            raise ValueError(f"missing construction zone bbox for {zone_id}")
        zones.append(
            ConstructionZone(
                zone_id=str(zone_id),
                label=construction_zone_display_name(str(zone_id)),
                bbox_xyxy=tuple(float(v) for v in raw),  # type: ignore[arg-type]
                fill_rgb=fills[str(zone_id)],
                outline_rgb=outlines[str(zone_id)],
            )
        )
    return tuple(zones)


def construction_zone_lookup(zones: Sequence[ConstructionZone]) -> Dict[str, ConstructionZone]:
    """Return zones keyed by zone id."""

    return {str(zone.zone_id): zone for zone in zones}


def place_construction_box(
    rng,
    zone_bbox: BBox,
    *,
    width: float,
    height: float,
    occupied: List[BBox],
    gap: float,
    protected: Sequence[BBox] | None = None,
    allow_overlap_fallback: bool = True,
    max_attempts: int = 180,
) -> BBox:
    """Place a non-overlapping bbox inside a construction zone."""

    width, height = clamp_box_size_to_region(
        width=float(width),
        height=float(height),
        region_bbox=zone_bbox,
        padding_x=18.0,
        padding_y=54.0,
        min_width=28.0,
        min_height=36.0,
    )
    x0_min = float(zone_bbox[0]) + 18.0
    x0_max = float(zone_bbox[2]) - float(width) - 18.0
    y0_min = float(zone_bbox[1]) + 50.0
    y0_max = float(zone_bbox[3]) - float(height) - 14.0
    if x0_max < x0_min:
        x0_max = x0_min
    if y0_max < y0_min:
        y0_max = y0_min
    gap_candidates = (float(gap), 0.0) if float(gap) > 0.0 else (0.0,)

    def try_place(blockers: Sequence[BBox], active_gap: float) -> BBox | None:
        for _ in range(int(max_attempts)):
            x0 = float(rng.uniform(x0_min, x0_max))
            y0 = float(rng.uniform(y0_min, y0_max))
            box = (x0, y0, x0 + float(width), y0 + float(height))
            if not any(_expanded_intersects(box, other, float(active_gap)) for other in blockers):
                return box
        return None

    for active_gap in gap_candidates:
        box = try_place(occupied, float(active_gap))
        if box is not None:
            occupied.append(box)
            return box

    protected_boxes = tuple(protected or ())
    if protected_boxes:
        for active_gap in gap_candidates:
            box = try_place(protected_boxes, float(active_gap))
            if box is not None:
                occupied.append(box)
                return box

    if bool(allow_overlap_fallback):
        x0 = float(rng.uniform(x0_min, x0_max))
        y0 = float(rng.uniform(y0_min, y0_max))
        box = (x0, y0, x0 + float(width), y0 + float(height))
        occupied.append(box)
        return box

    raise ValueError("could not place a non-overlapping construction item")


__all__ = [
    "build_construction_zones",
    "construction_zone_lookup",
    "place_construction_box",
    "sample_construction_layout",
]
