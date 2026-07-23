"""Canvas-profile bounds tests for migrated illustration scenes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task


MIGRATED_ILLUSTRATION_TASK_IDS: tuple[str, ...] = (
    "task_illustrations__construction_site__equipment_zone_count",
    "task_illustrations__construction_site__missing_patch_label",
    "task_illustrations__construction_site__rotated_tile_label",
    "task_illustrations__construction_site__worker_attribute_count",
    "task_illustrations__environment__crossing_feature_count",
    "task_illustrations__environment__feature_relation_object_count",
    "task_illustrations__environment__lit_window_count",
    "task_illustrations__environment__missing_patch_label",
    "task_illustrations__environment__rotated_tile_label",
    "task_illustrations__indoor_room__furniture_side_count",
    "task_illustrations__indoor_room__missing_patch_label",
    "task_illustrations__indoor_room__rotated_tile_label",
    "task_illustrations__indoor_room__swapped_tile_pair_label",
    "task_illustrations__indoor_room__surface_object_count",
    "task_illustrations__library__books_in_section_count",
    "task_illustrations__library__filtered_book_in_section_count",
    "task_illustrations__library__missing_patch_label",
    "task_illustrations__library__rotated_tile_label",
    "task_illustrations__library__swapped_tile_pair_label",
    "task_illustrations__park_playground__jigsaw_arrangement_label",
    "task_illustrations__park_playground__missing_patch_label",
    "task_illustrations__park_playground__person_count",
    "task_illustrations__park_playground__playground_equipment_count",
    "task_illustrations__park_playground__rotated_tile_label",
    "task_illustrations__park_playground__swapped_tile_pair_label",
    "task_illustrations__pixel_village__missing_patch_label",
    "task_illustrations__pixel_village__object_type_count",
    "task_illustrations__pixel_village__person_path_count",
    "task_illustrations__pixel_village__river_side_object_count",
    "task_illustrations__pixel_village__rotated_tile_label",
    "task_illustrations__pixel_village__swapped_tile_pair_label",
    "task_illustrations__pixel_village__territory_object_count",
    "task_illustrations__rpg_house__missing_patch_label",
    "task_illustrations__rpg_house__room_count",
    "task_illustrations__rpg_house__swapped_tile_pair_label",
)

MISSING_PATCH_TASK_IDS: tuple[str, ...] = (
    "task_illustrations__construction_site__missing_patch_label",
    "task_illustrations__environment__missing_patch_label",
    "task_illustrations__indoor_room__missing_patch_label",
    "task_illustrations__library__missing_patch_label",
    "task_illustrations__park_playground__missing_patch_label",
    "task_illustrations__pixel_village__missing_patch_label",
    "task_illustrations__rpg_house__missing_patch_label",
)

CANVAS_PROFILES: tuple[str, ...] = ("landscape", "square", "portrait")
FULL_BLEED_CONTEXT_TYPES: set[tuple[str, str]] = {
    ("environment", "environment_feature"),
}


def _iter_bbox_values(value: Any) -> Iterable[list[float]]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_bbox_values(item)
    elif isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            yield [float(item) for item in value]
        else:
            for item in value:
                yield from _iter_bbox_values(item)


def _iter_named_bboxes(value: Any, path: str = "") -> Iterable[tuple[str, Mapping[str, Any], list[float]]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            if (
                key in {"bbox", "bbox_xyxy", "support_bbox", "interior_bbox", "label_bbox"}
                and isinstance(item, list)
                and len(item) == 4
                and all(isinstance(coord, (int, float)) for coord in item)
            ):
                yield item_path, value, [float(coord) for coord in item]
            else:
                yield from _iter_named_bboxes(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_named_bboxes(item, f"{path}[{index}]")


def _bbox_is_inside(box: Sequence[float], *, width: int, height: int, tolerance: float = 0.5) -> bool:
    x0, y0, x1, y1 = [float(value) for value in box]
    return (
        -float(tolerance) <= x0 < x1 <= float(width) + float(tolerance)
        and -float(tolerance) <= y0 < y1 <= float(height) + float(tolerance)
    )


def _entity_type(parent: Mapping[str, Any]) -> str:
    return str(
        parent.get("entity_type")
        or parent.get("decor_type")
        or parent.get("feature_type")
        or parent.get("type")
        or parent.get("object_type")
        or ""
    )


@pytest.mark.parametrize("task_id", MIGRATED_ILLUSTRATION_TASK_IDS)
@pytest.mark.parametrize("canvas_profile", CANVAS_PROFILES)
def test_source_layout_illustration_tasks_keep_foreground_bboxes_inside_canvas(
    task_id: str,
    canvas_profile: str,
) -> None:
    out = create_task(task_id).generate(
        hash64(2026061601, f"{task_id}:{canvas_profile}", 0),
        params={"canvas_profile": canvas_profile},
        max_attempts=500,
    )
    trace = out.trace_payload
    width, height = [int(value) for value in trace["render_spec"]["canvas_size"]]

    for box in _iter_bbox_values(out.annotation_gt.value):
        assert _bbox_is_inside(box, width=width, height=height), (task_id, canvas_profile, "annotation", box, (width, height))

    render_map = trace.get("render_map", {})
    for key in ("option_bboxes_px_by_label", "tile_bboxes_px_by_label", "selected_option_bbox_px", "rotated_tile_bbox_px"):
        for box in _iter_bbox_values(render_map.get(key)):
            assert _bbox_is_inside(box, width=width, height=height), (task_id, canvas_profile, key, box, (width, height))

    source_size = render_map.get("source_size") or render_map.get("source_scene_canvas_size") or trace["query_spec"]["params"].get("source_size")
    is_composite_output = False
    if source_size:
        source_width, source_height = [int(value) for value in source_size]
        is_composite_output = [source_width, source_height] != [width, height]
        for key, value in render_map.items():
            if key.startswith("source_") and "bbox" in key and key != "source_feature_bboxes_px":
                for box in _iter_bbox_values(value):
                    assert _bbox_is_inside(box, width=source_width, height=source_height), (
                        task_id,
                        canvas_profile,
                        key,
                        box,
                        (source_width, source_height),
                    )

    for root_key in ("execution_trace", "scene_ir"):
        if is_composite_output:
            continue
        for path, parent, box in _iter_named_bboxes(trace.get(root_key, {})):
            if root_key == "execution_trace" and path.startswith("source_scene"):
                continue
            if (out.scene_id, _entity_type(parent)) in FULL_BLEED_CONTEXT_TYPES:
                continue
            assert _bbox_is_inside(box, width=width, height=height), (
                task_id,
                canvas_profile,
                root_key,
                path,
                _entity_type(parent),
                box,
                (width, height),
            )


@pytest.mark.parametrize("task_id", MISSING_PATCH_TASK_IDS)
@pytest.mark.parametrize("canvas_profile", CANVAS_PROFILES)
def test_missing_patch_tasks_use_source_relative_patch_ratios(task_id: str, canvas_profile: str) -> None:
    out = create_task(task_id).generate(
        hash64(2026061801, f"{task_id}:{canvas_profile}:patch-ratio", 0),
        params={"canvas_profile": canvas_profile},
        max_attempts=500,
    )
    params = out.trace_payload["query_spec"]["params"]
    patch_w, patch_h = [int(value) for value in params["patch_size"]]
    source_w, source_h = [int(value) for value in params["source_size"]]
    ratio_trace = params["patch_size_ratio"]

    width_ratio = float(patch_w) / float(source_w)
    height_ratio = float(patch_h) / float(source_h)
    area_ratio = float(patch_w * patch_h) / float(source_w * source_h)

    assert 0.15 <= width_ratio <= 0.30
    assert 0.15 <= height_ratio <= 0.26
    assert area_ratio <= 0.065
    assert ratio_trace["width_ratio_range"] == [0.15, 0.3]
    assert ratio_trace["height_ratio_range"] == [0.15, 0.26]
    assert ratio_trace["area_ratio_max"] == 0.065
