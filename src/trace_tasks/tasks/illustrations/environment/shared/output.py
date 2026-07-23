"""Scene-output serialization helpers for environment illustrations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ...shared.object_rendering import serialize_rendered_illustration_object

from .annotations import sort_bbox_centers_by_ids, sort_bboxes_by_ids, target_feature
from .defaults import FEATURE_RELATION_SIDE_DEFAULTS
from .labels import CROSSED_FEATURE_NAMES, CROSSING_NAMES, feature_name, feature_relation_phrase
from .rendering import ANNOTATION_BBOX_MIN_SIDE_PX, RenderedEnvironmentObjectScene
from .state import BoundCountResult, EnvironmentChoice


def _bbox_min_side(box: list[float]) -> float:
    return min(abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1])))


def _require_min_bbox_sides(*, label: str, bboxes: Sequence[list[float]]) -> None:
    for box in bboxes:
        if _bbox_min_side(box) < float(ANNOTATION_BBOX_MIN_SIDE_PX):
            raise ValueError(f"{label} bbox below {ANNOTATION_BBOX_MIN_SIDE_PX:g}px minimum side: {box}")


def serialize_environment_objects(scene: RenderedEnvironmentObjectScene) -> tuple[list[dict[str, Any]], Dict[str, list[float]], Dict[str, list[float]]]:
    """Serialize foreground objects and return object/part bbox maps."""

    serialized_objects = [serialize_rendered_illustration_object(obj) for obj in scene.objects]
    object_bboxes = {str(obj["object_id"]): list(obj["bbox"]) for obj in serialized_objects}
    part_bboxes = {
        str(part["part_id"]): list(part["bbox"])
        for obj in serialized_objects
        for part in obj["parts"]
    }
    return serialized_objects, object_bboxes, part_bboxes


def counted_side_object_ids(*, scene: Any, feature_id: str, relation: str) -> Tuple[str, ...]:
    """Return foreground object ids whose trace relation matches the requested side."""

    ids = []
    for placement in scene.placements:
        relation_info = placement.relations.get(str(feature_id))
        if isinstance(relation_info, Mapping) and str(relation_info.get("vertical_relation")) == str(relation):
            ids.append(str(placement.object_id))
    return tuple(ids)


def bind_feature_relation_result(
    scene: Any,
    choice: EnvironmentChoice,
    object_bboxes: Mapping[str, list[float]],
    _feature_bboxes: Mapping[str, list[float]],
    target_count: int,
) -> BoundCountResult:
    """Bind foreground-object witnesses for a sampled feature relation."""

    feature = target_feature(scene, str(choice.feature_type))
    if str(choice.relation) == "on":
        counted_object_ids = tuple(
            str(placement.object_id)
            for placement in scene.placements
            if str(placement.zone_id) == str(choice.feature_type)
        )
        if len(counted_object_ids) != int(target_count):
            raise ValueError(f"on-feature count {len(counted_object_ids)} did not match target {target_count}")
    else:
        counted_object_ids = counted_side_object_ids(
            scene=scene,
            feature_id=str(feature.feature_id),
            relation=str(choice.relation),
        )
        side_answer_min = int(FEATURE_RELATION_SIDE_DEFAULTS.target_count_min)
        side_answer_max = int(FEATURE_RELATION_SIDE_DEFAULTS.target_count_max)
        if len(counted_object_ids) < side_answer_min or len(counted_object_ids) > side_answer_max:
            raise ValueError(
                f"feature-relation side count {len(counted_object_ids)} outside "
                f"{side_answer_min}..{side_answer_max}"
            )
    counted_object_bboxes = sort_bboxes_by_ids(object_bboxes, counted_object_ids)
    _require_min_bbox_sides(label="feature-relation counted object", bboxes=counted_object_bboxes)
    counted_object_points = sort_bbox_centers_by_ids(object_bboxes, counted_object_ids)
    phrase = feature_relation_phrase(choice.feature_type, choice.relation)
    return BoundCountResult(
        answer=int(len(counted_object_ids)),
        annotation_value=list(counted_object_bboxes),
        render_map_extra={
            "counted_object_ids": list(counted_object_ids),
            "counted_object_bboxes_px": list(counted_object_bboxes),
            "counted_object_points_px": list(counted_object_points),
            "target_feature_id": str(feature.feature_id),
        },
        scene_relations={"feature_type": str(choice.feature_type), "feature_id": str(feature.feature_id), "relation": str(choice.relation)},
        execution_extra={
            "feature_type": str(choice.feature_type),
            "feature_id": str(feature.feature_id),
            "relation": str(choice.relation),
            "counted_object_ids": list(counted_object_ids),
            "object_zones": {placement.object_id: placement.zone_id for placement in scene.placements},
        },
        witness_symbolic={
            "counted_object_ids": list(counted_object_ids),
            "feature_id": str(feature.feature_id),
            "feature_type": str(choice.feature_type),
            "relation": str(choice.relation),
            "answer": int(len(counted_object_ids)),
        },
        operand_params={
            "feature_type": str(choice.feature_type),
            "feature_id": str(feature.feature_id),
            "feature_name": feature_name(choice.feature_type),
            "relation": str(choice.relation),
            "feature_relation_phrase": str(phrase),
            "feature_type_probabilities": dict(choice.feature_type_probabilities or {}),
            "relation_probabilities": dict(choice.relation_probabilities or {}),
        },
    )


def bind_crossing_result(
    scene: Any,
    choice: EnvironmentChoice,
    _object_bboxes: Mapping[str, list[float]],
    feature_bboxes: Mapping[str, list[float]],
    target_count: int,
) -> BoundCountResult:
    """Bind bridge/crosswalk feature bboxes as the counted witnesses."""

    counted_feature_ids = tuple(
        str(item.feature_id)
        for item in scene.features
        if str(item.feature_type) == str(choice.crossing_type)
    )
    if len(counted_feature_ids) != int(target_count):
        raise ValueError(f"crossing count {len(counted_feature_ids)} did not match target {target_count}")
    crossing_name = CROSSING_NAMES[str(choice.crossing_type)]
    crossed_feature_name = CROSSED_FEATURE_NAMES[str(choice.crossing_type)]
    counted_feature_bboxes = sort_bboxes_by_ids(feature_bboxes, counted_feature_ids)
    counted_feature_points = sort_bbox_centers_by_ids(feature_bboxes, counted_feature_ids)
    return BoundCountResult(
        answer=int(len(counted_feature_ids)),
        annotation_value=list(counted_feature_bboxes),
        render_map_extra={
            "counted_feature_ids": list(counted_feature_ids),
            "counted_feature_bboxes_px": list(counted_feature_bboxes),
            "counted_feature_points_px": list(counted_feature_points),
        },
        scene_relations={"crossing_type": str(choice.crossing_type)},
        execution_extra={"crossing_type": str(choice.crossing_type), "counted_feature_ids": list(counted_feature_ids)},
        witness_symbolic={
            "counted_feature_ids": list(counted_feature_ids),
            "crossing_type": str(choice.crossing_type),
            "answer": int(len(counted_feature_ids)),
        },
        operand_params={
            "crossing_type": str(choice.crossing_type),
            "crossing_name": str(crossing_name),
            "crossed_feature_name": str(crossed_feature_name),
            "crossing_type_probabilities": dict(choice.crossing_type_probabilities or {}),
        },
    )


def window_bboxes(scene: Any, window_mode: str) -> Tuple[Tuple[str, list[float]], ...]:
    """Return lit-window bbox records keyed by stable building/window ids."""

    items = []
    for building in scene.buildings:
        for index, bbox in enumerate(building.lit_window_bboxes):
            items.append((f"{building.building_id}_{window_mode}_window_{index:02d}", [round(float(v), 3) for v in bbox]))
    return tuple(items)


def bind_window_result(
    scene: Any,
    choice: EnvironmentChoice,
    _object_bboxes: Mapping[str, list[float]],
    _feature_bboxes: Mapping[str, list[float]],
    target_count: int,
) -> BoundCountResult:
    """Bind lit-window boxes as minimal visual witnesses for building windows."""

    window_items = window_bboxes(scene, str(choice.window_mode))
    if len(window_items) != int(target_count):
        raise ValueError(f"rendered {len(window_items)} lit windows, expected {target_count}")
    window_bbox_map = {item_id: bbox for item_id, bbox in window_items}
    counted_window_ids = tuple(item_id for item_id, _bbox in window_items)
    counted_window_bboxes = sort_bboxes_by_ids(window_bbox_map, counted_window_ids)
    _require_min_bbox_sides(label="lit window", bboxes=counted_window_bboxes)
    counted_window_points = sort_bbox_centers_by_ids(window_bbox_map, counted_window_ids)
    return BoundCountResult(
        answer=int(len(counted_window_ids)),
        annotation_value=list(counted_window_bboxes),
        render_map_extra={
            "window_bboxes_px": dict(window_bbox_map),
            "counted_window_ids": list(counted_window_ids),
            "counted_window_bboxes_px": list(counted_window_bboxes),
            "counted_window_points_px": list(counted_window_points),
        },
        scene_relations={"window_mode": str(choice.window_mode)},
        execution_extra={
            "window_mode": str(choice.window_mode),
            "building_count": int(len(scene.buildings)),
            "counted_window_ids": list(counted_window_ids),
        },
        witness_symbolic={
            "counted_window_ids": list(counted_window_ids),
            "window_mode": str(choice.window_mode),
            "answer": int(len(counted_window_ids)),
        },
        operand_params={
            "window_mode": str(choice.window_mode),
            "window_phrase": "lit windows",
            "window_mode_probabilities": dict(choice.window_mode_probabilities or {}),
        },
    )


__all__ = [
    "bind_crossing_result",
    "bind_feature_relation_result",
    "bind_window_result",
    "counted_side_object_ids",
    "serialize_environment_objects",
    "window_bboxes",
]
