"""Trace fragment helpers for pixel-village public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.config_defaults import group_default, required_group_defaults
from .rendering import PixelVillageScene
from .sampling import (
    _DEFAULTS,
    _build_object_sample,
    _build_path_sample,
    _build_river_side_object_sample,
    _build_territory_object_sample,
    _counted_object_entities,
    _entity_bbox_map,
    _path_people,
    _render_metadata,
    _render_scene,
    _river_bounds,
    _river_side_object_entities,
    _scene_entities,
    _scene_territories,
    _territory_object_entities,
)
from .state import PixelVillageCountBinding


def pixel_village_scene_ir(
    *,
    domain: str,
    scene_id: str,
    scene: PixelVillageScene,
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the common scene-IR fragment for a pixel-village scene."""

    return {
        "domain": str(domain),
        "scene_id": str(scene_id),
        "entities": _scene_entities(scene),
        "territories": _scene_territories(scene),
        "relations": dict(relations),
    }


def pixel_village_render_spec(scene: PixelVillageScene, *, scene_id: str) -> dict[str, Any]:
    """Return the common render-spec fragment for a pixel-village scene."""

    return {
        "canvas_size": [int(scene.image.size[0]), int(scene.image.size[1])],
        "coord_space": "pixel",
        "scene_id": str(scene_id),
        "style": {
            "renderer_id": str(scene.trace.get("renderer_id", "")),
            "style_id": "top_down_pixel_rpg",
            "theme_id": str(scene.trace.get("theme_id", "")),
            "tile_px": int(scene.trace.get("tile_px", 0)),
            "grid_cols": int(scene.trace.get("grid_cols", 0)),
            "grid_rows": int(scene.trace.get("grid_rows", 0)),
        },
    }


def pixel_village_render_map(
    *,
    entity_bboxes: Mapping[str, Sequence[float]],
    counted_entity_ids: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the common render-map fragment with optional task fields."""

    counted_bboxes = _bbox_values(entity_bboxes, counted_entity_ids)
    counted_points = _annotation_values(entity_bboxes, counted_entity_ids)
    payload = {
        "image_id": "img0",
        "entity_bboxes_px": {str(key): [round(float(v), 3) for v in value] for key, value in entity_bboxes.items()},
        "counted_entity_ids": [str(entity_id) for entity_id in counted_entity_ids],
        "counted_entity_bboxes_px": counted_bboxes,
        "counted_entity_points_px": counted_points,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def renderer_execution_fields(scene: PixelVillageScene) -> dict[str, Any]:
    """Return renderer fields used by count-task execution traces."""

    return {
        "category_counts": dict(scene.trace.get("category_counts", {})),
        "public_name_counts": dict(scene.trace.get("public_name_counts", {})),
        "renderer": _render_metadata(scene),
    }


def _bbox_values(entity_bboxes: Mapping[str, Sequence[float]], counted_entity_ids: Sequence[str]) -> list[list[float]]:
    return [[round(float(value), 3) for value in entity_bboxes[str(entity_id)]] for entity_id in counted_entity_ids]


def _annotation_values(entity_bboxes: Mapping[str, Sequence[float]], counted_entity_ids: Sequence[str]) -> list[list[float]]:
    bboxes = _bbox_values(entity_bboxes, counted_entity_ids)
    return [
        [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        for box in bboxes
    ]


def _bbox_set_projection(annotation_value: Sequence[Sequence[float]]) -> dict[str, Any]:
    bboxes = [[round(float(value), 3) for value in bbox[:4]] for bbox in annotation_value]
    return {"type": "bbox_set", "bbox_set": bboxes, "pixel_bbox_set": bboxes}


def bind_category_result(
    *,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
) -> PixelVillageCountBinding:
    """Bind visible-category count fragments for one sampled target object."""

    last_error: Exception | None = None
    sample = _build_object_sample(
        instance_seed=int(instance_seed),
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    answer_count_max = int(
        params.get(
            "target_answer_count_max",
            group_default(generation_defaults, "target_answer_count_max", _DEFAULTS.object_answer_count_max),
        )
    )
    scene = None
    counted_entities: tuple[Any, ...] = ()
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_params: Mapping[str, Any] = params
            if sample.target_object == "tree":
                scene_params = {**params, "cemetery_mode": "none"}
            scene = _render_scene(
                namespace=str(namespace),
                instance_seed=int(instance_seed),
                params=scene_params,
                render_defaults=rendering_defaults,
                attempt_index=int(attempt),
                path_person_count=0,
            )
            counted_entities = _counted_object_entities(scene, sample.target_object)
            answer_count = len(counted_entities)
            if 0 < answer_count <= answer_count_max:
                break
            last_error = RuntimeError(f"sampled target object count {answer_count} is outside allowed range 1..{answer_count_max}")
            scene = None
        except Exception as exc:  # pragma: no cover
            last_error = exc
            scene = None
    if scene is None:
        raise RuntimeError(f"could not generate pixel-village category count: {last_error}") from last_error

    pad = float(params.get("annotation_padding_px", group_default(rendering_defaults, "annotation_padding_px", _DEFAULTS.annotation_padding_px)))
    entity_bboxes = _entity_bbox_map(scene, pad=pad)
    counted_entity_ids = tuple(sorted(str(entity.entity_id) for entity in counted_entities))
    annotation_value = _bbox_values(entity_bboxes, counted_entity_ids)
    answer = int(len(counted_entity_ids))
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_pixel_village_object_type",
            "annotation_hint_pixel_village_object_type",
            "json_example_pixel_village_object_type",
            "json_example_answer_only_pixel_village_object_type",
        ],
        context="prompt defaults for pixel-village category count",
    )
    return PixelVillageCountBinding(
        prompt_defaults=required_defaults,
        slots={
            "target_plural": str(sample.target_plural),
            "target_unit": str(sample.target_unit),
            "json_output_contract": str(required_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(required_defaults["answer_hint_pixel_village_object_type"]).format(target_plural=str(sample.target_plural)),
            "annotation_hint": str(required_defaults["annotation_hint_pixel_village_object_type"]).format(target_unit=str(sample.target_unit)),
            "json_example": str(required_defaults["json_example_pixel_village_object_type"]),
            "json_example_answer_only": str(required_defaults["json_example_answer_only_pixel_village_object_type"]),
        },
        answer=answer,
        annotation_value=annotation_value,
        scene=scene,
        entity_bboxes=entity_bboxes,
        counted_entity_ids=counted_entity_ids,
        scene_relations={"target_object": str(sample.target_object), "target_public_name": str(sample.target_public_name)},
        branch_params={
            "target_object": str(sample.target_object),
            "target_plural": str(sample.target_plural),
            "target_public_name": str(sample.target_public_name),
            "target_count": answer,
            "target_answer_count_max": int(answer_count_max),
            "render_constraints": {"cemetery_mode": "none" if sample.target_object == "tree" else "task_default"},
            "target_object_probabilities": dict(sample.target_object_probabilities),
            "renderer": _render_metadata(scene),
        },
        render_map_extra={"path_tiles": list(scene.trace.get("path_tiles", []))},
        execution_trace={
            "target_object": str(sample.target_object),
            "target_public_name": str(sample.target_public_name),
            "answer": answer,
            "counted_entity_ids": list(counted_entity_ids),
            **renderer_execution_fields(scene),
        },
        witness_symbolic={
            "counted_entity_ids": list(counted_entity_ids),
            "target_object": str(sample.target_object),
            "count": answer,
            "answer": answer,
        },
        projected_annotation=_bbox_set_projection(annotation_value),
    )


def bind_path_result(
    *,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
) -> PixelVillageCountBinding:
    """Bind path-person count fragments using tile-footprint intersection."""

    last_error: Exception | None = None
    sample = _build_path_sample(
        instance_seed=int(instance_seed),
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    scene = None
    counted_entities: tuple[Any, ...] = ()
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene = _render_scene(
                namespace=str(namespace),
                instance_seed=int(instance_seed),
                params=params,
                render_defaults=rendering_defaults,
                attempt_index=int(attempt),
                path_person_count=int(sample.path_person_count),
                background_person_path_clearance=int(
                    params.get(
                        "background_person_path_clearance",
                        group_default(rendering_defaults, "background_person_path_clearance", _DEFAULTS.background_person_path_clearance),
                    )
                ),
            )
            counted_entities = _path_people(scene)
            if len(counted_entities) == int(sample.path_person_count):
                break
            last_error = RuntimeError("rendered path-person count did not match requested count")
            scene = None
        except Exception as exc:  # pragma: no cover
            last_error = exc
            scene = None
    if scene is None:
        raise RuntimeError(f"could not generate pixel-village path count: {last_error}") from last_error

    pad = float(params.get("annotation_padding_px", group_default(rendering_defaults, "annotation_padding_px", _DEFAULTS.annotation_padding_px)))
    entity_bboxes = _entity_bbox_map(scene, pad=pad)
    counted_entity_ids = tuple(sorted(str(entity.entity_id) for entity in counted_entities))
    annotation_value = _bbox_values(entity_bboxes, counted_entity_ids)
    answer = int(len(counted_entity_ids))
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_pixel_village_person_path",
            "annotation_hint_pixel_village_person_path",
            "json_example_pixel_village_person_path",
            "json_example_answer_only_pixel_village_person_path",
        ],
        context="prompt defaults for pixel-village path count",
    )
    return PixelVillageCountBinding(
        prompt_defaults=required_defaults,
        slots={
            "json_output_contract": str(required_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(required_defaults["answer_hint_pixel_village_person_path"]),
            "annotation_hint": str(required_defaults["annotation_hint_pixel_village_person_path"]),
            "json_example": str(required_defaults["json_example_pixel_village_person_path"]),
            "json_example_answer_only": str(required_defaults["json_example_answer_only_pixel_village_person_path"]),
        },
        answer=answer,
        annotation_value=annotation_value,
        scene=scene,
        entity_bboxes=entity_bboxes,
        counted_entity_ids=counted_entity_ids,
        scene_relations={"path_membership_rule": "person tile footprint intersects a path tile"},
        branch_params={
            "path_person_count": int(sample.path_person_count),
            "path_person_count_probabilities": dict(sample.path_person_count_probabilities),
            "renderer": _render_metadata(scene),
        },
        render_map_extra={"path_tiles": list(scene.trace.get("path_tiles", []))},
        execution_trace={
            "answer": answer,
            "counted_entity_ids": list(counted_entity_ids),
            "path_tiles": list(scene.trace.get("path_tiles", [])),
            "path_person_requested_count": int(scene.trace.get("path_person_requested_count", 0)),
            "path_person_placed_count": int(scene.trace.get("path_person_placed_count", 0)),
            "background_person_path_clearance": int(scene.trace.get("background_person_path_clearance", 0)),
            **renderer_execution_fields(scene),
        },
        witness_symbolic={
            "counted_entity_ids": list(counted_entity_ids),
            "path_membership_rule": "tile_footprint_intersects_path_tile",
            "count": answer,
            "answer": answer,
        },
        projected_annotation=_bbox_set_projection(annotation_value),
    )


def bind_territory_result(
    *,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
) -> PixelVillageCountBinding:
    """Bind territory-scoped object count fragments from rendered metadata."""

    last_error: Exception | None = None
    sample = _build_territory_object_sample(
        instance_seed=int(instance_seed),
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    answer_count_max = int(
        params.get(
            "target_answer_count_max",
            group_default(generation_defaults, "target_answer_count_max", _DEFAULTS.territory_object_answer_count_max),
        )
    )
    scene = None
    counted_entities: tuple[Any, ...] = ()
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene = _render_scene(
                namespace=str(namespace),
                instance_seed=int(instance_seed),
                params={**params, sample.force_mode_param: "force"},
                render_defaults=rendering_defaults,
                attempt_index=int(attempt),
                path_person_count=0,
            )
            counted_entities = _territory_object_entities(scene, sample)
            answer_count = len(counted_entities)
            if 0 < answer_count <= answer_count_max:
                break
            last_error = RuntimeError(f"territory object count {answer_count} is outside allowed range 1..{answer_count_max}")
            scene = None
        except Exception as exc:  # pragma: no cover
            last_error = exc
            scene = None
    if scene is None:
        raise RuntimeError(f"could not generate pixel-village territory count: {last_error}") from last_error

    pad = float(params.get("annotation_padding_px", group_default(rendering_defaults, "annotation_padding_px", _DEFAULTS.annotation_padding_px)))
    entity_bboxes = _entity_bbox_map(scene, pad=pad)
    counted_entity_ids = tuple(sorted(str(entity.entity_id) for entity in counted_entities))
    annotation_value = _bbox_values(entity_bboxes, counted_entity_ids)
    answer = int(len(counted_entity_ids))
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_pixel_village_territory_object",
            "annotation_hint_pixel_village_territory_object",
            "json_example_pixel_village_territory_object",
            "json_example_answer_only_pixel_village_territory_object",
        ],
        context="prompt defaults for pixel-village territory count",
    )
    return PixelVillageCountBinding(
        prompt_defaults=required_defaults,
        slots={
            "territory_name": str(sample.territory_name),
            "target_plural": str(sample.target_plural),
            "target_unit": str(sample.target_unit),
            "json_output_contract": str(required_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(required_defaults["answer_hint_pixel_village_territory_object"]).format(target_plural=str(sample.target_plural), territory_name=str(sample.territory_name)),
            "annotation_hint": str(required_defaults["annotation_hint_pixel_village_territory_object"]).format(target_unit=str(sample.target_unit), territory_name=str(sample.territory_name)),
            "json_example": str(required_defaults["json_example_pixel_village_territory_object"]),
            "json_example_answer_only": str(required_defaults["json_example_answer_only_pixel_village_territory_object"]),
        },
        answer=answer,
        annotation_value=annotation_value,
        scene=scene,
        entity_bboxes=entity_bboxes,
        counted_entity_ids=counted_entity_ids,
        scene_relations={
            "territory_id": str(sample.territory_id),
            "territory_type": str(sample.territory_type),
            "target_public_name": str(sample.target_public_name),
        },
        branch_params={
            "territory_object": str(sample.target_key),
            "territory_id": str(sample.territory_id),
            "territory_type": str(sample.territory_type),
            "territory_name": str(sample.territory_name),
            "target_public_name": str(sample.target_public_name),
            "target_count": answer,
            "target_answer_count_max": int(answer_count_max),
            "render_constraints": {str(sample.force_mode_param): "force"},
            "territory_object_probabilities": dict(sample.target_probabilities),
            "renderer": _render_metadata(scene),
        },
        render_map_extra={},
        execution_trace={
            "territory_object": str(sample.target_key),
            "territory_id": str(sample.territory_id),
            "territory_type": str(sample.territory_type),
            "target_public_name": str(sample.target_public_name),
            "answer": answer,
            "counted_entity_ids": list(counted_entity_ids),
            **renderer_execution_fields(scene),
        },
        witness_symbolic={
            "counted_entity_ids": list(counted_entity_ids),
            "territory_id": str(sample.territory_id),
            "target_public_name": str(sample.target_public_name),
            "count": answer,
            "answer": answer,
        },
        projected_annotation=_bbox_set_projection(annotation_value),
    )


def bind_river_side_result(
    *,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
) -> PixelVillageCountBinding:
    """Bind strict-side-of-river object count fragments from rendered metadata."""

    last_error: Exception | None = None
    sample = _build_river_side_object_sample(
        instance_seed=int(instance_seed),
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    answer_count_max = int(
        params.get(
            "target_answer_count_max",
            group_default(generation_defaults, "target_answer_count_max", _DEFAULTS.river_side_answer_count_max),
        )
    )
    scene = None
    counted_entities: tuple[Any, ...] = ()
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_params: Mapping[str, Any] = {
                **params,
                "river_mode": "force",
                "river_orientation": str(sample.river_orientation),
                "river_placement": "balanced",
            }
            if sample.target_object == "tree":
                scene_params = {**scene_params, "cemetery_mode": "none"}
            scene = _render_scene(
                namespace=str(namespace),
                instance_seed=int(instance_seed),
                params=scene_params,
                render_defaults=rendering_defaults,
                attempt_index=int(attempt),
                path_person_count=0,
            )
            if str(scene.trace.get("river_orientation", "")) != str(sample.river_orientation):
                raise RuntimeError("rendered river orientation did not match requested side")
            counted_entities = _river_side_object_entities(scene, sample)
            answer_count = len(counted_entities)
            if answer_count == int(sample.target_count):
                break
            last_error = RuntimeError(f"river-side object count {answer_count} did not match requested count {sample.target_count}")
            scene = None
        except Exception as exc:  # pragma: no cover
            last_error = exc
            scene = None
    if scene is None:
        raise RuntimeError(f"could not generate pixel-village river-side count: {last_error}") from last_error

    pad = float(params.get("annotation_padding_px", group_default(rendering_defaults, "annotation_padding_px", _DEFAULTS.annotation_padding_px)))
    entity_bboxes = _entity_bbox_map(scene, pad=pad)
    counted_entity_ids = tuple(sorted(str(entity.entity_id) for entity in counted_entities))
    annotation_value = _bbox_values(entity_bboxes, counted_entity_ids)
    answer = int(len(counted_entity_ids))
    river_bounds = _river_bounds(scene)
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_pixel_village_river_side_object",
            "annotation_hint_pixel_village_river_side_object",
            "json_example_pixel_village_river_side_object",
            "json_example_answer_only_pixel_village_river_side_object",
        ],
        context="prompt defaults for pixel-village river-side count",
    )
    return PixelVillageCountBinding(
        prompt_defaults=required_defaults,
        slots={
            "target_plural": str(sample.target_plural),
            "target_unit": str(sample.target_unit),
            "river_side": str(sample.river_side),
            "river_relation": str(sample.river_relation),
            "json_output_contract": str(required_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(required_defaults["answer_hint_pixel_village_river_side_object"]).format(target_plural=str(sample.target_plural), river_relation=str(sample.river_relation)),
            "annotation_hint": str(required_defaults["annotation_hint_pixel_village_river_side_object"]).format(target_unit=str(sample.target_unit), river_relation=str(sample.river_relation)),
            "json_example": str(required_defaults["json_example_pixel_village_river_side_object"]),
            "json_example_answer_only": str(required_defaults["json_example_answer_only_pixel_village_river_side_object"]),
        },
        answer=answer,
        annotation_value=annotation_value,
        scene=scene,
        entity_bboxes=entity_bboxes,
        counted_entity_ids=counted_entity_ids,
        scene_relations={
            "target_object": str(sample.target_object),
            "target_public_name": str(sample.target_public_name),
            "river_side": str(sample.river_side),
            "river_orientation": str(sample.river_orientation),
            "river_bounds": dict(river_bounds),
            "side_membership_rule": "entity tile footprint lies strictly on requested side of river bounds",
        },
        branch_params={
            "target_object": str(sample.target_object),
            "target_plural": str(sample.target_plural),
            "target_public_name": str(sample.target_public_name),
            "river_side": str(sample.river_side),
            "river_orientation": str(sample.river_orientation),
            "river_relation": str(sample.river_relation),
            "target_count": answer,
            "requested_target_count": int(sample.target_count),
            "target_count_support": [int(value) for value in sample.target_count_support],
            "target_answer_count_max": int(answer_count_max),
            "render_constraints": {
                "river_mode": "force",
                "river_orientation": str(sample.river_orientation),
                "river_placement": "balanced",
                "cemetery_mode": "none" if sample.target_object == "tree" else "task_default",
            },
            "target_object_probabilities": dict(sample.target_object_probabilities),
            "river_side_probabilities": dict(sample.river_side_probabilities),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "renderer": _render_metadata(scene),
        },
        render_map_extra={"water_tiles": list(scene.trace.get("water_tiles", [])), "river_bounds": dict(river_bounds)},
        execution_trace={
            "target_object": str(sample.target_object),
            "target_public_name": str(sample.target_public_name),
            "river_side": str(sample.river_side),
            "river_orientation": str(sample.river_orientation),
            "river_bounds": dict(river_bounds),
            "requested_target_count": int(sample.target_count),
            "answer": answer,
            "counted_entity_ids": list(counted_entity_ids),
            "water_tiles": list(scene.trace.get("water_tiles", [])),
            **renderer_execution_fields(scene),
        },
        witness_symbolic={
            "counted_entity_ids": list(counted_entity_ids),
            "target_object": str(sample.target_object),
            "river_side": str(sample.river_side),
            "river_orientation": str(sample.river_orientation),
            "river_bounds": dict(river_bounds),
            "side_membership_rule": "strict_tile_footprint_side_of_river_bounds",
            "count": answer,
            "answer": answer,
        },
        projected_annotation=_bbox_set_projection(annotation_value),
    )


__all__ = [
    "pixel_village_render_map",
    "pixel_village_render_spec",
    "pixel_village_scene_ir",
    "renderer_execution_fields",
    "bind_category_result",
    "bind_path_result",
    "bind_river_side_result",
    "bind_territory_result",
]
