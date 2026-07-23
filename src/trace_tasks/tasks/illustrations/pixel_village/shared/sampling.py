"""Sampling and metadata helpers for top-down pixel village tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import hash64, spawn_rng
from .....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....shared.config_defaults import group_default
from ...shared.rpg_tile_profiles import resolve_rpg_tile_render_params
from ...shared.task_support import bounds, sample_count, uniform_string_probability_map
from .rendering import PixelVillageEntity, PixelVillageScene, render_pixel_village_map


SCENE_ID = "pixel_village"

TARGET_OBJECT_KEYS: Tuple[str, ...] = (
    "building",
    "person",
    "tree",
    "lamp_post",
    "well",
    "pond",
)
TARGET_PUBLIC_NAME: Dict[str, str] = {
    "building": "building",
    "person": "person",
    "tree": "tree",
    "lamp_post": "lamp post",
    "well": "well",
    "pond": "pond",
}
TARGET_PROMPT_PLURAL: Dict[str, str] = {
    "building": "buildings",
    "person": "people",
    "tree": "trees",
    "lamp_post": "lamp posts",
    "well": "wells",
    "pond": "ponds",
}
TARGET_PROMPT_UNIT: Dict[str, str] = {
    "building": "building",
    "person": "person",
    "tree": "tree",
    "lamp_post": "lamp post",
    "well": "well",
    "pond": "pond",
}
RIVER_SIDE_TARGET_KEYS: Tuple[str, ...] = (
    "building",
    "person",
    "tree",
)
RIVER_SIDE_KEYS: Tuple[str, ...] = (
    "left",
    "right",
    "above",
    "below",
)
RIVER_SIDE_ORIENTATION: Dict[str, str] = {
    "left": "vertical",
    "right": "vertical",
    "above": "horizontal",
    "below": "horizontal",
}
RIVER_SIDE_PROMPT_RELATION: Dict[str, str] = {
    "left": "left of",
    "right": "right of",
    "above": "above",
    "below": "below",
}


def sample_option_answer_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    seed_scope: str,
    option_labels: Sequence[str],
) -> Tuple[int, Dict[str, float]]:
    """Sample a correct MCQ option index from visible option labels."""

    labels = tuple(str(label) for label in option_labels)
    explicit = params.get("correct_index")
    if explicit is not None:
        value = int(explicit)
        if value < 0 or value >= len(labels):
            raise ValueError("correct_index outside option label support")
        return int(value), {str(value): 1.0}
    if params.get("answer_label") is not None:
        label = str(params["answer_label"])
        if label not in set(labels):
            raise ValueError("answer_label outside option label support")
        value = int(labels.index(label))
        return int(value), {str(value): 1.0}
    if params.get("_sample_cursor") is not None:
        rng = spawn_rng(int(instance_seed), f"{seed_scope}:answer:{int(params['_sample_cursor'])}")
    else:
        rng = spawn_rng(int(instance_seed), f"{seed_scope}:answer")
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        tuple(range(len(labels))),
        sort_keys=True,
    )
    return int(selected), dict(probabilities)

TERRITORY_OBJECT_KEYS: Tuple[str, ...] = (
    "cemetery_grave_marker",
    "orchard_tree",
)
TERRITORY_OBJECT_SPECS: Dict[str, Dict[str, str]] = {
    "cemetery_grave_marker": {
        "territory_id": "cemetery_0",
        "territory_type": "cemetery",
        "territory_name": "cemetery",
        "target_public_name": "grave marker",
        "target_plural": "grave markers",
        "target_unit": "grave marker",
        "force_mode_param": "cemetery_mode",
    },
    "orchard_tree": {
        "territory_id": "orchard_0",
        "territory_type": "orchard",
        "territory_name": "orchard",
        "target_public_name": "tree",
        "target_plural": "trees",
        "target_unit": "tree",
        "force_mode_param": "orchard_mode",
    },
}


@dataclass(frozen=True)
class _Defaults:
    canvas_width: int = 960
    canvas_height: int = 720
    tile_px: int = 48
    theme_mode: str = "auto"
    cemetery_mode: str = "auto"
    orchard_mode: str = "auto"
    windmill_mode: str = "auto"
    path_person_count_min: int = 2
    path_person_count_max: int = 6
    background_person_path_clearance: int = 1
    object_answer_count_max: int = 8
    territory_object_answer_count_max: int = 8
    river_side_answer_count_max: int = 8
    annotation_padding_px: float = 0.0


@dataclass(frozen=True)
class _ObjectTypeSample:
    target_object: str
    target_plural: str
    target_unit: str
    target_public_name: str
    target_object_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _PathPeopleSample:
    path_person_count: int
    path_person_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _TerritoryObjectSample:
    target_key: str
    territory_id: str
    territory_type: str
    territory_name: str
    target_public_name: str
    target_plural: str
    target_unit: str
    force_mode_param: str
    target_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _RiverSideObjectSample:
    target_object: str
    target_plural: str
    target_unit: str
    target_public_name: str
    river_side: str
    river_orientation: str
    river_relation: str
    target_count: int
    target_count_support: Tuple[int, ...]
    target_object_probabilities: Dict[str, float]
    river_side_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
def _normalize_target_key(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text == "lamp":
        text = "lamp_post"
    if text == "people":
        text = "person"
    if text.endswith("s") and text[:-1] in TARGET_OBJECT_KEYS:
        text = text[:-1]
    return str(text)


def _normalize_river_side(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "north": "above",
        "south": "below",
        "west": "left",
        "east": "right",
        "top": "above",
        "bottom": "below",
    }
    return aliases.get(text, text)


def _normalize_territory_object_key(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cemetery": "cemetery_grave_marker",
        "grave_marker": "cemetery_grave_marker",
        "grave_markers": "cemetery_grave_marker",
        "orchard": "orchard_tree",
        "orchard_trees": "orchard_tree",
        "trees_in_orchard": "orchard_tree",
    }
    return aliases.get(text, text)


def _target_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("target_object_support", group_default(defaults, "target_object_support", TARGET_OBJECT_KEYS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("target_object_support must be a sequence")
    support = tuple(dict.fromkeys(_normalize_target_key(value) for value in raw))
    invalid = [value for value in support if value not in set(TARGET_OBJECT_KEYS)]
    if invalid:
        raise ValueError(f"unsupported pixel village target objects: {invalid}")
    if not support:
        raise ValueError("target_object_support must contain at least one supported target")
    return support


def _river_side_target_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get(
        "river_side_target_object_support",
        group_default(defaults, "river_side_target_object_support", RIVER_SIDE_TARGET_KEYS),
    )
    if raw is None:
        raw = params.get("target_object_support", group_default(defaults, "target_object_support", RIVER_SIDE_TARGET_KEYS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("river_side_target_object_support must be a sequence")
    support = tuple(dict.fromkeys(_normalize_target_key(value) for value in raw))
    invalid = [value for value in support if value not in set(RIVER_SIDE_TARGET_KEYS)]
    if invalid:
        raise ValueError(f"unsupported pixel village river-side target objects: {invalid}")
    if not support:
        raise ValueError("river_side_target_object_support must contain at least one supported target")
    return support


def _river_side_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("river_side_support", group_default(defaults, "river_side_support", RIVER_SIDE_KEYS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("river_side_support must be a sequence")
    support = tuple(dict.fromkeys(_normalize_river_side(value) for value in raw))
    invalid = [value for value in support if value not in set(RIVER_SIDE_KEYS)]
    if invalid:
        raise ValueError(f"unsupported pixel village river sides: {invalid}")
    if not support:
        raise ValueError("river_side_support must contain at least one supported side")
    return support


def _river_side_target_count_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    answer_count_max = int(
        params.get(
            "target_answer_count_max",
            group_default(defaults, "target_answer_count_max", _DEFAULTS.river_side_answer_count_max),
        )
    )
    raw = params.get("target_count_support", group_default(defaults, "target_count_support", None))
    if raw is None:
        raw = tuple(range(1, int(answer_count_max) + 1))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("target_count_support must be a sequence")
    support = tuple(dict.fromkeys(int(value) for value in raw))
    invalid = [value for value in support if int(value) < 1 or int(value) > int(answer_count_max)]
    if invalid:
        raise ValueError(f"target_count_support values must be within 1..{answer_count_max}: {invalid}")
    if not support:
        raise ValueError("target_count_support must contain at least one supported count")
    return support


def _territory_object_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get(
        "territory_object_support",
        group_default(defaults, "territory_object_support", TERRITORY_OBJECT_KEYS),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("territory_object_support must be a sequence")
    support = tuple(dict.fromkeys(_normalize_territory_object_key(value) for value in raw))
    invalid = [value for value in support if value not in set(TERRITORY_OBJECT_KEYS)]
    if invalid:
        raise ValueError(f"unsupported pixel village territory-object targets: {invalid}")
    if not support:
        raise ValueError("territory_object_support must contain at least one supported target")
    return support


def _resolve_target_object(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    support = _target_support(params, defaults)
    explicit = params.get("target_object")
    if explicit is None:
        explicit = params.get("target_public_name")
    if explicit is not None:
        target = _normalize_target_key(explicit)
        if target not in set(support):
            raise ValueError("target_object is outside configured support")
        return str(target), uniform_string_probability_map(support, selected=str(target))
    rng = spawn_rng(int(instance_seed), f"{namespace}:target_object")
    target, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(target), dict(probabilities)


def _resolve_river_side_target_object(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    support = _river_side_target_support(params, defaults)
    explicit = params.get("target_object")
    if explicit is None:
        explicit = params.get("target_public_name")
    if explicit is not None:
        target = _normalize_target_key(explicit)
        if target not in set(support):
            raise ValueError("target_object is outside configured river-side support")
        return str(target), uniform_string_probability_map(support, selected=str(target))
    rng = spawn_rng(int(instance_seed), f"{namespace}:target_object")
    target, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(target), dict(probabilities)


def _resolve_river_side(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    support = _river_side_support(params, defaults)
    explicit = params.get("river_side")
    if explicit is None:
        explicit = params.get("side")
    if explicit is not None:
        side = _normalize_river_side(explicit)
        if side not in set(support):
            raise ValueError("river_side is outside configured support")
        return str(side), uniform_string_probability_map(support, selected=str(side))
    rng = spawn_rng(int(instance_seed), f"{namespace}:river_side")
    side, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(side), dict(probabilities)


def _resolve_river_side_target_count(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[int, Tuple[int, ...], Dict[str, float]]:
    support = _river_side_target_count_support(params, defaults)
    explicit = params.get("target_count")
    if explicit is not None:
        count = int(explicit)
        if count not in set(support):
            raise ValueError("target_count is outside configured support")
        return int(count), support, support_probability_map(support, selected=int(count), sort_keys=True)
    cursor = params.get("_sample_cursor")
    if cursor is not None:
        count = int(support[int(cursor) % len(support)])
        return int(count), support, support_probability_map(support, sort_keys=True)
    rng = spawn_rng(int(instance_seed), f"{namespace}:target_count")
    count, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(count), support, dict(probabilities)


def _resolve_territory_object(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    support = _territory_object_support(params, defaults)
    explicit = params.get("territory_object")
    if explicit is None:
        explicit = params.get("target_territory_object")
    if explicit is not None:
        target = _normalize_territory_object_key(explicit)
        if target not in set(support):
            raise ValueError("territory_object is outside configured support")
        return str(target), uniform_string_probability_map(support, selected=str(target))
    rng = spawn_rng(int(instance_seed), f"{namespace}:territory_object")
    target, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(target), dict(probabilities)


def _scene_seed(namespace: str, instance_seed: int, attempt_index: int) -> int:
    if int(attempt_index) == 0:
        return int(instance_seed)
    return int(hash64(int(instance_seed), f"{namespace}:scene_attempt", int(attempt_index)))


def _render_scene(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    attempt_index: int,
    path_person_count: int = 0,
    background_person_path_clearance: int = 0,
) -> PixelVillageScene:
    profile_params = resolve_rpg_tile_render_params(
        params,
        render_defaults,
        tile_px_key="pixel_village_tile_px",
        fallback_tile_px=_DEFAULTS.tile_px,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:canvas_profile",
    )
    return render_pixel_village_map(
        _scene_seed(str(namespace), int(instance_seed), int(attempt_index)),
        width=int(profile_params["canvas_width"]),
        height=int(profile_params["canvas_height"]),
        tile_px=int(profile_params["tile_px"]),
        grid_cols=params.get("grid_cols", group_default(render_defaults, "pixel_village_grid_cols", profile_params["grid_cols"])),
        grid_rows=params.get("grid_rows", group_default(render_defaults, "pixel_village_grid_rows", profile_params["grid_rows"])),
        cemetery_mode=str(params.get("cemetery_mode", group_default(render_defaults, "pixel_village_cemetery_mode", _DEFAULTS.cemetery_mode))),
        orchard_mode=str(params.get("orchard_mode", group_default(render_defaults, "pixel_village_orchard_mode", _DEFAULTS.orchard_mode))),
        windmill_mode=str(params.get("windmill_mode", group_default(render_defaults, "pixel_village_windmill_mode", _DEFAULTS.windmill_mode))),
        theme_mode=str(params.get("theme_mode", group_default(render_defaults, "pixel_village_theme_mode", _DEFAULTS.theme_mode))),
        river_mode=str(params.get("river_mode", group_default(render_defaults, "pixel_village_river_mode", "auto"))),
        river_orientation=str(params.get("river_orientation", group_default(render_defaults, "pixel_village_river_orientation", "auto"))),
        river_placement=str(params.get("river_placement", group_default(render_defaults, "pixel_village_river_placement", "edge"))),
        path_person_count=int(path_person_count),
        background_person_path_clearance=int(background_person_path_clearance),
    )


def _bbox(entity: PixelVillageEntity, *, pad: float = 0.0, image_size: tuple[int, int]) -> list[float]:
    x0, y0, x1, y1 = entity.bbox_xyxy
    width, height = image_size
    p = max(0.0, float(pad))
    return [
        round(max(0.0, float(x0) - p), 3),
        round(max(0.0, float(y0) - p), 3),
        round(min(float(width), float(x1) + p), 3),
        round(min(float(height), float(y1) + p), 3),
    ]


def _entity_bbox_map(scene: PixelVillageScene, *, pad: float = 0.0) -> Dict[str, list[float]]:
    return {
        str(entity.entity_id): _bbox(entity, pad=float(pad), image_size=scene.image.size)
        for entity in scene.entities
    }


def _entity_footprint(entity: PixelVillageEntity) -> set[tuple[int, int]]:
    x, y, w, h = entity.tile_xywh
    return {
        (int(xx), int(yy))
        for xx in range(int(x), int(x) + int(w))
        for yy in range(int(y), int(y) + int(h))
    }


def _counted_object_entities(scene: PixelVillageScene, target_object: str) -> Tuple[PixelVillageEntity, ...]:
    target = str(target_object)
    public_name = TARGET_PUBLIC_NAME[target]
    if target == "building":
        return tuple(entity for entity in scene.entities if str(entity.category) == "building")
    if target == "person":
        return tuple(entity for entity in scene.entities if str(entity.category) == "person")
    return tuple(entity for entity in scene.entities if str(entity.public_name) == str(public_name))


def _path_people(scene: PixelVillageScene) -> Tuple[PixelVillageEntity, ...]:
    path_tiles = {tuple(int(v) for v in tile) for tile in scene.trace.get("path_tiles", [])}
    return tuple(
        entity
        for entity in scene.entities
        if str(entity.category) == "person" and bool(_entity_footprint(entity) & path_tiles)
    )


def _territory_object_entities(scene: PixelVillageScene, sample: _TerritoryObjectSample) -> Tuple[PixelVillageEntity, ...]:
    return tuple(
        entity
        for entity in scene.entities
        if str(entity.public_name) == str(sample.target_public_name)
        and str(entity.metadata.get("territory_id", "")) == str(sample.territory_id)
    )


def _river_bounds(scene: PixelVillageScene) -> dict[str, int]:
    raw = scene.trace.get("river_bounds")
    if isinstance(raw, Mapping):
        return {
            "min_x": int(raw["min_x"]),
            "max_x": int(raw["max_x"]),
            "min_y": int(raw["min_y"]),
            "max_y": int(raw["max_y"]),
        }
    water_tiles = {tuple(int(v) for v in tile) for tile in scene.trace.get("water_tiles", [])}
    if not water_tiles:
        raise ValueError("river-side task requires a visible river")
    xs = [x for x, _ in water_tiles]
    ys = [y for _, y in water_tiles]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


def _entity_is_strictly_on_river_side(entity: PixelVillageEntity, *, side: str, river_bounds: Mapping[str, int]) -> bool:
    footprint = _entity_footprint(entity)
    if not footprint:
        return False
    xs = [x for x, _ in footprint]
    ys = [y for _, y in footprint]
    if side == "left":
        return max(xs) < int(river_bounds["min_x"])
    if side == "right":
        return min(xs) > int(river_bounds["max_x"])
    if side == "above":
        return max(ys) < int(river_bounds["min_y"])
    if side == "below":
        return min(ys) > int(river_bounds["max_y"])
    raise ValueError(f"unsupported river side: {side}")


def _river_side_object_entities(scene: PixelVillageScene, sample: _RiverSideObjectSample) -> Tuple[PixelVillageEntity, ...]:
    river_bounds = _river_bounds(scene)
    return tuple(
        entity
        for entity in _counted_object_entities(scene, sample.target_object)
        if _entity_is_strictly_on_river_side(entity, side=str(sample.river_side), river_bounds=river_bounds)
    )


def _scene_entities(scene: PixelVillageScene) -> list[dict[str, Any]]:
    return [entity.as_dict() for entity in scene.entities]


def _scene_territories(scene: PixelVillageScene) -> list[dict[str, Any]]:
    return [territory.as_dict() for territory in scene.territories]


def _render_metadata(scene: PixelVillageScene) -> dict[str, Any]:
    return {
        "renderer_id": str(scene.trace.get("renderer_id", "")),
        "theme_mode": str(scene.trace.get("theme_mode", "")),
        "theme_id": str(scene.trace.get("theme_id", "")),
        "grid_cols": int(scene.trace.get("grid_cols", 0)),
        "grid_rows": int(scene.trace.get("grid_rows", 0)),
        "tile_px": int(scene.trace.get("tile_px", 0)),
        "map_offset_xy": list(scene.trace.get("map_offset_xy", [])),
        "map_size_px": list(scene.trace.get("map_size_px", [])),
        "river_present": bool(scene.trace.get("river_present", False)),
        "river_orientation": str(scene.trace.get("river_orientation", "")),
        "river_placement": str(scene.trace.get("river_placement", "")),
        "river_bounds": dict(scene.trace.get("river_bounds", {}) or {}),
        "bridge_box": list(scene.trace.get("bridge_box", []) or []),
        "cemetery_present": bool(scene.trace.get("cemetery_present", False)),
        "orchard_present": bool(scene.trace.get("orchard_present", False)),
        "windmill_present": bool(scene.trace.get("windmill_present", False)),
    }


def _build_object_sample(*, instance_seed: int, params: Mapping[str, Any], defaults: Mapping[str, Any], namespace: str) -> _ObjectTypeSample:
    target_object, target_probs = _resolve_target_object(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
    )
    return _ObjectTypeSample(
        target_object=str(target_object),
        target_plural=str(TARGET_PROMPT_PLURAL[str(target_object)]),
        target_unit=str(TARGET_PROMPT_UNIT[str(target_object)]),
        target_public_name=str(TARGET_PUBLIC_NAME[str(target_object)]),
        target_object_probabilities=dict(target_probs),
    )


def _build_path_sample(*, instance_seed: int, params: Mapping[str, Any], defaults: Mapping[str, Any], namespace: str) -> _PathPeopleSample:
    count_min, count_max = bounds(
        params,
        defaults,
        "path_person_count_min",
        "path_person_count_max",
        _DEFAULTS.path_person_count_min,
        _DEFAULTS.path_person_count_max,
    )
    count, probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:path_person_count",
        low=int(count_min),
        high=int(count_max),
        explicit_key="path_person_count",
    )
    return _PathPeopleSample(
        path_person_count=int(count),
        path_person_count_probabilities=dict(probabilities),
    )


def _build_territory_object_sample(*, instance_seed: int, params: Mapping[str, Any], defaults: Mapping[str, Any], namespace: str) -> _TerritoryObjectSample:
    target_key, probabilities = _resolve_territory_object(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
    )
    spec = TERRITORY_OBJECT_SPECS[str(target_key)]
    return _TerritoryObjectSample(
        target_key=str(target_key),
        territory_id=str(spec["territory_id"]),
        territory_type=str(spec["territory_type"]),
        territory_name=str(spec["territory_name"]),
        target_public_name=str(spec["target_public_name"]),
        target_plural=str(spec["target_plural"]),
        target_unit=str(spec["target_unit"]),
        force_mode_param=str(spec["force_mode_param"]),
        target_probabilities=dict(probabilities),
    )


def _build_river_side_object_sample(*, instance_seed: int, params: Mapping[str, Any], defaults: Mapping[str, Any], namespace: str) -> _RiverSideObjectSample:
    target_object, target_probs = _resolve_river_side_target_object(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
    )
    river_side, side_probs = _resolve_river_side(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
    )
    target_count, target_count_support, target_count_probs = _resolve_river_side_target_count(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
    )
    return _RiverSideObjectSample(
        target_object=str(target_object),
        target_plural=str(TARGET_PROMPT_PLURAL[str(target_object)]),
        target_unit=str(TARGET_PROMPT_UNIT[str(target_object)]),
        target_public_name=str(TARGET_PUBLIC_NAME[str(target_object)]),
        river_side=str(river_side),
        river_orientation=str(RIVER_SIDE_ORIENTATION[str(river_side)]),
        river_relation=str(RIVER_SIDE_PROMPT_RELATION[str(river_side)]),
        target_count=int(target_count),
        target_count_support=tuple(int(value) for value in target_count_support),
        target_object_probabilities=dict(target_probs),
        river_side_probabilities=dict(side_probs),
        target_count_probabilities=dict(target_count_probs),
    )










__all__ = [
    "RIVER_SIDE_KEYS",
    "RIVER_SIDE_ORIENTATION",
    "RIVER_SIDE_TARGET_KEYS",
    "SCENE_ID",
    "TARGET_OBJECT_KEYS",
    "TERRITORY_OBJECT_KEYS",
    "sample_option_answer_index",
    "_DEFAULTS",
    "_build_object_sample",
    "_build_path_sample",
    "_build_river_side_object_sample",
    "_build_territory_object_sample",
    "_counted_object_entities",
    "_entity_bbox_map",
    "_path_people",
    "_render_metadata",
    "_render_scene",
    "_river_bounds",
    "_river_side_object_entities",
    "_scene_entities",
    "_scene_territories",
    "_territory_object_entities",
]
