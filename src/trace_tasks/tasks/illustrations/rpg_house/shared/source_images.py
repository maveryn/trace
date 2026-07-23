"""Source-image helpers for RPG house reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.illustrations.shared.option_rendering import image_detail_score
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import resolve_rpg_tile_profile
from trace_tasks.tasks.shared.config_defaults import group_default

from .rendering import (
    DEFAULT_TILE_PX,
    MAX_ROOM_COUNT,
    MIN_ROOM_COUNT,
    RENDERER_ID,
    SCENE_ID,
    render_rpg_house_profile_scene,
)
from .state import RpgHouseScene


@dataclass(frozen=True)
class RpgHouseSourceSceneSpec:
    """Dense RPG house source scene shared by visual reconstruction tasks."""

    source_room_count: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    source_room_count_probabilities: Dict[str, float]


def sample_int_range(
    *,
    seed_namespace: str,
    instance_seed: int,
    attempt_index: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> int:
    """Sample one integer from a task-owned inclusive range."""

    low = int(params.get(str(low_key), group_default(defaults, str(low_key), int(fallback_low))))
    high = int(params.get(str(high_key), group_default(defaults, str(high_key), int(fallback_high))))
    if low > high:
        raise ValueError(f"{low_key}/{high_key} leaves no feasible integer range")
    rng = spawn_rng(
        int(instance_seed),
        str(seed_namespace),
        int(attempt_index),
        str(low_key),
        str(high_key),
    )
    return int(rng.randint(int(low), int(high)))


def sample_support_index(
    *,
    seed_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    support: Sequence[int],
    explicit_key: str,
) -> Tuple[int, Dict[str, float]]:
    """Select one integer value from finite support."""

    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError(f"{explicit_key} has empty support")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(values):
            raise ValueError(f"{explicit_key} must be one of {values}")
        return int(value), support_probability_map(values, selected=int(value), sort_keys=True)
    namespace = str(seed_namespace)
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    value, probabilities = uniform_choice_with_probabilities(rng, values, sort_keys=True)
    return int(value), dict(probabilities)


def option_count_support(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Return task option-count support filtered to the caller fallback set."""

    fallback_values = tuple(int(value) for value in fallback)
    raw = params.get("option_count_support", group_default(defaults, "option_count_support", fallback_values))
    raw_values = (raw,) if isinstance(raw, int) else tuple(raw)
    support = tuple(dict.fromkeys(int(value) for value in raw_values if int(value) in set(fallback_values)))
    if not support:
        raise ValueError(f"option_count_support must include one of {fallback_values}")
    return support


def _range_support(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    min_key: str,
    max_key: str,
    support_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, ...]:
    raw_support = params.get(str(support_key), group_default(defaults, str(support_key), None))
    if raw_support is not None:
        if isinstance(raw_support, int):
            values = (int(raw_support),)
        else:
            values = tuple(int(value) for value in raw_support)
        support = tuple(dict.fromkeys(value for value in values if int(fallback_min) <= int(value) <= int(fallback_max)))
        if not support:
            raise ValueError(f"{support_key} must include at least one value in [{fallback_min}, {fallback_max}]")
        return support
    low = int(params.get(str(min_key), group_default(defaults, str(min_key), int(fallback_min))))
    high = int(params.get(str(max_key), group_default(defaults, str(max_key), int(fallback_max))))
    low = max(int(fallback_min), low)
    high = min(int(fallback_max), high)
    if low > high:
        raise ValueError(f"{min_key}/{max_key} leaves no feasible source room count")
    return tuple(range(int(low), int(high) + 1))


def sample_rpg_house_source_scene_spec(
    *,
    seed_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    source_room_count_min: int,
    source_room_count_max: int,
    source_width: int,
    source_height: int,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> RpgHouseSourceSceneSpec:
    """Resolve source render dimensions and room count for reconstruction tasks."""

    support = _range_support(
        params=params,
        defaults=generation_defaults,
        min_key="source_room_count_min",
        max_key="source_room_count_max",
        support_key="source_room_count_support",
        fallback_min=int(source_room_count_min),
        fallback_max=int(source_room_count_max),
    )
    explicit = params.get("source_room_count")
    if explicit is not None:
        source_room_count = int(explicit)
        if source_room_count not in set(support):
            raise ValueError(f"source_room_count must be one of {support}")
        probabilities = support_probability_map(support, selected=int(source_room_count), sort_keys=True)
    else:
        rng = spawn_rng(int(instance_seed), f"{seed_namespace}:source_room_count")
        source_room_count, probabilities = uniform_choice_with_probabilities(
            rng,
            support,
            sort_keys=True,
        )
        source_room_count = int(source_room_count)
        probabilities = dict(probabilities)

    profile = resolve_rpg_tile_profile(
        params=params,
        defaults=generation_defaults,
        tile_px_key="rpg_house_tile_px",
        fallback_tile_px=DEFAULT_TILE_PX,
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:source_profile",
        width_key="source_width",
        height_key="source_height",
    )
    width = int(profile.width)
    height = int(profile.height)
    trace = dict(profile.trace())
    if grid_rows is not None and grid_cols is not None:
        if width % int(grid_cols) != 0 or height % int(grid_rows) != 0:
            raise ValueError("RPG house source profile must align with the requested reconstruction grid")
        trace["grid_alignment"] = {"rows": int(grid_rows), "cols": int(grid_cols)}
    return RpgHouseSourceSceneSpec(
        source_room_count=int(source_room_count),
        source_size=(int(width), int(height)),
        source_profile_trace=trace,
        source_room_count_probabilities=probabilities,
    )


def render_rpg_house_source_scene(
    *,
    seed_namespace: str,
    instance_seed: int,
    attempt_index: int,
    source: RpgHouseSourceSceneSpec,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RpgHouseScene:
    """Render one dense source RPG house panel."""

    tile_px = int(
        params.get(
            "source_tile_px",
            params.get("tile_px", group_default(render_defaults, "rpg_house_tile_px", DEFAULT_TILE_PX)),
        )
    )
    render_params = {
        "canvas_width": int(source.source_size[0]),
        "canvas_height": int(source.source_size[1]),
        "tile_px": int(tile_px),
        "grid_cols": int(source.source_profile_trace.get("rpg_tile_profile", {}).get("grid_cols", 0)),
        "grid_rows": int(source.source_profile_trace.get("rpg_tile_profile", {}).get("grid_rows", 0)),
        **dict(source.source_profile_trace),
    }
    return render_rpg_house_profile_scene(
        hash64(int(instance_seed), f"{seed_namespace}:source_scene", int(attempt_index)),
        render_params=render_params,
        tile_px=tile_px,
        room_count=int(source.source_room_count),
        sample_mixed_door_states=True,
    )


def candidate_crop_boxes_from_scene(
    *,
    scene: RpgHouseScene,
    patch_size: Sequence[int],
    margin_px: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Build patch-sized crop candidates centered on meaningful house features."""

    width, height = scene.image.size
    patch_w, patch_h = int(patch_size[0]), int(patch_size[1])
    margin = int(margin_px)
    max_x0 = int(width) - int(patch_w) - margin
    max_y0 = int(height) - int(patch_h) - margin
    if max_x0 < margin or max_y0 < margin:
        return tuple()

    def crop_for_box(box: Sequence[float]) -> Tuple[int, int, int, int]:
        cx = int(round((float(box[0]) + float(box[2])) * 0.5))
        cy = int(round((float(box[1]) + float(box[3])) * 0.5))
        x0 = min(max(margin, cx - patch_w // 2), max_x0)
        y0 = min(max(margin, cy - patch_h // 2), max_y0)
        return (int(x0), int(y0), int(x0 + patch_w), int(y0 + patch_h))

    raw_boxes: list[Tuple[int, int, int, int]] = []
    for entity in scene.entities:
        raw_boxes.append(crop_for_box(entity.bbox_xyxy))
    for door in scene.doors:
        raw_boxes.append(crop_for_box(door.bbox_xyxy))
    for room in scene.rooms:
        raw_boxes.append(crop_for_box(room.bbox_xyxy))

    ranked: list[Tuple[float, Tuple[int, int, int, int]]] = []
    seen: set[Tuple[int, int, int, int]] = set()
    for box in raw_boxes:
        if box in seen:
            continue
        seen.add(box)
        if box[0] < margin or box[1] < margin or box[2] > width - margin or box[3] > height - margin:
            continue
        ranked.append((float(image_detail_score(scene.image.crop(box))), box))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(box for _score, box in ranked)


def rpg_house_source_style_trace(scene: RpgHouseScene) -> Dict[str, Any]:
    """Return source-render style metadata for reconstruction tasks."""

    return {
        "source_renderer_id": str(scene.trace.get("renderer_id", RENDERER_ID)),
        "source_style_id": str(scene.trace.get("renderer_style", "")),
        "source_theme_id": str(scene.trace.get("theme_id", "")),
        "source_tile_px": int(scene.trace.get("tile_px", DEFAULT_TILE_PX)),
        "source_grid_cols": int(scene.trace.get("grid_cols", 0)),
        "source_grid_rows": int(scene.trace.get("grid_rows", 0)),
        "source_canvas_profile": str(scene.trace.get("canvas_profile", "")),
        "source_canvas_profile_probabilities": dict(scene.trace.get("canvas_profile_probabilities", {})),
    }


__all__ = [
    "RpgHouseSourceSceneSpec",
    "candidate_crop_boxes_from_scene",
    "option_count_support",
    "render_rpg_house_source_scene",
    "rpg_house_source_style_trace",
    "sample_rpg_house_source_scene_spec",
    "sample_int_range",
    "sample_support_index",
]
