"""Count visible terrain tiles on the highest isometric farmstead layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile

from .shared.output import bbox_projection, isometric_farmstead_render_spec, isometric_farmstead_scene_ir, rounded_bbox
from .shared.prompts import build_isometric_farmstead_prompt_artifacts
from .shared.rendering import SCENE_ID, render_isometric_farmstead_scene
from .shared.state import IsoFarmsteadScene, IsoFarmsteadTile


TASK_ID = "task_illustrations__isometric_farmstead__highest_terrain_tile_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
PROMPT_QUERY_KEY = "highest_terrain_tile_count"


@dataclass(frozen=True)
class _SampleSpec:
    target_count: int
    target_count_probabilities: dict[str, float]
    answer_count_support: tuple[int, ...]
    answer_count_probabilities: dict[str, float]
    canvas_width: int
    canvas_height: int
    canvas_profile: str
    canvas_profile_probabilities: dict[str, float]


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _support_ints(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    support_key: str,
    fallback: Sequence[int],
) -> tuple[int, ...]:
    raw = params.get(str(support_key), group_default(defaults, str(support_key), tuple(fallback)))
    values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in values))
    if not support:
        raise ValueError(f"{support_key} must include at least one value")
    return support


def _select_target_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    support = _support_ints(params, defaults, support_key="answer_count_support", fallback=tuple(range(4, 9)))
    explicit = params.get("target_count", params.get("answer_count"))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"target_count must be one of {support}")
        return value, support_probability_map(support, selected=value, sort_keys=True), support
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:target_count")
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(value), dict(probabilities), support


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    target_count, target_count_probabilities, answer_count_support = _select_target_count(
        instance_seed=int(instance_seed),
        params=params,
        defaults=_GEN_DEFAULTS,
    )
    profile = resolve_canvas_profile(
        params=params,
        defaults=_RENDER_DEFAULTS,
        fallback_width=1200,
        fallback_height=800,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:canvas_profile",
    )
    return _SampleSpec(
        target_count=int(target_count),
        target_count_probabilities=dict(target_count_probabilities),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        answer_count_probabilities=dict(support_probability_map(answer_count_support, sort_keys=True)),
        canvas_width=int(profile.width),
        canvas_height=int(profile.height),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
    )


def _highest_level_tiles(scene: IsoFarmsteadScene) -> tuple[int, tuple[IsoFarmsteadTile, ...]]:
    levels = tuple(sorted({int(tile.level) for tile in scene.tiles}))
    if not levels:
        raise ValueError("farmstead scene has no terrain tiles")
    target_level = max(levels)
    tiles = tuple(
        sorted(
            (tile for tile in scene.tiles if int(tile.level) == int(target_level)),
            key=lambda tile: (int(tile.row), int(tile.col)),
        )
    )
    return int(target_level), tiles


def _tile_union_bbox(tiles: Sequence[IsoFarmsteadTile]) -> list[float]:
    if not tiles:
        raise ValueError("cannot build bbox for empty tile set")
    return rounded_bbox(
        (
            min(float(tile.bbox_xyxy[0]) for tile in tiles),
            min(float(tile.bbox_xyxy[1]) for tile in tiles),
            max(float(tile.bbox_xyxy[2]) for tile in tiles),
            max(float(tile.bbox_xyxy[3]) for tile in tiles),
        )
    )


def _tiles_are_connected(tiles: Sequence[IsoFarmsteadTile]) -> bool:
    tile_ids = {str(tile.tile_id) for tile in tiles}
    if not tile_ids:
        return False
    tiles_by_cell = {(int(tile.col), int(tile.row)): tile for tile in tiles}
    start = next(iter(tiles))
    stack = [start]
    visited = {str(start.tile_id)}
    while stack:
        current = stack.pop()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = tiles_by_cell.get((int(current.col) + dc, int(current.row) + dr))
            if neighbor is None or str(neighbor.tile_id) in visited:
                continue
            visited.add(str(neighbor.tile_id))
            stack.append(neighbor)
    return visited == tile_ids


def _prompt_slots(prompt_defaults: Mapping[str, Any]) -> dict[str, str]:
    return {
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_highest_terrain_tile_count"]),
        "annotation_hint": str(prompt_defaults["annotation_hint_highest_terrain_tile_count"]),
        "json_example": str(prompt_defaults["json_example_highest_terrain_tile_count"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_highest_terrain_tile_count"]),
    }


@register_task
class IllustrationsIsometricFarmsteadHighestTerrainTileCountTask:
    """Count the visible terrain tiles on the highest elevation layer."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'logical_composition')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one highest-layer tile count instance with scalar bbox annotation."""

        sample = _sample_spec(instance_seed=int(instance_seed), params=params)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_highest_terrain_tile_count",
                "annotation_hint_highest_terrain_tile_count",
                "json_example_highest_terrain_tile_count",
                "json_example_answer_only_highest_terrain_tile_count",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )

        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                scene_seed = int(instance_seed) + int(attempt) * 1009
                scene = render_isometric_farmstead_scene(
                    scene_seed,
                    width=sample.canvas_width,
                    height=sample.canvas_height,
                    canvas_profile=sample.canvas_profile,
                    canvas_profile_probabilities=sample.canvas_profile_probabilities,
                    highest_level_tile_count=int(sample.target_count),
                    reserve_highest_level_tiles=True,
                    render_style_params=params,
                    render_style_defaults=_RENDER_DEFAULTS,
                )
                target_level, counted_tiles = _highest_level_tiles(scene)
                if len(counted_tiles) != int(sample.target_count):
                    raise ValueError(f"highest tile count {len(counted_tiles)} did not match target {sample.target_count}")
                if not _tiles_are_connected(counted_tiles):
                    raise ValueError("highest terrain tiles are not one connected layer")
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        annotation_value = _tile_union_bbox(counted_tiles)
        counted_tile_ids = [str(tile.tile_id) for tile in counted_tiles]
        counted_tile_bboxes = [rounded_bbox(tile.bbox_xyxy) for tile in counted_tiles]
        prompt_artifacts = build_isometric_farmstead_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=_prompt_slots(prompt_defaults),
            instance_seed=int(instance_seed),
        )
        render_map = {
            "image_id": "img0",
            "target_level": int(target_level),
            "answer_count": int(len(counted_tiles)),
            "counted_tile_ids": list(counted_tile_ids),
            "counted_tile_bboxes_px": [list(bbox) for bbox in counted_tile_bboxes],
            "highest_level_bbox_px": list(annotation_value),
            "level_tile_counts": dict(scene.trace.get("level_tile_counts", {})),
        }
        query_params = {
            "query_id": "single",
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": {"single": 1.0},
            "target_level": int(target_level),
            "target_count": int(sample.target_count),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "answer_count_support": list(sample.answer_count_support),
            "answer_count_probabilities": dict(sample.answer_count_probabilities),
            "answer_count": int(len(counted_tiles)),
            "counted_tile_ids": list(counted_tile_ids),
            "canvas_profile": str(sample.canvas_profile),
            "canvas_profile_probabilities": dict(sample.canvas_profile_probabilities),
        }
        trace_payload = {
            "scene_ir": isometric_farmstead_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "operation": "count_tiles_on_highest_elevation_layer",
                    "target_level": int(target_level),
                    "answer_count": int(len(counted_tiles)),
                    "counted_tile_ids": list(counted_tile_ids),
                    "highest_level_bbox_px": list(annotation_value),
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": "single",
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": query_params,
            },
            "render_spec": isometric_farmstead_render_spec(scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": "single",
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": int(len(counted_tiles)),
                "target_level": int(target_level),
                "counted_tile_ids": list(counted_tile_ids),
                "renderer": dict(scene.trace),
            },
            "witness_symbolic": {
                "answer_count": int(len(counted_tiles)),
                "target_level": int(target_level),
                "counted_tile_ids": list(counted_tile_ids),
                "counted_tile_bboxes": [list(bbox) for bbox in counted_tile_bboxes],
                "highest_level_bbox": list(annotation_value),
            },
            "projected_annotation": bbox_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(len(counted_tiles))),
            annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id="single",
        )


__all__ = [
    "IllustrationsIsometricFarmsteadHighestTerrainTileCountTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
