"""Count all terrain tiles reachable by a tactical RPG movement budget."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.output import (
    bbox_set_projection,
    movement_reachable_count_render_map,
    rounded_bbox,
    rpg_tactical_map_render_spec,
    rpg_tactical_map_scene_ir,
)
from .shared.prompts import (
    build_rpg_tactical_map_task_prompt_with_default_slots,
    rpg_tactical_map_terrain_rules_text,
)
from .shared.relations import (
    TERRAIN_MOVEMENT_COSTS,
    TERRAIN_WATER,
    shortest_movement_costs,
)
from .shared.rendering import (
    DEFAULT_TILE_PX,
    SCENE_ID,
    render_rpg_tactical_map_scene,
    resolve_tactical_map_render_params,
)
from .shared.sampling import select_int_from_support
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__movement_reachable_tile_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "movement_reachable_tile_count"
DEFAULT_MOVEMENT_BUDGET_SUPPORT: tuple[int, ...] = (2, 3, 4)
DEFAULT_MIN_ANSWER_COUNT = 3
DEFAULT_MAX_ANSWER_COUNT = 15

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _tiles_by_coord(scene: RpgTacticalMapScene) -> dict[tuple[int, int], RpgTacticalTile]:
    return {(int(tile.row), int(tile.col)): tile for tile in scene.tiles}


def _tile_by_id(scene: RpgTacticalMapScene) -> dict[str, RpgTacticalTile]:
    return {str(tile.tile_id): tile for tile in scene.tiles}


def _reachable_tiles_for_budget(
    *,
    scene: RpgTacticalMapScene,
    movement_budget: int,
) -> tuple[list[RpgTacticalTile], dict[str, int], str]:
    """Return non-start tiles reachable within the movement budget."""

    if not scene.units:
        raise ValueError("movement count task requires a blue unit")
    start_tile_id = str(scene.units[0].tile_id)
    tiles_by_id = _tile_by_id(scene)
    start_tile = tiles_by_id[start_tile_id]
    movement_costs = shortest_movement_costs(_tiles_by_coord(scene), start_coord=start_tile.coord)
    counted_tiles = [
        tile
        for tile in scene.tiles
        if str(tile.tile_id) != start_tile_id
        and str(tile.tile_id) in movement_costs
        and int(movement_costs[str(tile.tile_id)]) <= int(movement_budget)
    ]
    counted_tiles.sort(key=lambda tile: (int(tile.row), int(tile.col), str(tile.tile_id)))
    return counted_tiles, movement_costs, start_tile_id


@register_task
class IllustrationsRpgTacticalMapMovementReachableTileCountTask:
    """Count all tiles the blue unit can reach within a small movement budget."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one movement-area count instance with bbox-set annotation."""

        resolved_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        movement_budget, movement_budget_probabilities = select_int_from_support(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            support_key="movement_budget_support",
            explicit_key="movement_budget",
            fallback_support=DEFAULT_MOVEMENT_BUDGET_SUPPORT,
            namespace=f"{TASK_ID}:movement_budget",
        )
        min_answer_count = int(task_params.get("min_answer_count", group_default(_GEN_DEFAULTS, "min_answer_count", DEFAULT_MIN_ANSWER_COUNT)))
        max_answer_count = int(task_params.get("max_answer_count", group_default(_GEN_DEFAULTS, "max_answer_count", DEFAULT_MAX_ANSWER_COUNT)))
        if min_answer_count < 0 or max_answer_count < min_answer_count:
            raise ValueError("reachable tile count range must satisfy 0 <= min <= max")
        render_params = resolve_tactical_map_render_params(
            task_params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:canvas_profile",
        )
        _prompt_defaults, prompt_artifacts = build_rpg_tactical_map_task_prompt_with_default_slots(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            prompt_query_key=PROMPT_QUERY_KEY,
            answer_hint_key="answer_hint_movement_reachable_tile_count",
            annotation_hint_key="annotation_hint_movement_reachable_tile_count",
            json_example_key="json_example_movement_reachable_tile_count",
            json_example_answer_only_key="json_example_answer_only_movement_reachable_tile_count",
            context_label=TASK_ID,
            slots={
                "movement_points": str(int(movement_budget)),
                "terrain_rules": rpg_tactical_map_terrain_rules_text(),
            },
            instance_seed=int(instance_seed),
        )

        selected_scene: RpgTacticalMapScene | None = None
        counted_tiles: list[RpgTacticalTile] | None = None
        selected_costs: dict[str, int] | None = None
        start_tile_id: str | None = None
        attempt_errors: list[str] = []
        for attempt in range(max(1, int(max_attempts))):
            scene_seed = int(instance_seed) + int(attempt) * 7919
            scene = render_rpg_tactical_map_scene(
                scene_seed,
                width=int(render_params["canvas_width"]),
                height=int(render_params["canvas_height"]),
                grid_cols=int(render_params["grid_cols"]),
                grid_rows=int(render_params["grid_rows"]),
                tile_px=int(render_params.get("tile_px", DEFAULT_TILE_PX)),
                render_metadata=render_params,
            )
            try:
                probe_tiles, movement_costs, probe_start_tile_id = _reachable_tiles_for_budget(
                    scene=scene,
                    movement_budget=int(movement_budget),
                )
            except ValueError as exc:
                attempt_errors.append(str(exc))
                continue
            answer_count = len(probe_tiles)
            if not (int(min_answer_count) <= int(answer_count) <= int(max_answer_count)):
                attempt_errors.append(f"reachable tile count {answer_count} outside {min_answer_count}..{max_answer_count}")
                continue
            selected_scene = scene
            counted_tiles = probe_tiles
            selected_costs = movement_costs
            start_tile_id = probe_start_tile_id
            break
        if selected_scene is None or counted_tiles is None or selected_costs is None or start_tile_id is None:
            raise ValueError("failed to generate valid movement-reachability count: " + "; ".join(attempt_errors[-3:]))

        counted_tile_ids = [str(tile.tile_id) for tile in counted_tiles]
        annotation_value = [rounded_bbox(tile.bbox_xyxy) for tile in counted_tiles]
        render_map = movement_reachable_count_render_map(
            scene=selected_scene,
            counted_tile_ids=counted_tile_ids,
            movement_costs_by_tile_id=selected_costs,
            movement_budget=int(movement_budget),
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "movement_budget": int(movement_budget),
            "movement_budget_probabilities": dict(movement_budget_probabilities),
            "answer_count": int(len(counted_tiles)),
            "answer_count_range": [int(min_answer_count), int(max_answer_count)],
            "counted_tile_ids": list(counted_tile_ids),
            "start_tile_id": str(start_tile_id),
            "canvas_profile": str(render_params.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        }
        trace_payload = {
            "scene_ir": rpg_tactical_map_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=selected_scene,
                relations={
                    "operation": "count_tiles_reachable_under_movement_budget",
                    "movement_budget": int(movement_budget),
                    "terrain_movement_costs": dict(TERRAIN_MOVEMENT_COSTS),
                    "blocked_terrain": [TERRAIN_WATER],
                    "start_tile_id": str(start_tile_id),
                    "counted_tile_ids": list(counted_tile_ids),
                    "answer_count": int(len(counted_tiles)),
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": query_params,
            },
            "render_spec": rpg_tactical_map_render_spec(selected_scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": int(len(counted_tiles)),
                "movement_budget": int(movement_budget),
                "movement_costs_by_tile_id": {str(key): int(value) for key, value in selected_costs.items()},
                "start_tile_id": str(start_tile_id),
                "counted_tile_ids": list(counted_tile_ids),
                "renderer": dict(selected_scene.trace),
            },
            "witness_symbolic": {
                "answer_count": int(len(counted_tiles)),
                "movement_budget": int(movement_budget),
                "start_tile_id": str(start_tile_id),
                "counted_tile_ids": list(counted_tile_ids),
                "counted_tile_bboxes": [list(bbox) for bbox in annotation_value],
            },
            "projected_annotation": bbox_set_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(len(counted_tiles))),
            annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_value]),
            image=selected_scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(resolved_query_id),
        )


__all__ = [
    "IllustrationsRpgTacticalMapMovementReachableTileCountTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
