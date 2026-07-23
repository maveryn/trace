"""Read the shortest movement cost to a marked tactical RPG destination tile."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import (
    RpgTacticalMapValueAttempt,
    RpgTacticalMapValuePlan,
    run_rpg_tactical_map_value_lifecycle,
)
from .shared.output import movement_cost_value_render_map
from .shared.prompts import rpg_tactical_map_terrain_rules_text
from .shared.relations import TERRAIN_MOVEMENT_COSTS, TERRAIN_WATER, shortest_movement_costs_and_paths
from .shared.rendering import SCENE_ID
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__movement_cost_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "movement_cost_value"
DEFAULT_MIN_MOVEMENT_COST = 3
DEFAULT_MAX_MOVEMENT_COST = 10

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _tiles_by_coord(scene: RpgTacticalMapScene) -> dict[tuple[int, int], RpgTacticalTile]:
    return {(int(tile.row), int(tile.col)): tile for tile in scene.tiles}


def _tile_by_id(scene: RpgTacticalMapScene) -> dict[str, RpgTacticalTile]:
    return {str(tile.tile_id): tile for tile in scene.tiles}


def _tile_manhattan(first: RpgTacticalTile, second: RpgTacticalTile) -> int:
    return abs(int(first.row) - int(second.row)) + abs(int(first.col) - int(second.col))


def _select_movement_cost_attempt(
    *,
    scene: RpgTacticalMapScene,
    min_movement_cost: int,
    max_movement_cost: int,
    instance_seed: int,
) -> RpgTacticalMapValueAttempt:
    """Select one marked destination tile with a bounded shortest path cost."""

    if not scene.units:
        raise ValueError("movement cost task requires a blue unit")
    if int(min_movement_cost) < 1 or int(max_movement_cost) < int(min_movement_cost):
        raise ValueError("movement cost range must satisfy 1 <= min <= max")
    tiles_by_id = _tile_by_id(scene)
    start_tile_id = str(scene.units[0].tile_id)
    start_tile = tiles_by_id[start_tile_id]
    movement_costs, movement_paths = shortest_movement_costs_and_paths(_tiles_by_coord(scene), start_coord=start_tile.coord)
    target_pool = [
        tile
        for tile in scene.tiles
        if str(tile.tile_id) != start_tile_id
        and bool(tile.passable)
        and str(tile.tile_id) in movement_costs
        and int(min_movement_cost) <= int(movement_costs[str(tile.tile_id)]) <= int(max_movement_cost)
    ]
    if not target_pool:
        raise ValueError("no destination tile in requested movement-cost range")

    rng = random.Random(f"{int(instance_seed)}:movement_cost_value:{int(min_movement_cost)}:{int(max_movement_cost)}")
    tile_jitter = {str(tile.tile_id): float(rng.random()) for tile in scene.tiles}
    available_costs = sorted({int(movement_costs[str(tile.tile_id)]) for tile in target_pool})
    desired_cost = int(available_costs[int(rng.randrange(len(available_costs)))])
    cost_pool = [
        tile
        for tile in target_pool
        if int(movement_costs[str(tile.tile_id)]) == int(desired_cost)
    ]
    preferred_pool = [
        tile
        for tile in cost_pool
        if int(movement_costs[str(tile.tile_id)]) > _tile_manhattan(start_tile, tile)
    ] or cost_pool

    def target_sort_key(tile: RpgTacticalTile) -> tuple[int, int, int, float]:
        movement_cost = int(movement_costs[str(tile.tile_id)])
        manhattan = _tile_manhattan(start_tile, tile)
        terrain_delta = int(movement_cost) - int(manhattan)
        return (
            -int(terrain_delta),
            -int(movement_cost),
            -int(manhattan),
            tile_jitter[str(tile.tile_id)],
        )

    target_tile = sorted(preferred_pool, key=target_sort_key)[0]
    answer_value = int(movement_costs[str(target_tile.tile_id)])
    target_manhattan = _tile_manhattan(start_tile, target_tile)
    shortest_path_tile_ids = [str(tile_id) for tile_id in movement_paths[str(target_tile.tile_id)]]
    shortest_path_terrains = [str(tiles_by_id[str(tile_id)].terrain) for tile_id in shortest_path_tile_ids]
    shortest_path_entry_costs = [
        0 if index == 0 else int(tiles_by_id[str(tile_id)].movement_cost or 0)
        for index, tile_id in enumerate(shortest_path_tile_ids)
    ]
    relation_fields = {
        "operation": "shortest_movement_cost_to_marked_tile",
        "terrain_movement_costs": dict(TERRAIN_MOVEMENT_COSTS),
        "blocked_terrain": [TERRAIN_WATER],
        "start_tile_id": str(start_tile_id),
        "target_tile_id": str(target_tile.tile_id),
        "target_terrain": str(target_tile.terrain),
        "target_manhattan_distance": int(target_manhattan),
        "target_shortest_movement_cost": int(answer_value),
        "shortest_path_tile_ids": list(shortest_path_tile_ids),
        "shortest_path_terrains": list(shortest_path_terrains),
        "shortest_path_entry_costs": list(shortest_path_entry_costs),
        "sampled_target_cost_bucket": int(desired_cost),
        "available_target_cost_buckets": list(available_costs),
        "movement_cost_range": [int(min_movement_cost), int(max_movement_cost)],
        "terrain_cost_affects_answer": bool(answer_value > target_manhattan),
    }
    execution_fields = {
        **relation_fields,
        "movement_costs_by_tile_id": {str(tile_id): int(cost) for tile_id, cost in movement_costs.items()},
    }
    return RpgTacticalMapValueAttempt(
        target_tile_id=str(target_tile.tile_id),
        answer_value=int(answer_value),
        annotation_tile_id_map={
            "player_cell": str(start_tile_id),
            "target_cell": str(target_tile.tile_id),
        },
        relation_fields=relation_fields,
        execution_fields=execution_fields,
        witness_fields={
            "target_tile_id": str(target_tile.tile_id),
            "target_terrain": str(target_tile.terrain),
            "target_manhattan_distance": int(target_manhattan),
            "target_shortest_movement_cost": int(answer_value),
            "shortest_path_tile_ids": list(shortest_path_tile_ids),
            "shortest_path_terrains": list(shortest_path_terrains),
            "shortest_path_entry_costs": list(shortest_path_entry_costs),
            "terrain_cost_affects_answer": bool(answer_value > target_manhattan),
        },
    )


def _build_movement_cost_plan(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    query_id: str,
) -> RpgTacticalMapValuePlan:
    """Bind movement-cost range axes for the marked-destination value task."""

    min_movement_cost = int(
        task_params.get(
            "min_movement_cost",
            group_default(_GEN_DEFAULTS, "min_movement_cost", DEFAULT_MIN_MOVEMENT_COST),
        )
    )
    max_movement_cost = int(
        task_params.get(
            "max_movement_cost",
            group_default(_GEN_DEFAULTS, "max_movement_cost", DEFAULT_MAX_MOVEMENT_COST),
        )
    )

    def build_attempt(scene: RpgTacticalMapScene, attempt_seed: int) -> RpgTacticalMapValueAttempt:
        return _select_movement_cost_attempt(
            scene=scene,
            min_movement_cost=int(min_movement_cost),
            max_movement_cost=int(max_movement_cost),
            instance_seed=int(attempt_seed),
        )

    def build_render_map(scene: RpgTacticalMapScene, attempt: RpgTacticalMapValueAttempt) -> Mapping[str, Any]:
        execution = attempt.execution_fields
        return movement_cost_value_render_map(
            scene=scene,
            target_tile_id=str(attempt.target_tile_id),
            shortest_path_tile_ids=execution["shortest_path_tile_ids"],
            movement_costs_by_tile_id=execution["movement_costs_by_tile_id"],
            start_tile_id=str(execution["start_tile_id"]),
            target_manhattan_distance=int(execution["target_manhattan_distance"]),
            answer_value=int(attempt.answer_value),
        )

    return RpgTacticalMapValuePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={
            "terrain_rules": rpg_tactical_map_terrain_rules_text(),
        },
        answer_hint_key="answer_hint_movement_cost_value",
        annotation_hint_key="annotation_hint_movement_cost_value",
        json_example_key="json_example_movement_cost_value",
        json_example_answer_only_key="json_example_answer_only_movement_cost_value",
        query_params={
            "query_id": str(query_id),
            "query_id_probabilities": dict(query_probabilities),
            "min_movement_cost": int(min_movement_cost),
            "max_movement_cost": int(max_movement_cost),
        },
        build_attempt=build_attempt,
        build_render_map=build_render_map,
        failure_label="movement cost value target",
    )


@register_task
class IllustrationsRpgTacticalMapMovementCostValueTask:
    """Return the shortest movement-point cost to a marked destination tile."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_rpg_tactical_map_value_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_plan=_build_movement_cost_plan,
        )


__all__ = [
    "IllustrationsRpgTacticalMapMovementCostValueTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
