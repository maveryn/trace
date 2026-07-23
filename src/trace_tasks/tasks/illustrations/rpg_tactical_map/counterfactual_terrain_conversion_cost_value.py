"""Compute shortest movement cost after one optimal terrain conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .shared.output import counterfactual_terrain_conversion_cost_render_map
from .shared.prompts import rpg_tactical_map_terrain_rules_text
from .shared.relations import (
    TERRAIN_FOREST,
    TERRAIN_MOUNTAIN,
    TERRAIN_MOVEMENT_COSTS,
    TERRAIN_ROAD,
    TERRAIN_WATER,
    shortest_movement_costs_and_paths,
)
from .shared.rendering import SCENE_ID
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__counterfactual_terrain_conversion_cost_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "counterfactual_terrain_conversion_cost_value"
DEFAULT_MIN_COUNTERFACTUAL_COST = 2
DEFAULT_MAX_COUNTERFACTUAL_COST = 6
CONVERTIBLE_TERRAINS: tuple[str, ...] = (TERRAIN_WATER, TERRAIN_MOUNTAIN, TERRAIN_FOREST)

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _CounterfactualCandidate:
    target_tile_id: str
    changed_tile_id: str
    changed_original_terrain: str
    answer_value: int
    original_target_cost: int | None
    shortest_path_tile_ids: tuple[str, ...]
    shortest_path_terrains: tuple[str, ...]
    shortest_path_entry_costs_after_conversion: tuple[int, ...]
    counterfactual_movement_costs_by_tile_id: Mapping[str, int]
    improvement: int | None


def _tiles_by_coord(scene: RpgTacticalMapScene) -> dict[tuple[int, int], RpgTacticalTile]:
    return {(int(tile.row), int(tile.col)): tile for tile in scene.tiles}


def _tile_by_id(scene: RpgTacticalMapScene) -> dict[str, RpgTacticalTile]:
    return {str(tile.tile_id): tile for tile in scene.tiles}


def _converted_tiles_by_coord(
    scene: RpgTacticalMapScene,
    *,
    changed_tile_id: str,
) -> dict[tuple[int, int], RpgTacticalTile]:
    """Return a tile map where one tile has become road terrain."""

    converted: dict[tuple[int, int], RpgTacticalTile] = {}
    for tile in scene.tiles:
        if str(tile.tile_id) == str(changed_tile_id):
            tile = replace(tile, terrain=TERRAIN_ROAD, movement_cost=1, passable=True)
        converted[(int(tile.row), int(tile.col))] = tile
    return converted


def _path_entry_costs_after_conversion(
    *,
    scene: RpgTacticalMapScene,
    path_tile_ids: tuple[str, ...],
    changed_tile_id: str,
) -> tuple[int, ...]:
    tiles_by_id = _tile_by_id(scene)
    return tuple(
        0
        if index == 0
        else 1
        if str(tile_id) == str(changed_tile_id)
        else int(tiles_by_id[str(tile_id)].movement_cost or 0)
        for index, tile_id in enumerate(path_tile_ids)
    )


def _counterfactual_candidates_for_scene(
    *,
    scene: RpgTacticalMapScene,
    start_tile: RpgTacticalTile,
    original_movement_costs: Mapping[str, int],
    min_cost: int,
    max_cost: int,
) -> list[_CounterfactualCandidate]:
    """Return targets with one unique best terrain-to-road conversion."""

    tiles_by_id = _tile_by_id(scene)
    start_tile_id = str(start_tile.tile_id)
    converted_results_by_target: dict[str, list[_CounterfactualCandidate]] = {}
    for changed_tile in scene.tiles:
        changed_tile_id = str(changed_tile.tile_id)
        if changed_tile_id == start_tile_id:
            continue
        if str(changed_tile.terrain) not in set(CONVERTIBLE_TERRAINS):
            continue
        converted_tiles = _converted_tiles_by_coord(scene, changed_tile_id=changed_tile_id)
        movement_costs, movement_paths = shortest_movement_costs_and_paths(
            converted_tiles,
            start_coord=start_tile.coord,
        )
        for target_tile_id, answer_value in movement_costs.items():
            target_tile_id = str(target_tile_id)
            if target_tile_id in {start_tile_id, changed_tile_id}:
                continue
            target_tile = tiles_by_id[str(target_tile_id)]
            if not bool(target_tile.passable):
                continue
            answer_value = int(answer_value)
            if not (int(min_cost) <= int(answer_value) <= int(max_cost)):
                continue
            path_tile_ids = tuple(str(tile_id) for tile_id in movement_paths[target_tile_id])
            if changed_tile_id not in set(path_tile_ids):
                continue
            original_target_cost = (
                int(original_movement_costs[target_tile_id])
                if target_tile_id in original_movement_costs
                else None
            )
            if original_target_cost is not None and int(original_target_cost) <= int(answer_value):
                continue
            path_terrains = tuple(str(tiles_by_id[str(tile_id)].terrain) for tile_id in path_tile_ids)
            path_entry_costs = _path_entry_costs_after_conversion(
                scene=scene,
                path_tile_ids=path_tile_ids,
                changed_tile_id=changed_tile_id,
            )
            converted_results_by_target.setdefault(target_tile_id, []).append(
                _CounterfactualCandidate(
                    target_tile_id=target_tile_id,
                    changed_tile_id=changed_tile_id,
                    changed_original_terrain=str(changed_tile.terrain),
                    answer_value=int(answer_value),
                    original_target_cost=original_target_cost,
                    shortest_path_tile_ids=path_tile_ids,
                    shortest_path_terrains=path_terrains,
                    shortest_path_entry_costs_after_conversion=path_entry_costs,
                    counterfactual_movement_costs_by_tile_id={
                        str(tile_id): int(cost)
                        for tile_id, cost in movement_costs.items()
                    },
                    improvement=(
                        None
                        if original_target_cost is None
                        else int(original_target_cost) - int(answer_value)
                    ),
                )
            )

    unique_candidates: list[_CounterfactualCandidate] = []
    for converted_results in converted_results_by_target.values():
        best_cost = min(int(candidate.answer_value) for candidate in converted_results)
        best_candidates = [
            candidate
            for candidate in converted_results
            if int(candidate.answer_value) == int(best_cost)
        ]
        if len(best_candidates) == 1:
            unique_candidates.append(best_candidates[0])
    return unique_candidates


def _select_counterfactual_conversion_attempt(
    *,
    scene: RpgTacticalMapScene,
    min_cost: int,
    max_cost: int,
    instance_seed: int,
) -> RpgTacticalMapValueAttempt:
    """Select one target where one unique terrain-to-road conversion is optimal."""

    if not scene.units:
        raise ValueError("counterfactual terrain conversion task requires a blue unit")
    if int(min_cost) < 1 or int(max_cost) < int(min_cost):
        raise ValueError("counterfactual answer range must satisfy 1 <= min <= max")
    tiles_by_id = _tile_by_id(scene)
    start_tile_id = str(scene.units[0].tile_id)
    start_tile = tiles_by_id[start_tile_id]
    original_movement_costs, _original_paths = shortest_movement_costs_and_paths(
        _tiles_by_coord(scene),
        start_coord=start_tile.coord,
    )
    candidates = _counterfactual_candidates_for_scene(
        scene=scene,
        start_tile=start_tile,
        original_movement_costs=original_movement_costs,
        min_cost=int(min_cost),
        max_cost=int(max_cost),
    )
    if not candidates:
        raise ValueError("no unique one-tile terrain conversion found in requested answer range")

    rng = random.Random(f"{int(instance_seed)}:counterfactual_terrain_conversion:{int(min_cost)}:{int(max_cost)}")
    jitter = {str(candidate.changed_tile_id): float(rng.random()) for candidate in candidates}
    terrain_priority = {TERRAIN_WATER: 0, TERRAIN_MOUNTAIN: 1, TERRAIN_FOREST: 2}

    def candidate_sort_key(candidate: _CounterfactualCandidate) -> tuple[int, int, int, float]:
        unreachable_bonus = 0 if candidate.original_target_cost is None else 1
        improvement = 99 if candidate.improvement is None else int(candidate.improvement)
        return (
            terrain_priority.get(str(candidate.changed_original_terrain), 9),
            unreachable_bonus,
            -int(improvement),
            jitter[str(candidate.changed_tile_id)],
        )

    selected = sorted(candidates, key=candidate_sort_key)[0]
    relation_fields = {
        "operation": "shortest_movement_cost_after_unique_one_tile_to_road_conversion",
        "terrain_movement_costs": dict(TERRAIN_MOVEMENT_COSTS),
        "blocked_terrain": [TERRAIN_WATER],
        "conversion_rule": "change_exactly_one_non_road_tile_to_road_before_moving",
        "convertible_terrains": list(CONVERTIBLE_TERRAINS),
        "counterfactual_terrain": TERRAIN_ROAD,
        "start_tile_id": str(start_tile_id),
        "target_tile_id": str(selected.target_tile_id),
        "changed_tile_id": str(selected.changed_tile_id),
        "changed_tile_original_terrain": str(selected.changed_original_terrain),
        "original_target_cost": selected.original_target_cost,
        "counterfactual_shortest_movement_cost": int(selected.answer_value),
        "shortest_path_tile_ids": list(selected.shortest_path_tile_ids),
        "shortest_path_terrains_original": list(selected.shortest_path_terrains),
        "shortest_path_entry_costs_after_conversion": list(selected.shortest_path_entry_costs_after_conversion),
        "answer_range": [int(min_cost), int(max_cost)],
        "unique_best_changed_tile": True,
    }
    execution_fields = {
        **relation_fields,
        "original_movement_costs_by_tile_id": {
            str(tile_id): int(cost)
            for tile_id, cost in original_movement_costs.items()
        },
        "counterfactual_movement_costs_by_tile_id": dict(selected.counterfactual_movement_costs_by_tile_id),
    }
    return RpgTacticalMapValueAttempt(
        target_tile_id=str(selected.target_tile_id),
        answer_value=int(selected.answer_value),
        annotation_tile_id_map={
            "player_cell": str(start_tile_id),
            "target_cell": str(selected.target_tile_id),
            "changed_cell": str(selected.changed_tile_id),
        },
        relation_fields=relation_fields,
        execution_fields=execution_fields,
        witness_fields={
            "changed_tile_id": str(selected.changed_tile_id),
            "changed_tile_original_terrain": str(selected.changed_original_terrain),
            "changed_tile_counterfactual_terrain": TERRAIN_ROAD,
            "original_target_cost": selected.original_target_cost,
            "counterfactual_shortest_movement_cost": int(selected.answer_value),
            "shortest_path_tile_ids": list(selected.shortest_path_tile_ids),
            "shortest_path_entry_costs_after_conversion": list(selected.shortest_path_entry_costs_after_conversion),
            "unique_best_changed_tile": True,
        },
    )


def _build_counterfactual_conversion_plan(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    query_id: str,
) -> RpgTacticalMapValuePlan:
    """Bind answer range axes for the terrain-conversion value task."""

    min_cost = int(
        task_params.get(
            "min_counterfactual_cost",
            group_default(_GEN_DEFAULTS, "min_counterfactual_cost", DEFAULT_MIN_COUNTERFACTUAL_COST),
        )
    )
    max_cost = int(
        task_params.get(
            "max_counterfactual_cost",
            group_default(_GEN_DEFAULTS, "max_counterfactual_cost", DEFAULT_MAX_COUNTERFACTUAL_COST),
        )
    )

    def build_attempt(scene: RpgTacticalMapScene, attempt_seed: int) -> RpgTacticalMapValueAttempt:
        return _select_counterfactual_conversion_attempt(
            scene=scene,
            min_cost=int(min_cost),
            max_cost=int(max_cost),
            instance_seed=int(attempt_seed),
        )

    def build_render_map(scene: RpgTacticalMapScene, attempt: RpgTacticalMapValueAttempt) -> Mapping[str, Any]:
        execution = attempt.execution_fields
        return counterfactual_terrain_conversion_cost_render_map(
            scene=scene,
            target_tile_id=str(attempt.target_tile_id),
            changed_tile_id=str(execution["changed_tile_id"]),
            shortest_path_tile_ids=execution["shortest_path_tile_ids"],
            counterfactual_movement_costs_by_tile_id=execution["counterfactual_movement_costs_by_tile_id"],
            start_tile_id=str(execution["start_tile_id"]),
            original_target_cost=execution["original_target_cost"],
            answer_value=int(attempt.answer_value),
        )

    return RpgTacticalMapValuePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={
            "terrain_rules": rpg_tactical_map_terrain_rules_text(),
            "conversion_rule": "Before moving, exactly one water, mountain, or forest tile may be changed into a road tile.",
        },
        answer_hint_key="answer_hint_counterfactual_terrain_conversion_cost_value",
        annotation_hint_key="annotation_hint_counterfactual_terrain_conversion_cost_value",
        json_example_key="json_example_counterfactual_terrain_conversion_cost_value",
        json_example_answer_only_key="json_example_answer_only_counterfactual_terrain_conversion_cost_value",
        query_params={
            "query_id": str(query_id),
            "query_id_probabilities": dict(query_probabilities),
            "min_counterfactual_cost": int(min_cost),
            "max_counterfactual_cost": int(max_cost),
            "convertible_terrains": list(CONVERTIBLE_TERRAINS),
            "counterfactual_terrain": TERRAIN_ROAD,
        },
        build_attempt=build_attempt,
        build_render_map=build_render_map,
        failure_label="counterfactual terrain conversion cost value target",
    )


@register_task
class IllustrationsRpgTacticalMapCounterfactualTerrainConversionCostValueTask:
    """Return shortest movement cost after the best one-tile road conversion."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology', 'state_update')
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
            prepare_plan=_build_counterfactual_conversion_plan,
        )


__all__ = [
    "CONVERTIBLE_TERRAINS",
    "IllustrationsRpgTacticalMapCounterfactualTerrainConversionCostValueTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
