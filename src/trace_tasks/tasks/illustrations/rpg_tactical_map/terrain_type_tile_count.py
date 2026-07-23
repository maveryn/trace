"""Count one special terrain type in a tactical RPG map."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import RpgTacticalMapTileCountPlan, run_rpg_tactical_map_tile_count_lifecycle
from .shared.output import terrain_type_count_render_map
from .shared.relations import TERRAIN_FOREST, TERRAIN_MOUNTAIN, TERRAIN_WATER
from .shared.rendering import SCENE_ID
from .shared.sampling import select_string_from_support
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__terrain_type_tile_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "terrain_type_tile_count"
DEFAULT_TARGET_TERRAIN_SUPPORT: tuple[str, ...] = (TERRAIN_FOREST, TERRAIN_MOUNTAIN, TERRAIN_WATER)
DEFAULT_MIN_ANSWER_COUNT = 1
DEFAULT_MAX_ANSWER_COUNT = 18

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _tiles_matching(scene: RpgTacticalMapScene, terrain_name: str) -> list[RpgTacticalTile]:
    matches = [tile for tile in scene.tiles if str(tile.terrain) == str(terrain_name)]
    return sorted(matches, key=lambda tile: (int(tile.row), int(tile.col), str(tile.tile_id)))


def _build_terrain_count_plan(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    _query_id: str,
) -> RpgTacticalMapTileCountPlan:
    """Bind the special-terrain count target and annotation witnesses."""

    target_terrain, target_probabilities = select_string_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        support_key="target_terrain_support",
        explicit_key="target_terrain",
        fallback_support=DEFAULT_TARGET_TERRAIN_SUPPORT,
        namespace=f"{TASK_ID}:target_terrain",
    )
    if target_terrain not in DEFAULT_TARGET_TERRAIN_SUPPORT:
        raise ValueError(f"target_terrain must be one of {DEFAULT_TARGET_TERRAIN_SUPPORT}")
    min_count = int(task_params.get("min_answer_count", group_default(_GEN_DEFAULTS, "min_answer_count", DEFAULT_MIN_ANSWER_COUNT)))
    max_count = int(task_params.get("max_answer_count", group_default(_GEN_DEFAULTS, "max_answer_count", DEFAULT_MAX_ANSWER_COUNT)))

    def select_tiles(scene: RpgTacticalMapScene) -> list[RpgTacticalTile]:
        return _tiles_matching(scene, str(target_terrain))

    def render_map(scene: RpgTacticalMapScene, tile_ids: tuple[str, ...] | list[str]) -> Mapping[str, Any]:
        return terrain_type_count_render_map(
            scene=scene,
            target_terrain=str(target_terrain),
            counted_tile_ids=tuple(str(tile_id) for tile_id in tile_ids),
        )

    return RpgTacticalMapTileCountPlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={"terrain_label": str(target_terrain)},
        answer_hint_key="answer_hint_terrain_type_tile_count",
        annotation_hint_key="annotation_hint_terrain_type_tile_count",
        json_example_key="json_example_terrain_type_tile_count",
        json_example_answer_only_key="json_example_answer_only_terrain_type_tile_count",
        min_answer_count=int(min_count),
        max_answer_count=int(max_count),
        query_params={
            "target_terrain": str(target_terrain),
            "target_terrain_probabilities": dict(target_probabilities),
        },
        relation_fields={
            "operation": "count_tiles_matching_terrain_type",
            "target_terrain": str(target_terrain),
        },
        execution_fields={
            "target_terrain": str(target_terrain),
        },
        witness_fields={"target_terrain": str(target_terrain)},
        select_tiles=select_tiles,
        build_render_map=render_map,
        failure_label=f"{target_terrain} tile count",
    )


@register_task
class IllustrationsRpgTacticalMapTerrainTypeTileCountTask:
    """Count visible forest, mountain, or water tiles."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_rpg_tactical_map_tile_count_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_plan=_build_terrain_count_plan,
        )


__all__ = [
    "IllustrationsRpgTacticalMapTerrainTypeTileCountTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
