"""Count reachable treasure chests in an RPG dungeon scene."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    CountTarget,
    CountWitnesses,
    count_task_defaults,
    make_count_task_config,
    run_count_task,
)
from .shared.output import (
    entity_bbox_map,
    player_entity,
    rpg_dungeon_reachable_chest_count_render_map,
)
from .shared.rendering import (
    MAX_REACHABLE_CHEST_COUNT,
    MIN_REACHABLE_CHEST_COUNT,
)
from .shared.state import RpgDungeonScene


TASK_ID = "task_illustrations__rpg_dungeon__reachable_chest_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = count_task_defaults(TASK_ID)
_CONFIG = make_count_task_config(
    task_identifier=TASK_ID,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    prompt_query_key="reachable_chest_count",
    operation="count_reachable_chests_from_player",
    target_support_key="reachable_chest_count_support",
    target_explicit_key="reachable_chest_count",
    target_fallback_support=tuple(range(MIN_REACHABLE_CHEST_COUNT, MAX_REACHABLE_CHEST_COUNT + 1)),
    target_namespace_suffix="reachable_chest_count",
)


def _build_scene_kwargs(target: CountTarget, _instance_seed: int):
    return {
        "total_chest_count": int(target.total_value),
        "reachable_chest_count": int(target.target_value),
    }


def _bind_witnesses(
    scene: RpgDungeonScene,
    target: CountTarget,
    _scene_kwargs,
) -> CountWitnesses:
    """Bind counted reachable chests from the rendered scene."""

    answer = int(len(scene.reachable_chest_ids))
    if answer != int(target.target_value):
        raise ValueError(f"renderer produced {answer} reachable chests for target {target.target_value}")
    if player_entity(scene) is None:
        raise ValueError("RPG dungeon reachable-chest task requires a player entity")
    reachable_bboxes = entity_bbox_map(scene, scene.reachable_chest_ids)
    fields = {
        "reachable_chest_count": int(answer),
        "total_chest_count": int(len(scene.chest_entity_ids)),
        "player_entity_id": str(scene.player_entity_id),
        "reachable_chest_ids": [str(entity_id) for entity_id in scene.reachable_chest_ids],
    }
    return CountWitnesses(
        answer=answer,
        annotation_bboxes=[reachable_bboxes[str(entity_id)] for entity_id in scene.reachable_chest_ids],
        relations={
            **fields,
            "chest_entity_ids": [str(entity_id) for entity_id in scene.chest_entity_ids],
            "answer": int(answer),
        },
        execution_fields={"answer": int(answer), **fields},
        witness_fields=fields,
    )


@register_task
class IllustrationsRpgDungeonReachableChestCountTask:
    """Count chests reachable from the player through unblocked dungeon floor."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params, max_attempts):
        return run_count_task(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            config=_CONFIG,
            build_scene_kwargs=_build_scene_kwargs,
            bind_scene=_bind_witnesses,
            build_render_map=rpg_dungeon_reachable_chest_count_render_map,
        )


__all__ = [
    "IllustrationsRpgDungeonReachableChestCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
