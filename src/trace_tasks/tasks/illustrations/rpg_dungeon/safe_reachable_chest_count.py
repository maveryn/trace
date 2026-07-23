"""Count reachable non-monster-chamber chests in an RPG dungeon scene."""

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
    monster_entities,
    player_entity,
    rpg_dungeon_safe_reachable_chest_count_render_map,
    safe_reachable_chest_ids,
)
from .shared.rendering import (
    MAX_MONSTER_CHAMBER_COUNT,
    MAX_TOTAL_CHEST_COUNT,
)
from .shared.sampling import choose_hazard_render_counts
from .shared.state import RpgDungeonScene


TASK_ID = "task_illustrations__rpg_dungeon__safe_reachable_chest_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = count_task_defaults(TASK_ID)
_CONFIG = make_count_task_config(
    task_identifier=TASK_ID,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    prompt_query_key="safe_reachable_chest_count",
    operation="count_reachable_chests_outside_monster_chambers",
    target_support_key="safe_reachable_chest_count_support",
    target_explicit_key="safe_reachable_chest_count",
    target_fallback_support=tuple(range(0, int(MAX_TOTAL_CHEST_COUNT))),
    target_namespace_suffix="safe_reachable_chest_count",
    target_max_delta=-1,
)


def _build_scene_kwargs(target: CountTarget, instance_seed: int):
    hazards = choose_hazard_render_counts(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:hazard_counts:{target.query_id}",
        total_chambers=int(target.total_value),
        target_safe_total=int(target.target_value),
        max_hazard_total=MAX_MONSTER_CHAMBER_COUNT,
    )
    return {
        "total_chest_count": int(target.total_value),
        "reachable_chest_count": int(hazards.reachable_total),
        "monster_chamber_count": int(hazards.hazard_total),
        "reachable_monster_chamber_count": int(hazards.reachable_hazard_total),
    }


def _bind_witnesses(
    scene: RpgDungeonScene,
    target: CountTarget,
    _scene_kwargs,
) -> CountWitnesses:
    """Bind reachable chest witnesses after excluding monster chambers."""

    if player_entity(scene) is None:
        raise ValueError("safe reachable chest task requires a player entity")
    counted_ids = safe_reachable_chest_ids(scene)
    if len(counted_ids) != int(target.target_value):
        raise ValueError(f"renderer produced {len(counted_ids)} safe reachable chests for target {target.target_value}")
    counted_bboxes = entity_bbox_map(scene, counted_ids)
    fields = {
        "safe_reachable_chest_count": int(len(counted_ids)),
        "player_entity_id": str(scene.player_entity_id),
        "reachable_chest_ids": [str(entity_id) for entity_id in scene.reachable_chest_ids],
        "monster_chamber_ids": [str(entity.chamber_id) for entity in monster_entities(scene)],
        "counted_chest_ids": [str(entity_id) for entity_id in counted_ids],
    }
    return CountWitnesses(
        answer=int(len(counted_ids)),
        annotation_bboxes=[counted_bboxes[str(entity_id)] for entity_id in counted_ids],
        relations={**fields, "answer": int(len(counted_ids))},
        execution_fields={
            "answer": int(len(counted_ids)),
            "safe_reachable_chest_count": int(len(counted_ids)),
            "reachable_chest_ids": list(fields["reachable_chest_ids"]),
            "monster_chamber_ids": list(fields["monster_chamber_ids"]),
            "counted_chest_ids": list(fields["counted_chest_ids"]),
        },
        witness_fields=fields,
    )


@register_task
class IllustrationsRpgDungeonSafeReachableChestCountTask:
    """Count reachable treasure chests outside monster chambers."""

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
            build_render_map=rpg_dungeon_safe_reachable_chest_count_render_map,
        )


__all__ = [
    "IllustrationsRpgDungeonSafeReachableChestCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
