"""Count monster-occupied treasure chambers in an RPG dungeon scene."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    CountWitnesses,
    count_task_defaults,
    make_count_task_config,
    run_count_task,
)
from .shared.output import (
    monster_entities,
    rpg_dungeon_monster_chamber_count_render_map,
)
from .shared.rendering import (
    MAX_MONSTER_CHAMBER_COUNT,
)


TASK_ID = "task_illustrations__rpg_dungeon__monster_chamber_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = count_task_defaults(TASK_ID)
_CONFIG = make_count_task_config(
    task_identifier=TASK_ID,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    prompt_query_key="monster_chamber_count",
    operation="count_chambers_containing_monsters",
    target_support_key="monster_chamber_count_support",
    target_explicit_key="monster_chamber_count",
    target_fallback_support=tuple(range(1, MAX_MONSTER_CHAMBER_COUNT + 1)),
    target_namespace_suffix="monster_chamber_count",
)


def _build_scene_kwargs(target, _instance_seed):
    return {
        "total_chest_count": int(target.total_value),
        "reachable_chest_count": int(target.total_value),
        "monster_chamber_count": int(target.target_value),
    }


def _bind_witnesses(scene, target, _scene_kwargs):
    monsters = monster_entities(scene)
    chamber_ids = [str(entity.chamber_id) for entity in monsters]
    if len(set(chamber_ids)) != len(monsters):
        raise ValueError(f"monster chamber witnesses are not unique: {chamber_ids}")
    if len(monsters) != int(target.target_value):
        raise ValueError(f"renderer produced {len(monsters)} monster chambers for target {target.target_value}")
    object_types = [str(entity.object_type) for entity in monsters]
    if any(not object_type.startswith("monster_") for object_type in object_types):
        raise ValueError(f"unexpected non-monster witness: {object_types}")
    fields = {
        "monster_chamber_count": int(len(monsters)),
        "monster_entity_ids": [str(entity.entity_id) for entity in monsters],
        "monster_chamber_ids": chamber_ids,
        "monster_object_types": object_types,
        "total_chest_count": int(len(scene.chest_entity_ids)),
    }
    return CountWitnesses(
        answer=int(len(monsters)),
        annotation_bboxes=[[round(float(value), 3) for value in entity.bbox_xyxy] for entity in monsters],
        relations={**fields, "answer": int(len(monsters))},
        execution_fields={"answer": int(len(monsters)), **fields},
        witness_fields=fields,
    )


@register_task
class IllustrationsRpgDungeonMonsterChamberCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
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
            build_render_map=rpg_dungeon_monster_chamber_count_render_map,
        )
