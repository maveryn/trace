"""Count rooms directly adjacent to the player's room in an RPG house layout."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .shared.adjacent_room_count import generate_adjacent_room_count_output
from .shared.rendering import SCENE_ID


TASK_ID = "task_illustrations__rpg_house__adjacent_room_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "adjacent_room_count"
_IDENTITY_KEYS = {"public": "task_id", "branch": "query_id"}
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@register_task
class IllustrationsRpgHouseAdjacentRoomCountTask:
    """Count rooms that directly share a doorway with the player's room."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return generate_adjacent_room_count_output(
            public_name=TASK_ID,
            domain_name=self.domain,
            branch_names=SUPPORTED_QUERY_IDS,
            prompt_query_key=PROMPT_QUERY_KEY,
            identity_key_names=_IDENTITY_KEYS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
        )


__all__ = [
    "IllustrationsRpgHouseAdjacentRoomCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
