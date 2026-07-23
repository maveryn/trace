"""Count people placed directly on path tiles in a top-down pixel village."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ._lifecycle import PixelVillageCountPlan, run_pixel_village_count_lifecycle
from .shared.output import bind_path_result
from .shared.sampling import SCENE_ID


TASK_ID = "task_illustrations__pixel_village__person_path_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "people_on_path_count"
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _build_plan() -> PixelVillageCountPlan:
    """Build the public-owned person-on-path count objective plan."""

    return PixelVillageCountPlan(
        public_id=TASK_ID,
        prompt_query_key=PROMPT_QUERY_KEY,
        bind_result=bind_path_result,
        supported_query_ids=SUPPORTED_QUERY_IDS,
    )


@register_task
class IllustrationsPixelVillagePersonPathCountTask:
    """Count people whose tile footprint intersects a visible path tile."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_pixel_village_count_lifecycle(
            task=self,
            plan=_build_plan(),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["IllustrationsPixelVillagePersonPathCountTask", "SUPPORTED_QUERY_IDS", "TASK_ID", "_build_plan"]
