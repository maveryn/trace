"""Count target objects strictly on one side of the pixel-village river."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ._lifecycle import PixelVillageCountPlan, run_pixel_village_count_lifecycle
from .shared.output import bind_river_side_result
from .shared.sampling import SCENE_ID


TASK_ID = "task_illustrations__pixel_village__river_side_object_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "river_side_object_count"
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _build_plan() -> PixelVillageCountPlan:
    """Build the public-owned strict river-side count objective plan."""

    return PixelVillageCountPlan(
        public_id=TASK_ID,
        prompt_query_key=PROMPT_QUERY_KEY,
        bind_result=bind_river_side_result,
        supported_query_ids=SUPPORTED_QUERY_IDS,
    )


@register_task
class IllustrationsPixelVillageRiverSideObjectCountTask:
    """Count target objects strictly on one side of the pixel-village river."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'spatial_relations')
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


__all__ = ["IllustrationsPixelVillageRiverSideObjectCountTask", "SUPPORTED_QUERY_IDS", "TASK_ID", "_build_plan"]
