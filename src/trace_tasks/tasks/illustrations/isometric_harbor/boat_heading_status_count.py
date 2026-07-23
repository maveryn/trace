"""Count harbor boats by shoreline-relative heading direction."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    HarborHeadingCountConfig,
    build_harbor_heading_count_plan,
    run_harbor_count_lifecycle,
)
from .shared.rendering import SCENE_ID


TASK_ID = "task_illustrations__isometric_harbor__boat_heading_status_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("toward_shoreline_boat_count", "away_from_shoreline_boat_count")
QUERY_TO_HEADING_STATUS: Mapping[str, str] = {
    "toward_shoreline_boat_count": "toward_shoreline",
    "away_from_shoreline_boat_count": "away_from_shoreline",
}
HEADING_STATUS_LABELS: Mapping[str, str] = {
    "toward_shoreline": "toward the shoreline",
    "away_from_shoreline": "away from the shoreline",
}
_REQUIRED_PROMPT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_boat_heading_status_count",
    "annotation_hint_boat_heading_status_count",
    "json_example_boat_heading_status_count",
    "json_example_answer_only_boat_heading_status_count",
)


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _build_plan() -> Any:
    """Bind this public objective's query/status maps to the neutral heading-count lifecycle."""

    return build_harbor_heading_count_plan(
        HarborHeadingCountConfig(
            public_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            query_to_heading_status=QUERY_TO_HEADING_STATUS,
            heading_status_labels=HEADING_STATUS_LABELS,
            required_prompt_keys=_REQUIRED_PROMPT_KEYS,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
        )
    )


@register_task
class IllustrationsIsometricHarborBoatHeadingStatusCountTask:
    """Count boats facing toward or away from the shoreline."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_harbor_count_lifecycle(
            plan=_build_plan(),
            domain=self.domain,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "HEADING_STATUS_LABELS",
    "IllustrationsIsometricHarborBoatHeadingStatusCountTask",
    "QUERY_TO_HEADING_STATUS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
