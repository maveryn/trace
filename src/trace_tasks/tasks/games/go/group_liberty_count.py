"""Count liberties or shared liberties of a marked Go group."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import GoObjectivePlan, prepare_go_marked_group_count_objective, run_go_lifecycle
from .shared.rules import GO_RULE_GROUP_LIBERTIES, GO_RULE_SHARED_LIBERTIES
from .shared.state import DEFAULTS, GoIntegerAxis, GoSceneAxes, SCENE_ID


TASK_ID = "task_games__go__group_liberty_count"
SUPPORTED_QUERY_IDS = ("marked_group_liberty_count", "marked_group_shared_liberty_count")
TARGET_SUPPORT_KEY_BY_QUERY_ID = {
    "marked_group_liberty_count": "liberty_count_support",
    "marked_group_shared_liberty_count": "shared_liberty_count_support",
}
FALLBACK_SUPPORT_BY_QUERY_ID = {
    "marked_group_liberty_count": DEFAULTS.liberty_count_support,
    "marked_group_shared_liberty_count": DEFAULTS.shared_liberty_count_support,
}
RULE_MODE_BY_QUERY_ID = {
    "marked_group_liberty_count": GO_RULE_GROUP_LIBERTIES,
    "marked_group_shared_liberty_count": GO_RULE_SHARED_LIBERTIES,
}
ANNOTATION_COORD_ATTR_BY_QUERY_ID = {
    "marked_group_liberty_count": "liberty_coords",
    "marked_group_shared_liberty_count": "shared_liberty_coords",
}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_liberty_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
    scene_axes: GoSceneAxes,
    board_size_axis: GoIntegerAxis,
) -> GoObjectivePlan:
    """Bind the selected liberty query to a marked Go group construction."""

    selected_query = str(query_id)
    return prepare_go_marked_group_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        prompt_query_key=selected_query,
        rule_mode=str(RULE_MODE_BY_QUERY_ID[selected_query]),
        support_key=str(TARGET_SUPPORT_KEY_BY_QUERY_ID[selected_query]),
        fallback_support=FALLBACK_SUPPORT_BY_QUERY_ID[selected_query],
        target_namespace=f"{selected_query}.target_answer",
        board_size_axis=board_size_axis,
        scene_axes=scene_axes,
        annotation_coord_attr=str(ANNOTATION_COORD_ATTR_BY_QUERY_ID[selected_query]),
        attempt_namespace=f"games.go.{selected_query}",
    )


@register_task
class GamesGoGroupLibertyCountTask:
    """Count marked-group liberties or liberties shared with opponent stones."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_go_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_liberty_objective,
        )


__all__ = ["GamesGoGroupLibertyCountTask"]
