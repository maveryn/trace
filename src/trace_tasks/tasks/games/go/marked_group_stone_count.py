"""Count stones in the connected group containing one marked Go stone."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import prepare_go_marked_group_count_objective, run_go_lifecycle
from .shared.rules import GO_RULE_MARKED_GROUP_STONES
from .shared.state import DEFAULTS, GoIntegerAxis, GoSceneAxes, SCENE_ID


TASK_ID = "task_games__go__marked_group_stone_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "marked_group_stone_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_stone_group_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
    scene_axes: GoSceneAxes,
    board_size_axis: GoIntegerAxis,
):
    """Bind marked-stone group-size counting to a Go board construction."""

    del query_id, _query_probabilities
    return prepare_go_marked_group_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        prompt_query_key=PROMPT_QUERY_KEY,
        rule_mode=GO_RULE_MARKED_GROUP_STONES,
        support_key="marked_group_stone_count_support",
        fallback_support=DEFAULTS.marked_group_stone_count_support,
        target_namespace=f"{PROMPT_QUERY_KEY}.target_answer",
        board_size_axis=board_size_axis,
        scene_axes=scene_axes,
        annotation_coord_attr="marked_group_coords",
        annotation_kind="stone_bbox_set",
        attempt_namespace=f"games.go.{PROMPT_QUERY_KEY}",
        mark_reference_stone_only=True,
        prompt_example_annotation_points=((258, 198, 302, 242), (331, 198, 375, 242), (331, 271, 375, 315), (404, 271, 448, 315)),
        prompt_example_answer=4,
    )


@register_task
class GamesGoMarkedGroupStoneCountTask:
    """Count stones in the same-color group containing the marked stone."""

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
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_stone_group_objective,
        )


__all__ = ["GamesGoMarkedGroupStoneCountTask"]
