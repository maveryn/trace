"""Count opponent stones adjacent to a marked Go group."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import GoObjectivePlan, prepare_go_marked_group_count_objective, run_go_lifecycle
from .shared.rules import GO_RULE_ADJACENT_ENEMY_STONES
from .shared.state import DEFAULTS, GoIntegerAxis, GoSceneAxes, SCENE_ID


TASK_ID = "task_games__go__group_adjacent_enemy_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "marked_group_adjacent_enemy_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_adjacent_enemy_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _query_id: str,
    _query_probabilities: Mapping[str, float],
    scene_axes: GoSceneAxes,
    board_size_axis: GoIntegerAxis,
) -> GoObjectivePlan:
    """Bind the adjacent-enemy count to a marked Go group construction."""

    return prepare_go_marked_group_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        prompt_query_key=PROMPT_QUERY_KEY,
        rule_mode=GO_RULE_ADJACENT_ENEMY_STONES,
        support_key="adjacent_enemy_count_support",
        fallback_support=DEFAULTS.adjacent_enemy_count_support,
        target_namespace=f"{PROMPT_QUERY_KEY}.target_answer",
        board_size_axis=board_size_axis,
        scene_axes=scene_axes,
        annotation_coord_attr="adjacent_enemy_coords",
        annotation_kind="stone_bbox_set",
        attempt_namespace=f"games.go.{PROMPT_QUERY_KEY}",
        prompt_example_annotation_points=((258, 198, 302, 242), (331, 198, 375, 242), (404, 271, 448, 315)),
        prompt_example_answer=3,
    )


@register_task
class GamesGoGroupAdjacentEnemyCountTask:
    """Count opponent stones touching the marked group orthogonally."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'spatial_relations', 'topology')
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
            prepare_objective=_prepare_adjacent_enemy_objective,
        )


__all__ = ["GamesGoGroupAdjacentEnemyCountTask"]
