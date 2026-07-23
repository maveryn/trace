"""Count dots-and-boxes cells that currently have three drawn sides."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import DotsAndBoxesObjectivePlan, make_count_objective_plan, run_dots_and_boxes_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.sampling import resolve_dots_and_boxes_target_axis
from .shared.state import DotsAndBoxesBoardShapeAxis


TASK_ID = "task_games__dots_and_boxes__three_sided_box_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "three_sided_box_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_three_sided_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    query_id: str,
    board_shape: DotsAndBoxesBoardShapeAxis,
) -> DotsAndBoxesObjectivePlan:
    """Bind the single-query task to the three-drawn-sides box count."""

    del query_probabilities, query_id
    target_axis = resolve_dots_and_boxes_target_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="three_sided_box_count_support",
        fallback_support=DEFAULTS.three_sided_box_count_support,
        namespace=f"{PROMPT_QUERY_KEY}.target_answer",
    )
    return make_count_objective_plan(
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_example_shape="bbox_set",
        annotation_kind="box",
        annotation_entity_attr="counted_box_ids",
        target_axis=target_axis,
        board_shape=board_shape,
        count_mode="three_sided_box",
        query_params_extra={"prompt_query_key": PROMPT_QUERY_KEY},
        attempt_namespace=f"games.dots_and_boxes.{PROMPT_QUERY_KEY}",
    )


@register_task
class GamesDotsAndBoxesThreeSidedBoxCountTask:
    """Count boxes that are one missing edge away from completion."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_dots_and_boxes_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_three_sided_objective,
        )


__all__ = ["GamesDotsAndBoxesThreeSidedBoxCountTask"]
