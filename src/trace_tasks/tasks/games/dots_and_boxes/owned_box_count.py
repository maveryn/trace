"""Count completed dots-and-boxes cells owned by a player marker."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from ._lifecycle import (
    DotsAndBoxesObjectivePlan,
    make_count_objective_plan,
    run_dots_and_boxes_lifecycle,
)
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.sampling import resolve_dots_and_boxes_target_axis
from .shared.state import DotsAndBoxesBoardShapeAxis

TASK_ID = "task_games__dots_and_boxes__owned_box_count"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "owned_box_count"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _prepare_owned_box_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    query_id: str,
    board_shape: DotsAndBoxesBoardShapeAxis,
) -> DotsAndBoxesObjectivePlan:
    """Bind the selected player query to a neutral owned-box count."""

    del query_probabilities
    selected_query = str(query_id)
    if selected_query != QUERY_ID:
        raise ValueError(
            f"unsupported dots-and-boxes owned-box query: {selected_query}"
        )
    owner, owner_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{TASK_ID}.target_owner",
        explicit_key="target_owner",
        weights_key="target_owner_weights",
        balance_flag_key="balanced_target_owner_sampling",
        supported_variants=("A", "B"),
    )
    target_axis = resolve_dots_and_boxes_target_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="owned_box_count_support",
        fallback_support=DEFAULTS.owned_box_count_support,
        namespace="owned_box_count.target_answer",
    )
    return make_count_objective_plan(
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_example_shape="bbox_set",
        annotation_kind="box",
        annotation_entity_attr="counted_box_ids",
        target_axis=target_axis,
        board_shape=board_shape,
        count_mode="owned_box",
        owner=str(owner),
        query_params_extra={
            "target_owner": str(owner),
            "target_owner_probabilities": dict(owner_probabilities),
        },
        prompt_dynamic_slots={"target_owner": str(owner)},
        attempt_namespace=f"games.dots_and_boxes.owned_box_count.{str(owner)}",
    )


@register_task
class GamesDotsAndBoxesOwnedBoxCountTask:
    """Count boxes claimed by player A or B."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        return run_dots_and_boxes_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_owned_box_objective,
        )


__all__ = ["GamesDotsAndBoxesOwnedBoxCountTask"]
