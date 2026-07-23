from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveReversiPlan, run_reversi_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import resolve_current_player, resolve_reversi_target_axis, sample_marked_move_flip_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_games__reversi__marked_move_flip_count"
PROMPT_QUERY_KEY = "flip_count_for_marked_move"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_marked_move_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    query_id: str,
) -> ObjectiveReversiPlan:
    if str(query_id) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported Reversi marked-move query: {query_id}")
    target_axis = resolve_reversi_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="flip_count_support",
        fallback_support=DEFAULTS.flip_count_support,
        namespace=f"{SCENE_NAMESPACE}.marked_move_flip.target_answer",
        use_instance_seed_cycle=True,
    )
    return ObjectiveReversiPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.marked_move_flip",
        prompt_query_key=PROMPT_QUERY_KEY,
        target_axis=target_axis,
        annotation_kind="disc_point_set",
        query_params={"move_role": "marked_legal_move"},
        construct_attempt=lambda rng, axes: sample_marked_move_flip_scene(
            rng=rng,
            board_size=int(axes.board_size),
            current_player=resolve_current_player(rng, params=params),
            target_answer=int(target_axis.target_answer),
        ),
    )


@register_task
class GamesReversiMarkedMoveFlipCountTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_reversi_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_marked_move_objective,
        )


__all__ = ["GamesReversiMarkedMoveFlipCountTask", "TASK_ID"]
