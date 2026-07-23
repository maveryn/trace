"""Count legal Reversi destination cells for the current player."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import ObjectiveReversiPlan, run_reversi_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import (
    resolve_current_player,
    resolve_reversi_target_axis,
    sample_legal_destination_scene,
)
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_games__reversi__legal_destination_count"
SUPPORTED_QUERY_IDS = ("single",)
DEFAULT_QUERY_ID = "single"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_legal_destination_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    query_id: str,
) -> ObjectiveReversiPlan:
    """Prepare target support and sampler hooks for legal-destination counts."""

    if str(query_id) == "single":
        target_axis = resolve_reversi_target_axis(
            int(instance_seed),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            support_key="legal_move_count_support",
            fallback_support=DEFAULTS.legal_move_count_support,
            namespace=f"{SCENE_NAMESPACE}.legal_destination.target_answer",
        )
        return ObjectiveReversiPlan(
            attempt_namespace=f"{SCENE_NAMESPACE}.legal_destination",
            prompt_query_key="legal_move_count",
            target_axis=target_axis,
            annotation_kind="cell_bbox_set",
            query_params={"destination_filter": "all_legal_moves"},
            construct_attempt=lambda rng, axes: sample_legal_destination_scene(
                rng=rng,
                board_size=int(axes.board_size),
                current_player=resolve_current_player(rng, params=params),
                target_answer=int(target_axis.target_answer),
            ),
        )
    raise ValueError(f"unsupported Reversi legal-destination query: {query_id}")


@register_task
class GamesReversiLegalDestinationCountTask:
    """Count all legal destination squares for the current player."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_reversi_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_legal_destination_objective,
        )


__all__ = ["GamesReversiLegalDestinationCountTask", "TASK_ID"]
