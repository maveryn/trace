"""Find the longest capture chain for a marked Checkers king."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import (
    CheckersObjectivePlan,
    checkers_target_trace_params,
    resolve_checkers_task_target,
    run_checkers_lifecycle,
)
from .shared.rules import player_name
from .shared.sampling import sample_king_capture_chain_scene, scene_object_description
from .shared.state import SCENE_ID, SampledCheckersScene


TASK_ID = "task_games__checkers__max_capture_chain_length"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "max_capture_chain_length"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
MAX_CAPTURE_CHAIN_LENGTH_SUPPORT = (1, 2, 3, 4, 5)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_capture_chain_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _query_id: str,
    _query_probabilities: Mapping[str, float],
) -> CheckersObjectivePlan:
    """Bind the marked-king chain target and capture-chain sampler."""

    target = resolve_checkers_task_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="max_capture_chain_length_support",
        fallback_support=MAX_CAPTURE_CHAIN_LENGTH_SUPPORT,
        namespace=f"{TASK_ID}.target_answer",
    )

    def construct_attempt(rng, axes):
        return sample_king_capture_chain_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            target_answer=int(target.target_answer),
        )

    def prompt_slots(sample: SampledCheckersScene) -> dict[str, str]:
        return {
            "object_description": scene_object_description(str(sample.scene_variant)),
            "current_player_name": player_name(int(sample.current_player)),
        }

    def execution_fields(sample: SampledCheckersScene) -> dict[str, Any]:
        chain = sample.evaluation.selected_capture_chain
        return {
            "max_capture_chain_length": None if chain is None else int(len(chain.captured)),
        }

    return CheckersObjectivePlan(
        attempt_namespace="games.checkers.max_capture_chain_length",
        prompt_query_key=PROMPT_QUERY_KEY,
        target=target,
        query_params={
            **checkers_target_trace_params(target),
            "prompt_query_key": PROMPT_QUERY_KEY,
        },
        construct_attempt=construct_attempt,
        build_prompt_dynamic_slots=prompt_slots,
        build_execution_extra=execution_fields,
    )


@register_task
class GamesCheckersMaxCaptureChainLengthTask:
    """Find the maximum capture-chain length for the marked king checker."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a marked-king chain scene with captured-piece annotations."""

        return run_checkers_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_capture_chain_objective,
        )


__all__ = ["GamesCheckersMaxCaptureChainLengthTask"]
