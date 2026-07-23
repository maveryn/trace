"""Winning-payline count task for slot-machine games."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import build_fixed_query_objective_plan, build_slot_attempt_result, run_fixed_query_slot_machine_lifecycle
from .shared.defaults import SCENE_ID, SCENE_NAMESPACE, WINNING_PAYLINE_COUNT_SUPPORT
from .shared.rules import resolve_target_count, target_count_trace_params
from .shared.sampling import sample_slot_machine_grid
from .shared.state import SlotMachineAxes


TASK_ID = "task_games__slot_machine__winning_payline_count"
PROMPT_QUERY_KEY = "winning_payline_count"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_winning_payline_count_objective(
    instance_seed,
    task_params,
    _selected_query,
    query_probabilities,
    axes: SlotMachineAxes,
):
    """Bind the count target and prompt slots for winning paylines."""

    target = resolve_target_count(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="winning_payline_count_support",
        explicit_key="target_winning_payline_count",
        fallback_support=WINNING_PAYLINE_COUNT_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.winning_payline_count.target",
        balanced_flag_key="balanced_winning_payline_count_sampling",
    )
    target_params = target_count_trace_params(
        target=target,
        explicit_key="target_winning_payline_count",
        support_key="winning_payline_count_support",
        probabilities_key="winning_payline_count_probabilities",
    )

    def construct_attempt(rng, resolved_axes: SlotMachineAxes):
        scene = sample_slot_machine_grid(
            rng=rng,
            axes=resolved_axes,
            target_winning_count=int(target.value),
        )
        return build_slot_attempt_result(
            scene=scene,
            answer_value=len(scene.winning_payline_ids),
            annotation_payline_ids=tuple(str(payline_id) for payline_id in scene.winning_payline_ids),
            execution_extra=dict(target_params),
        )

    return build_fixed_query_objective_plan(
        attempt_namespace=f"{SCENE_NAMESPACE}.winning_payline_count.{int(target.value)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        query_probabilities=query_probabilities,
        query_params=target_params,
        example_answer_value=2,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesSlotMachineWinningPaylineCountTask:
    """Count row or diagonal paylines whose three visible symbols all match."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'matching')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_fixed_query_slot_machine_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_winning_payline_count_objective,
        )


__all__ = ["GamesSlotMachineWinningPaylineCountTask", "TASK_ID"]
