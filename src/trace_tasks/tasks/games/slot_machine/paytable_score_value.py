"""Paytable score task for slot-machine games."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import build_fixed_query_objective_plan, build_slot_attempt_result, run_fixed_query_slot_machine_lifecycle
from .shared.defaults import SCENE_ID, SCENE_NAMESPACE, SCORE_TASK_WINNING_PAYLINE_COUNT_SUPPORT
from .shared.rules import resolve_target_count, target_count_trace_params
from .shared.sampling import sample_slot_machine_grid, sample_slot_paytable
from .shared.state import SlotMachineAxes, total_winning_payline_score


TASK_ID = "task_games__slot_machine__paytable_score_value"
PROMPT_QUERY_KEY = "paytable_score_value"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_paytable_score_objective(
    instance_seed,
    task_params,
    _selected_query,
    query_probabilities,
    axes: SlotMachineAxes,
):
    """Bind score-table generation, score answer, and annotation witnesses."""

    target = resolve_target_count(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="score_task_winning_payline_count_support",
        explicit_key="target_score_winning_payline_count",
        fallback_support=SCORE_TASK_WINNING_PAYLINE_COUNT_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.paytable_score_value.target",
        balanced_flag_key="balanced_score_task_winning_payline_count_sampling",
    )
    target_params = target_count_trace_params(
        target=target,
        explicit_key="target_score_winning_payline_count",
        support_key="score_task_winning_payline_count_support",
        probabilities_key="score_task_winning_payline_count_probabilities",
    )

    def construct_attempt(rng, resolved_axes: SlotMachineAxes):
        paytable_entries = sample_slot_paytable(rng)
        scene = sample_slot_machine_grid(
            rng=rng,
            axes=resolved_axes,
            target_winning_count=int(target.value),
            paytable_entries=paytable_entries,
        )
        score_value = total_winning_payline_score(scene)
        return build_slot_attempt_result(
            scene=scene,
            answer_value=score_value,
            annotation_payline_ids=tuple(str(payline_id) for payline_id in scene.winning_payline_ids),
            execution_extra=dict(target_params),
        )

    return build_fixed_query_objective_plan(
        attempt_namespace=f"{SCENE_NAMESPACE}.paytable_score_value.{int(target.value)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        query_probabilities=query_probabilities,
        query_params=target_params,
        example_answer_value=4,
        payline_annotation_schema="segment",
        prompt_extra_slots={
            "object_description": "a toy slot machine with three reels, three visible rows, simple symbol icons, and a side paytable of symbol scores",
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesSlotMachinePaytableScoreValueTask:
    """Compute the total score from winning row or diagonal paylines."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
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
            prepare_objective=_prepare_paytable_score_objective,
        )


__all__ = ["GamesSlotMachinePaytableScoreValueTask", "TASK_ID"]
