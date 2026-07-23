from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import HexAttemptResult, HexObjectivePlan, run_hex_lifecycle
from .shared.sampling import resolve_hex_integer_axis, sample_gap_count_scene
from .shared.state import SCENE_ID


TASK_ID = "task_games__hex__connection_gap_count"
SUPPORTED_QUERY_IDS = ("single",)
PROMPT_QUERY_KEY = "connection_gap_count"
GAP_COUNT_SUPPORT = (1, 2, 3, 4, 5)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_connection_gap_objective(
    instance_seed,
    task_params,
    _selected_query_id,
    _branch_probabilities,
    scene_axes,
):
    del scene_axes
    target_axis = resolve_hex_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="connection_gap_count_support",
        explicit_key="target_answer",
        fallback_support=GAP_COUNT_SUPPORT,
        namespace=f"{PROMPT_QUERY_KEY}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )

    def construct_attempt(rng, axes):
        sample = sample_gap_count_scene(
            rng=rng,
            scene_axes=axes,
            target_answer=int(target_axis.value),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        return HexAttemptResult(
            sample=sample,
            annotation_coords=tuple(sample.annotation_coords),
        )

    return HexObjectivePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_gt=TypedValue(type="integer", value=int(target_axis.value)),
        target_axis=target_axis,
        candidate_count_axis=None,
        extra_query_params={},
        attempt_namespace=f"games.hex.{PROMPT_QUERY_KEY}",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesHexConnectionGapCountTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_hex_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_connection_gap_objective,
        )


__all__ = ["GamesHexConnectionGapCountTask"]
