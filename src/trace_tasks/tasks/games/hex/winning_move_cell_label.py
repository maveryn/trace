from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import HexAttemptResult, HexObjectivePlan, run_hex_lifecycle
from .shared.sampling import (
    resolve_hex_candidate_count_axis,
    resolve_hex_string_choice,
    sample_winning_move_scene,
)
from .shared.state import DEFAULTS, SCENE_ID


TASK_ID = "task_games__hex__winning_move_cell_label"
SUPPORTED_QUERY_IDS = ("single",)
PROMPT_QUERY_KEY = "winning_move_cell_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_winning_move_objective(
    instance_seed,
    task_params,
    _selected_query_id,
    _branch_probabilities,
    scene_axes,
):
    """Bind the target candidate label to a unique immediate Hex win."""

    candidate_count_axis = resolve_hex_candidate_count_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{PROMPT_QUERY_KEY}.candidate_count",
    )
    target_label_axis = resolve_hex_string_choice(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="winning_move_label_support",
        explicit_key="target_label",
        fallback_support=DEFAULTS.winning_move_label_support,
        namespace=f"{PROMPT_QUERY_KEY}.target_label",
        balanced_flag_key="balanced_target_label_sampling",
    )
    target_label = str(target_label_axis.value)
    candidate_count = max(int(candidate_count_axis.value), ord(target_label[0]) - ord("A") + 1)

    def construct_attempt(rng, axes):
        del axes
        sample = sample_winning_move_scene(
            rng=rng,
            scene_axes=scene_axes,
            target_label=target_label,
            candidate_count=int(candidate_count),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        return HexAttemptResult(
            sample=sample,
            annotation_coords=(tuple(sample.winning_move_coord),),
            annotation_contract="point",
        )

    return HexObjectivePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_gt=TypedValue(type="string", value=target_label),
        target_axis=target_label_axis,
        candidate_count_axis=candidate_count_axis,
        extra_query_params={},
        attempt_namespace=f"games.hex.{PROMPT_QUERY_KEY}",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesHexWinningMoveCellLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'topology', 'state_update')
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
            prepare_objective=_prepare_winning_move_objective,
        )


__all__ = ["GamesHexWinningMoveCellLabelTask"]
