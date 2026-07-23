from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import run_fixed_numeric_balance_lifecycle
from .shared.constraints import build_missing_weight_dataset
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__balance_scale__missing_object_weight_value"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "missing_object_weight_value"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "puzzles",
    SCENE_ID,
    task_id=TASK_ID,
)


def _construct_missing_weight_dataset(
    *,
    task_params,
    gen_defaults,
    instance_seed,
    answer_value,
    answer_support,
    axes,
    task_name,
    prompt_query_key,
):
    """Bind the missing-weight objective to the neutral balance equation builder."""

    return build_missing_weight_dataset(
        params=task_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        answer_value=int(answer_value),
        answer_support=answer_support,
        scene_variant=str(axes.scene_variant),
        target_cue_mode=str(axes.target_cue_mode),
        namespace=str(task_name),
        prompt_query_key=str(prompt_query_key),
    )


@register_task
class PuzzlesBalanceScaleMissingObjectWeightValueTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate balanced equations that determine one missing object weight."""

        return run_fixed_numeric_balance_lifecycle(
            task_name=TASK_ID,
            domain=self.domain,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_query_key=PROMPT_QUERY_KEY,
            fallback_min=1,
            fallback_max=20,
            dataset_hook=_construct_missing_weight_dataset,
        )
