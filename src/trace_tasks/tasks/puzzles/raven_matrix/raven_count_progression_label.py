from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import run_raven_matrix_task
from .shared.defaults import resolve_count_bounds
from .shared.rules import build_count_progression_dataset
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_puzzles__raven_matrix__raven_count_progression_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "raven_count_progression_label_query"
PROMPT_QUERY_KEY = "raven_count_progression_label"
RULE_CODE = "count_progression_matrix"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.count_progression"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesRavenMatrixCountProgressionLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        attempts = max(1, int(max_attempts))
        return _run_count_progression(int(instance_seed), params=params, max_attempts=attempts)


def _run_count_progression(instance_seed, *, params, max_attempts):
    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{_NAMESPACE_BASE}.branch",
    )
    return run_raven_matrix_task(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults=_PROMPT_DEFAULTS,
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        namespace_base=_NAMESPACE_BASE,
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        dataset_factory=_build_dataset,
        task_field_factory=_task_fields,
        question_format=PROMPT_QUERY_KEY,
        view_family="raven_count_progression_matrix",
    )


def _build_dataset(attempt_seed, params, axes):
    count_min, count_max = resolve_count_bounds(params, _GEN_DEFAULTS)
    return build_count_progression_dataset(
        rng=spawn_rng(int(attempt_seed), f"{_NAMESPACE_BASE}.dataset"),
        count_min=int(count_min),
        count_max=int(count_max),
        option_count=int(axes.option_count),
        correct_option_index=int(axes.answer_option_index),
    )


def _task_fields(_axes, _dataset):
    return {"raven_rule_code": RULE_CODE}


__all__ = ["PuzzlesRavenMatrixCountProgressionLabelTask", "TASK_ID"]
