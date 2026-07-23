from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import run_public_rubiks_task
from .shared.sampling import build_result_sample
from .shared.state import DOMAIN, SCENE_ID

TASK_ID = "task_puzzles__rubiks_net__rubiks_move_result_label"
DIRECT_QUERY_ID = "direct_sequence_result_label"
INVERSE_QUERY_ID = "inverse_sequence_result_label"
SUPPORTED_QUERY_IDS = (DIRECT_QUERY_ID, INVERSE_QUERY_ID)
PROMPT_TASK_KEY = "rubiks_move_result_label_query"
OBJECT_DESCRIPTION_KEY = "object_description_move_result_label"
RULE_CODE = "rubiks_sequence_result_match"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.move_result"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesRubiksNetMoveResultLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_public_rubiks_task(
            public_identity=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=DIRECT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            namespace_base=_NAMESPACE_BASE,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_query_key=None,
            object_description_key=OBJECT_DESCRIPTION_KEY,
            dataset_factory_factory=_build_dataset_for_branch,
            task_field_factory=_task_fields,
            question_format=None,
        )


def _build_dataset_for_branch(selected_branch):
    def _build_dataset(rng, _attempt_seed, params, axes):
        result_mode = (
            "inverse" if str(selected_branch) == INVERSE_QUERY_ID else "direct"
        )
        return build_result_sample(
            rng=rng,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            option_count=int(axes.option_count),
            answer_option_index=int(axes.answer_option_index),
            result_mode=str(result_mode),
        )

    return _build_dataset


def _task_fields(_axes, dataset):
    return {
        "rubiks_rule_code": RULE_CODE,
        "move_count": len(dataset.get("query_sequence", [])),
        "result_mode": str(dataset.get("result_mode", "")),
    }


__all__ = ["PuzzlesRubiksNetMoveResultLabelTask", "TASK_ID"]
