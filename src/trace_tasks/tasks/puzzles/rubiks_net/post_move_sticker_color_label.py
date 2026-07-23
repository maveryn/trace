from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import run_public_rubiks_task
from .shared.sampling import build_post_move_sticker_sample
from .shared.state import DOMAIN, SCENE_ID

TASK_ID = "task_puzzles__rubiks_net__post_move_sticker_color_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "post_move_sticker_color_label_query"
PROMPT_QUERY_KEY = "post_move_sticker_color_label"
OBJECT_DESCRIPTION_KEY = "object_description_sticker_color_label"
RULE_CODE = "post_move_sticker_color_readout"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.post_move_sticker_color"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesRubiksNetPostMoveStickerColorLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_public_rubiks_task(
            public_identity=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=SINGLE_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            namespace_base=_NAMESPACE_BASE,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_query_key=PROMPT_QUERY_KEY,
            object_description_key=OBJECT_DESCRIPTION_KEY,
            dataset_factory=_build_dataset,
            task_field_factory=_task_fields,
            question_format=PROMPT_QUERY_KEY,
        )


def _build_dataset(rng, _attempt_seed, params, axes):
    return build_post_move_sticker_sample(
        rng=rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        option_count=int(axes.option_count),
        answer_option_index=int(axes.answer_option_index),
    )


def _task_fields(_axes, dataset):
    return {
        "rubiks_rule_code": RULE_CODE,
        "move_count": len(dataset.get("query_sequence", [])),
        "target_sticker_id": str(dataset["target_sticker_id"]),
    }


__all__ = ["PuzzlesRubiksNetPostMoveStickerColorLabelTask", "TASK_ID"]
