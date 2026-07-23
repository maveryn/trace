"""Public word-search task for selecting a present option word."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.tasks.registry import register_task

from ._lifecycle import make_word_search_binding, run_word_search_single_query_case
from .shared.annotations import segment_for_cell_pair
from .shared.sampling import (
    present_word_segments,
    resolve_scene_variant,
    sample_present_word_option_dataset,
)
from .shared.state import DOMAIN, OPTION_LABELS, SCENE_ID

TASK_ID = "task_puzzles__word_search__present_word_option_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.present_word_option_label"


def _build_present_word_option_scene(params, generation_defaults, rng):
    scene_variant, scene_probs = resolve_scene_variant(params, generation_defaults, rng)
    option_count = 4
    answer_label = str(uniform_choice(rng, OPTION_LABELS[:option_count]))
    return sample_present_word_option_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        option_count=int(option_count),
        answer_label=str(answer_label),
    )


def _bind_present_word_option_output(dataset, visual):
    correct_option = next(spec for spec in dataset.option_specs if bool(spec.is_correct))
    segments = present_word_segments(dataset)
    return make_word_search_binding(
        answer_type="option_letter",
        answer_value=str(correct_option.label),
        annotation_result=segment_for_cell_pair(
            visual["rendered_scene"].cell_centers_px,
            segments[0],
        ),
        semantic_params={
            "answer_schema": "option_letter",
            "option_count": int(len(dataset.option_specs)),
        },
        execution_fields={
            "annotation_policy": "segment_present_word_start_to_end",
            "supporting_annotation_source": "cell_centers_px",
            "option_specs": [
                {
                    "label": str(spec.label),
                    "word": str(spec.word),
                    "is_correct": bool(spec.is_correct),
                }
                for spec in dataset.option_specs
            ],
        },
    )


@register_task
class PuzzlesWordSearchPresentWordOptionLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('topology', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_word_search_single_query_case(
            task_id=TASK_ID,
            supported_query_ids=self.supported_query_ids,
            namespace=_NAMESPACE_BASE,
            prompt_task_key="present_word_option_label_query",
            prompt_query_key="present_word_option_label",
            instance_seed=int(instance_seed),
            params=params,
            sample_builder=_build_present_word_option_scene,
            output_binder=_bind_present_word_option_output,
            attempt_limit=int(max_attempts),
        )
