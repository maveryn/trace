"""Public word-search task for locating a target word."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.tasks.registry import register_task

from ._lifecycle import make_word_search_binding, run_word_search_single_query_case
from .shared.annotations import bbox_sequence_for_cells
from .shared.defaults import get_int_range
from .shared.sampling import resolve_scene_variant, sample_location_dataset
from .shared.state import DOMAIN, OPTION_LABELS, SCENE_ID, WordSearchDataset

TASK_ID = "task_puzzles__word_search__search_location_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "search_location_label_query"
PROMPT_QUERY_KEY = "search_location_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.search_location_label"


def _build_location_scene(params, generation_defaults, rng):
    """Construct one target-word location dataset."""

    scene_variant, scene_probs = resolve_scene_variant(params, generation_defaults, rng)
    option_min, option_max = get_int_range(
        params,
        generation_defaults,
        min_key="option_count_min",
        max_key="option_count_max",
        fallback_min=6,
        fallback_max=8,
    )
    option_count, _probabilities = integer_range_choice(rng, option_min, option_max)
    answer_label = str(uniform_choice(rng, OPTION_LABELS[: int(option_count)]))
    return sample_location_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        option_count=int(option_count),
        answer_label=str(answer_label),
    )


def _bind_location_output(dataset, visual):
    """Bind option-label answer and bbox-sequence over the found word cells."""

    correct_options = [spec for spec in dataset.option_specs if bool(spec.is_correct)]
    if len(correct_options) != 1:
        raise ValueError("word-search location task requires one correct option")
    answer_label = str(correct_options[0].label)
    if str(dataset.answer_value) != answer_label:
        raise ValueError("word-search location answer label is inconsistent")
    return make_word_search_binding(
        answer_type="option_letter",
        answer_value=answer_label,
        annotation_result=bbox_sequence_for_cells(
            visual["rendered_scene"].item_bbox_map,
            dataset.target_cells,
        ),
        semantic_params={
            "answer_schema": "option_letter",
            "option_count": int(len(dataset.option_specs)),
        },
        execution_fields={
            "annotation_policy": "bbox_sequence_found_word_cells",
            "option_specs": [
                {
                    "label": str(spec.label),
                    "row_1based": int(spec.row_1based),
                    "col_1based": int(spec.col_1based),
                    "direction": str(spec.direction),
                    "is_correct": bool(spec.is_correct),
                }
                for spec in dataset.option_specs
            ],
        },
    )


@register_task
class PuzzlesWordSearchLocationLabelTask:
    """Choose the option that locates the target word."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'topology', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one word-location option case."""

        return run_word_search_single_query_case(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            namespace=_NAMESPACE_BASE,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_query_key=PROMPT_QUERY_KEY,
            instance_seed=int(instance_seed),
            params=params,
            sample_builder=_build_location_scene,
            output_binder=_bind_location_output,
            attempt_limit=int(max_attempts),
        )


__all__ = ["PuzzlesWordSearchLocationLabelTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
