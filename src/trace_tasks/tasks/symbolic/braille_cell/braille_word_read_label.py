"""Select the word written by a Braille word plate."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_braille_choice_instance
from .shared.rendering import render_braille_word_read_scene
from .shared.sampling import build_plate_source_word_choice
from .shared.state import WORD_OPTION_LABELS


TASK_ID = "task_symbolic__braille_cell__braille_word_read_label"
INTERNAL_QUERY_KEY = "braille_word_read_label"
TASK_PROMPT_KEY = "braille_word_read_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "symbolic",
    "braille_cell",
    task_id=TASK_ID,
)


@register_task
class SymbolicBrailleWordReadLabelTask:
    """Choose the word encoded by a multi-cell Braille plate."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate source-plate word reading with task-owned option and annotation binding."""

        factory = lambda retry_seed, variant, probs: build_plate_source_word_choice(
            rng=spawn_rng(int(retry_seed), f"{TASK_ID}.dataset"),
            scene_variant=str(variant),
            scene_variant_probabilities=probs,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            labels=WORD_OPTION_LABELS,
        )
        roles = lambda dataset: {
            "source_plate": str(dataset.annotation_item_ids[0]),
            "selected_option": str(dataset.annotation_item_ids[1]),
        }
        render_kwargs = lambda dataset: {"source_plate": dataset.source_plate, "word_options": dataset.word_options}
        return run_braille_choice_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_id="braille_cell",
            internal_query_key=INTERNAL_QUERY_KEY,
            scene_variant_namespace=f"{TASK_ID}.scene_variant",
            failure_message=f"failed to generate Braille word read instance for {TASK_ID}",
            dataset_factory=factory,
            render_scene=render_braille_word_read_scene,
            render_kwargs_factory=render_kwargs,
            task_prompt_key=TASK_PROMPT_KEY,
            object_description_key_factory=lambda dataset: f"object_description_braille_word_read_label_{dataset.scene_variant}",
            annotation_hint_key="annotation_hint_braille_word_read_label",
            answer_hint_key="answer_hint_braille_word_read_label",
            json_example_key="json_example_braille_word_read_label",
            json_example_answer_only_key="json_example_answer_only_braille_word_read_label",
            annotation_roles_factory=roles,
            answer_value_factory=lambda dataset: str(dataset.answer_value),
            answer_type="string",
            answer_support_factory=lambda dataset: dataset.target_answer_support,
            annotation_item_ids_factory=lambda dataset: dataset.annotation_item_ids,
            braille_metadata_factory=lambda dataset: dataset.metadata,
        )
