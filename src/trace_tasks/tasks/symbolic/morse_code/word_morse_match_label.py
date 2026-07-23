"""Select the Morse-code strip matching a source word."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_morse_choice_instance
from .shared.rendering import render_word_morse_match_scene
from .shared.sampling import build_word_source_code_choice
from .shared.state import OPTION_LABELS


TASK_ID = "task_symbolic__morse_code__word_morse_match_label"
INTERNAL_QUERY_KEY = "word_morse_match_label"
TASK_PROMPT_KEY = "word_morse_match_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "symbolic",
    "morse_code",
    task_id=TASK_ID,
)


@register_task
class SymbolicMorseWordMatchLabelTask:
    """Choose the Morse-code card that encodes the source word."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate source-word Morse matching with task-owned option and annotation binding."""

        factory = lambda retry_seed, variant, probs: build_word_source_code_choice(
            rng=spawn_rng(int(retry_seed), f"{TASK_ID}.dataset"),
            scene_variant=str(variant),
            scene_variant_probabilities=probs,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            labels=OPTION_LABELS,
        )
        roles = lambda dataset: {
            "source_word": str(dataset.annotation_item_ids[0]),
            "selected_option": str(dataset.annotation_item_ids[1]),
        }
        render_kwargs = lambda dataset: {"source_word": dataset.source_word, "code_options": dataset.code_options}
        return run_morse_choice_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_id="morse_code",
            internal_query_key=INTERNAL_QUERY_KEY,
            scene_variant_namespace=f"{TASK_ID}.scene_variant",
            failure_message=f"failed to generate Morse word match instance for {TASK_ID}",
            dataset_factory=factory,
            render_scene=render_word_morse_match_scene,
            render_kwargs_factory=render_kwargs,
            task_prompt_key=TASK_PROMPT_KEY,
            object_description_key_factory=lambda dataset: f"object_description_word_morse_match_label_{dataset.scene_variant}",
            annotation_hint_key="annotation_hint_word_morse_match_label",
            answer_hint_key="answer_hint_word_morse_match_label",
            json_example_key="json_example_word_morse_match_label",
            json_example_answer_only_key="json_example_answer_only_word_morse_match_label",
            annotation_roles_factory=roles,
            answer_value_factory=lambda dataset: str(dataset.answer_value),
            answer_type="string",
            answer_support_factory=lambda dataset: dataset.target_answer_support,
            annotation_item_ids_factory=lambda dataset: dataset.annotation_item_ids,
            morse_metadata_factory=lambda dataset: dataset.metadata,
        )
