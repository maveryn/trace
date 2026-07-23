from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ._lifecycle import TruthTableTaskBinding, render_pattern_scene, run_truth_table_instance
from .shared.sampling import build_pattern_dataset


TASK_ID = "task_symbolic__truth_table__truth_pattern_label"
INTERNAL_QUERY_KEY = "truth_pattern_label"
TASK_PROMPT_KEY = "truth_pattern_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "symbolic",
    "truth_table",
    task_id=TASK_ID,
)
_BINDING = TruthTableTaskBinding(
    seed_namespace=TASK_ID,
    internal_query_key=INTERNAL_QUERY_KEY,
    task_prompt_key=TASK_PROMPT_KEY,
    object_description_prefix="object_description_truth_pattern_label",
    annotation_hint_key="annotation_hint_truth_pattern_label",
    answer_hint_key="answer_hint_truth_pattern_label",
    json_example_key="json_example_truth_pattern_label",
    json_example_answer_only_key="json_example_answer_only_truth_pattern_label",
    answer_type="string",
    failure_message=f"failed to generate truth-table truth-pattern instance for {TASK_ID}",
)


def _build_annotation(rendered_scene, dataset):
    return bbox_annotation_artifacts(rendered_scene.item_bboxes[str(dataset.selected_option_id)])


@register_task
class SymbolicTruthTableTruthPatternLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'formula_evaluation', 'matching')
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        factory = lambda retry_seed, variant, probs: build_pattern_dataset(
            rng=spawn_rng(int(retry_seed), f"{TASK_ID}.dataset"),
            scene_variant=str(variant),
            scene_variant_probabilities=probs,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
        )
        return run_truth_table_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            binding=_BINDING,
            dataset_factory=factory,
            render_scene=render_pattern_scene,
            render_kwargs_factory=lambda dataset: {
                "expression": dataset.expression,
                "options": dataset.options,
            },
            annotation_factory=_build_annotation,
            answer_value_factory=lambda dataset: str(dataset.answer_value),
            answer_support_factory=lambda dataset: tuple(dataset.target_answer_support),
            annotation_item_ids_factory=lambda dataset: (str(dataset.selected_option_id),),
            metadata_factory=lambda dataset: dict(dataset.metadata),
            annotation_source="item_bboxes_px",
        )


__all__ = ["SymbolicTruthTableTruthPatternLabelTask", "TASK_ID"]
