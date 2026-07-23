"""Solve one missing coefficient in a simple chemical equation."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ._lifecycle import (
    ChemicalEquationTaskBinding,
    load_chemical_equation_defaults,
    run_chemical_equation_instance,
)
from .shared.sampling import build_missing_coefficient_dataset

TASK_ID = "task_symbolic__chemical_equation__missing_coefficient_value"
INTERNAL_QUERY_ID = "missing_coefficient_value"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_chemical_equation_defaults(
    TASK_ID
)

_BINDING = ChemicalEquationTaskBinding(
    public_task_id=TASK_ID,
    internal_query_id=INTERNAL_QUERY_ID,
    task_prompt_key="missing_coefficient_value",
    object_description_prefix="object_description_missing_coefficient_value",
    annotation_hint_key="annotation_hint_missing_coefficient_value",
    answer_hint_key="answer_hint_missing_coefficient_value",
    json_example_key="json_example_missing_coefficient_value",
    json_example_answer_only_key="json_example_answer_only_missing_coefficient_value",
    answer_type="integer",
    annotation_source="item_bboxes_px",
    failure_message=f"failed to generate chemical-equation missing-coefficient instance for {TASK_ID}",
)


def _annotation_item_ids(dataset) -> tuple[str, ...]:
    hidden_slot = int(dataset.hidden_slot_index)
    return (
        f"coefficient_slot_{hidden_slot + 1}",
        *tuple(str(term.molecule_card_id) for term in dataset.terms),
    )


def _build_annotation(rendered_scene, dataset):
    return bbox_set_annotation_artifacts(
        [rendered_scene.item_bboxes[item_id] for item_id in _annotation_item_ids(dataset)]
    )


@register_task
class SymbolicChemicalEquationMissingCoefficientTask:
    """Return the missing coefficient that balances the shown equation."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        return run_chemical_equation_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            binding=_BINDING,
            dataset_factory=lambda rng, scene_variant, scene_probs: build_missing_coefficient_dataset(
                rng=rng,
                params=params,
                gen_defaults=_GEN_DEFAULTS,
                scene_variant=str(scene_variant),
                scene_variant_probabilities=scene_probs,
            ),
            annotation_factory=_build_annotation,
            annotation_item_ids_factory=_annotation_item_ids,
        )


__all__ = [
    "INTERNAL_QUERY_ID",
    "TASK_ID",
    "SymbolicChemicalEquationMissingCoefficientTask",
]
