"""Select the assignment row that satisfies a visible logic-gate circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.visual_defaults import load_symbolic_noise_defaults

from .shared.annotations import projected_bbox_map
from .shared.output import assignment_trace, circuit_trace
from .shared.prompts import build_logic_gate_prompt
from .shared.rendering import render_assignment_bundle, render_payload_sections, rounded_render_maps
from .shared.rules import all_three_input_assignments, build_exact_assignment_circuit, evaluate_logic_circuit
from .shared.state import (
    CandidateAssignmentSpec,
    INPUT_LABELS,
    OPTION_LABELS,
    SCENE_ID,
    SCENE_VARIANTS,
    SUPPORTED_GATE_TYPES,
    LogicCircuitSpec,
)


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__logic_gate_circuit__satisfying_assignment_label"
ASSIGNMENT_OUTPUTS_ONE_QUERY_ID = "assignment_outputs_one_label"
ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID = "assignment_outputs_zero_label"
SUPPORTED_QUERY_IDS = (ASSIGNMENT_OUTPUTS_ONE_QUERY_ID, ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID)
TASK_PROMPT_KEY = "logic_gate_satisfying_assignment_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.20)


@dataclass(frozen=True)
class _AssignmentDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    selected_query_id: str
    selected_query_probabilities: dict[str, float]
    target_output_value: int
    answer_label: str
    target_answer_support: tuple[str, ...]
    source_circuit: LogicCircuitSpec
    candidates: tuple[CandidateAssignmentSpec, ...]
    annotation_item_ids: tuple[str, str]
    metadata: dict[str, Any]


def _target_output_value(selected_query_id: str) -> int:
    if str(selected_query_id) == ASSIGNMENT_OUTPUTS_ONE_QUERY_ID:
        return 1
    if str(selected_query_id) == ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID:
        return 0
    raise ValueError(f"unsupported logic-gate assignment branch: {selected_query_id}")


def _resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=TASK_ID,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _build_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    selected_query_id: str,
    selected_query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> _AssignmentDataset:
    """Build one source circuit and four assignment rows with a unique answer.

    The invariant is that the source circuit is synthesized from the chosen
    satisfying assignment, then all visible rows are evaluated so exactly one
    candidate matches the branch target output.
    """

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_output = _target_output_value(str(selected_query_id))
    option_count = int(params.get("option_count", gen_defaults.get("option_count", 4)))
    if int(option_count) != 4:
        raise ValueError("logic-gate assignment task requires exactly four visual options")

    labels = tuple(str(label) for label in OPTION_LABELS[: int(option_count)])
    answer_label = str(
        params.get(
            "answer_label",
            params.get("correct_label", uniform_choice(rng, labels, sort_keys=False)),
        )
    ).upper()
    if answer_label not in labels:
        raise ValueError(f"answer_label must be one of {labels}")

    assignments = list(all_three_input_assignments())
    correct_values = dict(params.get("correct_values", rng.choice(assignments)))
    correct_values = {str(key): int(value) for key, value in correct_values.items()}
    if set(correct_values) != set(INPUT_LABELS) or any(int(value) not in {0, 1} for value in correct_values.values()):
        raise ValueError("correct_values must provide x/y/z values in {0,1}")

    source_circuit = build_exact_assignment_circuit(
        target_output_value=int(target_output),
        correct_values=correct_values,
    )
    distractor_assignments = [assignment for assignment in assignments if dict(assignment) != dict(correct_values)]
    rng.shuffle(distractor_assignments)
    selected_distractors = distractor_assignments[: int(option_count) - 1]
    candidates: list[CandidateAssignmentSpec] = []
    distractor_index = 0
    for label in labels:
        if str(label) == str(answer_label):
            values = dict(correct_values)
            is_correct = True
        else:
            values = dict(selected_distractors[int(distractor_index)])
            distractor_index += 1
            is_correct = False
        output_value = int(evaluate_logic_circuit(source_circuit, values))
        candidates.append(
            CandidateAssignmentSpec(
                item_id=f"option_{label}",
                label=str(label),
                values=dict(values),
                output_value=int(output_value),
                is_correct=bool(is_correct),
            )
        )

    correct_candidates = [candidate for candidate in candidates if int(candidate.output_value) == int(target_output)]
    if len(correct_candidates) != 1 or str(correct_candidates[0].label) != str(answer_label):
        raise RuntimeError("logic-gate assignment construction failed to make a unique correct option")
    return _AssignmentDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        selected_query_id=str(selected_query_id),
        selected_query_probabilities={str(key): float(value) for key, value in selected_query_probabilities.items()},
        target_output_value=int(target_output),
        answer_label=str(answer_label),
        target_answer_support=tuple(labels),
        source_circuit=source_circuit,
        candidates=tuple(candidates),
        annotation_item_ids=("source_circuit", f"option_{answer_label}"),
        metadata={
            "target_output_value": int(target_output),
            "correct_assignment": dict(correct_values),
            "correct_option_label": str(answer_label),
            "option_assignments": {
                str(candidate.label): {str(key): int(value) for key, value in candidate.values.items()}
                for candidate in candidates
            },
            "supported_gate_types": list(SUPPORTED_GATE_TYPES),
        },
    )


@register_task
class SymbolicLogicGateSatisfyingAssignmentLabelTask:
    """Select the assignment option that makes the source circuit output the requested value."""

    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'topology', 'formula_evaluation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one assignment-selection instance with role-bound annotation.

        This method owns branch selection, unique-answer construction, prompt
        slots, source/option annotation binding, trace fields, and final
        TaskOutput construction. Scene shared helpers only render the circuit
        grammar and expose final pixel geometry.
        """

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=ASSIGNMENT_OUTPUTS_ONE_QUERY_ID,
            task_id=TASK_ID,
        )
        scene_variant, scene_variant_probabilities = _resolve_scene_variant(
            task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )

        last_error: Exception | None = None
        dataset: _AssignmentDataset | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                dataset = _build_dataset(
                    instance_seed=int(instance_seed) + int(attempt_index),
                    scene_variant=str(scene_variant),
                    scene_variant_probabilities=scene_variant_probabilities,
                    selected_query_id=str(selected_query_id),
                    selected_query_probabilities=query_probabilities,
                    params=task_params,
                    gen_defaults=_GEN_DEFAULTS,
                )
                break
            except Exception as exc:
                last_error = exc
        if dataset is None:
            raise RuntimeError(f"failed to generate logic-gate instance for {TASK_ID}") from last_error

        render_bundle = render_assignment_bundle(
            instance_seed=int(instance_seed),
            circuit=dataset.source_circuit,
            candidates=dataset.candidates,
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt, prompt_variants, prompt_meta, _prompt_artifacts = build_logic_gate_prompt(
            domain=DOMAIN,
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_variant=str(dataset.scene_variant),
            prompt_key=TASK_PROMPT_KEY,
            branch_key=str(dataset.selected_query_id),
            instance_seed=int(instance_seed),
            annotation_hint_key=f"annotation_hint_{dataset.selected_query_id}",
            answer_hint_key=f"answer_hint_{dataset.selected_query_id}",
            json_example_key=f"json_example_{dataset.selected_query_id}",
            json_example_answer_only_key=f"json_example_answer_only_{dataset.selected_query_id}",
        )

        item_bboxes, output_points, signal_points = rounded_render_maps(render_bundle)
        keyed_bboxes = {
            "source_circuit": list(item_bboxes[str(dataset.annotation_item_ids[0])]),
            "selected_option": list(item_bboxes[str(dataset.annotation_item_ids[1])]),
        }
        projected_annotation = projected_bbox_map(keyed_bboxes)
        answer_gt = TypedValue(type="option_letter", value=str(dataset.answer_label))
        annotation_gt = TypedValue(type="bbox_map", value=dict(keyed_bboxes))
        render_spec, render_map = render_payload_sections(
            render_bundle,
            item_bboxes=item_bboxes,
            output_points=output_points,
            signal_points=signal_points,
            annotation_source="item_bboxes_px",
        )
        render_spec["scene_variant"] = str(dataset.scene_variant)

        query_params = {
            "query_id": str(dataset.selected_query_id),
            "query_id_probabilities": dict(dataset.selected_query_probabilities),
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "target_output_value": int(dataset.target_output_value),
            "target_answer_support": list(dataset.target_answer_support),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": SCENE_ID,
                "entities": [dict(entity) for entity in render_bundle.rendered.entities],
                "relations": {
                    "query_id": str(dataset.selected_query_id),
                    "scene_id": SCENE_ID,
                    "scene_variant": str(dataset.scene_variant),
                    "target_output_value": int(dataset.target_output_value),
                    "answer_value": str(dataset.answer_label),
                },
            },
            "query_spec": {
                "query_id": str(dataset.selected_query_id),
                "template_id": str(prompt_meta["bundle_id"]),
                "prompt_variant": dict(prompt_meta["prompt_variant"]),
                "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
                "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
                "params": dict(query_params),
            },
            "render_spec": dict(render_spec),
            "render_map": dict(render_map),
            "execution_trace": {
                **dict(query_params),
                "answer_value": str(dataset.answer_label),
                "answer_type": "option_letter",
                "annotation_item_ids": [str(item) for item in dataset.annotation_item_ids],
                "logic_gate_metadata": dict(dataset.metadata),
                "source_circuit": circuit_trace(dataset.source_circuit),
                "candidates": [assignment_trace(candidate) for candidate in dataset.candidates],
                "question_format": str(dataset.selected_query_id),
            },
            "witness_symbolic": {"type": "bbox_map", "value": dict(keyed_bboxes)},
            "projected_annotation": dict(projected_annotation),
            "answer_gt": answer_gt.to_dict(),
            "annotation_gt": annotation_gt.to_dict(),
        }
        return TaskOutput(
            prompt=str(prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=render_bundle.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(dataset.selected_query_id),
            prompt_variants=dict(prompt_variants),
        )


__all__ = [
    "ASSIGNMENT_OUTPUTS_ONE_QUERY_ID",
    "ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID",
    "SCENE_ID",
    "SUPPORTED_GATE_TYPES",
    "SUPPORTED_QUERY_IDS",
    "SymbolicLogicGateSatisfyingAssignmentLabelTask",
    "TASK_ID",
]
