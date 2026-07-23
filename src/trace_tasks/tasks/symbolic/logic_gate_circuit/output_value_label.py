"""Select the circuit option with a target final output value."""

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
from ..shared.common import get_int_range as _get_range
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.visual_defaults import load_symbolic_noise_defaults

from .shared.annotations import projected_bbox
from .shared.output import circuit_trace
from .shared.prompts import build_logic_gate_prompt
from .shared.rendering import render_option_bundle, render_payload_sections, rounded_render_maps
from .shared.rules import sample_circuit_with_output
from .shared.state import (
    LogicCircuitSpec,
    OPTION_LABELS,
    SCENE_ID,
    SCENE_VARIANTS,
    SUPPORTED_GATE_TYPES,
)


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__logic_gate_circuit__output_value_label"
OUTPUT_ONE_LABEL_QUERY_ID = "output_one_label"
OUTPUT_ZERO_LABEL_QUERY_ID = "output_zero_label"
SUPPORTED_QUERY_IDS = (OUTPUT_ONE_LABEL_QUERY_ID, OUTPUT_ZERO_LABEL_QUERY_ID)
TASK_PROMPT_KEY = "logic_gate_output_value_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.20)


@dataclass(frozen=True)
class _LabelDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    selected_query_id: str
    selected_query_probabilities: dict[str, float]
    target_output_value: int
    answer_label: str
    target_answer_support: tuple[str, ...]
    circuits: tuple[LogicCircuitSpec, ...]
    annotation_item_id: str
    metadata: dict[str, Any]


def _target_output_value(selected_query_id: str) -> int:
    if str(selected_query_id) == OUTPUT_ONE_LABEL_QUERY_ID:
        return 1
    if str(selected_query_id) == OUTPUT_ZERO_LABEL_QUERY_ID:
        return 0
    raise ValueError(f"unsupported logic-gate output-label branch: {selected_query_id}")


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
) -> _LabelDataset:
    """Construct four circuit options with exactly one target-output answer."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_output = _target_output_value(str(selected_query_id))

    option_count = int(params.get("option_count", gen_defaults.get("option_count", 4)))
    if int(option_count) != 4:
        raise ValueError("logic-gate output-label scenes require exactly four visual options")
    labels = tuple(str(label) for label in OPTION_LABELS[: int(option_count)])
    answer_label = str(params.get("answer_label", uniform_choice(rng, labels, sort_keys=False)))
    answer_label = answer_label.upper()
    if answer_label not in labels:
        raise ValueError(f"answer_label must be one of {labels}")

    input_count_min, input_count_max = _get_range(
        params,
        gen_defaults,
        min_key="input_count_min",
        max_key="input_count_max",
        fallback_min=2,
        fallback_max=2,
    )
    gate_count_min, gate_count_max = _get_range(
        params,
        gen_defaults,
        min_key="gate_count_min",
        max_key="gate_count_max",
        fallback_min=1,
        fallback_max=3,
    )
    if int(input_count_min) != 2 or int(input_count_max) != 2:
        raise ValueError("logic-gate output-label panels require exactly two visible inputs")
    if int(gate_count_min) < 1 or int(gate_count_max) > 3:
        raise ValueError("logic-gate output-label gate support must stay within 1..3")

    circuits: list[LogicCircuitSpec] = []
    for index, label in enumerate(labels):
        desired_output = int(target_output) if str(label) == str(answer_label) else 1 - int(target_output)
        circuits.append(
            sample_circuit_with_output(
                rng,
                item_id=f"circuit_{index + 1}",
                label=str(label),
                input_count_min=int(input_count_min),
                input_count_max=int(input_count_max),
                gate_count_min=int(gate_count_min),
                gate_count_max=int(gate_count_max),
                target_output_value=int(desired_output),
            )
        )

    annotation_item_id = next(circuit.item_id for circuit in circuits if str(circuit.label) == str(answer_label))
    return _LabelDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        selected_query_id=str(selected_query_id),
        selected_query_probabilities={str(key): float(value) for key, value in selected_query_probabilities.items()},
        target_output_value=int(target_output),
        answer_label=str(answer_label),
        target_answer_support=tuple(labels),
        circuits=tuple(circuits),
        annotation_item_id=str(annotation_item_id),
        metadata={
            "option_count": int(option_count),
            "matching_circuit_label": str(answer_label),
            "target_output_value": int(target_output),
            "supported_gate_types": list(SUPPORTED_GATE_TYPES),
        },
    )


def _build_label_trace_payload(
    *,
    dataset: _LabelDataset,
    prompt_meta: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    render_map: Mapping[str, Any],
    rendered_entities: tuple[dict[str, Any], ...],
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    annotation_bbox: list[float],
    query_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble output-label trace fields from the same dataset used for answer binding."""

    scene_relations = dict(query_params)
    scene_relations["answer_value"] = str(dataset.answer_label)

    payload: dict[str, Any] = {}
    payload["scene_ir"] = {
        "scene_kind": SCENE_ID,
        "entities": [dict(entity) for entity in rendered_entities],
        "relations": scene_relations,
    }
    payload["query_spec"] = {
        "query_id": str(dataset.selected_query_id),
        "template_id": str(prompt_meta["bundle_id"]),
        "prompt_variant": dict(prompt_meta["prompt_variant"]),
        "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
        "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
        "params": dict(query_params),
    }
    payload["render_spec"] = dict(render_spec)
    payload["render_map"] = dict(render_map)
    payload["execution_trace"] = {
        **dict(query_params),
        "answer_value": str(dataset.answer_label),
        "answer_type": "option_letter",
        "annotation_item_id": str(dataset.annotation_item_id),
        "logic_gate_metadata": dict(dataset.metadata),
        "circuits": [circuit_trace(circuit) for circuit in dataset.circuits],
        "question_format": str(dataset.selected_query_id),
    }
    payload["witness_symbolic"] = {"type": "bbox", "value": list(annotation_bbox)}
    payload["projected_annotation"] = projected_bbox(annotation_bbox)
    payload["answer_gt"] = answer_gt.to_dict()
    payload["annotation_gt"] = annotation_gt.to_dict()
    return payload


@register_task
class SymbolicLogicGateOutputValueLabelTask:
    """Select the circuit whose final OUT node evaluates to a target value."""

    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'topology', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one output-label instance with task-owned answer binding."""

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=OUTPUT_ONE_LABEL_QUERY_ID,
            task_id=TASK_ID,
        )
        scene_variant, scene_variant_probabilities = _resolve_scene_variant(
            task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )

        last_error: Exception | None = None
        dataset: _LabelDataset | None = None
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

        render_bundle = render_option_bundle(
            instance_seed=int(instance_seed),
            circuits=dataset.circuits,
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
        annotation_bbox = [round(float(value), 3) for value in item_bboxes[str(dataset.annotation_item_id)]]
        answer_gt = TypedValue(type="option_letter", value=str(dataset.answer_label))
        annotation_gt = TypedValue(type="bbox", value=list(annotation_bbox))
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
        trace_payload = _build_label_trace_payload(
            dataset=dataset,
            prompt_meta=prompt_meta,
            render_spec=render_spec,
            render_map=render_map,
            rendered_entities=tuple(dict(entity) for entity in render_bundle.rendered.entities),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            annotation_bbox=list(annotation_bbox),
            query_params=query_params,
        )
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
    "OUTPUT_ONE_LABEL_QUERY_ID",
    "OUTPUT_VALUE_LABEL_TASK_ID",
    "OUTPUT_ZERO_LABEL_QUERY_ID",
    "SCENE_ID",
    "SUPPORTED_GATE_TYPES",
    "SUPPORTED_QUERY_IDS",
    "SymbolicLogicGateOutputValueLabelTask",
    "TASK_ID",
]

OUTPUT_VALUE_LABEL_TASK_ID = TASK_ID
