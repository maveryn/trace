"""Count gates whose computed internal output equals a requested value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ..shared.common import get_int_param as _get_int_param
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.visual_defaults import load_symbolic_noise_defaults

from ._lifecycle import build_single_circuit_count_artifacts
from .shared.rules import build_expression_tree_circuit, evaluate_logic_circuit_trace
from .shared.state import LogicCircuitSpec, SCENE_ID, SCENE_VARIANTS, SUPPORTED_GATE_TYPES


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__logic_gate_circuit__internal_output_count"
INTERNAL_OUTPUT_ONE_QUERY_ID = "internal_output_one_count"
INTERNAL_OUTPUT_ZERO_QUERY_ID = "internal_output_zero_count"
SUPPORTED_QUERY_IDS = (INTERNAL_OUTPUT_ONE_QUERY_ID, INTERNAL_OUTPUT_ZERO_QUERY_ID)
TASK_PROMPT_KEY = "logic_gate_internal_output_count"
COUNT_INPUT_LABELS: tuple[str, ...] = ("a", "b", "c", "d", "e")

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.20)


@dataclass(frozen=True)
class _InternalOutputCountDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    selected_query_id: str
    selected_query_probabilities: dict[str, float]
    target_output_value: int
    answer_value: int
    target_answer_support: tuple[int, ...]
    target_answer_probabilities: dict[str, float]
    circuit: LogicCircuitSpec
    gate_outputs: dict[str, int]
    signal_values: dict[str, int]
    annotation_item_ids: tuple[str, ...]
    metadata: dict[str, Any]


def _target_output_value(selected_query_id: str) -> int:
    if str(selected_query_id) == INTERNAL_OUTPUT_ONE_QUERY_ID:
        return 1
    if str(selected_query_id) == INTERNAL_OUTPUT_ZERO_QUERY_ID:
        return 0
    raise ValueError(f"unsupported logic-gate internal-output branch: {selected_query_id}")


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


def _resolve_target_answer(
    rng: Any,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    gate_count: int,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    low = _get_int_param(params, gen_defaults, "target_answer_min", 0)
    high = _get_int_param(params, gen_defaults, "target_answer_max", int(gate_count))
    if int(low) < 0 or int(high) > int(gate_count) or int(low) > int(high):
        raise ValueError("target answer range must be within 0..gate_count")
    support = tuple(range(int(low), int(high) + 1))
    answer = int(params.get("answer_value", params.get("target_answer", rng.choice(support))))
    if int(answer) not in support:
        raise ValueError(f"answer_value must be one of {support}")
    probability = 1.0 / float(len(support))
    return int(answer), tuple(int(value) for value in support), {str(value): float(probability) for value in support}


def _sample_gate_types(rng: Any, gate_count: int) -> tuple[str, ...]:
    return tuple(str(rng.choice(SUPPORTED_GATE_TYPES)) for _ in range(int(gate_count)))


def _build_matching_internal_output_circuit(
    rng: Any,
    *,
    target_output_value: int,
    target_count: int,
    gate_count: int,
) -> tuple[LogicCircuitSpec, dict[str, Any], tuple[str, ...]]:
    """Find a four-gate circuit whose intermediate gate outputs match the target count."""

    for _attempt in range(10000):
        gate_types = _sample_gate_types(rng, int(gate_count))
        circuit = build_expression_tree_circuit(
            rng,
            item_id="source_circuit",
            label="",
            input_labels=COUNT_INPUT_LABELS,
            gate_types=gate_types,
            input_id_prefix="internal_in",
            gate_id_prefix="internal_gate",
            signal_id_prefix="internal_sig",
            role="source_circuit",
        )
        evaluation = evaluate_logic_circuit_trace(circuit)
        matching_gate_ids = tuple(
            str(gate.item_id)
            for gate in circuit.gates
            if int(evaluation["gate_outputs"][str(gate.item_id)]) == int(target_output_value)
        )
        if len(matching_gate_ids) == int(target_count):
            return circuit, dict(evaluation), tuple(matching_gate_ids)
    raise RuntimeError("failed to construct circuit with requested internal output count")


def _build_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    selected_query_id: str,
    selected_query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> _InternalOutputCountDataset:
    """Construct one circuit with an exact count of gates producing the target internal output."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    gate_count = _get_int_param(params, gen_defaults, "gate_count", 4)
    if int(gate_count) != 4:
        raise ValueError("logic-gate internal-output-count task currently requires exactly four gates")
    target_output = _target_output_value(str(selected_query_id))
    answer_value, answer_support, answer_probabilities = _resolve_target_answer(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        gate_count=int(gate_count),
    )
    circuit, evaluation, matching_gate_ids = _build_matching_internal_output_circuit(
        rng,
        target_output_value=int(target_output),
        target_count=int(answer_value),
        gate_count=int(gate_count),
    )
    gate_outputs = {str(key): int(value) for key, value in dict(evaluation["gate_outputs"]).items()}
    signal_values = {str(key): int(value) for key, value in dict(evaluation["signal_values"]).items()}
    output_counts = {
        "0": sum(1 for value in gate_outputs.values() if int(value) == 0),
        "1": sum(1 for value in gate_outputs.values() if int(value) == 1),
    }
    if int(output_counts[str(target_output)]) != int(answer_value):
        raise RuntimeError("internal-output-count construction produced inconsistent target count")
    return _InternalOutputCountDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        selected_query_id=str(selected_query_id),
        selected_query_probabilities={str(key): float(value) for key, value in selected_query_probabilities.items()},
        target_output_value=int(target_output),
        answer_value=int(answer_value),
        target_answer_support=tuple(int(value) for value in answer_support),
        target_answer_probabilities=dict(answer_probabilities),
        circuit=circuit,
        gate_outputs=dict(gate_outputs),
        signal_values=dict(signal_values),
        annotation_item_ids=tuple(str(item_id) for item_id in matching_gate_ids),
        metadata={
            "gate_count": int(gate_count),
            "target_output_value": int(target_output),
            "gate_output_counts": dict(output_counts),
            "gate_outputs": dict(gate_outputs),
            "signal_values": dict(signal_values),
            "matching_gate_ids": [str(item_id) for item_id in matching_gate_ids],
            "target_answer_probabilities": dict(answer_probabilities),
            "supported_gate_types": list(SUPPORTED_GATE_TYPES),
        },
    )


@register_task
class SymbolicLogicGateInternalOutputCountTask:
    """Count gates whose evaluated intermediate output equals the requested bit."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own intermediate-output sampling, gate annotation binding, prompt slots, and trace assembly."""

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=INTERNAL_OUTPUT_ONE_QUERY_ID,
            task_id=TASK_ID,
        )
        scene_variant, scene_variant_probabilities = _resolve_scene_variant(
            task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        last_error: Exception | None = None
        dataset: _InternalOutputCountDataset | None = None
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

        query_params = {
            "query_id": str(dataset.selected_query_id),
            "query_id_probabilities": dict(dataset.selected_query_probabilities),
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "target_output_value": int(dataset.target_output_value),
            "target_answer_support": [int(value) for value in dataset.target_answer_support],
        }
        artifacts = build_single_circuit_count_artifacts(
            task_id=TASK_ID,
            domain=DOMAIN,
            prompt_defaults=_PROMPT_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            instance_seed=int(instance_seed),
            params=task_params,
            circuit=dataset.circuit,
            annotation_item_ids=dataset.annotation_item_ids,
            answer_value=int(dataset.answer_value),
            scene_variant=str(dataset.scene_variant),
            prompt_key=TASK_PROMPT_KEY,
            branch_key=str(dataset.selected_query_id),
            annotation_hint_key=f"annotation_hint_{dataset.selected_query_id}",
            answer_hint_key=f"answer_hint_{dataset.selected_query_id}",
            json_example_key=f"json_example_{dataset.selected_query_id}",
            json_example_answer_only_key=f"json_example_answer_only_{dataset.selected_query_id}",
            prompt_extra_slots=None,
            query_payload=query_params,
            execution_extra={
                "question_format": str(dataset.selected_query_id),
                "gate_outputs": dict(dataset.gate_outputs),
                "signal_values": dict(dataset.signal_values),
            },
            logic_gate_metadata=dataset.metadata,
            public_query_value=str(dataset.selected_query_id),
            show_fixed_input_values=True,
        )
        return TaskOutput(
            prompt=artifacts.prompt,
            answer_gt=artifacts.answer_gt,
            annotation_gt=artifacts.annotation_gt,
            image=artifacts.image,
            image_id="img0",
            trace_payload=artifacts.trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(dataset.selected_query_id),
            prompt_variants=artifacts.prompt_variants,
        )


__all__ = [
    "INTERNAL_OUTPUT_ONE_QUERY_ID",
    "INTERNAL_OUTPUT_ZERO_QUERY_ID",
    "SCENE_ID",
    "SUPPORTED_GATE_TYPES",
    "SUPPORTED_QUERY_IDS",
    "SymbolicLogicGateInternalOutputCountTask",
    "TASK_ID",
]
