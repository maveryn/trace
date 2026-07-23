"""Count gates of a requested type in one visible logic-gate circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.common import get_int_param as _get_int_param
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.visual_defaults import load_symbolic_noise_defaults

from ._lifecycle import build_single_circuit_count_artifacts
from .shared.rules import build_expression_tree_circuit
from .shared.state import LogicCircuitSpec, SCENE_ID, SCENE_VARIANTS, SUPPORTED_GATE_TYPES


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__logic_gate_circuit__gate_type_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
INTERNAL_QUERY_ID = "gate_type_count"
TASK_PROMPT_KEY = "logic_gate_gate_type_count"
COUNT_INPUT_LABELS: tuple[str, ...] = ("a", "b", "c", "d", "e")

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.20)


@dataclass(frozen=True)
class _GateTypeCountDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    target_gate_type: str
    answer_value: int
    target_answer_support: tuple[int, ...]
    target_answer_probabilities: dict[str, float]
    circuit: LogicCircuitSpec
    annotation_item_ids: tuple[str, ...]
    metadata: dict[str, Any]


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


def _target_gate_type_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[str, ...]:
    raw_support = params.get("target_gate_type_support", gen_defaults.get("target_gate_type_support", SUPPORTED_GATE_TYPES))
    support = tuple(str(item).upper() for item in raw_support)
    if not support:
        raise ValueError("target_gate_type_support cannot be empty")
    unsupported = [gate_type for gate_type in support if gate_type not in SUPPORTED_GATE_TYPES]
    if unsupported:
        raise ValueError(f"unsupported target gate type(s): {unsupported}")
    return support


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


def _sample_gate_types(
    rng: Any,
    *,
    target_gate_type: str,
    target_count: int,
    gate_count: int,
) -> tuple[str, ...]:
    non_target_pool = tuple(gate_type for gate_type in SUPPORTED_GATE_TYPES if str(gate_type) != str(target_gate_type))
    gate_types = [str(target_gate_type) for _ in range(int(target_count))]
    gate_types.extend(str(rng.choice(non_target_pool)) for _ in range(int(gate_count) - int(target_count)))
    rng.shuffle(gate_types)
    return tuple(str(gate_type) for gate_type in gate_types)


def _build_exact_gate_type_circuit(
    rng: Any,
    *,
    target_gate_type: str,
    target_count: int,
    gate_count: int,
) -> LogicCircuitSpec:
    """Build a fanout-free expression tree with an exact target-gate count."""

    gate_types = _sample_gate_types(
        rng,
        target_gate_type=str(target_gate_type),
        target_count=int(target_count),
        gate_count=int(gate_count),
    )
    return build_expression_tree_circuit(
        rng,
        item_id="source_circuit",
        label="",
        input_labels=COUNT_INPUT_LABELS,
        gate_types=gate_types,
        input_id_prefix="count_in",
        gate_id_prefix="count_gate",
        signal_id_prefix="count_sig",
        role="source_circuit",
    )


def _build_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> _GateTypeCountDataset:
    """Construct one circuit with an exact count of the sampled target gate type."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    gate_count = _get_int_param(params, gen_defaults, "gate_count", 4)
    if int(gate_count) != 4:
        raise ValueError("logic-gate gate-type-count task currently requires exactly four gates")
    target_support = _target_gate_type_support(params, gen_defaults)
    target_gate_type = str(params.get("target_gate_type", rng.choice(target_support))).upper()
    if target_gate_type not in target_support:
        raise ValueError(f"target_gate_type must be one of {target_support}")
    answer_value, answer_support, answer_probabilities = _resolve_target_answer(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        gate_count=int(gate_count),
    )
    circuit = _build_exact_gate_type_circuit(
        rng,
        target_gate_type=str(target_gate_type),
        target_count=int(answer_value),
        gate_count=int(gate_count),
    )
    matching_gate_ids = tuple(
        str(gate.item_id)
        for gate in circuit.gates
        if str(gate.gate_type).upper() == str(target_gate_type)
    )
    if len(matching_gate_ids) != int(answer_value):
        raise RuntimeError("logic-gate gate-type-count construction produced inconsistent target count")
    gate_type_counts = {
        str(gate_type): sum(1 for gate in circuit.gates if str(gate.gate_type).upper() == str(gate_type))
        for gate_type in SUPPORTED_GATE_TYPES
    }
    return _GateTypeCountDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        target_gate_type=str(target_gate_type),
        answer_value=int(answer_value),
        target_answer_support=tuple(int(value) for value in answer_support),
        target_answer_probabilities=dict(answer_probabilities),
        circuit=circuit,
        annotation_item_ids=tuple(matching_gate_ids),
        metadata={
            "gate_count": int(gate_count),
            "target_gate_type": str(target_gate_type),
            "target_gate_type_support": list(target_support),
            "gate_type_counts": dict(gate_type_counts),
            "matching_gate_ids": [str(item_id) for item_id in matching_gate_ids],
            "target_answer_probabilities": dict(answer_probabilities),
            "supported_gate_types": list(SUPPORTED_GATE_TYPES),
        },
    )


@register_task
class SymbolicLogicGateGateTypeCountTask:
    """Count visible standard gate symbols whose type matches a requested gate type."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own exact-count sampling, gate annotation binding, prompt slots, and trace assembly."""

        scene_variant, scene_variant_probabilities = _resolve_scene_variant(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        last_error: Exception | None = None
        dataset: _GateTypeCountDataset | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                dataset = _build_dataset(
                    instance_seed=int(instance_seed) + int(attempt_index),
                    scene_variant=str(scene_variant),
                    scene_variant_probabilities=scene_variant_probabilities,
                    params=params,
                    gen_defaults=_GEN_DEFAULTS,
                )
                break
            except Exception as exc:
                last_error = exc
        if dataset is None:
            raise RuntimeError(f"failed to generate logic-gate instance for {TASK_ID}") from last_error

        query_params = {
            "query_id": SINGLE_QUERY_ID,
            "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
            "internal_query_id": INTERNAL_QUERY_ID,
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "target_gate_type": str(dataset.target_gate_type),
            "target_answer_support": [int(value) for value in dataset.target_answer_support],
        }
        artifacts = build_single_circuit_count_artifacts(
            task_id=TASK_ID,
            domain=DOMAIN,
            prompt_defaults=_PROMPT_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            circuit=dataset.circuit,
            annotation_item_ids=dataset.annotation_item_ids,
            answer_value=int(dataset.answer_value),
            scene_variant=str(dataset.scene_variant),
            prompt_key=TASK_PROMPT_KEY,
            branch_key=INTERNAL_QUERY_ID,
            annotation_hint_key=f"annotation_hint_{INTERNAL_QUERY_ID}",
            answer_hint_key=f"answer_hint_{INTERNAL_QUERY_ID}",
            json_example_key=f"json_example_{INTERNAL_QUERY_ID}",
            json_example_answer_only_key=f"json_example_answer_only_{INTERNAL_QUERY_ID}",
            prompt_extra_slots={"target_gate_type": str(dataset.target_gate_type)},
            query_payload=query_params,
            execution_extra={"question_format": INTERNAL_QUERY_ID},
            logic_gate_metadata=dataset.metadata,
            public_query_value=SINGLE_QUERY_ID,
            show_fixed_input_values=False,
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
            query_id=SINGLE_QUERY_ID,
            prompt_variants=artifacts.prompt_variants,
        )


__all__ = [
    "INTERNAL_QUERY_ID",
    "SCENE_ID",
    "SUPPORTED_GATE_TYPES",
    "SUPPORTED_QUERY_IDS",
    "SymbolicLogicGateGateTypeCountTask",
    "TASK_ID",
]
