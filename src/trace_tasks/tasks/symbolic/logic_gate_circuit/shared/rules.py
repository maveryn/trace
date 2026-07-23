"""Boolean logic-gate evaluation and neutral circuit construction helpers."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

from .....core.sampling import uniform_choice

from .state import INPUT_LABELS, LogicCircuitSpec, LogicGateSpec, LogicInputSpec, SUPPORTED_GATE_TYPES


def evaluate_logic_gate(gate_type: str, input_values: Sequence[int]) -> int:
    """Evaluate one supported Boolean gate over 0/1 input values."""

    gate = str(gate_type).upper()
    values = tuple(1 if int(value) else 0 for value in input_values)
    if gate not in SUPPORTED_GATE_TYPES:
        raise ValueError(f"unsupported logic gate type: {gate_type}")
    if gate == "NOT":
        if len(values) != 1:
            raise ValueError("NOT gate requires exactly one input")
        return 0 if values[0] else 1
    if len(values) != 2:
        raise ValueError(f"{gate} gate requires exactly two inputs")
    a, b = int(values[0]), int(values[1])
    if gate == "AND":
        return 1 if a and b else 0
    if gate == "OR":
        return 1 if a or b else 0
    if gate == "XOR":
        return 1 if a != b else 0
    if gate == "NAND":
        return 0 if a and b else 1
    if gate == "NOR":
        return 0 if a or b else 1
    raise AssertionError(f"unhandled logic gate type: {gate}")


def evaluate_logic_circuit(circuit: LogicCircuitSpec, assignment: Mapping[str, int] | None = None) -> int:
    """Evaluate a topologically ordered circuit spec."""

    return int(evaluate_logic_circuit_trace(circuit, assignment=assignment)["final_output"])


def evaluate_logic_circuit_trace(circuit: LogicCircuitSpec, assignment: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Evaluate a circuit and expose signal plus per-gate output values."""

    assignment_values = {str(key): int(value) for key, value in dict(assignment or {}).items()}
    signal_values: dict[str, int] = {}
    for input_spec in circuit.inputs:
        if input_spec.value is None:
            if str(input_spec.label) not in assignment_values:
                raise ValueError(f"missing assignment value for input {input_spec.label!r}")
            value = int(assignment_values[str(input_spec.label)])
        else:
            value = int(input_spec.value)
        signal_values[str(input_spec.item_id)] = 1 if value else 0
    gate_outputs: dict[str, int] = {}
    for gate in circuit.gates:
        inputs = [signal_values[str(signal_id)] for signal_id in gate.input_signal_ids]
        output_value = evaluate_logic_gate(str(gate.gate_type), inputs)
        signal_values[str(gate.output_signal_id)] = int(output_value)
        gate_outputs[str(gate.item_id)] = int(output_value)
    return {
        "final_output": int(signal_values[str(circuit.output_signal_id)]),
        "signal_values": {str(key): int(value) for key, value in signal_values.items()},
        "gate_outputs": {str(key): int(value) for key, value in gate_outputs.items()},
    }


def build_expression_tree_circuit(
    rng: Any,
    *,
    item_id: str,
    label: str,
    input_labels: Sequence[str],
    gate_types: Sequence[str],
    input_id_prefix: str,
    gate_id_prefix: str,
    signal_id_prefix: str,
    role: str,
) -> LogicCircuitSpec:
    """Build a planar fanout-free expression tree from an explicit gate-type sequence.

    The signal pool is kept in visual top-to-bottom order. Binary gates only
    combine adjacent signals and replace that adjacent pair in-place. This
    preserves contiguous input spans for every subtree, which lets the renderer
    draw circuit diagrams without crossing unrelated wires.
    """

    normalized_gate_types = tuple(str(gate_type).upper() for gate_type in gate_types)
    if any(gate_type not in SUPPORTED_GATE_TYPES for gate_type in normalized_gate_types):
        raise ValueError("gate_types contains unsupported logic gate labels")
    binary_count = sum(1 for gate_type in normalized_gate_types if str(gate_type) != "NOT")
    required_input_count = max(1, int(binary_count) + 1)
    labels = tuple(str(label) for label in input_labels)
    if len(labels) < int(required_input_count):
        raise ValueError("not enough input labels for the requested binary gate count")

    inputs = tuple(
        LogicInputSpec(
            item_id=f"{input_id_prefix}_{labels[index]}",
            label=str(labels[index]),
            value=int(rng.randrange(2)),
        )
        for index in range(int(required_input_count))
    )
    pool = [str(input_spec.item_id) for input_spec in inputs]
    gates: list[LogicGateSpec] = []

    for gate_type in normalized_gate_types:
        output_signal_id = f"{signal_id_prefix}_{len(gates) + 1}"
        if str(gate_type) == "NOT":
            source_index = int(uniform_choice(rng, tuple(range(len(pool))), sort_keys=True))
            input_signal_ids = (str(pool[source_index]),)
            pool[source_index] = str(output_signal_id)
        else:
            if len(pool) < 2:
                raise RuntimeError("binary gate construction reached fewer than two available signals")
            left_index = int(uniform_choice(rng, tuple(range(len(pool) - 1)), sort_keys=True))
            left_signal = str(pool[left_index])
            right_signal = str(pool[left_index + 1])
            input_signal_ids = (left_signal, right_signal)
            pool[left_index : left_index + 2] = [str(output_signal_id)]
        gates.append(
            LogicGateSpec(
                item_id=f"{gate_id_prefix}_{len(gates) + 1}",
                gate_type=str(gate_type),
                input_signal_ids=tuple(input_signal_ids),
                output_signal_id=str(output_signal_id),
            )
        )

    if len(pool) != 1:
        raise RuntimeError("expression-tree construction did not end with one output signal")
    circuit = LogicCircuitSpec(
        item_id=str(item_id),
        label=str(label),
        inputs=tuple(inputs),
        gates=tuple(gates),
        output_signal_id=str(pool[0]),
        output_value=None,
        role=str(role),
    )
    return LogicCircuitSpec(
        item_id=str(circuit.item_id),
        label=str(circuit.label),
        inputs=tuple(circuit.inputs),
        gates=tuple(circuit.gates),
        output_signal_id=str(circuit.output_signal_id),
        output_value=int(evaluate_logic_circuit(circuit)),
        role=str(circuit.role),
    )


def gate_arity(gate_type: str) -> int:
    """Return the supported input arity for one gate label."""

    return 1 if str(gate_type).upper() == "NOT" else 2


def output_dependency_signal_ids(circuit: LogicCircuitSpec) -> set[str]:
    """Return signal ids that contribute to the final output signal."""

    dependencies: dict[str, set[str]] = {str(input_spec.item_id): {str(input_spec.item_id)} for input_spec in circuit.inputs}
    for gate in circuit.gates:
        gate_dependencies: set[str] = set()
        for signal_id in gate.input_signal_ids:
            gate_dependencies.update(dependencies[str(signal_id)])
        gate_dependencies.add(str(gate.output_signal_id))
        dependencies[str(gate.output_signal_id)] = set(gate_dependencies)
    return set(dependencies[str(circuit.output_signal_id)])


def sample_random_circuit(
    rng: Any,
    *,
    item_id: str,
    label: str,
    input_count: int,
    gate_count: int,
) -> LogicCircuitSpec:
    """Sample one planar fanout-free expression-tree circuit and compute its output."""

    if int(input_count) < 1:
        raise ValueError("logic-gate circuits require at least one input")
    minimum_gate_count = max(0, int(input_count) - 1)
    if int(gate_count) < int(minimum_gate_count):
        raise ValueError("fanout-free logic-gate circuits need at least input_count - 1 gates")

    inputs = tuple(
        LogicInputSpec(
            item_id=f"{item_id}_in_{INPUT_LABELS[index]}",
            label=str(INPUT_LABELS[index]),
            value=int(rng.randrange(2)),
        )
        for index in range(int(input_count))
    )
    pool = [str(input_spec.item_id) for input_spec in inputs]
    binary_gate_types = tuple(gate for gate in SUPPORTED_GATE_TYPES if str(gate).upper() != "NOT")
    binary_remaining = max(0, len(pool) - 1)
    unary_remaining = int(gate_count) - int(binary_remaining)
    gates: list[LogicGateSpec] = []

    def _append_gate(gate_type: str, selected_inputs: Sequence[str]) -> str:
        output_signal_id = f"{item_id}_sig_{len(gates) + 1}"
        gates.append(
            LogicGateSpec(
                item_id=f"{item_id}_gate_{len(gates) + 1}",
                gate_type=str(gate_type),
                input_signal_ids=tuple(selected_inputs),
                output_signal_id=str(output_signal_id),
            )
        )
        return str(output_signal_id)

    while len(pool) > 1:
        if int(unary_remaining) > 0 and int(rng.randrange(2)) == 1:
            unary_index = int(uniform_choice(rng, tuple(range(len(pool))), sort_keys=True))
            pool[unary_index] = _append_gate("NOT", (str(pool[unary_index]),))
            unary_remaining -= 1

        left_index = int(uniform_choice(rng, tuple(range(len(pool) - 1)), sort_keys=True))
        left_signal = str(pool[left_index])
        right_signal = str(pool[left_index + 1])
        output_signal = _append_gate(str(rng.choice(binary_gate_types)), (left_signal, right_signal))
        pool[left_index : left_index + 2] = [str(output_signal)]

    while int(unary_remaining) > 0:
        pool[0] = _append_gate("NOT", (str(pool[0]),))
        unary_remaining -= 1

    circuit = LogicCircuitSpec(
        item_id=str(item_id),
        label=str(label),
        inputs=tuple(inputs),
        gates=tuple(gates),
        output_signal_id=str(pool[0]),
        output_value=None,
        role="candidate_circuit",
    )
    return LogicCircuitSpec(
        item_id=str(circuit.item_id),
        label=str(circuit.label),
        inputs=tuple(circuit.inputs),
        gates=tuple(circuit.gates),
        output_signal_id=str(circuit.output_signal_id),
        output_value=int(evaluate_logic_circuit(circuit)),
        role=str(circuit.role),
    )


def sample_circuit_with_output(
    rng: Any,
    *,
    item_id: str,
    label: str,
    input_count_min: int,
    input_count_max: int,
    gate_count_min: int,
    gate_count_max: int,
    target_output_value: int,
) -> LogicCircuitSpec:
    """Sample a circuit whose final output equals the requested value."""

    last_circuit: LogicCircuitSpec | None = None
    candidate_sizes = [
        (int(input_count), int(gate_count))
        for input_count in range(int(input_count_min), int(input_count_max) + 1)
        for gate_count in range(int(gate_count_min), int(gate_count_max) + 1)
        if int(gate_count) >= max(0, int(input_count) - 1)
    ]
    if not candidate_sizes:
        raise ValueError("no compatible input/gate-count pairs for fanout-free logic-gate circuits")
    for _attempt in range(1000):
        input_count, gate_count = rng.choice(candidate_sizes)
        circuit = sample_random_circuit(
            rng,
            item_id=str(item_id),
            label=str(label),
            input_count=int(input_count),
            gate_count=int(gate_count),
        )
        last_circuit = circuit
        required_input_ids = {str(input_spec.item_id) for input_spec in circuit.inputs}
        required_gate_signal_ids = {str(gate.output_signal_id) for gate in circuit.gates}
        dependency_signal_ids = output_dependency_signal_ids(circuit)
        if not required_input_ids.issubset(dependency_signal_ids):
            continue
        if not required_gate_signal_ids.issubset(dependency_signal_ids):
            continue
        if int(circuit.output_value or 0) == int(target_output_value):
            return circuit
    if last_circuit is None:
        raise RuntimeError("failed to sample a candidate circuit")
    raise RuntimeError(f"failed to sample circuit with output {target_output_value}")


def assignment_to_values(bits: Sequence[int]) -> dict[str, int]:
    """Map a three-bit assignment tuple to x/y/z values."""

    return {str(label): int(value) for label, value in zip(INPUT_LABELS, bits)}


def all_three_input_assignments() -> tuple[dict[str, int], ...]:
    """Return all x/y/z Boolean assignments."""

    return tuple(assignment_to_values(bits) for bits in product((0, 1), repeat=3))


def build_exact_assignment_circuit(*, target_output_value: int, correct_values: Mapping[str, int]) -> LogicCircuitSpec:
    """Build a simple circuit satisfied by exactly one assignment."""

    inputs = tuple(LogicInputSpec(item_id=f"source_in_{label}", label=str(label), value=None) for label in INPUT_LABELS)
    gates: list[LogicGateSpec] = []
    literal_signals: list[str] = []
    for label in INPUT_LABELS:
        input_signal = f"source_in_{label}"
        if int(correct_values[str(label)]) == 1:
            literal_signals.append(str(input_signal))
        else:
            output_signal = f"source_not_{label}"
            gates.append(
                LogicGateSpec(
                    item_id=f"source_gate_not_{label}",
                    gate_type="NOT",
                    input_signal_ids=(str(input_signal),),
                    output_signal_id=str(output_signal),
                )
            )
            literal_signals.append(str(output_signal))
    gates.append(
        LogicGateSpec(
            item_id="source_gate_and_1",
            gate_type="AND",
            input_signal_ids=(str(literal_signals[0]), str(literal_signals[1])),
            output_signal_id="source_and_1",
        )
    )
    final_gate_type = "AND" if int(target_output_value) == 1 else "NAND"
    gates.append(
        LogicGateSpec(
            item_id="source_gate_final",
            gate_type=str(final_gate_type),
            input_signal_ids=("source_and_1", str(literal_signals[2])),
            output_signal_id="source_out",
        )
    )
    return LogicCircuitSpec(
        item_id="source_circuit",
        label="",
        inputs=tuple(inputs),
        gates=tuple(gates),
        output_signal_id="source_out",
        output_value=None,
        role="source_circuit",
    )
