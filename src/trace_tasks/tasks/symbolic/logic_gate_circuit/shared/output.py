"""Trace serialization helpers for logic-gate circuit scenes."""

from __future__ import annotations

from typing import Any

from .state import CandidateAssignmentSpec, LogicCircuitSpec


def circuit_trace(circuit: LogicCircuitSpec) -> dict[str, Any]:
    """Serialize one visible circuit grammar into trace metadata."""

    return {
        "item_id": str(circuit.item_id),
        "label": str(circuit.label),
        "role": str(circuit.role),
        "inputs": [
            {
                "item_id": str(input_spec.item_id),
                "label": str(input_spec.label),
                "value": None if input_spec.value is None else int(input_spec.value),
            }
            for input_spec in circuit.inputs
        ],
        "gates": [
            {
                "item_id": str(gate.item_id),
                "gate_type": str(gate.gate_type),
                "input_signal_ids": [str(signal_id) for signal_id in gate.input_signal_ids],
                "output_signal_id": str(gate.output_signal_id),
            }
            for gate in circuit.gates
        ],
        "output_signal_id": str(circuit.output_signal_id),
        "output_value": None if circuit.output_value is None else int(circuit.output_value),
    }


def assignment_trace(candidate: CandidateAssignmentSpec) -> dict[str, Any]:
    """Serialize one visible assignment row into trace metadata."""

    return {
        "item_id": str(candidate.item_id),
        "label": str(candidate.label),
        "values": {str(key): int(value) for key, value in candidate.values.items()},
        "output_value": int(candidate.output_value),
        "is_correct": bool(candidate.is_correct),
    }
