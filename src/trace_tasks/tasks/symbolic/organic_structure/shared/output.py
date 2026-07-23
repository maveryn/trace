"""Trace serialization helpers for organic-structure scenes."""

from __future__ import annotations

from typing import Any, Mapping

from .state import OrganicStructureSpec


def atom_trace_records(spec: OrganicStructureSpec, *, branch_item_ids: set[str] | None = None, degrees: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """Serialize implicit line-angle atoms for execution traces."""

    branch_ids = set(branch_item_ids or set())
    degree_map = {str(key): int(value) for key, value in dict(degrees or {}).items()}
    records: list[dict[str, Any]] = []
    for atom in spec.atoms:
        record: dict[str, Any] = {
            "item_id": atom.item_id,
            "element": atom.element,
            "implicit": bool(atom.implicit),
            "x": round(float(atom.x), 6),
            "y": round(float(atom.y), 6),
        }
        if degree_map:
            record["degree"] = int(degree_map[str(atom.item_id)])
            record["is_branch_point"] = str(atom.item_id) in branch_ids
        records.append(record)
    return records


def bond_trace_records(spec: OrganicStructureSpec) -> list[dict[str, Any]]:
    """Serialize organic bonds for execution traces."""

    return [
        {
            "item_id": bond.item_id,
            "from_vertex": int(bond.atom_a),
            "to_vertex": int(bond.atom_b),
            "from_atom": int(bond.atom_a),
            "to_atom": int(bond.atom_b),
            "bond_order": bond.order,
            "bond_role": bond.role,
            "ring_index": None if bond.ring_index is None else int(bond.ring_index),
        }
        for bond in spec.bonds
    ]


def text_label_trace_records(spec: OrganicStructureSpec) -> list[dict[str, Any]]:
    """Serialize visible non-answer text labels in the structure."""

    return [
        {
            "item_id": label.item_id,
            "text": label.text,
            "x": round(float(label.x), 6),
            "y": round(float(label.y), 6),
            "role": label.role,
            "anchor_atom": None if label.anchor_atom is None else int(label.anchor_atom),
        }
        for label in spec.text_labels
    ]


def ring_trace_records(spec: OrganicStructureSpec, *, target_ring_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Serialize organic rings for execution traces."""

    selected = set(target_ring_ids or set())
    return [
        {
            "item_id": f"ring_{ring_index + 1:02d}",
            "ring_size": int(len(ring_atoms)),
            "atom_ids": [str(spec.atoms[int(idx)].item_id) for idx in ring_atoms],
            "atom_indices": [int(idx) for idx in ring_atoms],
            "is_target_ring": f"ring_{ring_index + 1:02d}" in selected,
        }
        for ring_index, ring_atoms in enumerate(spec.ring_atom_sets)
    ]


def atom_degrees(spec: OrganicStructureSpec) -> dict[str, int]:
    """Return topological degree for each visible line-angle atom."""

    degree_map = {str(atom.item_id): 0 for atom in spec.atoms}
    for bond in spec.bonds:
        degree_map[str(spec.atoms[int(bond.atom_a)].item_id)] += 1
        degree_map[str(spec.atoms[int(bond.atom_b)].item_id)] += 1
    return degree_map


__all__ = [
    "atom_degrees",
    "atom_trace_records",
    "bond_trace_records",
    "ring_trace_records",
    "text_label_trace_records",
]
