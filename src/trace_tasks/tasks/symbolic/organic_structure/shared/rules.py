"""Organic-structure scaffold construction and validation rules."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence, Tuple

from .state import (
    BOND_ORDER_VALUES,
    ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT,
    ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT,
    SUPPORTED_ORGANIC_RING_SIZES,
    OrganicAtom,
    OrganicBond,
    OrganicConstraintReport,
    OrganicStructureSpec,
    OrganicTextLabel,
)


def _regular_ring_vertices(cx: float, cy: float, radius: float, sides: int, phase: float) -> Tuple[Tuple[float, float], ...]:
    return tuple(
        (
            float(cx) + float(radius) * math.cos(float(phase) + 2.0 * math.pi * idx / int(sides)),
            float(cy) + float(radius) * math.sin(float(phase) + 2.0 * math.pi * idx / int(sides)),
        )
        for idx in range(int(sides))
    )


def _new_atom(
    atoms: list[OrganicAtom],
    x: float,
    y: float,
    *,
    element: str = "C",
    implicit: bool = True,
) -> int:
    atom_index = len(atoms)
    atoms.append(
        OrganicAtom(
            item_id=f"atom_{atom_index:02d}",
            x=float(x),
            y=float(y),
            element=str(element),
            implicit=bool(implicit),
        )
    )
    return atom_index


def _new_text_label(
    labels: list[OrganicTextLabel],
    text: str,
    x: float,
    y: float,
    *,
    role: str = "substituent_label",
    anchor_atom: int | None = None,
) -> str:
    label_id = f"label_{len(labels) + 1:02d}"
    labels.append(
        OrganicTextLabel(
            item_id=label_id,
            text=str(text),
            x=float(x),
            y=float(y),
            role=str(role),
            anchor_atom=None if anchor_atom is None else int(anchor_atom),
        )
    )
    return label_id


def _new_bond(
    bonds: list[OrganicBond],
    atom_a: int,
    atom_b: int,
    order: str,
    *,
    role: str = "backbone",
    ring_index: int | None = None,
) -> str:
    bond_id = f"bond_{len(bonds) + 1:02d}"
    bonds.append(
        OrganicBond(
            item_id=bond_id,
            atom_a=int(atom_a),
            atom_b=int(atom_b),
            order=str(order),
            role=str(role),
            ring_index=ring_index,
        )
    )
    return bond_id


def _bond_key(atom_a: int, atom_b: int) -> Tuple[int, int]:
    left, right = sorted((int(atom_a), int(atom_b)))
    return int(left), int(right)


def _find_bond_index(bonds: Sequence[OrganicBond], atom_a: int, atom_b: int) -> int | None:
    target = _bond_key(atom_a, atom_b)
    for bond_index, bond in enumerate(bonds):
        if _bond_key(bond.atom_a, bond.atom_b) == target:
            return int(bond_index)
    return None


def _new_bond_if_missing(
    bonds: list[OrganicBond],
    atom_a: int,
    atom_b: int,
    order: str,
    *,
    role: str = "backbone",
    ring_index: int | None = None,
) -> int:
    existing = _find_bond_index(bonds, atom_a, atom_b)
    if existing is not None:
        return int(existing)
    _new_bond(bonds, atom_a, atom_b, order, role=role, ring_index=ring_index)
    return len(bonds) - 1


def _set_bond_orders(
    bonds: Sequence[OrganicBond],
    selected_indices: set[int],
    *,
    selected_order: str,
) -> Tuple[OrganicBond, ...]:
    return tuple(
        replace(bond, order=str(selected_order)) if int(index) in selected_indices else bond
        for index, bond in enumerate(bonds)
    )


def _target_bond_ids(spec: OrganicStructureSpec, target_bond_order: str) -> Tuple[str, ...]:
    return tuple(str(bond.item_id) for bond in spec.bonds if str(bond.order) == str(target_bond_order))


def _atom_degrees(spec: OrganicStructureSpec) -> Tuple[int, ...]:
    degrees = [0 for _ in spec.atoms]
    for bond in spec.bonds:
        degrees[int(bond.atom_a)] += 1
        degrees[int(bond.atom_b)] += 1
    return tuple(int(value) for value in degrees)


def _degree_counts_from_bonds(bonds: Sequence[OrganicBond], atom_count: int) -> Tuple[int, ...]:
    degrees = [0 for _ in range(int(atom_count))]
    for bond in bonds:
        degrees[int(bond.atom_a)] += 1
        degrees[int(bond.atom_b)] += 1
    return tuple(int(value) for value in degrees)


def organic_branch_point_atom_indices(spec: OrganicStructureSpec) -> Tuple[int, ...]:
    """Return skeletal vertices where three or more drawn bonds meet."""

    return tuple(idx for idx, degree in enumerate(_atom_degrees(spec)) if int(degree) >= 3)


def organic_ring_item_ids(spec: OrganicStructureSpec, target_ring_size: int) -> Tuple[str, ...]:
    """Return ring item ids whose rendered polygon has the requested vertex count."""

    return tuple(
        f"ring_{ring_index + 1:02d}"
        for ring_index, ring_atoms in enumerate(spec.ring_atom_sets)
        if len(ring_atoms) == int(target_ring_size)
    )


def _find_atom_at(atoms: Sequence[OrganicAtom], x: float, y: float) -> int | None:
    for atom_index, atom in enumerate(atoms):
        if abs(float(atom.x) - float(x)) < 1e-6 and abs(float(atom.y) - float(y)) < 1e-6:
            return int(atom_index)
    return None


def _add_ring_from_vertices(
    atoms: list[OrganicAtom],
    bonds: list[OrganicBond],
    ring_atom_sets: list[Tuple[int, ...]],
    vertices: Sequence[Tuple[float, float]],
    *,
    role: str,
) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
    ring_index = len(ring_atom_sets)
    ring_atoms: list[int] = []
    for x, y in vertices:
        existing = _find_atom_at(atoms, x, y)
        ring_atoms.append(existing if existing is not None else _new_atom(atoms, x, y))

    ring_bonds: list[int] = []
    for edge_index in range(len(ring_atoms)):
        ring_bonds.append(
            _new_bond_if_missing(
                bonds,
                ring_atoms[edge_index],
                ring_atoms[(edge_index + 1) % len(ring_atoms)],
                "single",
                role=role,
                ring_index=ring_index,
            )
        )
    ring_atom_sets.append(tuple(ring_atoms))
    return tuple(ring_atoms), tuple(ring_bonds)


def _add_hex_ring(
    atoms: list[OrganicAtom],
    bonds: list[OrganicBond],
    ring_atom_sets: list[Tuple[int, ...]],
    *,
    cx: float,
    cy: float,
    radius: float = 1.0,
    role: str = "aromatic_ring",
) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
    return _add_ring_from_vertices(
        atoms,
        bonds,
        ring_atom_sets,
        _regular_ring_vertices(float(cx), float(cy), float(radius), 6, math.pi / 6.0),
        role=role,
    )


def _add_regular_ring(
    atoms: list[OrganicAtom],
    bonds: list[OrganicBond],
    ring_atom_sets: list[Tuple[int, ...]],
    *,
    cx: float,
    cy: float,
    sides: int,
    radius: float = 1.0,
    role: str = "ring",
) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
    phase = math.pi / 6.0 if int(sides) == 6 else -math.pi / 2.0
    return _add_ring_from_vertices(
        atoms,
        bonds,
        ring_atom_sets,
        _regular_ring_vertices(float(cx), float(cy), float(radius), int(sides), phase),
        role=role,
    )


def _set_atom_label(atoms: list[OrganicAtom], atom_index: int, element: str) -> None:
    atom = atoms[int(atom_index)]
    atoms[int(atom_index)] = replace(atom, element=str(element), implicit=False)


def _select_nonadjacent_bond_indices(
    rng: Any,
    bonds: Sequence[OrganicBond],
    candidates: Sequence[int],
    count: int,
) -> set[int]:
    for _attempt in range(24):
        shuffled = [int(index) for index in candidates]
        rng.shuffle(shuffled)
        selected: list[int] = []
        used_atoms: set[int] = set()
        for bond_index in shuffled:
            bond = bonds[int(bond_index)]
            if int(bond.atom_a) in used_atoms or int(bond.atom_b) in used_atoms:
                continue
            selected.append(int(bond_index))
            used_atoms.add(int(bond.atom_a))
            used_atoms.add(int(bond.atom_b))
            if len(selected) == int(count):
                return set(selected)
    raise ValueError(f"could not select {int(count)} nonadjacent bonds")


def _add_labeled_substituent(
    atoms: list[OrganicAtom],
    bonds: list[OrganicBond],
    labels: list[OrganicTextLabel],
    anchor: int,
    *,
    angle: float,
    text: str,
    length: float = 0.92,
) -> int:
    atom = atoms[int(anchor)]
    tail = _new_atom(
        atoms,
        float(atom.x) + float(length) * math.cos(float(angle)),
        float(atom.y) + float(length) * math.sin(float(angle)),
    )
    _new_bond(bonds, int(anchor), int(tail), "single", role="labeled_substituent")
    _new_text_label(
        labels,
        str(text),
        float(atoms[tail].x) + 0.22 * math.cos(float(angle)),
        float(atoms[tail].y) + 0.22 * math.sin(float(angle)),
        anchor_atom=int(tail),
    )
    return int(tail)


def _add_carbonyl_substituent(
    atoms: list[OrganicAtom],
    bonds: list[OrganicBond],
    anchor: int,
    *,
    angle: float,
) -> None:
    atom = atoms[int(anchor)]
    carbon = _new_atom(
        atoms,
        float(atom.x) + 0.78 * math.cos(float(angle)),
        float(atom.y) + 0.78 * math.sin(float(angle)),
    )
    oxygen = _new_atom(
        atoms,
        float(atoms[carbon].x) + 0.68 * math.cos(float(angle)),
        float(atoms[carbon].y) + 0.68 * math.sin(float(angle)),
        element="O",
        implicit=False,
    )
    _new_bond(bonds, int(anchor), int(carbon), "single", role="carbonyl_substituent")
    _new_bond(bonds, int(carbon), int(oxygen), "double", role="carbonyl_substituent")


def _ring_center(atoms: Sequence[OrganicAtom], ring_atoms: Sequence[int]) -> Tuple[float, float]:
    xs = [float(atoms[int(idx)].x) for idx in ring_atoms]
    ys = [float(atoms[int(idx)].y) for idx in ring_atoms]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _center_layout(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    mean_x = sum(float(x) for x, _y in points) / max(1, len(points))
    mean_y = sum(float(y) for _x, y in points) / max(1, len(points))
    return tuple((float(x) - mean_x, float(y) - mean_y) for x, y in points)


def _hex_fused_ring_centers(rng: Any, ring_count: int) -> Tuple[Tuple[float, float], ...]:
    """Return non-linear fused hex-ring centers on the regular hex lattice."""

    count = int(ring_count)
    step_x = math.sqrt(3.0)
    step_y = 1.5
    half_x = step_x / 2.0
    templates: dict[int, Tuple[Tuple[Tuple[float, float], ...], ...]] = {
        2: (
            ((0.0, 0.0), (step_x, 0.0)),
            ((0.0, 0.0), (half_x, step_y)),
        ),
        3: (
            ((0.0, 0.0), (step_x, 0.0), (half_x, step_y)),
            ((0.0, 0.0), (half_x, step_y), (half_x, -step_y)),
            ((0.0, 0.0), (step_x, 0.0), (step_x + half_x, step_y)),
        ),
        4: (
            ((0.0, 0.0), (step_x, 0.0), (half_x, step_y), (step_x + half_x, step_y)),
            ((0.0, 0.0), (step_x, 0.0), (half_x, -step_y), (step_x + half_x, -step_y)),
            ((0.0, 0.0), (half_x, step_y), (half_x, -step_y), (step_x, 0.0)),
        ),
        5: (
            ((0.0, 0.0), (step_x, 0.0), (2.0 * step_x, 0.0), (half_x, step_y), (step_x + half_x, step_y)),
            ((0.0, 0.0), (step_x, 0.0), (half_x, step_y), (half_x, -step_y), (step_x + half_x, -step_y)),
            ((0.0, 0.0), (step_x, 0.0), (half_x, step_y), (step_x + half_x, step_y), (step_x + half_x, -step_y)),
        ),
    }
    return _center_layout(rng.choice(templates.get(count, templates[5])))


def _linked_ring_cluster_layout(
    rng: Any,
    ring_count: int,
) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[int, int], ...], str]:
    """Return chemistry-like central-ring layouts with terminal attached rings."""

    count = int(ring_count)
    spacing = 3.18
    direction_sets: dict[int, Tuple[Tuple[float, ...], ...]] = {
        2: ((0.0,), (math.pi / 6.0,)),
        3: ((0.0, 2.0 * math.pi / 3.0), (math.pi / 3.0, -math.pi / 3.0)),
        4: ((0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0), (math.pi / 6.0, 5.0 * math.pi / 6.0, -math.pi / 2.0)),
        5: ((0.0, math.pi / 3.0, math.pi, -math.pi / 3.0), (math.pi / 6.0, 5.0 * math.pi / 6.0, -5.0 * math.pi / 6.0, -math.pi / 6.0)),
    }
    directions = rng.choice(direction_sets.get(count, direction_sets[5]))
    centers = [(0.0, 0.0)] + [(spacing * math.cos(angle), spacing * math.sin(angle)) for angle in directions]
    links = tuple((0, index) for index in range(1, len(centers)))
    layout_id = f"radial_{max(1, len(centers) - 1)}_arm"
    return _center_layout(centers), tuple((int(a), int(b)) for a, b in links), str(layout_id)


def _ring_anchor_toward(
    atoms: Sequence[OrganicAtom],
    ring_atoms: Sequence[int],
    center: Tuple[float, float],
    target: Tuple[float, float],
    *,
    excluded_atoms: set[int] | None = None,
) -> int:
    dx = float(target[0]) - float(center[0])
    dy = float(target[1]) - float(center[1])
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    excluded = set() if excluded_atoms is None else {int(value) for value in excluded_atoms}
    candidates = [int(atom_index) for atom_index in ring_atoms if int(atom_index) not in excluded]
    if not candidates:
        candidates = [int(atom_index) for atom_index in ring_atoms]
    return max(
        candidates,
        key=lambda atom_index: (
            (float(atoms[int(atom_index)].x) - float(center[0])) * ux
            + (float(atoms[int(atom_index)].y) - float(center[1])) * uy
        ),
    )


def _link_ring_pair(
    atoms: Sequence[OrganicAtom],
    bonds: list[OrganicBond],
    ring_atom_sets: Sequence[Tuple[int, ...]],
    left_ring_index: int,
    right_ring_index: int,
) -> None:
    left_ring = ring_atom_sets[int(left_ring_index)]
    right_ring = ring_atom_sets[int(right_ring_index)]
    left_center = _ring_center(atoms, left_ring)
    right_center = _ring_center(atoms, right_ring)
    left_atom = _ring_anchor_toward(atoms, left_ring, left_center, right_center)
    right_atom = _ring_anchor_toward(atoms, right_ring, right_center, left_center)
    if int(left_atom) != int(right_atom) and _find_bond_index(bonds, int(left_atom), int(right_atom)) is None:
        _new_bond(bonds, int(left_atom), int(right_atom), "single", role="ring_link")


def _available_ring_anchor(
    atoms: Sequence[OrganicAtom],
    bonds: Sequence[OrganicBond],
    ring_atoms: Sequence[int],
    *,
    preferred_angle: float,
) -> int:
    degrees = _degree_counts_from_bonds(bonds, len(atoms))
    center = _ring_center(atoms, ring_atoms)
    ux = math.cos(float(preferred_angle))
    uy = math.sin(float(preferred_angle))
    return max(
        ring_atoms,
        key=lambda atom_index: (
            -int(degrees[int(atom_index)]),
            (float(atoms[int(atom_index)].x) - center[0]) * ux
            + (float(atoms[int(atom_index)].y) - center[1]) * uy,
        ),
    )


def _make_fused_aromatic_scaffold(rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    """Build a fused hex-ring motif with exactly the requested visible double bonds."""

    if str(target_bond_order) != "double":
        raise ValueError("fused aromatic scaffold is only for double-bond targets")
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    labels: list[OrganicTextLabel] = []
    ring_atom_sets: list[Tuple[int, ...]] = []
    ring_count = max(2, min(5, int(answer_count) + rng.choice((0, 1))))
    ring_bond_candidates: list[int] = []
    for ring_index, (cx, cy) in enumerate(_hex_fused_ring_centers(rng, ring_count)):
        ring_atoms, ring_bonds = _add_hex_ring(
            atoms,
            bonds,
            ring_atom_sets,
            cx=cx,
            cy=cy,
            radius=1.0,
            role="fused_aromatic_ring",
        )
        ring_bond_candidates.extend(int(index) for index in ring_bonds)
        if ring_index == 0 and rng.random() < 0.55:
            _set_atom_label(atoms, ring_atoms[1], "N")

    unique_candidates = tuple(dict.fromkeys(ring_bond_candidates))
    selected = _select_nonadjacent_bond_indices(rng, bonds, unique_candidates, int(answer_count))
    final_bonds = list(_set_bond_orders(bonds, selected, selected_order="double"))

    outer_atoms = [idx for idx, degree in enumerate(_degree_counts_from_bonds(final_bonds, len(atoms))) if degree == 2]
    rng.shuffle(outer_atoms)
    label_choices = ("CH3", "OH", "Br")
    for anchor in outer_atoms[: rng.choice((1, 2))]:
        center_x = sum(atom.x for atom in atoms) / len(atoms)
        center_y = sum(atom.y for atom in atoms) / len(atoms)
        angle = math.atan2(float(atoms[anchor].y) - center_y, float(atoms[anchor].x) - center_x)
        _add_labeled_substituent(
            atoms,
            final_bonds,
            labels,
            int(anchor),
            angle=angle,
            text=str(rng.choice(label_choices)),
            length=0.82,
        )

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(final_bonds),
        ring_atom_sets=tuple(ring_atom_sets),
        scaffold_id=f"fused_aromatic_{ring_count}_{answer_count}",
        scaffold_family="fused_aromatic",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="motif_valence_and_line_angle_geometry_v2",
        text_labels=tuple(labels),
    )


def _add_aromatic_distractor_bonds(
    rng: Any,
    bonds: list[OrganicBond],
    ring_bonds: Sequence[int],
    *,
    count: int = 3,
) -> None:
    selected = _select_nonadjacent_bond_indices(rng, bonds, tuple(dict.fromkeys(ring_bonds)), int(count))
    for bond_index, bond in enumerate(tuple(bonds)):
        if int(bond_index) in selected:
            bonds[int(bond_index)] = replace(bond, order="double")


def _make_aryl_polyyne_scaffold(rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    """Build phenyl-ended linear alkyne chains with exactly the requested triple bonds."""

    if str(target_bond_order) != "triple":
        raise ValueError("aryl-polyyne scaffold is only for triple-bond targets")
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    labels: list[OrganicTextLabel] = []
    ring_atom_sets: list[Tuple[int, ...]] = []
    center_y = 0.0
    chain_angle = math.pi / 6.0
    chain_dx = math.cos(chain_angle)
    chain_dy = math.sin(chain_angle)
    link_length = 0.76
    left_ring, left_bonds = _add_hex_ring(
        atoms,
        bonds,
        ring_atom_sets,
        cx=0.0,
        cy=center_y,
        role="phenyl_end_ring",
    )
    _add_aromatic_distractor_bonds(rng, bonds, left_bonds, count=3)
    left_anchor = left_ring[0]

    step = 0.82
    chain_start_x = float(atoms[left_anchor].x) + link_length * chain_dx
    chain_start_y = float(atoms[left_anchor].y) + link_length * chain_dy
    chain_atoms: list[int] = []
    for atom_index in range(int(answer_count) * 2):
        chain_atoms.append(
            _new_atom(
                atoms,
                chain_start_x + step * atom_index * chain_dx,
                chain_start_y + step * atom_index * chain_dy,
            )
        )
    _new_bond(bonds, int(left_anchor), chain_atoms[0], "single", role="aryl_polyyne_link")
    for triple_index in range(int(answer_count)):
        left = chain_atoms[triple_index * 2]
        right = chain_atoms[triple_index * 2 + 1]
        _new_bond(bonds, left, right, "triple", role="polyyne_backbone")
        if right != chain_atoms[-1]:
            _new_bond(bonds, right, chain_atoms[triple_index * 2 + 2], "single", role="polyyne_backbone")

    right_center_x = float(atoms[chain_atoms[-1]].x) + (link_length + 1.0) * chain_dx
    right_center_y = float(atoms[chain_atoms[-1]].y) + (link_length + 1.0) * chain_dy
    right_ring, right_bonds = _add_hex_ring(
        atoms,
        bonds,
        ring_atom_sets,
        cx=right_center_x,
        cy=right_center_y,
        role="phenyl_end_ring",
    )
    _add_aromatic_distractor_bonds(rng, bonds, right_bonds, count=3)
    right_anchor = right_ring[3]
    _new_bond(bonds, chain_atoms[-1], int(right_anchor), "single", role="aryl_polyyne_link")

    if rng.random() < 0.65:
        _set_atom_label(atoms, right_ring[1], "N")
    left_center = _ring_center(atoms, left_ring)
    left_substituent_anchor = left_ring[3]
    _add_labeled_substituent(
        atoms,
        bonds,
        labels,
        int(left_substituent_anchor),
        angle=math.atan2(
            float(atoms[int(left_substituent_anchor)].y) - left_center[1],
            float(atoms[int(left_substituent_anchor)].x) - left_center[0],
        ),
        text=str(rng.choice(("CH3", "C2H5", "OH"))),
        length=0.82,
    )

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=tuple(ring_atom_sets),
        scaffold_id=f"aryl_polyyne_{answer_count}",
        scaffold_family="aryl_polyyne",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="motif_valence_and_line_angle_geometry_v2",
        text_labels=tuple(labels),
    )


def _make_rich_ring_size_scaffold(rng: Any, *, target_ring_size: int, answer_count: int) -> OrganicStructureSpec:
    """Build branched linked-ring motifs with exact explicit target-ring cardinality."""

    target_ring_size = int(target_ring_size)
    answer_count = int(answer_count)
    if target_ring_size not in SUPPORTED_ORGANIC_RING_SIZES:
        raise ValueError(f"unsupported target ring size: {target_ring_size}")
    if answer_count < 1 or answer_count > ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT:
        raise ValueError(f"rich ring-size scaffold supports counts 1..{ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT}")

    distractor_size = 5 if target_ring_size == 6 else 6
    distractor_count = 1 if answer_count > 0 and answer_count < ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT else 0

    if int(target_ring_size) == 5 and distractor_count:
        ring_sizes = [distractor_size] + [target_ring_size for _ in range(answer_count)]
    else:
        ring_sizes = [target_ring_size] + [target_ring_size for _ in range(answer_count - 1)] + [
            distractor_size for _ in range(distractor_count)
        ]
        if len(ring_sizes) > 2:
            satellites = ring_sizes[1:]
            rng.shuffle(satellites)
            ring_sizes = [ring_sizes[0]] + satellites

    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    labels: list[OrganicTextLabel] = []
    ring_atom_sets: list[Tuple[int, ...]] = []
    centers, links, layout_id = _linked_ring_cluster_layout(rng, len(ring_sizes))
    for ring_index, (ring_size, center) in enumerate(zip(ring_sizes, centers)):
        ring_atoms, ring_bonds = _add_regular_ring(
            atoms,
            bonds,
            ring_atom_sets,
            cx=float(center[0]),
            cy=float(center[1]),
            sides=int(ring_size),
            radius=1.0 if int(ring_size) == 6 else 0.94,
            role="linked_ring_cluster",
        )
        _add_aromatic_distractor_bonds(rng, bonds, ring_bonds, count=3 if int(ring_size) == 6 else 2)
        if int(ring_size) == 5 and rng.random() < 0.8:
            _set_atom_label(atoms, ring_atoms[0], str(rng.choice(("N", "O"))))

    used_central_link_atoms: set[int] = set()
    for left_ring_index, right_ring_index in links:
        if int(left_ring_index) == 0:
            central_ring = ring_atom_sets[0]
            satellite_ring = ring_atom_sets[int(right_ring_index)]
            central_center = _ring_center(atoms, central_ring)
            satellite_center = _ring_center(atoms, satellite_ring)
            central_atom = _ring_anchor_toward(
                atoms,
                central_ring,
                central_center,
                satellite_center,
                excluded_atoms=used_central_link_atoms,
            )
            satellite_atom = _ring_anchor_toward(atoms, satellite_ring, satellite_center, central_center)
            used_central_link_atoms.add(int(central_atom))
            if _find_bond_index(bonds, int(central_atom), int(satellite_atom)) is None:
                _new_bond(bonds, int(central_atom), int(satellite_atom), "single", role="ring_link")
        else:
            _link_ring_pair(atoms, bonds, ring_atom_sets, int(left_ring_index), int(right_ring_index))

    if ring_atom_sets:
        first_ring = ring_atom_sets[0]
        first_center = _ring_center(atoms, first_ring)
        anchor = _available_ring_anchor(atoms, bonds, first_ring, preferred_angle=math.pi)
        angle = math.atan2(float(atoms[anchor].y) - first_center[1], float(atoms[anchor].x) - first_center[0])
        _add_labeled_substituent(atoms, bonds, labels, int(anchor), angle=angle, text=str(rng.choice(("CH3", "OH", "Br"))))
    if ring_atom_sets and rng.random() < 0.7:
        last_ring = ring_atom_sets[-1]
        last_center = _ring_center(atoms, last_ring)
        carbonyl_anchor = _available_ring_anchor(atoms, bonds, last_ring, preferred_angle=0.0)
        carbonyl_angle = math.atan2(
            float(atoms[carbonyl_anchor].y) - last_center[1],
            float(atoms[carbonyl_anchor].x) - last_center[0],
        )
        _add_carbonyl_substituent(atoms, bonds, int(carbonyl_anchor), angle=carbonyl_angle)

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=tuple(ring_atom_sets),
        scaffold_id=f"rich_ring_cluster_{target_ring_size}_{answer_count}_{len(ring_sizes)}_{layout_id}",
        scaffold_family="rich_ring_cluster",
        target_bond_order="not_applicable",
        target_answer_value=int(answer_count),
        constraint_policy="motif_valence_and_line_angle_geometry_v2",
        text_labels=tuple(labels),
    )


def _make_polyene_scaffold(rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    edge_count = int(answer_count) * 2 + rng.choice((1, 2))
    atom_count = edge_count + 1
    for idx in range(atom_count):
        _new_atom(atoms, float(idx) * 0.92, 0.42 if idx % 2 else -0.42)

    target_edge_indices = set(range(0, int(answer_count) * 2, 2))
    for edge_index in range(edge_count):
        order = "double" if edge_index in target_edge_indices else "single"
        _new_bond(bonds, edge_index, edge_index + 1, order, role="polyene_backbone")

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=tuple(),
        scaffold_id=f"polyene_chain_{answer_count}",
        scaffold_family="polyene_chain",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="basic_carbon_valence_and_line_angle_geometry_v1",
    )


def _make_cycloalkene_scaffold(_rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    if int(answer_count) > 3:
        raise ValueError("cycloalkene scaffold supports at most three target double bonds")
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    for x, y in _regular_ring_vertices(0.0, 0.0, 1.08, 6, math.pi / 6):
        _new_atom(atoms, x, y)

    double_edges_by_count = {1: (0,), 2: (0, 3), 3: (0, 2, 4)}
    target_edges = set(double_edges_by_count[int(answer_count)])
    for edge_index in range(6):
        order = "double" if edge_index in target_edges else "single"
        _new_bond(bonds, edge_index, (edge_index + 1) % 6, order, role="ring", ring_index=0)

    anchor = 5
    center_x = sum(atom.x for atom in atoms[:6]) / 6.0
    center_y = sum(atom.y for atom in atoms[:6]) / 6.0
    ax = atoms[anchor].x
    ay = atoms[anchor].y
    angle = math.atan2(ay - center_y, ax - center_x)
    tail = _new_atom(atoms, ax + 0.95 * math.cos(angle), ay + 0.95 * math.sin(angle))
    _new_bond(bonds, anchor, tail, "single", role="side_chain")

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=((0, 1, 2, 3, 4, 5),),
        scaffold_id=f"cycloalkene_ring_{answer_count}",
        scaffold_family="cycloalkene_ring",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="basic_carbon_valence_and_line_angle_geometry_v1",
    )


def _make_polyyne_scaffold(_rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    atom_count = int(answer_count) * 2 + 2
    for idx in range(atom_count):
        _new_atom(atoms, float(idx) * 0.78, 0.0)

    target_edges = set(range(1, int(answer_count) * 2, 2))
    for edge_index in range(atom_count - 1):
        order = "triple" if edge_index in target_edges else "single"
        _new_bond(bonds, edge_index, edge_index + 1, order, role="polyyne_backbone")

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=tuple(),
        scaffold_id=f"polyyne_chain_{answer_count}",
        scaffold_family="polyyne_chain",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="basic_carbon_valence_and_line_angle_geometry_v1",
    )


def _make_alkynyl_ring_scaffold(_rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    if int(answer_count) > 3:
        raise ValueError("alkynyl ring scaffold supports at most three target triple bonds")
    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    ring_points = _regular_ring_vertices(0.0, 0.0, 1.08, 6, math.pi / 6)
    for x, y in ring_points:
        _new_atom(atoms, x, y)

    for edge_index in range(6):
        order = "double" if edge_index in (0, 2, 4) else "single"
        _new_bond(bonds, edge_index, (edge_index + 1) % 6, order, role="aromatic_ring", ring_index=0)

    anchors_by_count = {1: (1,), 2: (1, 4), 3: (1, 3, 5)}
    center_x = sum(atom.x for atom in atoms[:6]) / 6.0
    center_y = sum(atom.y for atom in atoms[:6]) / 6.0
    for anchor in anchors_by_count[int(answer_count)]:
        ax = atoms[anchor].x
        ay = atoms[anchor].y
        angle = math.atan2(ay - center_y, ax - center_x)
        first = _new_atom(atoms, ax + 0.72 * math.cos(angle), ay + 0.72 * math.sin(angle))
        second = _new_atom(atoms, atoms[first].x + 0.94 * math.cos(angle), atoms[first].y + 0.94 * math.sin(angle))
        _new_bond(bonds, anchor, first, "single", role="alkynyl_substituent")
        _new_bond(bonds, first, second, "triple", role="alkynyl_substituent")

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=((0, 1, 2, 3, 4, 5),),
        scaffold_id=f"alkynyl_ring_{answer_count}",
        scaffold_family="alkynyl_ring",
        target_bond_order=str(target_bond_order),
        target_answer_value=int(answer_count),
        constraint_policy="basic_carbon_valence_and_line_angle_geometry_v1",
    )


def _make_separated_ring_size_scaffold(rng: Any, *, target_ring_size: int, answer_count: int) -> OrganicStructureSpec:
    """Build a ring-chain scaffold with exact target ring cardinality.

    Rings are generated as separated pentagons/hexagons and linked by single
    bonds, so ring-size annotation can use one bbox per ring without fused-ring
    ambiguity or crossing bonds.
    """

    target_ring_size = int(target_ring_size)
    answer_count = int(answer_count)
    if target_ring_size not in SUPPORTED_ORGANIC_RING_SIZES:
        raise ValueError(f"unsupported target ring size: {target_ring_size}")
    if answer_count < 1 or answer_count > ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT:
        raise ValueError(f"ring-size scaffold supports counts 1..{ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT}")

    distractor_ring_size = 5 if target_ring_size == 6 else 6
    max_total_rings = ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT
    max_distractors = max(0, max_total_rings - answer_count)
    distractor_count = int(rng.choice(tuple(range(min(2, max_distractors) + 1))))

    ring_sizes = [target_ring_size for _ in range(answer_count)] + [distractor_ring_size for _ in range(distractor_count)]
    rng.shuffle(ring_sizes)

    atoms: list[OrganicAtom] = []
    bonds: list[OrganicBond] = []
    ring_atom_sets: list[Tuple[int, ...]] = []
    ring_centers: list[Tuple[float, float]] = []
    ring_spacing = 3.15
    for ring_index, ring_size in enumerate(ring_sizes):
        cx = (float(ring_index) - (len(ring_sizes) - 1) / 2.0) * ring_spacing
        cy = 0.18 if ring_index % 2 else -0.18
        radius = 1.02 if int(ring_size) == 6 else 0.98
        ring_indices: list[int] = []
        for x, y in _regular_ring_vertices(cx, cy, radius, int(ring_size), 0.0):
            ring_indices.append(_new_atom(atoms, x, y))
        current_ring_index = len(ring_atom_sets)
        for edge_index in range(int(ring_size)):
            _new_bond(
                bonds,
                ring_indices[edge_index],
                ring_indices[(edge_index + 1) % int(ring_size)],
                "single",
                role="ring",
                ring_index=current_ring_index,
            )
        ring_atom_sets.append(tuple(ring_indices))
        ring_centers.append((float(cx), float(cy)))

    for left_ring_index in range(max(0, len(ring_atom_sets) - 1)):
        right_ring_index = left_ring_index + 1
        left_center = ring_centers[left_ring_index]
        right_center = ring_centers[right_ring_index]
        left_atom = max(
            ring_atom_sets[left_ring_index],
            key=lambda atom_index: (
                atoms[int(atom_index)].x - left_center[0],
                -abs(atoms[int(atom_index)].y - left_center[1]),
            ),
        )
        right_atom = min(
            ring_atom_sets[right_ring_index],
            key=lambda atom_index: (
                atoms[int(atom_index)].x - right_center[0],
                abs(atoms[int(atom_index)].y - right_center[1]),
            ),
        )
        _new_bond(bonds, int(left_atom), int(right_atom), "single", role="ring_link")

    return OrganicStructureSpec(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        ring_atom_sets=tuple(ring_atom_sets),
        scaffold_id=f"separated_rings_{target_ring_size}_{answer_count}_{len(ring_sizes)}",
        scaffold_family="separated_ring_chain",
        target_bond_order="not_applicable",
        target_answer_value=int(answer_count),
        constraint_policy="basic_carbon_valence_and_line_angle_geometry_v1",
    )


def build_constrained_organic_structure(rng: Any, *, target_bond_order: str, answer_count: int) -> OrganicStructureSpec:
    """Build a notation-plausible skeletal structure with exact target count."""

    target_bond_order = str(target_bond_order)
    answer_count = int(answer_count)
    if target_bond_order not in ("double", "triple"):
        raise ValueError(f"unsupported target bond order: {target_bond_order!r}")
    if answer_count < 1 or answer_count > ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT:
        raise ValueError(f"organic structures support answer counts 1..{ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT}; got {answer_count}")

    builders: list[Any]
    if target_bond_order == "double":
        builders = [_make_fused_aromatic_scaffold]
    else:
        builders = [_make_aryl_polyyne_scaffold]

    rng.shuffle(builders)
    errors: list[str] = []
    for builder in builders:
        try:
            spec = builder(rng, target_bond_order=target_bond_order, answer_count=answer_count)
            report = validate_organic_structure(spec)
            target_ids = _target_bond_ids(spec, target_bond_order)
            if len(target_ids) != answer_count:
                raise ValueError(f"target count mismatch for {spec.scaffold_id}: {len(target_ids)} != {answer_count}")
            if report.crossing_count:
                raise ValueError(f"crossed bonds in {spec.scaffold_id}")
            return spec
        except ValueError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"no constrained organic scaffold fit the request: {'; '.join(errors)}")


def build_constrained_organic_ring_size_structure(rng: Any, *, target_ring_size: int, answer_count: int) -> OrganicStructureSpec:
    """Build a notation-plausible ring structure with exact target ring count."""

    target_ring_size = int(target_ring_size)
    answer_count = int(answer_count)
    if target_ring_size not in SUPPORTED_ORGANIC_RING_SIZES:
        raise ValueError(f"unsupported target ring size: {target_ring_size}")
    if answer_count < 1 or answer_count > ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT:
        raise ValueError(f"organic ring-size structures support answer counts 1..{ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT}; got {answer_count}")

    errors: list[str] = []
    for _attempt in range(8):
        try:
            spec = _make_rich_ring_size_scaffold(rng, target_ring_size=target_ring_size, answer_count=answer_count)
            report = validate_organic_structure(spec)
            matching_ids = organic_ring_item_ids(spec, target_ring_size)
            if len(matching_ids) != answer_count:
                raise ValueError(f"ring-size count mismatch for {spec.scaffold_id}: {len(matching_ids)} != {answer_count}")
            if report.crossing_count:
                raise ValueError(f"crossed bonds in {spec.scaffold_id}")
            return spec
        except ValueError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"no constrained organic ring-size scaffold fit the request: {'; '.join(errors)}")


def validate_organic_structure(spec: OrganicStructureSpec) -> OrganicConstraintReport:
    """Check the chemistry-style notation constraints used by this scene.

    The validator enforces only the synthetic scene grammar: atom valence,
    simple multiple-bond geometry, supported ring sizes, non-cramped branch
    angles, and no crossed bonds. It does not certify molecule identity.
    """

    atom_count = len(spec.atoms)
    valences = [0 for _ in range(atom_count)]
    neighbors: list[list[Tuple[int, OrganicBond]]] = [[] for _ in range(atom_count)]
    for bond in spec.bonds:
        if bond.order not in BOND_ORDER_VALUES:
            raise ValueError(f"unsupported bond order {bond.order!r}")
        if bond.atom_a == bond.atom_b:
            raise ValueError("self-bonds are not allowed")
        if bond.atom_a < 0 or bond.atom_a >= atom_count or bond.atom_b < 0 or bond.atom_b >= atom_count:
            raise ValueError("bond endpoint is out of range")
        order_value = int(BOND_ORDER_VALUES[str(bond.order)])
        valences[bond.atom_a] += order_value
        valences[bond.atom_b] += order_value
        neighbors[bond.atom_a].append((bond.atom_b, bond))
        neighbors[bond.atom_b].append((bond.atom_a, bond))

    over_valent = [spec.atoms[idx].item_id for idx, value in enumerate(valences) if value > 4]
    if over_valent:
        raise ValueError(f"atom valence exceeded for {over_valent}")

    multiple_counts = [0 for _ in range(atom_count)]
    for bond in spec.bonds:
        if bond.order in ("double", "triple"):
            multiple_counts[bond.atom_a] += 1
            multiple_counts[bond.atom_b] += 1
    crowded_multiple = [spec.atoms[idx].item_id for idx, value in enumerate(multiple_counts) if value > 1]
    if crowded_multiple:
        raise ValueError(f"adjacent cumulated multiple bonds are not supported: {crowded_multiple}")

    triple_linear_atom_ids: list[str] = []
    for bond in spec.bonds:
        if bond.order != "triple":
            continue
        for atom_index, partner_index in ((bond.atom_a, bond.atom_b), (bond.atom_b, bond.atom_a)):
            if len(neighbors[atom_index]) > 2:
                raise ValueError(f"triple-bond atom {spec.atoms[atom_index].item_id} is branched")
            if len(neighbors[atom_index]) == 2:
                other_index = next(idx for idx, _bond in neighbors[atom_index] if idx != partner_index)
                dot = _normalized_dot(spec.atoms[atom_index], spec.atoms[partner_index], spec.atoms[other_index])
                if dot > -0.94:
                    raise ValueError(f"triple-bond atom {spec.atoms[atom_index].item_id} is not linear")
                triple_linear_atom_ids.append(spec.atoms[atom_index].item_id)

    for ring in spec.ring_atom_sets:
        if len(ring) not in (5, 6):
            raise ValueError("v1 organic rings must be pentagons or hexagons")

    branch_indices = organic_branch_point_atom_indices(spec)
    branch_angles: list[float] = []
    for atom_index in branch_indices:
        branch_angles.append(_minimum_incident_angle_degrees(spec, int(atom_index), neighbors[int(atom_index)]))
    min_branch_angle = min(branch_angles) if branch_angles else None
    if min_branch_angle is not None and float(min_branch_angle) < 40.0:
        raise ValueError(f"branch angle is too cramped: {min_branch_angle:.2f} degrees")

    crossing_count = _count_crossings(spec)
    if crossing_count:
        raise ValueError(f"structure has {crossing_count} crossed bond(s)")

    return OrganicConstraintReport(
        valence_by_atom_id={spec.atoms[idx].item_id: int(value) for idx, value in enumerate(valences)},
        max_valence=max(valences) if valences else 0,
        branch_point_atom_ids=tuple(spec.atoms[idx].item_id for idx in branch_indices),
        min_branch_angle_degrees=None if min_branch_angle is None else float(min_branch_angle),
        triple_linear_atom_ids=tuple(triple_linear_atom_ids),
        ring_sizes=tuple(len(ring) for ring in spec.ring_atom_sets),
        crossing_count=int(crossing_count),
    )


def _minimum_incident_angle_degrees(
    spec: OrganicStructureSpec,
    atom_index: int,
    neighbors: Sequence[Tuple[int, OrganicBond]],
) -> float:
    origin = spec.atoms[int(atom_index)]
    angles: list[float] = []
    for left_index, (left_neighbor, _left_bond) in enumerate(neighbors):
        for right_neighbor, _right_bond in neighbors[left_index + 1 :]:
            dot = max(
                -1.0,
                min(1.0, _normalized_dot(origin, spec.atoms[int(left_neighbor)], spec.atoms[int(right_neighbor)])),
            )
            angles.append(math.degrees(math.acos(dot)))
    return min(angles) if angles else 180.0


def _normalized_dot(origin: OrganicAtom, a: OrganicAtom, b: OrganicAtom) -> float:
    ax = float(a.x) - float(origin.x)
    ay = float(a.y) - float(origin.y)
    bx = float(b.x) - float(origin.x)
    by = float(b.y) - float(origin.y)
    alen = math.hypot(ax, ay) or 1.0
    blen = math.hypot(bx, by) or 1.0
    return (ax * bx + ay * by) / (alen * blen)


def _count_crossings(spec: OrganicStructureSpec) -> int:
    crossings = 0
    for left_index, left in enumerate(spec.bonds):
        left_atoms = {left.atom_a, left.atom_b}
        a = spec.atoms[left.atom_a]
        b = spec.atoms[left.atom_b]
        for right in spec.bonds[left_index + 1 :]:
            if left_atoms.intersection({right.atom_a, right.atom_b}):
                continue
            c = spec.atoms[right.atom_a]
            d = spec.atoms[right.atom_b]
            if _segments_intersect((a.x, a.y), (b.x, b.y), (c.x, c.y), (d.x, d.y)):
                crossings += 1
    return crossings


def _segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> bool:
    def orient(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < -1e-9) and (o3 * o4 < -1e-9)


__all__ = [
    "build_constrained_organic_ring_size_structure",
    "build_constrained_organic_structure",
    "organic_branch_point_atom_indices",
    "organic_ring_item_ids",
    "validate_organic_structure",
]
