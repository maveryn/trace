"""Cube topology helpers for cube-net puzzle scenes."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product
from typing import Dict, Mapping, Sequence, Tuple

from .state import (
    FACE_BY_NORMAL,
    FACE_IDS,
    NET_COORDS,
    NORMAL_BY_FACE,
    SIDE_OFFSETS,
)


Vector3 = Tuple[int, int, int]
FaceBasis = Tuple[Vector3, Vector3, Vector3]
RotationMatrix = Tuple[Vector3, Vector3, Vector3]


def _neg(vec: Sequence[int]) -> Vector3:
    return (-int(vec[0]), -int(vec[1]), -int(vec[2]))


def _permutation_parity(perm: Sequence[int]) -> int:
    """Return +1 for even permutations and -1 for odd permutations."""

    inversions = 0
    values = [int(value) for value in perm]
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] > values[right]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def _apply_rotation(matrix: RotationMatrix, vec: Sequence[int]) -> Vector3:
    """Apply one signed permutation matrix to a cube normal vector."""

    return (
        int(matrix[0][0]) * int(vec[0])
        + int(matrix[0][1]) * int(vec[1])
        + int(matrix[0][2]) * int(vec[2]),
        int(matrix[1][0]) * int(vec[0])
        + int(matrix[1][1]) * int(vec[1])
        + int(matrix[1][2]) * int(vec[2]),
        int(matrix[2][0]) * int(vec[0])
        + int(matrix[2][1]) * int(vec[1])
        + int(matrix[2][2]) * int(vec[2]),
    )


@lru_cache(maxsize=1)
def cube_rotation_matrices() -> Tuple[RotationMatrix, ...]:
    """Return the 24 orientation-preserving rotations of a cube."""

    rotations: list[RotationMatrix] = []
    for perm in permutations((0, 1, 2)):
        parity = _permutation_parity(perm)
        for signs in product((-1, 1), repeat=3):
            if parity * int(signs[0]) * int(signs[1]) * int(signs[2]) != 1:
                continue
            rows: list[Vector3] = []
            for row_index in range(3):
                row = [0, 0, 0]
                row[int(perm[row_index])] = int(signs[row_index])
                rows.append((int(row[0]), int(row[1]), int(row[2])))
            rotations.append((rows[0], rows[1], rows[2]))
    if len(rotations) != 24:
        raise ValueError("cube rotation construction did not produce 24 rotations")
    return tuple(rotations)


def rotate_face_assignment(
    face_assignment: Mapping[str, str],
    rotation: RotationMatrix,
) -> Dict[str, str]:
    """Rotate one face-value assignment by a physical cube rotation."""

    rotated: Dict[str, str] = {}
    for face_id, value in face_assignment.items():
        normal = NORMAL_BY_FACE[str(face_id)]
        target_face = FACE_BY_NORMAL[_apply_rotation(rotation, normal)]
        rotated[str(target_face)] = str(value)
    if set(rotated) != set(FACE_IDS):
        raise ValueError("rotated face assignment did not cover all cube faces")
    return rotated


def canonical_face_assignment_signature(face_assignment: Mapping[str, str]) -> Tuple[str, ...]:
    """Normalize a face-value assignment over all whole-cube rotations."""

    signatures = [
        tuple(rotated[str(face)] for face in FACE_IDS)
        for rotated in (
            rotate_face_assignment(face_assignment, rotation)
            for rotation in cube_rotation_matrices()
        )
    ]
    return min(signatures)


def basis_across_side(
    *,
    normal: Vector3,
    up: Vector3,
    right: Vector3,
    side: str,
) -> FaceBasis:
    """Return the neighboring face basis created by folding across one net edge."""

    if str(side) == "top":
        return tuple(up), _neg(normal), tuple(right)
    if str(side) == "bottom":
        return _neg(up), tuple(normal), tuple(right)
    if str(side) == "right":
        return tuple(right), tuple(up), _neg(normal)
    if str(side) == "left":
        return _neg(right), tuple(up), tuple(normal)
    raise ValueError(f"unsupported side: {side}")


def net_face_bases() -> Dict[str, FaceBasis]:
    """Propagate 3-D face bases from the displayed cube net without task routing."""

    coord_to_face = {tuple(coord): str(face) for face, coord in NET_COORDS.items()}
    bases: Dict[str, FaceBasis] = {
        "F": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    }
    queue = ["F"]
    while queue:
        face = queue.pop(0)
        normal, up, right = bases[str(face)]
        x, y = NET_COORDS[str(face)]
        for side, (dx, dy) in SIDE_OFFSETS.items():
            neighbor = coord_to_face.get((int(x + dx), int(y + dy)))
            if neighbor is None or neighbor in bases:
                continue
            bases[str(neighbor)] = basis_across_side(
                normal=normal,
                up=up,
                right=right,
                side=str(side),
            )
            queue.append(str(neighbor))
    if set(bases) != set(FACE_IDS):
        raise ValueError("cube net basis propagation did not cover all faces")
    return bases


NET_FACE_BASES = net_face_bases()


def face_across_display_side(face_id: str, side: str) -> str:
    """Return the folded cube face across one visible side of a net face."""

    normal, up, right = NET_FACE_BASES[str(face_id)]
    if str(side) == "top":
        target_normal = tuple(up)
    elif str(side) == "bottom":
        target_normal = _neg(up)
    elif str(side) == "right":
        target_normal = tuple(right)
    elif str(side) == "left":
        target_normal = _neg(right)
    else:
        raise ValueError(f"unsupported marked side: {side}")
    return str(FACE_BY_NORMAL[tuple(target_normal)])


__all__ = [
    "NET_FACE_BASES",
    "basis_across_side",
    "canonical_face_assignment_signature",
    "cube_rotation_matrices",
    "face_across_display_side",
    "net_face_bases",
    "rotate_face_assignment",
]
