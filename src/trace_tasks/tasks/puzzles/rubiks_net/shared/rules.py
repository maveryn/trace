"""Pure Rubik quarter-turn mechanics and state serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .state import (
    FACE_ORDER,
    FACE_ORIENTATIONS,
    MOVE_FACES,
    NORMAL_TO_FACE,
    StickerKey,
    Vector3,
)


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (int(a[0] + b[0]), int(a[1] + b[1]), int(a[2] + b[2]))


def _vec_mul(a: Vector3, scale: int) -> Vector3:
    return (int(a[0] * scale), int(a[1] * scale), int(a[2] * scale))


def _dot(a: Vector3, b: Vector3) -> int:
    return int((a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]))


def _sticker_to_geometry(key: StickerKey) -> tuple[Vector3, Vector3]:
    face, row, col = str(key[0]), int(key[1]), int(key[2])
    normal, right, up = FACE_ORIENTATIONS[str(face)]
    position = _vec_add(
        _vec_add(tuple(normal), _vec_mul(tuple(right), int(col - 1))),
        _vec_mul(tuple(up), int(row - 1)),
    )
    return position, tuple(normal)


def _geometry_to_sticker(position: Vector3, normal: Vector3) -> StickerKey:
    face = str(NORMAL_TO_FACE[tuple(normal)])
    face_normal, right, up = FACE_ORIENTATIONS[str(face)]
    relative = (
        int(position[0] - face_normal[0]),
        int(position[1] - face_normal[1]),
        int(position[2] - face_normal[2]),
    )
    col = int(_dot(relative, tuple(right)) + 1)
    row = int(_dot(relative, tuple(up)) + 1)
    return (str(face), int(row), int(col))


def _rotate_vector_quarter(
    vec: Vector3,
    axis: Vector3,
    quarter_turns: int,
) -> Vector3:
    turns = int(quarter_turns) % 4
    x, y, z = int(vec[0]), int(vec[1]), int(vec[2])
    ax = tuple(int(v) for v in axis)
    for _ in range(turns):
        if ax == (1, 0, 0):
            x, y, z = x, -z, y
        elif ax == (-1, 0, 0):
            x, y, z = x, z, -y
        elif ax == (0, 1, 0):
            x, y, z = z, y, -x
        elif ax == (0, -1, 0):
            x, y, z = -z, y, x
        elif ax == (0, 0, 1):
            x, y, z = -y, x, z
        elif ax == (0, 0, -1):
            x, y, z = y, -x, z
        else:
            raise ValueError(f"unsupported rotation axis: {axis}")
    return (int(x), int(y), int(z))


def parse_move(move: str) -> tuple[str, bool]:
    """Parse one Rubik face-turn token into face and prime direction."""

    text = str(move).strip()
    if not text:
        raise ValueError("empty Rubik move")
    face = str(text[0]).upper()
    if face not in MOVE_FACES:
        raise ValueError(f"unsupported Rubik move face: {move}")
    return face, "'" in text


def invert_move(move: str) -> str:
    """Return the inverse of one face-turn token."""

    face, is_prime = parse_move(str(move))
    return str(face) if bool(is_prime) else f"{face}'"


def invert_sequence(sequence: Sequence[str]) -> list[str]:
    """Return inverse tokens for one face-turn sequence."""

    return [
        invert_move(str(move)) for move in reversed([str(item) for item in sequence])
    ]


def apply_move(
    state: Mapping[StickerKey, str],
    move: str,
) -> dict[StickerKey, str]:
    """Apply one quarter-turn to a sticker-color state."""

    face, is_prime = parse_move(str(move))
    axis = tuple(FACE_ORIENTATIONS[str(face)][0])
    # Clockwise from outside is a negative right-hand turn around the face normal.
    quarter_turns = 1 if bool(is_prime) else 3
    next_state: dict[StickerKey, str] = {}
    for key, color_name in state.items():
        position, normal = _sticker_to_geometry(tuple(key))
        if int(_dot(position, axis)) == 1:
            new_position = _rotate_vector_quarter(position, axis, int(quarter_turns))
            new_normal = _rotate_vector_quarter(normal, axis, int(quarter_turns))
            next_state[_geometry_to_sticker(new_position, new_normal)] = str(color_name)
        else:
            next_state[tuple(key)] = str(color_name)
    return next_state


def apply_sequence(
    state: Mapping[StickerKey, str],
    sequence: Sequence[str],
) -> dict[StickerKey, str]:
    """Apply a sequence of quarter-turn tokens."""

    current: dict[StickerKey, str] = {
        tuple(key): str(value) for key, value in state.items()
    }
    for move in [str(item) for item in sequence]:
        current = apply_move(current, str(move))
    return current


def state_signature(
    state: Mapping[StickerKey, str],
) -> tuple[tuple[str, int, int, str], ...]:
    """Return one stable sticker-color signature for duplicate rejection."""

    return tuple(
        (str(face), int(row), int(col), str(state[(str(face), int(row), int(col))]))
        for face in FACE_ORDER
        for row in range(3)
        for col in range(3)
    )


def make_solved_state(face_color_names: Mapping[str, str]) -> dict[StickerKey, str]:
    """Create a solved sticker state from face-to-color assignments."""

    state: dict[StickerKey, str] = {}
    for face in FACE_ORDER:
        for row in range(3):
            for col in range(3):
                state[(str(face), int(row), int(col))] = str(
                    face_color_names[str(face)]
                )
    return state


def sample_move(rng, *, previous_face: str | None = None) -> str:
    """Sample one quarter-turn while avoiding immediate same-face repeats."""

    faces = [str(face) for face in MOVE_FACES if str(face) != str(previous_face or "")]
    face = str(faces[int(rng.randrange(len(faces)))])
    return f"{face}'" if int(rng.randrange(2)) == 1 else str(face)


def sample_move_sequence(rng, *, length: int) -> list[str]:
    """Sample a sequence of quarter-turn tokens."""

    moves: list[str] = []
    previous_face: str | None = None
    for _ in range(int(length)):
        move = sample_move(rng, previous_face=previous_face)
        previous_face, _ = parse_move(str(move))
        moves.append(str(move))
    return moves


def format_sequence(sequence: Sequence[str]) -> str:
    """Format a move sequence for prompt text."""

    return " ".join(str(item) for item in sequence)


def sticker_id(face: str, row: int, col: int) -> str:
    """Return the stable sticker identifier used in trace and render maps."""

    return f"{str(face)}_r{int(row)}_c{int(col)}"


def face_color_count(
    state: Mapping[StickerKey, str],
    *,
    face: str,
    color_name: str,
) -> int:
    """Count stickers of one color on one face."""

    return sum(
        1
        for row in range(3)
        for col in range(3)
        if str(state[(str(face), int(row), int(col))]) == str(color_name)
    )


def state_for_trace(state: Mapping[StickerKey, str]) -> dict[str, str]:
    """Serialize a tuple-key sticker-color state for trace metadata."""

    serialized: dict[str, str] = {}
    for key, color_name in state.items():
        face, row, col = key
        serialized[f"{face}_r{int(row)}_c{int(col)}"] = str(color_name)
    return serialized


def signature_for_trace(
    signature: Sequence[tuple[str, int, int, str]],
) -> list[list[str | int]]:
    """Serialize a state signature into JSON-friendly rows."""

    return [
        [str(face), int(row), int(col), str(color_name)]
        for face, row, col, color_name in signature
    ]


__all__ = [
    "apply_move",
    "apply_sequence",
    "face_color_count",
    "format_sequence",
    "invert_sequence",
    "make_solved_state",
    "parse_move",
    "sample_move",
    "sample_move_sequence",
    "signature_for_trace",
    "state_for_trace",
    "state_signature",
    "sticker_id",
]
