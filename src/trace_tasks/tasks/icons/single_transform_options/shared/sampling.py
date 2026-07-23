"""Sampling helpers for single-transform option icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_assets import icon_transform_signature, resolve_icon_pool
from ...shared.icon_transform import IDENTITY_TRANSFORM_ID


ALL_OPTION_TRANSFORMS: Tuple[str, ...] = (
    IDENTITY_TRANSFORM_ID,
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
)


_TRANSFORM_MATRICES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    IDENTITY_TRANSFORM_ID: ((1, 0), (0, 1)),
    "rot90": ((0, -1), (1, 0)),
    "rot180": ((-1, 0), (0, -1)),
    "rot270": ((0, 1), (-1, 0)),
    "flip_h": ((-1, 0), (0, 1)),
    "flip_v": ((1, 0), (0, -1)),
    "flip_diag_main": ((0, 1), (1, 0)),
    "flip_diag_anti": ((0, -1), (-1, 0)),
}
_MATRIX_TRANSFORMS = {matrix: transform_id for transform_id, matrix in _TRANSFORM_MATRICES.items()}


def _matrix_multiply(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the product `left * right` for two 2x2 transform matrices."""

    return (
        (
            int(left[0][0] * right[0][0] + left[0][1] * right[1][0]),
            int(left[0][0] * right[0][1] + left[0][1] * right[1][1]),
        ),
        (
            int(left[1][0] * right[0][0] + left[1][1] * right[1][0]),
            int(left[1][0] * right[0][1] + left[1][1] * right[1][1]),
        ),
    )


def compose_transform_ids(*, after_transform_id: str, before_transform_id: str) -> str:
    """Return the canonical transform produced by applying `before`, then `after`."""

    after = str(after_transform_id)
    before = str(before_transform_id)
    if after not in _TRANSFORM_MATRICES:
        raise ValueError(f"unsupported after_transform_id: {after}")
    if before not in _TRANSFORM_MATRICES:
        raise ValueError(f"unsupported before_transform_id: {before}")
    matrix = _matrix_multiply(_TRANSFORM_MATRICES[after], _TRANSFORM_MATRICES[before])
    if matrix not in _MATRIX_TRANSFORMS:
        raise ValueError(f"unsupported composed transform matrix: {matrix}")
    return str(_MATRIX_TRANSFORMS[matrix])


def resolve_fixed_option_count(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback_defaults: Any,
) -> int:
    """Resolve the task-configured fixed option layout count."""

    raw = params.get("object_count", group_default(generation_defaults, "object_count_max", fallback_defaults.object_count_max))
    count = int(raw)
    low = int(group_default(generation_defaults, "object_count_min", fallback_defaults.object_count_min))
    high = int(group_default(generation_defaults, "object_count_max", fallback_defaults.object_count_max))
    if low != high:
        raise ValueError("single-transform option scenes require a fixed object_count range")
    if count != high:
        raise ValueError(f"single-transform option scenes require object_count={high}")
    return count


def icon_supports_transform_set(icon_id: str, *, check_size_px: int) -> bool:
    """Return whether one icon has distinct signatures for the supported transforms."""

    signatures = [
        icon_transform_signature(str(icon_id), int(check_size_px), str(transform_id))
        for transform_id in ALL_OPTION_TRANSFORMS
    ]
    return len(set(signatures)) == len(signatures)


def sample_transform_distinct_icon(
    rng,
    *,
    pool_manifest: str,
    transform_check_size_px: int,
) -> str:
    """Sample an icon whose option transforms remain visually distinct."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if not pool:
        raise ValueError("single-transform option pool resolved no icons")
    rng.shuffle(pool)
    for icon_id in pool:
        if icon_supports_transform_set(str(icon_id), check_size_px=int(transform_check_size_px)):
            return str(icon_id)
    raise ValueError("insufficient non-symmetric icons with distinct supported transforms")


def option_transforms_for_answer(*, answer_index: int, target_transform_id: str) -> Tuple[str, ...]:
    """Place one target transform among the fixed transform distractor set."""

    distractors = [str(value) for value in ALL_OPTION_TRANSFORMS if str(value) != str(target_transform_id)]
    if len(distractors) != 5:
        raise ValueError("single-transform option scenes require exactly five distractor transforms")
    option_transforms: list[str] = []
    cursor = 0
    for index in range(6):
        if int(index) == int(answer_index):
            option_transforms.append(str(target_transform_id))
        else:
            option_transforms.append(str(distractors[int(cursor)]))
            cursor += 1
    return tuple(str(value) for value in option_transforms)


def validate_option_transform_signatures(
    *,
    icon_id: str,
    transform_ids: Sequence[str],
    check_size_px: int,
) -> None:
    """Reject option sets with visually collapsed transform signatures."""

    signatures = [
        icon_transform_signature(str(icon_id), int(check_size_px), str(transform_id))
        for transform_id in transform_ids
    ]
    if len(set(signatures)) != len(signatures):
        raise ValueError("single-transform option transforms are not visually unique")


__all__ = [
    "ALL_OPTION_TRANSFORMS",
    "compose_transform_ids",
    "icon_supports_transform_set",
    "option_transforms_for_answer",
    "resolve_fixed_option_count",
    "sample_transform_distinct_icon",
    "validate_option_transform_signatures",
]
