"""Cross-domain helpers for assigning deterministic scene labels."""

from __future__ import annotations

from typing import Sequence, Tuple


LABEL_POOL_A_L: Tuple[str, ...] = tuple("ABCDEFGHIJKL")
LABEL_POOL_SAFE_UPPER: Tuple[str, ...] = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")


def assign_shuffled_labels(
    rng,
    *,
    object_count: int,
    label_pool: Sequence[str] = LABEL_POOL_A_L,
) -> Tuple[str, ...]:
    """Return one shuffled label subset for a multi-object scene.

    The helper intentionally keeps the pool compact (`A`..`L`) so labels stay
    legible in crowded review workbooks and prompts.
    """

    if int(object_count) > len(label_pool):
        raise ValueError("object_count exceeds available scene labels")
    labels = [str(label) for label in label_pool[: int(object_count)]]
    rng.shuffle(labels)
    return tuple(str(label) for label in labels)


def assign_random_shuffled_labels(
    rng,
    *,
    object_count: int,
    label_pool: Sequence[str] = LABEL_POOL_SAFE_UPPER,
) -> Tuple[str, ...]:
    """Return one shuffled random subset of labels without replacement.

    Unlike `assign_shuffled_labels`, this helper does not always take the first
    `object_count` labels from the pool. It is useful when the scene should not
    bias toward labels near the start of the alphabet.
    """

    if int(object_count) > len(label_pool):
        raise ValueError("object_count exceeds available scene labels")
    labels = [str(label) for label in rng.sample(list(label_pool), k=int(object_count))]
    rng.shuffle(labels)
    return tuple(str(label) for label in labels)


__all__ = [
    "LABEL_POOL_A_L",
    "LABEL_POOL_SAFE_UPPER",
    "assign_random_shuffled_labels",
    "assign_shuffled_labels",
]
