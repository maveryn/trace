"""Shared visually-confusable object policies for 3D count/readout tasks."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


VISUAL_CONFUSION_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("pen", "pencil", "ruler", "tube", "stick"),
    ("card", "bookmark", "small_box", "ticket", "mail_envelope", "open_book"),
    ("candy_disc", "cd", "berry", "button", "sphere", "marble", "bead", "coaster", "lid"),
    ("screw", "hex_nut", "clip", "socket", "bolt", "hook", "tape_roll", "torus"),
    ("fork", "spoon"),
    ("plate", "bowl", "cup", "jar", "can", "lid", "bottle", "tray", "coaster", "basket"),
    ("hammer", "paint_brush"),
    ("flower", "rose", "cactus", "leaf", "egg", "apple", "carrot", "tomato"),
    ("light_bulb", "lantern", "candle"),
    ("horseshoe", "umbrella"),
    ("mini_chair", "mini_table", "chair", "table", "stool"),
    ("small_box", "cabinet"),
    ("chess_piece", "trophy", "crown"),
)

_CONFUSABLE_BY_SHAPE: Mapping[str, Tuple[str, ...]] = {
    str(shape): tuple(str(other) for other in group if str(other) != str(shape))
    for group in VISUAL_CONFUSION_GROUPS
    for shape in group
}


def confusable_shape_names(shape_type: str) -> Tuple[str, ...]:
    """Return object names that are too visually close for object-type distractors."""

    return tuple(str(shape) for shape in _CONFUSABLE_BY_SHAPE.get(str(shape_type), ()))


def shapes_conflict(left: str, right: str) -> bool:
    """Return whether two object types should not be used as semantic distractors."""

    return str(right) in set(confusable_shape_names(str(left))) or str(left) in set(confusable_shape_names(str(right)))


def compatible_shape_names(
    support: Sequence[str],
    *,
    anchors: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Return support after removing anchors and their visual near-confusers."""

    blocked = {str(shape) for shape in exclude}
    for anchor in anchors:
        blocked.add(str(anchor))
        blocked.update(confusable_shape_names(str(anchor)))
    pool = tuple(str(shape) for shape in support if str(shape) not in blocked)
    if not pool:
        raise ValueError("3D object readout needs at least one compatible shape")
    return pool


def compatible_distractor_pool(target_shape_type: str, *, support: Sequence[str]) -> Tuple[str, ...]:
    """Choose object distractors while avoiding visually near-identical groups."""

    return compatible_shape_names(support, anchors=(str(target_shape_type),))


__all__ = [
    "VISUAL_CONFUSION_GROUPS",
    "compatible_distractor_pool",
    "compatible_shape_names",
    "confusable_shape_names",
    "shapes_conflict",
]
