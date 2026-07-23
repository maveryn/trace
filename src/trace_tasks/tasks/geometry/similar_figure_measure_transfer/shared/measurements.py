"""Numeric similar-figure measurement case builders."""

from __future__ import annotations

from typing import Sequence

from .state import SimilarMeasureCase, Side


def default_support_side(shape_kind: str) -> Side:
    """Return a readable second corresponding side for one figure shape."""

    if str(shape_kind) == "triangle":
        return (1, 2)
    return (2, 3)


def corresponding_side_case(
    *,
    construction_family: str,
    shape_kind: str,
    scale_factor: int,
    source_side: int,
    support_source: int,
) -> SimilarMeasureCase:
    """Build a case whose answer is the target corresponding side length."""

    support_side = default_support_side(str(shape_kind))
    labels = _labels_for_sides(str(shape_kind), (0, 1), support_side)
    return SimilarMeasureCase(
        construction_family=str(construction_family),
        shape_kind=str(shape_kind),
        layout_kind=_layout_for_family(str(construction_family)),
        answer=int(scale_factor) * int(source_side),
        target_name="A'B'",
        relation="corresponding_sides_of_similar_figures",
        scale_factor=int(scale_factor),
        source_target_side_value=int(source_side),
        target_target_side_value=int(scale_factor) * int(source_side),
        support_source_side_value=int(support_source),
        support_target_side_value=int(scale_factor) * int(support_source),
        target_side=(0, 1),
        support_side=support_side,
        annotation_labels=labels,
    )


def side_length_from_area_case(
    *,
    construction_family: str,
    shape_kind: str,
    scale_factor: int,
    source_side: int,
    source_area: int,
    use_ratio_label: bool,
) -> SimilarMeasureCase:
    """Build a side-transfer case whose linear scale comes from area data."""

    labels = _labels_for_sides(str(shape_kind), (0, 1), (0, 1))
    target_area = int(source_area) * int(scale_factor) * int(scale_factor)
    return SimilarMeasureCase(
        construction_family=str(construction_family),
        shape_kind=str(shape_kind),
        layout_kind=_layout_for_family(str(construction_family)),
        answer=int(scale_factor) * int(source_side),
        target_name="A'B'",
        relation="side_length_from_area_scale",
        scale_factor=int(scale_factor),
        source_target_side_value=int(source_side),
        target_target_side_value=int(scale_factor) * int(source_side),
        source_area=int(source_area),
        target_area=target_area,
        area_ratio_label=f"{int(scale_factor) * int(scale_factor)}:1",
        target_side=(0, 1),
        support_side=(0, 1),
        annotation_labels=labels,
    )


def _layout_for_family(construction_family: str) -> str:
    if str(construction_family) == "nested_side_transfer" or str(construction_family) == "area_known_side_nested":
        return "nested"
    if "rotated" in str(construction_family) or "two_pair" in str(construction_family):
        return "rotated_pair"
    return "side_by_side"


def _base_labels(shape_kind: str) -> tuple[str, ...]:
    count_by_shape = {"triangle": 3, "quadrilateral": 4, "pentagon": 5}
    count = int(count_by_shape.get(str(shape_kind), 4))
    return tuple(chr(ord("A") + index) for index in range(count))


def _labels_for_sides(shape_kind: str, target_side: Side, support_side: Side) -> tuple[str, ...]:
    base = _base_labels(str(shape_kind))
    selected: list[str] = []
    for side in (target_side, support_side):
        for index in side:
            label = base[int(index)]
            if label not in selected:
                selected.append(label)
            prime = f"{label}'"
            if prime not in selected:
                selected.append(prime)
    return tuple(selected)
