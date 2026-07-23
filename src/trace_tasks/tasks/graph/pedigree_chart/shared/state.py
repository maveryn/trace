"""State records and static topology templates for pedigree-chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

SUPPORTED_PEDIGREE_SCENE_VARIANTS: Tuple[str, ...] = (
    "classic_pedigree",
    "row_guided_pedigree",
    "paper_pedigree",
)
PEDIGREE_RELATIONSHIP_LABELS: Tuple[str, ...] = (
    "parent",
    "child",
    "sibling",
    "partner",
    "grandparent",
    "grandchild",
)
PEDIGREE_RELATEDNESS_LABELS: Tuple[str, ...] = ("0", "1/8", "1/4", "3/8", "1/2")
PEDIGREE_RELATEDNESS_OPTION_LABELS: Tuple[str, ...] = (
    "0",
    "1/16",
    "1/8",
    "1/4",
    "3/8",
    "1/2",
    "5/8",
    "3/4",
    "1",
)
PEDIGREE_SEX_LABELS: Mapping[str, str] = {
    "male": "male",
    "female": "female",
}
GENERATION_LABELS: Tuple[str, ...] = ("I", "II", "III", "IV")


@dataclass(frozen=True)
class PedigreePerson:
    """One person in a pedigree chart."""

    person_id: str
    generation_index: int
    sex: str
    affected: bool
    label: str

    @property
    def generation_label(self) -> str:
        return GENERATION_LABELS[int(self.generation_index)]


@dataclass(frozen=True)
class PedigreeFamily:
    """One couple and its children in a pedigree chart."""

    pedigree_id: str
    parent_ids: Tuple[str, str]
    child_ids: Tuple[str, ...]


@dataclass(frozen=True)
class PedigreeSample:
    """Trace-ready pedigree sample."""

    people: Tuple[PedigreePerson, ...]
    families: Tuple[PedigreeFamily, ...]
    target_count: int
    target_generation_index: int | None
    target_sex: str | None
    counted_person_ids: Tuple[str, ...]
    template_name: str


@dataclass(frozen=True)
class RenderedPedigreePerson:
    """Rendered projection for one pedigree person."""

    person_id: str
    label: str
    generation_index: int
    generation_label: str
    sex: str
    affected: bool
    center_xy: Point
    symbol_bbox_xyxy: BBox
    label_bbox_xyxy: BBox


@dataclass(frozen=True)
class RenderedPedigreeFamily:
    """Rendered connector projection for one pedigree family."""

    pedigree_id: str
    parent_ids: Tuple[str, str]
    child_ids: Tuple[str, ...]
    spouse_segment_px: Tuple[Point, Point]
    descent_segments_px: Tuple[Tuple[Point, Point], ...]


@dataclass(frozen=True)
class RenderedPedigreeScene:
    """Full render output for one pedigree scene."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    people: Tuple[RenderedPedigreePerson, ...]
    families: Tuple[RenderedPedigreeFamily, ...]
    scene_variant: str
    resolved_label_font_size_px: int
    resolved_label_stroke_width_px: int
    generation_label_bboxes: Dict[str, BBox]


@dataclass(frozen=True)
class PedigreeRelationshipQuerySample:
    """One relationship-label query over a pedigree."""

    sample: PedigreeSample
    answer: str
    person_a_id: str
    person_b_id: str
    annotation_roles: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PedigreeRelatednessQuerySample:
    """One coefficient-of-relatedness query over a pedigree."""

    sample: PedigreeSample
    answer: str
    person_a_id: str
    person_b_id: str
    annotation_roles: Tuple[Tuple[str, str], ...]
    contributing_paths: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class _TemplatePerson:
    """Template person before labels and affected-state assignment."""

    person_id: str
    generation_index: int
    sex: str


@dataclass(frozen=True)
class _Template:
    """Static pedigree topology template."""

    name: str
    people: Tuple[_TemplatePerson, ...]
    families: Tuple[PedigreeFamily, ...]


def _template_compact_three() -> _Template:
    people = (
        _TemplatePerson("p0", 0, "male"),
        _TemplatePerson("p1", 0, "female"),
        _TemplatePerson("p2", 1, "male"),
        _TemplatePerson("p3", 1, "female"),
        _TemplatePerson("p4", 1, "male"),
        _TemplatePerson("p5", 1, "male"),
        _TemplatePerson("p6", 1, "female"),
        _TemplatePerson("p7", 2, "female"),
        _TemplatePerson("p8", 2, "male"),
        _TemplatePerson("p9", 2, "male"),
        _TemplatePerson("p10", 2, "female"),
    )
    families = (
        PedigreeFamily("f0", ("p0", "p1"), ("p2", "p3", "p5")),
        PedigreeFamily("f1", ("p3", "p4"), ("p7", "p8")),
        PedigreeFamily("f2", ("p5", "p6"), ("p9", "p10")),
    )
    return _Template("compact_three_generation", people, families)


def _template_wide_three() -> _Template:
    people = (
        _TemplatePerson("p0", 0, "male"),
        _TemplatePerson("p1", 0, "female"),
        _TemplatePerson("p2", 1, "male"),
        _TemplatePerson("p3", 1, "female"),
        _TemplatePerson("p4", 1, "female"),
        _TemplatePerson("p5", 1, "male"),
        _TemplatePerson("p6", 1, "male"),
        _TemplatePerson("p7", 1, "female"),
        _TemplatePerson("p8", 1, "female"),
        _TemplatePerson("p9", 1, "male"),
        _TemplatePerson("p10", 2, "male"),
        _TemplatePerson("p11", 2, "female"),
        _TemplatePerson("p12", 2, "female"),
        _TemplatePerson("p13", 2, "male"),
        _TemplatePerson("p14", 2, "male"),
        _TemplatePerson("p15", 2, "female"),
    )
    families = (
        PedigreeFamily("f0", ("p0", "p1"), ("p2", "p4", "p6", "p8")),
        PedigreeFamily("f1", ("p2", "p3"), ("p10", "p11")),
        PedigreeFamily("f2", ("p4", "p5"), ("p12", "p13")),
        PedigreeFamily("f3", ("p6", "p7"), ("p14", "p15")),
        PedigreeFamily("f4", ("p8", "p9"), ()),
    )
    return _Template("wide_three_generation", people, families)


def _template_four_generation() -> _Template:
    people = (
        _TemplatePerson("p0", 0, "male"),
        _TemplatePerson("p1", 0, "female"),
        _TemplatePerson("p2", 1, "female"),
        _TemplatePerson("p3", 1, "male"),
        _TemplatePerson("p4", 1, "male"),
        _TemplatePerson("p5", 1, "female"),
        _TemplatePerson("p6", 2, "male"),
        _TemplatePerson("p7", 2, "female"),
        _TemplatePerson("p8", 2, "female"),
        _TemplatePerson("p9", 2, "male"),
        _TemplatePerson("p10", 3, "male"),
        _TemplatePerson("p11", 3, "female"),
        _TemplatePerson("p12", 3, "male"),
    )
    families = (
        PedigreeFamily("f0", ("p0", "p1"), ("p2", "p4", "p5")),
        PedigreeFamily("f1", ("p2", "p3"), ("p6", "p8", "p9")),
        PedigreeFamily("f2", ("p6", "p7"), ("p10", "p11", "p12")),
    )
    return _Template("four_generation", people, families)


def _template_compound_relatedness() -> _Template:
    people = (
        _TemplatePerson("p0", 0, "male"),
        _TemplatePerson("p1", 0, "female"),
        _TemplatePerson("p2", 1, "female"),
        _TemplatePerson("p3", 1, "female"),
        _TemplatePerson("p4", 1, "male"),
        _TemplatePerson("p5", 2, "female"),
        _TemplatePerson("p6", 2, "male"),
        _TemplatePerson("p7", 2, "female"),
        _TemplatePerson("p8", 2, "male"),
    )
    families = (
        PedigreeFamily("f0", ("p0", "p1"), ("p2", "p3")),
        PedigreeFamily("f1", ("p4", "p2"), ("p5", "p7")),
        PedigreeFamily("f2", ("p4", "p3"), ("p6", "p8")),
    )
    return _Template("compound_relatedness", people, families)


def _templates() -> Tuple[_Template, ...]:
    return (
        _template_compact_three(),
        _template_wide_three(),
        _template_four_generation(),
        _template_compound_relatedness(),
    )


def _generation_persons(template: _Template, generation_index: int) -> Tuple[_TemplatePerson, ...]:
    return tuple(person for person in template.people if int(person.generation_index) == int(generation_index))


def _assign_labels(template: _Template, affected_ids: Iterable[str]) -> Tuple[PedigreePerson, ...]:
    affected_set = {str(person_id) for person_id in affected_ids}
    counts_by_generation: Dict[int, int] = {}
    result: List[PedigreePerson] = []
    for person in template.people:
        generation_index = int(person.generation_index)
        counts_by_generation[generation_index] = int(counts_by_generation.get(generation_index, 0)) + 1
        label = f"{GENERATION_LABELS[generation_index]}-{counts_by_generation[generation_index]}"
        result.append(
            PedigreePerson(
                person_id=str(person.person_id),
                generation_index=int(generation_index),
                sex=str(person.sex),
                affected=str(person.person_id) in affected_set,
                label=str(label),
            )
        )
    return tuple(result)


def _person_map(sample: PedigreeSample) -> Dict[str, PedigreePerson]:
    return {str(person.person_id): person for person in sample.people}


def _template_by_name(name: str) -> _Template:
    for template in _templates():
        if str(template.name) == str(name):
            return template
    raise ValueError(f"unknown pedigree template: {name}")


def _label_for_id(sample: PedigreeSample, person_id: str) -> str:
    return _person_map(sample)[str(person_id)].label


def _parents_by_child(sample: PedigreeSample) -> Dict[str, Tuple[str, str]]:
    result: Dict[str, Tuple[str, str]] = {}
    for family in sample.families:
        for child_id in family.child_ids:
            result[str(child_id)] = tuple(str(parent_id) for parent_id in family.parent_ids)  # type: ignore[assignment]
    return result


def _children_by_parent(sample: PedigreeSample) -> Dict[str, Tuple[str, ...]]:
    result: Dict[str, List[str]] = {str(person.person_id): [] for person in sample.people}
    for family in sample.families:
        for parent_id in family.parent_ids:
            result.setdefault(str(parent_id), []).extend(str(child_id) for child_id in family.child_ids)
    return {str(parent_id): tuple(child_ids) for parent_id, child_ids in result.items()}


def _families_by_partner_pair(sample: PedigreeSample) -> Dict[Tuple[str, str], PedigreeFamily]:
    result: Dict[Tuple[str, str], PedigreeFamily] = {}
    for family in sample.families:
        key = tuple(sorted(str(parent_id) for parent_id in family.parent_ids))
        result[key] = family
    return result



__all__ = [
    "GENERATION_LABELS",
    "PEDIGREE_RELATEDNESS_LABELS",
    "PEDIGREE_RELATEDNESS_OPTION_LABELS",
    "PEDIGREE_RELATIONSHIP_LABELS",
    "PEDIGREE_SEX_LABELS",
    "SUPPORTED_PEDIGREE_SCENE_VARIANTS",
    "PedigreeFamily",
    "PedigreePerson",
    "PedigreeRelatednessQuerySample",
    "PedigreeRelationshipQuerySample",
    "PedigreeSample",
    "RenderedPedigreeFamily",
    "RenderedPedigreePerson",
    "RenderedPedigreeScene",
]
