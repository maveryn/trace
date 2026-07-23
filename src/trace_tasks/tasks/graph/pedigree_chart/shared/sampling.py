"""Pedigree relationship and relatedness query sampling."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .state import (
    PEDIGREE_RELATEDNESS_LABELS,
    PEDIGREE_RELATIONSHIP_LABELS,
    PedigreeRelatednessQuerySample,
    PedigreeRelationshipQuerySample,
    PedigreeSample,
    _assign_labels,
    _children_by_parent,
    _parents_by_child,
    _templates,
)

def _relationship_candidates(sample: PedigreeSample, relation: str) -> Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]:
    """Return `(person_a, person_b, extra_roles)` candidates for one relation."""

    parents_by_child = _parents_by_child(sample)
    children_by_parent = _children_by_parent(sample)
    candidates: List[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = []
    if str(relation) == "partner":
        for family in sample.families:
            left, right = tuple(str(parent_id) for parent_id in family.parent_ids)
            candidates.append((left, right, ()))
            candidates.append((right, left, ()))
    elif str(relation) == "parent":
        for child_id, parent_ids in parents_by_child.items():
            for parent_id in parent_ids:
                candidates.append((str(parent_id), str(child_id), ()))
    elif str(relation) == "child":
        for child_id, parent_ids in parents_by_child.items():
            for parent_id in parent_ids:
                candidates.append((str(child_id), str(parent_id), ()))
    elif str(relation) == "sibling":
        for family in sample.families:
            children = tuple(str(child_id) for child_id in family.child_ids)
            for index, child_a in enumerate(children):
                for child_b in children[index + 1 :]:
                    bridge = (("shared_parent_1", str(family.parent_ids[0])), ("shared_parent_2", str(family.parent_ids[1])))
                    candidates.append((str(child_a), str(child_b), bridge))
                    candidates.append((str(child_b), str(child_a), bridge))
    elif str(relation) in {"grandparent", "grandchild"}:
        for middle_id, grandparent_ids in parents_by_child.items():
            for grandchild_id in children_by_parent.get(str(middle_id), ()):
                for grandparent_id in grandparent_ids:
                    bridge = (("middle_parent", str(middle_id)),)
                    if str(relation) == "grandparent":
                        candidates.append((str(grandparent_id), str(grandchild_id), bridge))
                    else:
                        candidates.append((str(grandchild_id), str(grandparent_id), bridge))
    else:
        raise ValueError(f"unsupported pedigree relationship: {relation}")
    return tuple(candidates)


def sample_pedigree_relationship(
    instance_seed: int,
    *,
    target_relationship: str,
    max_attempts: int = 80,
) -> PedigreeRelationshipQuerySample:
    """Sample one relationship query with a unique relationship answer."""

    if str(target_relationship) not in set(PEDIGREE_RELATIONSHIP_LABELS):
        raise ValueError(f"unsupported relationship label: {target_relationship}")
    templates = _templates()
    for attempt in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), "pedigree_chart.relationship", int(attempt))
        template = templates[int(rng.randrange(len(templates)))]
        sample = PedigreeSample(
            people=_assign_labels(template, ()),
            families=tuple(template.families),
            target_count=0,
            target_generation_index=None,
            target_sex=None,
            counted_person_ids=(),
            template_name=str(template.name),
        )
        candidates = _relationship_candidates(sample, str(target_relationship))
        if not candidates:
            continue
        person_a_id, person_b_id, extra_roles = rng.choice(list(candidates))
        roles = (("person_a", str(person_a_id)), ("person_b", str(person_b_id)), *tuple(extra_roles))
        return PedigreeRelationshipQuerySample(
            sample=sample,
            answer=str(target_relationship),
            person_a_id=str(person_a_id),
            person_b_id=str(person_b_id),
            annotation_roles=tuple(roles),
        )
    raise ValueError("unable to sample pedigree relationship query")


def _ancestor_paths_by_person(sample: PedigreeSample, person_id: str) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    """Return all upward paths from one person to each ancestor, including self."""

    parents_by_child = _parents_by_child(sample)
    paths: Dict[str, List[Tuple[str, ...]]] = {}

    def visit(current_id: str, path: Tuple[str, ...]) -> None:
        paths.setdefault(str(current_id), []).append(tuple(path))
        for parent_id in parents_by_child.get(str(current_id), ()):
            if str(parent_id) in set(path):
                continue
            visit(str(parent_id), (*tuple(path), str(parent_id)))

    visit(str(person_id), (str(person_id),))
    return {str(key): tuple(value) for key, value in paths.items()}


def _format_relatedness_fraction(value: Fraction) -> str:
    normalized = Fraction(value)
    if int(normalized.denominator) == 1:
        return str(int(normalized.numerator))
    return f"{int(normalized.numerator)}/{int(normalized.denominator)}"


def _relatedness_fraction_and_paths(
    sample: PedigreeSample,
    person_a_id: str,
    person_b_id: str,
) -> Tuple[Fraction, Tuple[Dict[str, Any], ...]]:
    paths_a = _ancestor_paths_by_person(sample, str(person_a_id))
    paths_b = _ancestor_paths_by_person(sample, str(person_b_id))
    total = Fraction(0, 1)
    contributing: List[Dict[str, Any]] = []
    for ancestor_id in sorted(set(paths_a).intersection(paths_b)):
        for path_a in paths_a[str(ancestor_id)]:
            for path_b in paths_b[str(ancestor_id)]:
                if not set(path_a[:-1]).isdisjoint(set(path_b[:-1])):
                    continue
                edges = int(len(path_a) + len(path_b) - 2)
                contribution = Fraction(1, int(2**edges))
                total += contribution
                contributing.append(
                    {
                        "ancestor_id": str(ancestor_id),
                        "path_a": list(path_a),
                        "path_b": list(path_b),
                        "edge_count": int(edges),
                        "contribution": _format_relatedness_fraction(contribution),
                    }
                )
    return Fraction(total), tuple(contributing)


def _relatedness_annotation_roles(
    *,
    person_a_id: str,
    person_b_id: str,
    contributing_paths: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[str, str], ...]:
    roles: List[Tuple[str, str]] = [("person_a", str(person_a_id)), ("person_b", str(person_b_id))]
    seen_roles = {"person_a", "person_b"}
    for path_index, path_info in enumerate(contributing_paths, start=1):
        ancestor_id = str(path_info["ancestor_id"])
        role = f"shared_ancestor_{path_index}"
        if role not in seen_roles:
            roles.append((role, ancestor_id))
            seen_roles.add(role)
        for side_key, role_side in (("path_a", "a"), ("path_b", "b")):
            path = [str(item) for item in path_info.get(side_key, [])]
            middle_ids = tuple(person_id for person_id in path[1:-1] if person_id not in {str(person_a_id), str(person_b_id)})
            for middle_index, middle_id in enumerate(middle_ids, start=1):
                middle_role = f"path_{path_index}_{role_side}_{middle_index}"
                if middle_role in seen_roles:
                    continue
                roles.append((middle_role, str(middle_id)))
                seen_roles.add(middle_role)
    return tuple(roles)


def _relatedness_candidates(
    sample: PedigreeSample,
    target_relatedness: str,
) -> Tuple[PedigreeRelatednessQuerySample, ...]:
    candidates: List[PedigreeRelatednessQuerySample] = []
    people = tuple(sample.people)
    for index_a, person_a in enumerate(people):
        for person_b in people[index_a + 1 :]:
            fraction, contributing_paths = _relatedness_fraction_and_paths(
                sample,
                str(person_a.person_id),
                str(person_b.person_id),
            )
            answer = _format_relatedness_fraction(fraction)
            if str(answer) != str(target_relatedness):
                continue
            roles = _relatedness_annotation_roles(
                person_a_id=str(person_a.person_id),
                person_b_id=str(person_b.person_id),
                contributing_paths=contributing_paths,
            )
            candidates.append(
                PedigreeRelatednessQuerySample(
                    sample=sample,
                    answer=str(answer),
                    person_a_id=str(person_a.person_id),
                    person_b_id=str(person_b.person_id),
                    annotation_roles=tuple(roles),
                    contributing_paths=tuple(contributing_paths),
                )
            )
    return tuple(candidates)


def sample_pedigree_relatedness(
    instance_seed: int,
    *,
    target_relatedness: str,
    max_attempts: int = 100,
) -> PedigreeRelatednessQuerySample:
    """Sample one pedigree relatedness-coefficient query."""

    if str(target_relatedness) not in set(PEDIGREE_RELATEDNESS_LABELS):
        raise ValueError(f"unsupported relatedness label: {target_relatedness}")
    templates = _templates()
    for attempt in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), "pedigree_chart.relatedness", int(attempt))
        template = templates[int(rng.randrange(len(templates)))]
        sample = PedigreeSample(
            people=_assign_labels(template, ()),
            families=tuple(template.families),
            target_count=0,
            target_generation_index=None,
            target_sex=None,
            counted_person_ids=(),
            template_name=str(template.name),
        )
        candidates = _relatedness_candidates(sample, str(target_relatedness))
        if not candidates:
            continue
        return rng.choice(list(candidates))
    raise ValueError("unable to sample pedigree relatedness query")



__all__ = ["sample_pedigree_relatedness", "sample_pedigree_relationship"]
