"""Topology algorithms for rooted phylogeny trees."""

from __future__ import annotations

from typing import List, Tuple

from .state import PhylogenySample, _leaf_node_id_by_label, _node_map


def descendant_leaf_labels(sample: PhylogenySample, node_id: str) -> Tuple[str, ...]:
    """Return all descendant leaf labels for one node."""

    nodes = _node_map(sample)

    def visit(current_id: str) -> List[str]:
        node = nodes[str(current_id)]
        if node.leaf_label is not None:
            return [str(node.leaf_label)]
        labels: List[str] = []
        for child_id in node.child_ids:
            labels.extend(visit(str(child_id)))
        return labels

    return tuple(sorted(visit(str(node_id))))


def canonical_phylogeny_signature(sample: PhylogenySample) -> Tuple[Tuple[str, ...], ...]:
    """Return a rooted topology signature ignoring child order and layout."""

    all_count = int(sample.leaf_count)
    clades = []
    for node in sample.nodes:
        if node.leaf_label is not None:
            continue
        leaves = tuple(descendant_leaf_labels(sample, str(node.node_id)))
        if 2 <= len(leaves) < all_count:
            clades.append(tuple(str(label) for label in leaves))
    return tuple(sorted(clades, key=lambda value: (len(value), value)))


def internal_clade_node_ids(
    sample: PhylogenySample,
    *,
    min_leaf_count: int = 2,
    max_leaf_count: int | None = None,
    include_root: bool = False,
) -> Tuple[str, ...]:
    """Return internal node ids with clade sizes inside the requested bounds."""

    upper = int(sample.leaf_count if max_leaf_count is None else max_leaf_count)
    result = []
    for node in sample.nodes:
        if node.leaf_label is not None:
            continue
        if (not include_root) and str(node.node_id) == str(sample.root_id):
            continue
        count = len(descendant_leaf_labels(sample, str(node.node_id)))
        if int(min_leaf_count) <= int(count) <= int(upper):
            result.append(str(node.node_id))
    return tuple(sorted(result))


def cherry_pairs(sample: PhylogenySample) -> Tuple[Tuple[str, str, str], ...]:
    """Return `(target_leaf, sister_leaf, parent_id)` candidates."""

    nodes = _node_map(sample)
    pairs = []
    for node in sample.nodes:
        if len(node.child_ids) != 2:
            continue
        children = [nodes[str(child_id)] for child_id in node.child_ids]
        if not all(child.leaf_label is not None for child in children):
            continue
        left_label = str(children[0].leaf_label)
        right_label = str(children[1].leaf_label)
        pairs.append((left_label, right_label, str(node.node_id)))
        pairs.append((right_label, left_label, str(node.node_id)))
    return tuple(sorted(pairs))


def mrca_node_id(sample: PhylogenySample, leaf_label_a: str, leaf_label_b: str) -> str:
    """Return the most recent common ancestor node id for two leaf labels."""

    nodes = _node_map(sample)
    leaf_to_id = _leaf_node_id_by_label(sample)
    id_a = str(leaf_to_id[str(leaf_label_a)])
    id_b = str(leaf_to_id[str(leaf_label_b)])
    ancestors_a: set[str] = set()
    current: str | None = id_a
    while current is not None:
        ancestors_a.add(str(current))
        current = nodes[str(current)].parent_id
    current = id_b
    while current is not None:
        if str(current) in ancestors_a:
            return str(current)
        current = nodes[str(current)].parent_id
    raise ValueError("phylogeny leaves have no common ancestor")


def leaf_pair_for_mrca(sample: PhylogenySample, node_id: str, rng) -> Tuple[str, str]:
    """Choose two leaves whose MRCA is exactly `node_id`."""

    nodes = _node_map(sample)
    node = nodes[str(node_id)]
    if len(node.child_ids) < 2:
        raise ValueError("MRCA candidate must have at least two child clades")
    child_groups = [
        tuple(descendant_leaf_labels(sample, str(child_id)))
        for child_id in node.child_ids
    ]
    non_empty = [group for group in child_groups if group]
    if len(non_empty) < 2:
        raise ValueError("MRCA candidate must have two non-empty child clades")
    left_group, right_group = rng.sample(non_empty, 2)
    leaf_a = str(rng.choice(list(left_group)))
    leaf_b = str(rng.choice(list(right_group)))
    if str(leaf_a) == str(leaf_b):
        raise ValueError("MRCA pair must use distinct leaves")
    return tuple(sorted((leaf_a, leaf_b)))  # type: ignore[return-value]


__all__ = [
    "canonical_phylogeny_signature",
    "cherry_pairs",
    "descendant_leaf_labels",
    "internal_clade_node_ids",
    "leaf_pair_for_mrca",
    "mrca_node_id",
]
