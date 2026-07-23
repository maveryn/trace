"""Sampling primitives for phylogeny-tree graph scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import hash64, spawn_rng
from .algorithms import (
    canonical_phylogeny_signature,
    cherry_pairs,
    descendant_leaf_labels,
    internal_clade_node_ids,
    leaf_pair_for_mrca,
    mrca_node_id,
)
from .state import PHYLOGENY_TAXON_LABEL_POOL, PhylogenyNode, PhylogenySample


def _assign_depths(
    raw_nodes: Dict[str, Dict[str, Any]],
    *,
    root_id: str,
    depth: int = 0,
) -> None:
    raw_nodes[str(root_id)]["depth"] = int(depth)
    for child_id in raw_nodes[str(root_id)]["child_ids"]:
        _assign_depths(raw_nodes, root_id=str(child_id), depth=int(depth) + 1)


def sample_phylogeny_tree(
    instance_seed: int,
    *,
    leaf_count: int,
    labels: Sequence[str] | None = None,
    max_attempts: int = 200,
) -> PhylogenySample:
    """Sample one full binary rooted tree over the provided leaf labels."""

    if int(leaf_count) < 3:
        raise ValueError("phylogeny tree needs at least three leaves")
    rng = spawn_rng(int(instance_seed), "phylogeny_tree.sample")
    if labels is None:
        pool = list(PHYLOGENY_TAXON_LABEL_POOL)
        if int(leaf_count) > len(pool):
            raise ValueError("leaf_count exceeds phylogeny label pool")
        rng.shuffle(pool)
        leaf_labels = tuple(sorted(pool[: int(leaf_count)]))
    else:
        leaf_labels = tuple(str(label) for label in labels)
        if len(leaf_labels) != int(leaf_count):
            raise ValueError("labels length must match leaf_count")
        if len(set(leaf_labels)) != len(leaf_labels):
            raise ValueError("phylogeny labels must be unique")

    for attempt in range(max(1, int(max_attempts))):
        build_rng = spawn_rng(int(instance_seed), "phylogeny_tree.build", int(attempt))
        raw_nodes: Dict[str, Dict[str, Any]] = {
            f"leaf_{label}": {
                "parent_id": None,
                "child_ids": [],
                "leaf_label": str(label),
                "depth": 0,
            }
            for label in leaf_labels
        }
        clusters = [f"leaf_{label}" for label in leaf_labels]
        build_rng.shuffle(clusters)
        internal_index = 0
        while len(clusters) > 1:
            clusters.sort(key=lambda node_id: (len(_raw_descendant_labels(raw_nodes, node_id)), node_id))
            if len(clusters) >= 4 and build_rng.random() < 0.45:
                candidates = clusters[: max(3, len(clusters) // 2)]
            else:
                candidates = list(clusters)
            left_id = str(build_rng.choice(candidates))
            clusters.remove(left_id)
            candidates = [node_id for node_id in candidates if str(node_id) in set(clusters)] or list(clusters)
            right_id = str(build_rng.choice(candidates))
            clusters.remove(right_id)
            parent_id = f"internal_{internal_index}"
            internal_index += 1
            raw_nodes[parent_id] = {
                "parent_id": None,
                "child_ids": [left_id, right_id],
                "leaf_label": None,
                "depth": 0,
            }
            raw_nodes[left_id]["parent_id"] = parent_id
            raw_nodes[right_id]["parent_id"] = parent_id
            clusters.append(parent_id)
            build_rng.shuffle(clusters)
        root_id = str(clusters[0])
        _assign_depths(raw_nodes, root_id=root_id)
        nodes = tuple(
            PhylogenyNode(
                node_id=str(node_id),
                parent_id=(None if raw["parent_id"] is None else str(raw["parent_id"])),
                child_ids=tuple(str(child_id) for child_id in raw["child_ids"]),
                leaf_label=(None if raw["leaf_label"] is None else str(raw["leaf_label"])),
                depth=int(raw["depth"]),
            )
            for node_id, raw in sorted(raw_nodes.items(), key=lambda item: (int(item[1]["depth"]), str(item[0])))
        )
        sample = PhylogenySample(
            nodes=nodes,
            root_id=str(root_id),
            leaf_labels=tuple(sorted(leaf_labels)),
            canonical_signature=(),
        )
        signature = canonical_phylogeny_signature(sample)
        return PhylogenySample(
            nodes=nodes,
            root_id=str(root_id),
            leaf_labels=tuple(sorted(leaf_labels)),
            canonical_signature=tuple(signature),
        )
    raise ValueError("could not sample phylogeny tree")


def _raw_descendant_labels(raw_nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> Tuple[str, ...]:
    node = raw_nodes[str(node_id)]
    if node.get("leaf_label") is not None:
        return (str(node["leaf_label"]),)
    labels: List[str] = []
    for child_id in node.get("child_ids", []):
        labels.extend(_raw_descendant_labels(raw_nodes, str(child_id)))
    return tuple(sorted(labels))


def sample_phylogeny_with_clade_size(
    instance_seed: int,
    *,
    target_size: int,
    leaf_count_min: int,
    leaf_count_max: int,
    max_attempts: int,
) -> Tuple[PhylogenySample, str]:
    """Sample a tree with one non-root clade containing exactly target_size leaves."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            rng = spawn_rng(int(instance_seed), "phylogeny_tree.clade_size", int(attempt))
            low = max(int(target_size) + 1, int(leaf_count_min))
            high = max(low, int(leaf_count_max))
            leaf_count = int(rng.randint(low, high))
            sample = sample_phylogeny_tree(
                hash64(int(instance_seed), "phylogeny_tree.clade_tree", int(attempt)),
                leaf_count=int(leaf_count),
                max_attempts=50,
            )
            candidates = [
                node_id
                for node_id in internal_clade_node_ids(sample, min_leaf_count=int(target_size), max_leaf_count=int(target_size))
                if len(descendant_leaf_labels(sample, node_id)) == int(target_size)
            ]
            if not candidates:
                continue
            target_node_id = str(rng.choice(candidates))
            return sample, target_node_id
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not sample clade-size phylogeny: {last_error}")


def sample_phylogeny_with_cherry(
    instance_seed: int,
    *,
    leaf_count_min: int,
    leaf_count_max: int,
    max_attempts: int,
) -> Tuple[PhylogenySample, Tuple[str, str, str]]:
    """Sample a tree with a leaf sister pair."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            rng = spawn_rng(int(instance_seed), "phylogeny_tree.cherry", int(attempt))
            sample = sample_phylogeny_tree(
                hash64(int(instance_seed), "phylogeny_tree.cherry_tree", int(attempt)),
                leaf_count=int(rng.randint(int(leaf_count_min), int(leaf_count_max))),
                max_attempts=50,
            )
            candidates = list(cherry_pairs(sample))
            if not candidates:
                continue
            return sample, tuple(str(item) for item in rng.choice(candidates))  # type: ignore[return-value]
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not sample sister-leaf phylogeny: {last_error}")


def sample_phylogeny_with_mrca_size(
    instance_seed: int,
    *,
    target_size: int,
    leaf_count_min: int,
    leaf_count_max: int,
    max_attempts: int,
) -> Tuple[PhylogenySample, str, Tuple[str, str]]:
    """Sample a tree and leaf pair whose MRCA clade has target_size leaves."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            sample, node_id = sample_phylogeny_with_clade_size(
                hash64(int(instance_seed), "phylogeny_tree.mrca_tree", int(attempt)),
                target_size=int(target_size),
                leaf_count_min=int(leaf_count_min),
                leaf_count_max=int(leaf_count_max),
                max_attempts=20,
            )
            rng = spawn_rng(int(instance_seed), "phylogeny_tree.mrca_pair", int(attempt))
            leaf_pair = leaf_pair_for_mrca(sample, node_id, rng)
            if str(mrca_node_id(sample, leaf_pair[0], leaf_pair[1])) != str(node_id):
                continue
            return sample, str(node_id), tuple(str(label) for label in leaf_pair)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not sample MRCA-size phylogeny: {last_error}")


def sample_topology_outlier_options(
    instance_seed: int,
    *,
    leaf_count_min: int,
    leaf_count_max: int,
    option_count: int = 4,
    max_attempts: int = 200,
) -> Dict[str, Any]:
    """Build four option cladograms with one rooted-topology outlier."""

    if int(option_count) != 4:
        raise ValueError("phylogeny topology options require exactly four options")
    rng = spawn_rng(int(instance_seed), "phylogeny_tree.topology_options")
    leaf_count = int(rng.randint(int(leaf_count_min), int(leaf_count_max)))
    base = sample_phylogeny_tree(
        hash64(int(instance_seed), "phylogeny_tree.topology_base", 0),
        leaf_count=int(leaf_count),
        max_attempts=50,
    )
    outlier: PhylogenySample | None = None
    for attempt in range(max(1, int(max_attempts))):
        candidate = sample_phylogeny_tree(
            hash64(int(instance_seed), "phylogeny_tree.topology_outlier", int(attempt)),
            leaf_count=int(leaf_count),
            labels=base.leaf_labels,
            max_attempts=50,
        )
        if tuple(candidate.canonical_signature) != tuple(base.canonical_signature):
            outlier = candidate
            break
    if outlier is None:
        raise ValueError("could not build topology outlier")

    correct_index = int(rng.randrange(int(option_count)))
    option_specs = []
    for index in range(int(option_count)):
        label = chr(ord("A") + int(index))
        if int(index) == int(correct_index):
            sample = outlier
            role = "outlier"
        else:
            sample = base
            role = "equivalent"
        option_specs.append(
            {
                "option_label": str(label),
                "sample": sample,
                "role": str(role),
                "canonical_signature": tuple(sample.canonical_signature),
                "layout_seed": hash64(int(instance_seed), f"phylogeny_tree.option_layout.{label}", int(index)),
            }
        )
    return {
        "base_sample": base,
        "outlier_sample": outlier,
        "option_specs": tuple(option_specs),
        "answer_option_label": chr(ord("A") + int(correct_index)),
        "correct_option_index": int(correct_index),
        "leaf_count": int(leaf_count),
    }


__all__ = [
    "sample_phylogeny_tree",
    "sample_phylogeny_with_cherry",
    "sample_phylogeny_with_clade_size",
    "sample_phylogeny_with_mrca_size",
    "sample_topology_outlier_options",
]
