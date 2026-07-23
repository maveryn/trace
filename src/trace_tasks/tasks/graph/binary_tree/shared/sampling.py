"""Neutral sampling primitives for graph binary-tree scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .....core.seed import hash64, spawn_rng
from ....shared.config_defaults import group_default
from ...shared.label_assets import default_graph_label_bucket_weights, resolve_graph_node_labels
from .algorithms import (
    array_index_to_node_id,
    bst_insert_path,
    bst_search_path,
    choose_relation,
    labels_for_count_mode,
    node_type_counts,
    sort_node_ids_by_level,
)
from .defaults import BinaryTreeDefaults
from .state import BinaryTreeNode, BinaryTreeSample, OperationSelection, RelationSelection


def support_for_count_mode(
    count_mode: str,
    *,
    gen_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
) -> Tuple[int, ...]:
    """Return feasible answer support for one neutral count mode."""

    lower = int(group_default(gen_defaults, "target_count_min", defaults.target_count_min))
    upper = int(group_default(gen_defaults, "target_count_max", defaults.target_count_max))
    node_count_min = int(group_default(gen_defaults, "node_count_min", defaults.node_count_min))
    node_count_max = int(group_default(gen_defaults, "node_count_max", defaults.node_count_max))
    mode = str(count_mode)
    if mode == "leaf":
        return tuple(range(max(2, lower), int(upper) + 1))
    if mode == "two_child":
        return tuple(range(max(1, lower), int(upper) + 1))
    if mode == "internal":
        internal_min = int(group_default(gen_defaults, "internal_count_min", defaults.internal_count_min))
        internal_max = int(group_default(gen_defaults, "internal_count_max", defaults.internal_count_max))
        return tuple(
            internal_count
            for internal_count in range(int(internal_min), int(internal_max) + 1)
            if any(
                int(node_count_min) <= (int(internal_count) + int(two_child_count) + 1) <= int(node_count_max)
                for two_child_count in range(1, int(internal_count) + 1)
            )
        )
    return tuple(range(max(1, lower), int(upper) + 1))


def depth_support_for_target(
    target_count: int,
    *,
    gen_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
) -> Tuple[int, ...]:
    """Return feasible depth choices for a requested depth-level count."""

    lower = int(group_default(gen_defaults, "depth_min", defaults.depth_min))
    upper = int(group_default(gen_defaults, "depth_max", defaults.depth_max))
    return tuple(depth for depth in range(int(lower), int(upper) + 1) if int(target_count) <= (2 ** int(depth)))


def _children_for_type(rng, node_type: str) -> Tuple[str, ...]:
    if str(node_type) == "two":
        return ("L", "R")
    if bool(rng.randrange(2)):
        return ("L",)
    return ("R",)


def _child_key(side: str) -> str:
    if str(side) == "L":
        return "left_id"
    if str(side) == "R":
        return "right_id"
    raise ValueError(f"unsupported binary-tree child side: {side}")


def _build_tree_from_internal_counts(
    rng,
    *,
    two_child_count: int,
    single_child_count: int,
    max_depth: int,
    max_attempts: int = 200,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Construct an ordered binary tree with exact internal-node type counts."""

    two_count = int(two_child_count)
    single_count = int(single_child_count)
    if two_count < 1:
        raise ValueError("binary tree construction requires at least one two-child node")
    if single_count < 0:
        raise ValueError("single_child_count must be non-negative")
    max_allowed_depth = max(2, int(max_depth))

    for _attempt in range(max(1, int(max_attempts))):
        nodes: Dict[str, Dict[str, Any]] = {
            "": {"parent_id": None, "left_id": None, "right_id": None, "depth": 0}
        }
        remaining_types: List[str] = ["two"] * max(0, two_count - 1) + ["single"] * single_count
        rng.shuffle(remaining_types)
        open_slots: List[Tuple[str, str]] = [("", "L"), ("", "R")]
        failed = False
        while remaining_types:
            feasible_slot_indices = [
                index
                for index, (parent_id, _side) in enumerate(open_slots)
                if int(nodes[str(parent_id)]["depth"]) + 1 < max_allowed_depth
            ]
            if not feasible_slot_indices:
                failed = True
                break
            slot_index = int(rng.choice(feasible_slot_indices))
            parent_id, side = open_slots.pop(slot_index)
            node_type = remaining_types.pop(0)
            node_id = f"{parent_id}{side}"
            depth = int(nodes[str(parent_id)]["depth"]) + 1
            nodes[node_id] = {"parent_id": str(parent_id), "left_id": None, "right_id": None, "depth": int(depth)}
            nodes[str(parent_id)][_child_key(str(side))] = str(node_id)
            child_sides = list(_children_for_type(rng, str(node_type)))
            rng.shuffle(child_sides)
            for child_side in child_sides:
                open_slots.append((str(node_id), str(child_side)))
            rng.shuffle(open_slots)
        if failed:
            continue

        for parent_id, side in list(open_slots):
            node_id = f"{parent_id}{side}"
            depth = int(nodes[str(parent_id)]["depth"]) + 1
            if depth > max_allowed_depth:
                failed = True
                break
            nodes[node_id] = {"parent_id": str(parent_id), "left_id": None, "right_id": None, "depth": int(depth)}
            nodes[str(parent_id)][_child_key(str(side))] = str(node_id)
        if failed:
            continue
        return nodes, ""

    raise ValueError("could not construct binary tree with requested node-type counts")


def _binary_path_product(length: int) -> Tuple[Tuple[str, ...], ...]:
    if int(length) <= 0:
        return ((),)
    previous = _binary_path_product(int(length) - 1)
    return tuple((*prefix, side) for prefix in previous for side in ("L", "R"))


def _build_tree_for_depth_count(
    rng,
    *,
    target_depth: int,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
    max_depth: int,
    compact_selection: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Construct a tree with an exact number of nodes at a target depth."""

    depth = int(target_depth)
    count = int(target_count)
    if depth < 1 or count < 1:
        raise ValueError("target_depth and target_count must be positive for depth-level count")
    if count > 2 ** depth:
        raise ValueError("target_count is infeasible for the requested depth")

    positions = ["".join(bits) for bits in _binary_path_product(depth)]
    if bool(compact_selection):
        max_start = max(0, len(positions) - int(count))
        start = int(rng.randint(0, int(max_start))) if int(max_start) > 0 else 0
        selected = set(positions[int(start) : int(start) + int(count)])
    else:
        rng.shuffle(positions)
        selected = set(positions[:count])
    node_ids = {""}
    for path in selected:
        for length in range(1, len(path) + 1):
            node_ids.add(path[:length])

    if len(node_ids) > int(node_count_max):
        raise ValueError("depth target creates more nodes than node_count_max")

    max_allowed_depth = max(int(depth), int(max_depth))
    open_slots = [
        f"{node_id}{side}"
        for node_id in sorted(node_ids, key=lambda value: (len(value), value))
        for side in ("L", "R")
        if f"{node_id}{side}" not in node_ids
        and len(f"{node_id}{side}") != depth
        and len(f"{node_id}{side}") <= max_allowed_depth
    ]
    rng.shuffle(open_slots)
    target_total = int(rng.randint(max(len(node_ids), int(node_count_min)), int(node_count_max)))
    while len(node_ids) < target_total and open_slots:
        slot = str(open_slots.pop(0))
        if len(slot) == depth:
            continue
        parent_id = slot[:-1]
        if parent_id not in node_ids:
            continue
        node_ids.add(slot)
        if len(slot) < max_allowed_depth:
            child_slots = [
                f"{slot}{side}"
                for side in ("L", "R")
                if len(f"{slot}{side}") != depth and len(f"{slot}{side}") <= max_allowed_depth
            ]
            rng.shuffle(child_slots)
            open_slots.extend(child_slots)
            rng.shuffle(open_slots)

    nodes: Dict[str, Dict[str, Any]] = {
        node_id: {"parent_id": None if node_id == "" else node_id[:-1], "left_id": None, "right_id": None, "depth": len(node_id)}
        for node_id in node_ids
    }
    for node_id in list(node_ids):
        if node_id == "":
            continue
        parent_id = node_id[:-1]
        side = node_id[-1]
        nodes[str(parent_id)][_child_key(str(side))] = str(node_id)
    actual = sum(1 for node_id in node_ids if len(node_id) == depth)
    if int(actual) != int(count):
        raise ValueError("depth-level construction failed to preserve target count")
    if len(node_ids) < int(node_count_min) or len(node_ids) > int(node_count_max):
        raise ValueError("depth-level construction fell outside configured node count range")
    return nodes, ""


def assign_labels_to_tree(
    rng,
    *,
    nodes: Mapping[str, Mapping[str, Any]],
    label_variant: str,
    max_chars: int,
    min_chars: int | None = None,
    bucket_weights: Mapping[str, float] | None = None,
) -> BinaryTreeSample:
    """Attach graph-label resources to an ordered binary-tree topology."""

    node_ids = sort_node_ids_by_level(tuple(nodes.keys()))
    labels = resolve_graph_node_labels(
        rng,
        label_variant=str(label_variant),
        object_count=len(node_ids),
        max_chars=int(max_chars),
        min_chars=min_chars,
        bucket_weights=bucket_weights if bucket_weights is not None else default_graph_label_bucket_weights(),
        sequential_numbers=True,
    )
    label_by_id = {str(node_id): str(label) for node_id, label in zip(node_ids, labels.labels)}

    tree_nodes = tuple(
        BinaryTreeNode(
            node_id=str(node_id),
            label=str(label_by_id[str(node_id)]),
            parent_id=str(nodes[str(node_id)]["parent_id"]) if nodes[str(node_id)].get("parent_id") is not None else None,
            left_id=str(nodes[str(node_id)]["left_id"]) if nodes[str(node_id)].get("left_id") is not None else None,
            right_id=str(nodes[str(node_id)]["right_id"]) if nodes[str(node_id)].get("right_id") is not None else None,
            depth=int(nodes[str(node_id)]["depth"]),
        )
        for node_id in node_ids
    )
    return sample_from_nodes(
        tree_nodes,
        label_variant=str(labels.label_variant),
        label_source_kind=str(labels.label_source_kind),
        label_bucket=str(labels.label_bucket),
        label_manifest=str(labels.label_manifest),
        label_filter=dict(labels.label_filter),
        label_bucket_probabilities=dict(labels.label_bucket_probabilities),
    )


def sample_from_nodes(
    nodes: Tuple[BinaryTreeNode, ...],
    *,
    label_variant: str,
    label_source_kind: str,
    label_bucket: str,
    label_manifest: str,
    label_filter: Mapping[str, Any],
    label_bucket_probabilities: Mapping[str, float],
) -> BinaryTreeSample:
    """Build a ``BinaryTreeSample`` and derived traversal orders from nodes."""

    node_by_id = {str(node.node_id): node for node in nodes}

    def preorder(node_id: str) -> List[str]:
        node = node_by_id[str(node_id)]
        result = [str(node.label)]
        if node.left_id is not None:
            result.extend(preorder(str(node.left_id)))
        if node.right_id is not None:
            result.extend(preorder(str(node.right_id)))
        return result

    def inorder(node_id: str) -> List[str]:
        node = node_by_id[str(node_id)]
        result: List[str] = []
        if node.left_id is not None:
            result.extend(inorder(str(node.left_id)))
        result.append(str(node.label))
        if node.right_id is not None:
            result.extend(inorder(str(node.right_id)))
        return result

    def postorder(node_id: str) -> List[str]:
        node = node_by_id[str(node_id)]
        result: List[str] = []
        if node.left_id is not None:
            result.extend(postorder(str(node.left_id)))
        if node.right_id is not None:
            result.extend(postorder(str(node.right_id)))
        result.append(str(node.label))
        return result

    ordered_ids = sort_node_ids_by_level(tuple(node_by_id))
    return BinaryTreeSample(
        nodes=nodes,
        root_id="",
        label_variant=str(label_variant),
        label_source_kind=str(label_source_kind),
        label_bucket=str(label_bucket),
        label_manifest=str(label_manifest),
        label_filter=dict(label_filter),
        label_bucket_probabilities=dict(label_bucket_probabilities),
        preorder_labels=tuple(preorder("")),
        inorder_labels=tuple(inorder("")),
        postorder_labels=tuple(postorder("")),
        level_order_labels=tuple(str(node_by_id[node_id].label) for node_id in ordered_ids),
    )


def sample_structure_count_tree(
    instance_seed: int,
    *,
    count_mode: str,
    target_count: int,
    target_depth: int | None,
    node_count_min: int,
    node_count_max: int,
    max_depth: int,
    label_variant: str,
    label_max_chars: int,
    max_attempts: int = 200,
) -> BinaryTreeSample:
    """Sample a binary tree whose selected structural mode has the target count."""

    mode = str(count_mode)
    target = int(target_count)
    if target < 0:
        raise ValueError("target_count must be non-negative")
    rng = spawn_rng(int(instance_seed), f"binary_tree_count.{mode}.{target}.{target_depth}")

    if mode == "depth_level":
        if target_depth is None:
            raise ValueError("target_depth is required for depth-level count")
        for attempt in range(max(1, int(max_attempts))):
            try:
                nodes, _root = _build_tree_for_depth_count(
                    rng,
                    target_depth=int(target_depth),
                    target_count=int(target),
                    node_count_min=int(node_count_min),
                    node_count_max=int(node_count_max),
                    max_depth=int(max_depth),
                )
            except ValueError:
                continue
            sample = assign_labels_to_tree(
                spawn_rng(int(instance_seed), f"binary_tree_count.labels.{mode}.{attempt}"),
                nodes=nodes,
                label_variant=str(label_variant),
                max_chars=int(label_max_chars),
            )
            actual = sum(1 for node in sample.nodes if int(node.depth) == int(target_depth))
            if int(actual) == int(target):
                return sample
        for fallback_attempt in range(32):
            try:
                nodes, _root = _build_tree_for_depth_count(
                    rng,
                    target_depth=int(target_depth),
                    target_count=int(target),
                    node_count_min=int(node_count_min),
                    node_count_max=int(node_count_max),
                    max_depth=int(max_depth),
                    compact_selection=True,
                )
            except ValueError:
                continue
            sample = assign_labels_to_tree(
                spawn_rng(int(instance_seed), f"binary_tree_count.labels.{mode}.compact.{fallback_attempt}"),
                nodes=nodes,
                label_variant=str(label_variant),
                max_chars=int(label_max_chars),
            )
            actual = sum(1 for node in sample.nodes if int(node.depth) == int(target_depth))
            if int(actual) == int(target):
                return sample
        raise ValueError("could not sample depth-level binary tree")

    for attempt in range(max(1, int(max_attempts))):
        if mode == "leaf":
            two_count = max(1, target - 1)
            single_max = max(0, int(node_count_max) - ((2 * int(two_count)) + 1))
            single_min = max(0, int(node_count_min) - ((2 * int(two_count)) + 1))
            single_count = int(rng.randint(int(single_min), int(max(single_min, single_max)))) if single_max >= single_min else int(single_min)
        elif mode == "two_child":
            two_count = max(1, target)
            single_max = max(0, int(node_count_max) - ((2 * int(two_count)) + 1))
            single_min = max(0, int(node_count_min) - ((2 * int(two_count)) + 1))
            single_count = int(rng.randint(int(single_min), int(max(single_min, single_max)))) if single_max >= single_min else int(single_min)
        elif mode == "single_child":
            single_count = int(target)
            feasible_two = [
                two
                for two in range(1, max(2, int(node_count_max)))
                if int(node_count_min) <= ((2 * int(two)) + 1 + int(single_count)) <= int(node_count_max)
            ]
            if not feasible_two:
                raise ValueError("no feasible two-child count for requested single-child target")
            two_count = int(rng.choice(feasible_two))
        else:
            internal_count = int(target)
            feasible_two = [
                two
                for two in range(1, int(internal_count) + 1)
                if int(node_count_min) <= (int(internal_count) + int(two) + 1) <= int(node_count_max)
            ]
            if not feasible_two:
                raise ValueError("no feasible two-child count for requested internal-node target")
            two_count = int(rng.choice(feasible_two))
            single_count = int(internal_count) - int(two_count)

        try:
            nodes, _root = _build_tree_from_internal_counts(
                rng,
                two_child_count=int(two_count),
                single_child_count=int(single_count),
                max_depth=int(max_depth),
            )
        except ValueError:
            continue
        sample = assign_labels_to_tree(
            spawn_rng(int(instance_seed), f"binary_tree_count.labels.{mode}.{attempt}"),
            nodes=nodes,
            label_variant=str(label_variant),
            max_chars=int(label_max_chars),
        )
        counts = node_type_counts(sample)
        actual = int(counts[str(mode)])
        if actual == int(target):
            return sample
    raise ValueError("could not sample binary tree for count mode")


def sample_traversal_tree(
    instance_seed: int,
    *,
    node_count_min: int,
    node_count_max: int,
    max_depth: int,
    label_variant: str,
    label_max_chars: int,
    max_attempts: int = 200,
) -> BinaryTreeSample:
    """Sample a varied binary tree for traversal-order tasks."""

    rng = spawn_rng(int(instance_seed), "binary_tree_traversal.shape")
    feasible_pairs = [
        (two, single)
        for two in range(1, max(2, int(node_count_max)))
        for single in range(0, max(1, int(node_count_max)))
        if int(node_count_min) <= ((2 * int(two)) + 1 + int(single)) <= int(node_count_max)
    ]
    if not feasible_pairs:
        raise ValueError("no feasible binary-tree traversal support")
    for attempt in range(max(1, int(max_attempts))):
        two_count, single_count = rng.choice(feasible_pairs)
        try:
            nodes, _root = _build_tree_from_internal_counts(
                rng,
                two_child_count=int(two_count),
                single_child_count=int(single_count),
                max_depth=int(max_depth),
            )
        except ValueError:
            continue
        sample = assign_labels_to_tree(
            spawn_rng(int(instance_seed), f"binary_tree_traversal.labels.{attempt}"),
            nodes=nodes,
            label_variant=str(label_variant),
            max_chars=int(label_max_chars),
        )
        if int(sample.node_count) >= int(node_count_min) and int(sample.node_count) <= int(node_count_max):
            return sample
    raise ValueError("could not sample binary tree for traversal task")


def sample_relation_tree(
    instance_seed: int,
    *,
    relation_kind: str,
    relation_answer_scope: str = "",
    node_count_min: int,
    node_count_max: int,
    max_depth: int,
    label_variant: str,
    label_max_chars: int,
    max_attempts: int,
) -> Tuple[BinaryTreeSample, RelationSelection]:
    """Sample a tree and relation selection for a neutral relation kind."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            sample = sample_traversal_tree(
                hash64(int(instance_seed), "binary_tree.relation.tree", int(attempt)),
                node_count_min=int(node_count_min),
                node_count_max=int(node_count_max),
                max_depth=int(max_depth),
                label_variant=str(label_variant),
                label_max_chars=int(label_max_chars),
                max_attempts=max(20, int(max_attempts)),
            )
            rng = spawn_rng(int(instance_seed), f"binary_tree.relation.{relation_kind}.{attempt}")
            relation = choose_relation(
                rng,
                sample=sample,
                relation_kind=str(relation_kind),
                answer_scope=str(relation_answer_scope),
            )
            return sample, relation
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not sample binary-tree relation instance: {last_error}")


def _sample_unique_keys(rng, *, node_count: int, key_min: int, key_max: int) -> List[int]:
    support = list(range(int(key_min), int(key_max) + 1))
    if len(support) < int(node_count) + 6:
        raise ValueError("key range is too small for search-tree operation sampling")
    rng.shuffle(support)
    return sorted(int(value) for value in support[: int(node_count)])


def _sample_bst_nodes(
    instance_seed: int,
    *,
    node_count: int,
    key_min: int,
    key_max: int,
    max_depth: int,
    max_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    for attempt in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"binary_tree.operation.bst.{attempt}")
        keys = _sample_unique_keys(rng, node_count=int(node_count), key_min=int(key_min), key_max=int(key_max))
        insertion_order = list(keys)
        rng.shuffle(insertion_order)
        nodes: Dict[str, Dict[str, Any]] = {}
        failed = False
        for key in insertion_order:
            if not bst_insert_path(nodes, int(key), max_depth=int(max_depth)):
                failed = True
                break
        if not failed and len(nodes) == int(node_count):
            return nodes
    raise ValueError("could not sample a bounded-depth BST")


def _sample_heap_nodes(
    instance_seed: int,
    *,
    node_count: int,
    violation_gap_min: int,
    violation_gap_max: int,
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[str, str]]:
    rng = spawn_rng(int(instance_seed), "binary_tree.operation.heap")
    node_ids = [array_index_to_node_id(index) for index in range(int(node_count))]
    base_values = {node_id: 12 + (len(node_id) * 18) + index for index, node_id in enumerate(node_ids)}
    non_root_ids = [node_id for node_id in node_ids if node_id]
    violation_child_id = str(rng.choice(non_root_ids))
    violation_parent_id = violation_child_id[:-1]
    parent_value = int(base_values[violation_parent_id])
    min_gap = max(1, int(violation_gap_min))
    max_gap = max(min_gap, int(violation_gap_max))
    max_gap = min(max_gap, max(1, parent_value - 1))
    gap = int(rng.randint(min_gap, max_gap))
    base_values[violation_child_id] = parent_value - gap
    used = set()
    for node_id in node_ids:
        value = int(base_values[node_id])
        while value in used:
            value += 1
        base_values[node_id] = value
        used.add(value)
    nodes: Dict[str, Dict[str, Any]] = {}
    node_id_set = set(node_ids)
    for node_id in node_ids:
        left_id = f"{node_id}L" if f"{node_id}L" in node_id_set else None
        right_id = f"{node_id}R" if f"{node_id}R" in node_id_set else None
        nodes[node_id] = {
            "key": int(base_values[node_id]),
            "parent_id": node_id[:-1] if node_id else None,
            "left_id": left_id,
            "right_id": right_id,
            "depth": len(node_id),
        }
    violations = []
    for child_id in non_root_ids:
        parent_id = child_id[:-1]
        if int(nodes[parent_id]["key"]) > int(nodes[child_id]["key"]):
            violations.append((parent_id, child_id))
    if len(violations) != 1:
        raise ValueError("heap sampler failed to create exactly one min-heap violation")
    return nodes, tuple(violations[0])


def _numeric_sample_from_nodes(nodes: Mapping[str, Mapping[str, Any]]) -> BinaryTreeSample:
    node_ids = sort_node_ids_by_level(tuple(nodes.keys()))
    tree_nodes = tuple(
        BinaryTreeNode(
            node_id=str(node_id),
            label=str(nodes[str(node_id)]["key"]),
            parent_id=str(nodes[str(node_id)]["parent_id"]) if nodes[str(node_id)].get("parent_id") is not None else None,
            left_id=str(nodes[str(node_id)]["left_id"]) if nodes[str(node_id)].get("left_id") is not None else None,
            right_id=str(nodes[str(node_id)]["right_id"]) if nodes[str(node_id)].get("right_id") is not None else None,
            depth=int(nodes[str(node_id)]["depth"]),
        )
        for node_id in node_ids
    )
    values = [int(node.label) for node in tree_nodes]
    return sample_from_nodes(
        tree_nodes,
        label_variant="numeric_keys",
        label_source_kind="synthetic_numeric_keys",
        label_bucket="numeric_keys",
        label_manifest="synthetic_numeric_keys_v0",
        label_filter={"min": int(min(values)), "max": int(max(values))},
        label_bucket_probabilities={"numeric_keys": 1.0},
    )


def sample_search_tree_operation(
    instance_seed: int,
    *,
    operation_kind: str,
    node_count: int,
    key_min: int,
    key_max: int,
    heap_violation_gap_min: int,
    heap_violation_gap_max: int,
    max_depth: int,
    max_attempts: int,
) -> Tuple[BinaryTreeSample, OperationSelection]:
    """Sample a numeric BST or heap operation using a neutral operation kind."""

    operation = str(operation_kind)
    if operation == "heap_violation":
        nodes, (parent_id, child_id) = _sample_heap_nodes(
            int(instance_seed),
            node_count=int(node_count),
            violation_gap_min=int(heap_violation_gap_min),
            violation_gap_max=int(heap_violation_gap_max),
        )
        sample = _numeric_sample_from_nodes(nodes)
        parent_label = str(nodes[str(parent_id)]["key"])
        child_label = str(nodes[str(child_id)]["key"])
        return sample, OperationSelection(
            target_key=None,
            answer_label=str(child_label),
            annotation_labels=(str(parent_label), str(child_label)),
            query_node_ids=(str(parent_id), str(child_id)),
            answer_node_id=str(child_id),
            operation_kind="min_heap_property_violation",
        )

    nodes = _sample_bst_nodes(
        int(instance_seed),
        node_count=int(node_count),
        key_min=int(key_min),
        key_max=int(key_max),
        max_depth=int(max_depth),
        max_attempts=max(1, int(max_attempts)),
    )
    rng = spawn_rng(int(instance_seed), f"binary_tree.operation.{operation}")
    if operation == "bst_search_terminal" and rng.randrange(2):
        target_key = int(rng.choice([int(node["key"]) for node in nodes.values()]))
    else:
        existing = {int(node["key"]) for node in nodes.values()}
        candidates = [key for key in range(int(key_min), int(key_max) + 1) if key not in existing]
        if not candidates:
            raise ValueError("no missing key candidate")
        target_key = int(rng.choice(candidates))
    path_ids = bst_search_path(nodes, int(target_key))
    answer_node_id = str(path_ids[-1])
    sample = _numeric_sample_from_nodes(nodes)
    labels_by_id = {str(node.node_id): str(node.label) for node in sample.nodes}
    return sample, OperationSelection(
        target_key=int(target_key),
        answer_label=str(labels_by_id[str(answer_node_id)]),
        annotation_labels=tuple(str(labels_by_id[str(node_id)]) for node_id in path_ids),
        query_node_ids=tuple(str(node_id) for node_id in path_ids),
        answer_node_id=str(answer_node_id),
        operation_kind=str(operation),
    )


__all__ = [
    "assign_labels_to_tree",
    "depth_support_for_target",
    "labels_for_count_mode",
    "sample_from_nodes",
    "sample_relation_tree",
    "sample_search_tree_operation",
    "sample_structure_count_tree",
    "sample_traversal_tree",
    "support_for_count_mode",
]
