"""Scene-local algorithms for graph binary-tree tasks."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from .state import BinaryTreeSample, RelationSelection


COUNT_MODES: Tuple[str, ...] = (
    "leaf",
    "internal",
    "single_child",
    "two_child",
    "depth_level",
)
TRAVERSAL_ORDERS: Tuple[str, ...] = (
    "preorder",
    "inorder",
    "postorder",
    "level_order",
)
LOCAL_RELATION_KINDS: Tuple[str, ...] = (
    "parent",
    "left_child",
    "right_child",
    "sibling",
)


def sort_node_ids_by_level(node_ids: Sequence[str]) -> Tuple[str, ...]:
    """Return tree node ids in stable level-order path order."""

    return tuple(sorted((str(node_id) for node_id in node_ids), key=lambda value: (len(value), value)))


def node_type_counts(sample: BinaryTreeSample) -> Dict[str, int]:
    """Return structural node counts keyed by neutral count mode."""

    counts = {
        "leaf": 0,
        "internal": 0,
        "single_child": 0,
        "two_child": 0,
    }
    for node in sample.nodes:
        child_count = int(node.left_id is not None) + int(node.right_id is not None)
        if child_count == 0:
            counts["leaf"] += 1
        else:
            counts["internal"] += 1
        if child_count == 1:
            counts["single_child"] += 1
        if child_count == 2:
            counts["two_child"] += 1
    return counts


def labels_for_count_mode(
    sample: BinaryTreeSample,
    *,
    count_mode: str,
    target_depth: int | None = None,
) -> Tuple[str, ...]:
    """Return node labels selected by a neutral structural count mode."""

    mode = str(count_mode)
    labels: List[str] = []
    for node in sample.nodes:
        child_count = int(node.left_id is not None) + int(node.right_id is not None)
        if mode == "leaf" and child_count == 0:
            labels.append(str(node.label))
        elif mode == "internal" and child_count > 0:
            labels.append(str(node.label))
        elif mode == "single_child" and child_count == 1:
            labels.append(str(node.label))
        elif mode == "two_child" and child_count == 2:
            labels.append(str(node.label))
        elif mode == "depth_level" and target_depth is not None and int(node.depth) == int(target_depth):
            labels.append(str(node.label))
    return tuple(labels)


def traversal_labels(sample: BinaryTreeSample, traversal_order: str) -> Tuple[str, ...]:
    """Return ordered labels for one neutral traversal rule."""

    order = str(traversal_order)
    if order == "preorder":
        return tuple(sample.preorder_labels)
    if order == "inorder":
        return tuple(sample.inorder_labels)
    if order == "postorder":
        return tuple(sample.postorder_labels)
    if order == "level_order":
        return tuple(sample.level_order_labels)
    raise ValueError(f"unsupported binary-tree traversal order: {traversal_order}")


def ancestors_including_self(node_id: str) -> Tuple[str, ...]:
    """Return ancestor ids from one node up through the root."""

    return tuple(str(node_id)[:length] for length in range(len(str(node_id)), -1, -1))


def lowest_common_ancestor_id(node_id_a: str, node_id_b: str) -> str:
    """Return the node id for the lowest common ancestor of two node ids."""

    ancestors_a = set(ancestors_including_self(str(node_id_a)))
    for ancestor in ancestors_including_self(str(node_id_b)):
        if ancestor in ancestors_a:
            return str(ancestor)
    return ""


def choose_relation(
    rng,
    *,
    sample: BinaryTreeSample,
    relation_kind: str,
    answer_scope: str | None = None,
) -> RelationSelection:
    """Choose one satisfiable relation instance from a sampled tree."""

    node_by_id = {str(node.node_id): node for node in sample.nodes}
    relation = str(relation_kind)

    if relation == "parent":
        candidates = [node for node in sample.nodes if node.parent_id is not None]
        node = rng.choice(candidates)
        answer = node_by_id[str(node.parent_id)]
        return RelationSelection(
            query_labels=(str(node.label),),
            answer_label=str(answer.label),
            annotation_labels=(str(node.label), str(answer.label)),
            query_node_ids=(str(node.node_id),),
            answer_node_id=str(answer.node_id),
        )

    if relation == "left_child":
        candidates = [node for node in sample.nodes if node.left_id is not None]
        if not candidates:
            raise ValueError("no left-child candidates")
        node = rng.choice(candidates)
        answer = node_by_id[str(node.left_id)]
        return RelationSelection(
            query_labels=(str(node.label),),
            answer_label=str(answer.label),
            annotation_labels=(str(node.label), str(answer.label)),
            query_node_ids=(str(node.node_id),),
            answer_node_id=str(answer.node_id),
        )

    if relation == "right_child":
        candidates = [node for node in sample.nodes if node.right_id is not None]
        if not candidates:
            raise ValueError("no right-child candidates")
        node = rng.choice(candidates)
        answer = node_by_id[str(node.right_id)]
        return RelationSelection(
            query_labels=(str(node.label),),
            answer_label=str(answer.label),
            annotation_labels=(str(node.label), str(answer.label)),
            query_node_ids=(str(node.node_id),),
            answer_node_id=str(answer.node_id),
        )

    if relation == "sibling":
        candidates = []
        for node in sample.nodes:
            if node.parent_id is None:
                continue
            parent = node_by_id[str(node.parent_id)]
            if parent.left_id is not None and parent.right_id is not None:
                sibling_id = parent.right_id if str(parent.left_id) == str(node.node_id) else parent.left_id
                candidates.append((node, node_by_id[str(sibling_id)]))
        if not candidates:
            raise ValueError("no sibling candidates")
        node, answer = rng.choice(candidates)
        return RelationSelection(
            query_labels=(str(node.label),),
            answer_label=str(answer.label),
            annotation_labels=(str(node.label), str(answer.label)),
            query_node_ids=(str(node.node_id),),
            answer_node_id=str(answer.node_id),
        )

    if relation == "lowest_common_ancestor":
        candidates_lca_by_scope = {"root": [], "non_root": []}
        nodes = list(sample.nodes)
        for index_a, node_a in enumerate(nodes):
            for node_b in nodes[index_a + 1 :]:
                lca_id = lowest_common_ancestor_id(str(node_a.node_id), str(node_b.node_id))
                if lca_id in (str(node_a.node_id), str(node_b.node_id)):
                    continue
                if len(str(node_a.node_id)) <= len(lca_id) or len(str(node_b.node_id)) <= len(lca_id):
                    continue
                scope = "root" if str(lca_id) == "" else "non_root"
                candidates_lca_by_scope[str(scope)].append((node_a, node_b, node_by_id[str(lca_id)]))
        requested_scope = str(answer_scope or "")
        if requested_scope:
            if requested_scope not in candidates_lca_by_scope:
                raise ValueError(f"unsupported lowest-common-ancestor answer scope: {answer_scope}")
            candidates_lca = list(candidates_lca_by_scope[str(requested_scope)])
        else:
            candidates_lca = [
                *candidates_lca_by_scope["root"],
                *candidates_lca_by_scope["non_root"],
            ]
        if not candidates_lca:
            raise ValueError("no lowest-common-ancestor candidates")
        node_a, node_b, answer = rng.choice(candidates_lca)
        selected_scope = "root" if str(answer.node_id) == "" else "non_root"
        return RelationSelection(
            query_labels=(str(node_a.label), str(node_b.label)),
            answer_label=str(answer.label),
            annotation_labels=(str(node_a.label), str(node_b.label), str(answer.label)),
            query_node_ids=(str(node_a.node_id), str(node_b.node_id)),
            answer_node_id=str(answer.node_id),
            answer_scope=str(selected_scope),
        )

    raise ValueError(f"unsupported binary-tree relation kind: {relation_kind}")


def bst_insert_path(nodes: Dict[str, Dict[str, object]], key: int, *, max_depth: int) -> bool:
    """Insert one key into an in-memory BST while preserving a depth cap."""

    if not nodes:
        nodes[""] = {"key": int(key), "parent_id": None, "left_id": None, "right_id": None, "depth": 0}
        return True
    current_id = ""
    while True:
        current_key = int(nodes[str(current_id)]["key"])
        if int(key) == current_key:
            return False
        side = "L" if int(key) < current_key else "R"
        child_key = "left_id" if side == "L" else "right_id"
        child_id = nodes[str(current_id)][child_key]
        if child_id is None:
            node_id = f"{current_id}{side}"
            depth = len(node_id)
            if int(depth) > int(max_depth):
                return False
            nodes[node_id] = {
                "key": int(key),
                "parent_id": str(current_id),
                "left_id": None,
                "right_id": None,
                "depth": int(depth),
            }
            nodes[str(current_id)][child_key] = str(node_id)
            return True
        current_id = str(child_id)


def bst_search_path(nodes: Mapping[str, Mapping[str, object]], target_key: int) -> Tuple[str, ...]:
    """Return the visible BST comparison path for one target key."""

    path: List[str] = []
    current_id = ""
    while current_id in nodes:
        path.append(str(current_id))
        current_key = int(nodes[str(current_id)]["key"])
        if int(target_key) == current_key:
            break
        next_id = f"{current_id}{'L' if int(target_key) < current_key else 'R'}"
        if next_id not in nodes:
            break
        current_id = str(next_id)
    return tuple(path)


def array_index_to_node_id(index: int) -> str:
    """Map a heap array index to the corresponding binary-tree path id."""

    bits = bin(int(index) + 1)[3:]
    return "".join("L" if bit == "0" else "R" for bit in bits)


__all__ = [
    "COUNT_MODES",
    "LOCAL_RELATION_KINDS",
    "TRAVERSAL_ORDERS",
    "ancestors_including_self",
    "array_index_to_node_id",
    "bst_insert_path",
    "bst_search_path",
    "choose_relation",
    "labels_for_count_mode",
    "lowest_common_ancestor_id",
    "node_type_counts",
    "sort_node_ids_by_level",
    "traversal_labels",
]
