"""Sampling primitives for the sunburst chart scene."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.palette import composition_hsv_color, lighten_rgb as lighten
from trace_tasks.tasks.charts.shared.composition.values import select_unique_extremum
from trace_tasks.tasks.charts.shared.label_assets import normalize_chart_label_for_collision
from trace_tasks.tasks.shared.name_assets import load_label_manifest

from .defaults import generation_value, int_sequence, resolve_bounds
from .state import RGB, SunburstNode, SunburstTree


PARENT_LABEL_MANIFESTS: tuple[str, ...] = (
    "panel_titles/technical_topics.txt",
    "industries/industries_bls_qcew.txt",
    "categories/abstract_group_labels.txt",
)
SUBGROUP_LABEL_MANIFESTS: tuple[str, ...] = (
    "categories/product_labels.txt",
    "categories/status_labels.txt",
    "categories/priority_labels.txt",
    "categories/abstract_group_labels.txt",
    "panel_titles/technical_topics.txt",
)
LEAF_LABEL_MANIFESTS: tuple[str, ...] = (
    "categories/product_labels.txt",
    "categories/status_labels.txt",
    "categories/priority_labels.txt",
    "panel_titles/technical_topics.txt",
    "organizations/company_tickers_sec.txt",
)
LEGACY_PARENT_LABELS: tuple[str, ...] = (
    "Healthcare",
    "Transport",
    "Commercial",
    "Historical",
    "Athletics",
    "Culture",
    "Natural",
    "Education",
)


def parent_nodes(tree: SunburstTree) -> tuple[SunburstNode, ...]:
    nodes = nodes_by_id(tree)
    return tuple(nodes[str(parent_id)] for parent_id in tree.parent_ids)


def nodes_by_id(tree: SunburstTree) -> dict[str, SunburstNode]:
    return {str(node.node_id): node for node in tree.nodes}


def descendant_leaf_ids(node_lookup: Mapping[str, SunburstNode], node_ref: str) -> tuple[str, ...]:
    node = node_lookup[str(node_ref)]
    if node.level == "leaf":
        return (str(node_ref),)
    leaves: list[str] = []
    for child_id in node.child_ids:
        leaves.extend(descendant_leaf_ids(node_lookup, str(child_id)))
    return tuple(leaves)


def hierarchy_rows(tree: SunburstTree) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node.node_id),
            "label": str(node.label),
            "level": str(node.level),
            "parent_id": str(node.parent_id) if node.parent_id is not None else None,
            "value": int(node.value),
            "child_ids": [str(child_id) for child_id in node.child_ids],
        }
        for node in tree.nodes
    ]


def sample_tree(params: Mapping[str, Any], *, instance_seed: int) -> SunburstTree:
    """Build a balanced hierarchy with unique leaf values for deterministic tasks."""

    parent_min, parent_max = resolve_bounds(
        params,
        min_key="sunburst_parent_count_min",
        max_key="sunburst_parent_count_max",
        fallback_min=4,
        fallback_max=5,
    )
    subgroup_min, subgroup_max = resolve_bounds(
        params,
        min_key="sunburst_subgroup_count_min",
        max_key="sunburst_subgroup_count_max",
        fallback_min=2,
        fallback_max=3,
    )
    leaf_min, leaf_max = resolve_bounds(
        params,
        min_key="sunburst_leaf_count_min",
        max_key="sunburst_leaf_count_max",
        fallback_min=1,
        fallback_max=2,
    )
    value_min, value_max = resolve_bounds(
        params,
        min_key="sunburst_leaf_value_min",
        max_key="sunburst_leaf_value_max",
        fallback_min=5,
        fallback_max=40,
    )
    value_step = max(1, int(generation_value(params, "sunburst_leaf_value_step", 5)))
    rng = spawn_rng(int(instance_seed), "charts.sunburst.tree")
    parent_count = int(rng.randint(int(parent_min), int(parent_max)))
    parent_label_max_chars = int(generation_value(params, "sunburst_parent_label_max_chars", 10))
    subgroup_label_max_chars = int(generation_value(params, "sunburst_subgroup_label_max_chars", 8))
    leaf_label_max_chars = int(generation_value(params, "sunburst_leaf_label_max_chars", 7))
    subgroup_counts = [
        int(rng.randint(int(subgroup_min), int(subgroup_max)))
        for _ in range(int(parent_count))
    ]
    leaf_counts_by_parent = [
        [
            int(rng.randint(int(leaf_min), int(leaf_max)))
            for _ in range(int(subgroup_count))
        ]
        for subgroup_count in subgroup_counts
    ]
    parent_labels, subgroup_labels, leaf_labels = _sample_hierarchy_labels(
        parent_count=int(parent_count),
        subgroup_counts=tuple(subgroup_counts),
        leaf_counts_by_parent=tuple(tuple(counts) for counts in leaf_counts_by_parent),
        parent_max_chars=int(parent_label_max_chars),
        subgroup_max_chars=int(subgroup_label_max_chars),
        leaf_max_chars=int(leaf_label_max_chars),
        instance_seed=int(instance_seed),
    )

    built_nodes: dict[str, SunburstNode] = {}
    parent_ids: list[str] = []
    subgroup_ids: list[str] = []
    leaf_ids: list[str] = []
    leaf_value_low = int(math.ceil(int(value_min) / int(value_step)))
    leaf_value_high = int(math.floor(int(value_max) / int(value_step)))
    if leaf_value_low > leaf_value_high:
        raise ValueError("sunburst leaf value range is incompatible with value step")

    subgroup_label_index = 0
    leaf_label_index = 0
    for parent_index in range(int(parent_count)):
        parent_id = f"parent_{parent_index}"
        parent_ids.append(parent_id)
        parent_color = _parent_color(parent_index, parent_count, instance_seed=int(instance_seed))
        subgroup_count = int(subgroup_counts[int(parent_index)])
        value_units = list(range(int(leaf_value_low), int(leaf_value_high) + 1))
        rng.shuffle(value_units)
        parent_child_ids: list[str] = []
        parent_value = 0
        for subgroup_index in range(int(subgroup_count)):
            subgroup_id = f"{parent_id}_subgroup_{subgroup_index}"
            subgroup_ids.append(subgroup_id)
            parent_child_ids.append(subgroup_id)
            subgroup_label = str(subgroup_labels[int(subgroup_label_index)])
            subgroup_label_index += 1
            leaf_count = int(leaf_counts_by_parent[int(parent_index)][int(subgroup_index)])
            subgroup_child_ids: list[str] = []
            subgroup_value = 0
            for leaf_index in range(int(leaf_count)):
                leaf_label = str(leaf_labels[int(leaf_label_index)])
                leaf_label_index += 1
                leaf_id = f"{subgroup_id}_leaf_{leaf_index}"
                if value_units:
                    value_unit = int(value_units.pop())
                else:
                    value_unit = int(rng.randint(leaf_value_low, leaf_value_high))
                value = int(value_unit * int(value_step))
                leaf_ids.append(leaf_id)
                subgroup_child_ids.append(leaf_id)
                subgroup_value += int(value)
                built_nodes[leaf_id] = SunburstNode(
                    node_id=leaf_id,
                    label=str(leaf_label),
                    level="leaf",
                    parent_id=subgroup_id,
                    value=int(value),
                    child_ids=(),
                    color_rgb=lighten(parent_color, 0.20 + 0.08 * (leaf_index % 3)),
                )
            parent_value += int(subgroup_value)
            built_nodes[subgroup_id] = SunburstNode(
                node_id=subgroup_id,
                label=str(subgroup_label),
                level="subgroup",
                parent_id=parent_id,
                value=int(subgroup_value),
                child_ids=tuple(subgroup_child_ids),
                color_rgb=lighten(parent_color, 0.08),
            )
        built_nodes[parent_id] = SunburstNode(
            node_id=parent_id,
            label=str(parent_labels[int(parent_index)]),
            level="parent",
            parent_id="root",
            value=int(parent_value),
            child_ids=tuple(parent_child_ids),
            color_rgb=parent_color,
        )

    root_value = int(sum(int(built_nodes[parent_id].value) for parent_id in parent_ids))
    built_nodes["root"] = SunburstNode(
        node_id="root",
        label="Total",
        level="root",
        parent_id=None,
        value=int(root_value),
        child_ids=tuple(parent_ids),
        color_rgb=(224, 232, 190),
    )

    ordered_nodes = [built_nodes["root"]]
    for parent_id in parent_ids:
        ordered_nodes.append(built_nodes[parent_id])
        for subgroup_id in built_nodes[parent_id].child_ids:
            ordered_nodes.append(built_nodes[subgroup_id])
            for leaf_id in built_nodes[subgroup_id].child_ids:
                ordered_nodes.append(built_nodes[leaf_id])

    return SunburstTree(
        nodes=tuple(ordered_nodes),
        root_id="root",
        parent_ids=tuple(parent_ids),
        subgroup_ids=tuple(subgroup_ids),
        leaf_ids=tuple(leaf_ids),
        generation_ranges={
            "parent_count_range": [int(parent_min), int(parent_max)],
            "subgroup_count_range": [int(subgroup_min), int(subgroup_max)],
            "leaf_count_range": [int(leaf_min), int(leaf_max)],
            "leaf_value_range": [int(value_min), int(value_max)],
            "leaf_value_step": int(value_step),
            "label_max_chars": {
                "parent": int(parent_label_max_chars),
                "subgroup": int(subgroup_label_max_chars),
                "leaf": int(leaf_label_max_chars),
            },
            "label_manifests": {
                "parent": list(PARENT_LABEL_MANIFESTS),
                "subgroup": list(SUBGROUP_LABEL_MANIFESTS),
                "leaf": list(LEAF_LABEL_MANIFESTS),
            },
        },
    )


def choose_parent(tree: SunburstTree, *, instance_seed: int, namespace: str) -> SunburstNode:
    parents = parent_nodes(tree)
    if not parents:
        raise ValueError("sunburst tree contains no parent nodes")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return parents[int(rng.randrange(len(parents)))]


def unique_extreme_parent(tree: SunburstTree, *, direction: str) -> SunburstNode:
    parents = parent_nodes(tree)
    if str(direction) == "highest":
        select_largest = True
    elif str(direction) == "lowest":
        select_largest = False
    else:
        raise ValueError(f"unsupported sunburst extremum direction: {direction}")
    return select_unique_extremum(
        tuple((parent, int(parent.value)) for parent in parents),
        select_largest=bool(select_largest),
        min_margin=1,
        error_label="sunburst parent totals",
        item_label="parents",
    ).item


def threshold_leaf_case(
    tree: SunburstTree,
    *,
    comparator: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> dict[str, Any]:
    """Choose one parent and threshold with a nontrivial matching-leaf count."""

    node_lookup = nodes_by_id(tree)
    rng = spawn_rng(int(instance_seed), f"charts.sunburst.threshold.{comparator}")
    count_support = [int(value) for value in int_sequence(params, "sunburst_condition_count_support", (1, 2, 3, 4, 5))]
    ordered_counts = _ordered_count_support(count_support, params=params, rng=rng)
    max_requested_count = max(count_support) if count_support else 0
    parent_order = list(str(parent_id) for parent_id in tree.parent_ids)
    rng.shuffle(parent_order)
    comparison = str(comparator)
    if comparison not in {"above", "below"}:
        raise ValueError(f"unsupported threshold comparator: {comparator}")
    for parent_id in parent_order:
        leaf_ids = descendant_leaf_ids(node_lookup, str(parent_id))
        values = [int(node_lookup[leaf_id].value) for leaf_id in leaf_ids]
        if len(values) < 3 or len(values) <= int(max_requested_count):
            continue
        candidates_by_count: dict[int, list[int]] = defaultdict(list)
        for threshold in range(min(values), max(values) + 1):
            if comparison == "above":
                count = sum(1 for value in values if int(value) > int(threshold))
            else:
                count = sum(1 for value in values if int(value) < int(threshold))
            if 1 <= int(count) <= len(values) - 1:
                candidates_by_count[int(count)].append(int(threshold))
        available_counts = [count for count in ordered_counts if count in candidates_by_count]
        if not available_counts:
            continue
        answer = int(available_counts[0])
        thresholds = candidates_by_count[int(answer)]
        threshold = int(thresholds[int(rng.randrange(len(thresholds)))])
        return {
            "parent_id": str(parent_id),
            "parent_label": str(node_lookup[parent_id].label),
            "comparison_phrase": comparison,
            "threshold_value": int(threshold),
            "leaf_ids": tuple(str(leaf_id) for leaf_id in leaf_ids),
            "leaf_values": values,
            "answer": int(answer),
        }
    raise ValueError("unable to build nontrivial sunburst threshold count case")


def range_leaf_case(
    tree: SunburstTree,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> dict[str, Any]:
    """Choose one parent and inclusive range with a nontrivial leaf count."""

    node_lookup = nodes_by_id(tree)
    rng = spawn_rng(int(instance_seed), "charts.sunburst.range")
    count_support = [int(value) for value in int_sequence(params, "sunburst_condition_count_support", (1, 2, 3, 4, 5))]
    ordered_counts = _ordered_count_support(count_support, params=params, rng=rng)
    max_requested_count = max(count_support) if count_support else 0
    parent_order = list(str(parent_id) for parent_id in tree.parent_ids)
    rng.shuffle(parent_order)
    for parent_id in parent_order:
        leaf_ids = descendant_leaf_ids(node_lookup, str(parent_id))
        values = sorted(int(node_lookup[leaf_id].value) for leaf_id in leaf_ids)
        unique_values = sorted(set(values))
        if len(unique_values) < 3 or len(values) <= int(max_requested_count):
            continue
        candidates_by_count: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for low_index, low in enumerate(unique_values):
            for high in unique_values[low_index:]:
                count = sum(1 for value in values if int(low) <= int(value) <= int(high))
                if 1 <= int(count) <= len(values) - 1:
                    candidates_by_count[int(count)].append((int(low), int(high)))
        available_counts = [count for count in ordered_counts if count in candidates_by_count]
        if not available_counts:
            continue
        answer = int(available_counts[0])
        ranges = candidates_by_count[int(answer)]
        lower, upper = ranges[int(rng.randrange(len(ranges)))]
        return {
            "parent_id": str(parent_id),
            "parent_label": str(node_lookup[parent_id].label),
            "lower_value": int(lower),
            "upper_value": int(upper),
            "leaf_ids": tuple(str(leaf_id) for leaf_id in leaf_ids),
            "leaf_values": [int(node_lookup[str(leaf_id)].value) for leaf_id in leaf_ids],
            "answer": int(answer),
        }
    raise ValueError("unable to build nontrivial sunburst range count case")


def threshold_matching_leaf_ids(case: Mapping[str, Any], node_lookup: Mapping[str, SunburstNode]) -> tuple[str, ...]:
    """Return leaves whose values satisfy the sampled one-bound predicate."""

    threshold = int(case["threshold_value"])
    comparison = str(case["comparison_phrase"])
    matched = tuple(
        str(leaf_id)
        for leaf_id in tuple(str(item) for item in case["leaf_ids"])
        if (
            (comparison == "above" and int(node_lookup[str(leaf_id)].value) > threshold)
            or (comparison == "below" and int(node_lookup[str(leaf_id)].value) < threshold)
        )
    )
    if not matched:
        raise ValueError("sunburst threshold case produced zero matching leaves")
    return matched


def range_matching_leaf_ids(case: Mapping[str, Any], node_lookup: Mapping[str, SunburstNode]) -> tuple[str, ...]:
    """Return leaves whose values fall inside the sampled inclusive range."""

    lower = int(case["lower_value"])
    upper = int(case["upper_value"])
    matched = tuple(
        str(leaf_id)
        for leaf_id in tuple(str(item) for item in case["leaf_ids"])
        if lower <= int(node_lookup[str(leaf_id)].value) <= upper
    )
    if not matched:
        raise ValueError("sunburst range case produced zero matching leaves")
    return matched


def _sample_hierarchy_labels(
    *,
    parent_count: int,
    subgroup_counts: Sequence[int],
    leaf_counts_by_parent: Sequence[Sequence[int]],
    parent_max_chars: int,
    subgroup_max_chars: int,
    leaf_max_chars: int,
    instance_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve parent, subgroup, and leaf label tiers as one collision domain.

    The three returned tiers are sampled separately for readability constraints,
    but they share one used-label set so prompt-referenced labels are unambiguous.
    """

    rng = spawn_rng(int(instance_seed), "charts.sunburst.labels")
    used: set[str] = {
        normalize_chart_label_for_collision(label)
        for label in ("Total", *LEGACY_PARENT_LABELS)
    }
    parent_labels = _sample_unique_labels(
        rng=rng,
        manifests=PARENT_LABEL_MANIFESTS,
        count=int(parent_count),
        min_chars=2,
        max_chars=int(parent_max_chars),
        used=used,
    )
    subgroup_labels = _sample_unique_labels(
        rng=rng,
        manifests=SUBGROUP_LABEL_MANIFESTS,
        count=sum(int(count) for count in subgroup_counts),
        min_chars=2,
        max_chars=int(subgroup_max_chars),
        used=used,
    )
    leaf_labels = _sample_unique_labels(
        rng=rng,
        manifests=LEAF_LABEL_MANIFESTS,
        count=sum(sum(int(count) for count in parent_counts) for parent_counts in leaf_counts_by_parent),
        min_chars=2,
        max_chars=int(leaf_max_chars),
        used=used,
    )
    return tuple(parent_labels), tuple(subgroup_labels), tuple(leaf_labels)


def _sample_unique_labels(
    *,
    rng: Any,
    manifests: Sequence[str],
    count: int,
    min_chars: int,
    max_chars: int,
    used: set[str],
) -> tuple[str, ...]:
    """Sample one non-overlapping label tier from curated manifests.

    Labels are unique across the whole sunburst tree by normalized visible
    text, so a prompt never has two hierarchy nodes with the same label.
    """

    pool: list[str] = []
    seen_pool: set[str] = set()
    for manifest in manifests:
        labels = load_label_manifest(
            str(manifest),
            min_chars=int(min_chars),
            max_chars=int(max_chars),
            allow_spaces=False,
            allow_punctuation=False,
            ascii_only=True,
            compact_length=False,
        )
        for label in labels:
            value = str(label).strip()
            normalized = normalize_chart_label_for_collision(value)
            if not value or normalized in used or normalized in seen_pool:
                continue
            seen_pool.add(normalized)
            pool.append(value)
    if len(pool) < int(count):
        raise ValueError(f"sunburst label pool too small for {count} labels with max_chars={max_chars}")
    rng.shuffle(pool)
    selected: list[str] = []
    for label in pool:
        normalized = normalize_chart_label_for_collision(str(label))
        if normalized in used:
            continue
        selected.append(str(label))
        used.add(normalized)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise ValueError("sunburst label sampler could not build unique labels")
    return tuple(selected)


def _parent_color(index: int, count: int, *, instance_seed: int) -> RGB:
    return composition_hsv_color(
        int(index),
        int(count),
        instance_seed=int(instance_seed),
        namespace="charts.sunburst.palette",
        saturation_base=0.48,
        saturation_jitter=0.14,
        value_base=0.76,
        value_jitter=0.12,
    )


def _ordered_count_support(count_support: Sequence[int], *, params: Mapping[str, Any], rng: Any) -> list[int]:
    """Return count support in seeded random order for feasible-case search."""

    del params
    support = [int(value) for value in count_support]
    if not support:
        return []
    rng.shuffle(support)
    return list(support)


__all__ = [
    "choose_parent",
    "descendant_leaf_ids",
    "hierarchy_rows",
    "nodes_by_id",
    "parent_nodes",
    "range_leaf_case",
    "range_matching_leaf_ids",
    "sample_tree",
    "threshold_leaf_case",
    "threshold_matching_leaf_ids",
    "unique_extreme_parent",
]
