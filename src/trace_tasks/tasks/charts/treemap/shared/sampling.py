"""Data sampling primitives for treemap charts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.palette import (
    composition_hsv_color,
    darken_rgb as darken,
    lighten_rgb as lighten,
)

from .defaults import generation_range
from .state import RGB, TreemapDataset, TreemapLeaf, TreemapParent


_THEMES: tuple[dict[str, Any], ...] = (
    {
        "title": "Community Program Breakdown",
        "parent_axis": "program",
        "leaf_axis": "audience",
        "parents": ("Festival", "Tournament", "Health Fair", "Cleanup", "Workshop", "Forum"),
        "leaves": ("Adults", "Youth", "Seniors", "Families", "Volunteers", "Guests"),
    },
    {
        "title": "Method Adoption by Industry",
        "parent_axis": "industry",
        "leaf_axis": "method",
        "parents": ("Healthcare", "Finance", "Education", "Retail", "Manufacturing", "Logistics"),
        "leaves": ("Scrum", "Kanban", "Extreme", "Waterfall", "Lean", "Hybrid"),
    },
    {
        "title": "Habitat Species by Category",
        "parent_axis": "habitat",
        "leaf_axis": "species group",
        "parents": ("Polar", "Forest", "Wetland", "Desert", "Marine", "Grassland"),
        "leaves": ("Birds", "Mammals", "Reptiles", "Fish", "Plants", "Insects"),
    },
    {
        "title": "Department Time Allocation",
        "parent_axis": "department",
        "leaf_axis": "activity",
        "parents": ("Council", "Planning", "Budget", "Outreach", "Services", "Media"),
        "leaves": ("Meetings", "Research", "Review", "Events", "Relations", "Audits"),
    },
    {
        "title": "Household Expense Composition",
        "parent_axis": "expense group",
        "leaf_axis": "expense item",
        "parents": ("Housing", "Food", "Utilities", "Transport", "Debt", "Care"),
        "leaves": ("Rent", "Groceries", "Water", "Electricity", "Internet", "Fuel"),
    },
    {
        "title": "Streaming Revenue Composition",
        "parent_axis": "genre",
        "leaf_axis": "platform",
        "parents": ("Rock", "Hip-Hop", "Jazz", "Electronic", "Country", "Classical"),
        "leaves": ("Spotify", "Apple", "Amazon", "YouTube", "Tidal", "Deezer"),
    },
)


def _theme_color(index: int, count: int, *, instance_seed: int) -> RGB:
    return composition_hsv_color(
        int(index),
        int(count),
        instance_seed=int(instance_seed),
        namespace="charts.treemap.palette",
        saturation_base=0.45,
        saturation_jitter=0.18,
        value_base=0.74,
        value_jitter=0.12,
    )


def _build_values(
    *,
    parent_count: int,
    leaf_count: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = spawn_rng(int(instance_seed), "charts.treemap.values")
    matrix: list[list[int]] = [[0 for _ in range(int(leaf_count))] for _ in range(int(parent_count))]
    for leaf_index in range(int(leaf_count)):
        subtotal = 0
        for parent_index in range(max(0, int(parent_count) - 1)):
            value = int(rng.randint(int(value_min), int(value_max)))
            matrix[parent_index][leaf_index] = int(value)
            subtotal += int(value)
        valid_last_values = [
            value
            for value in range(int(value_min), int(value_max) + 1)
            if (int(subtotal) + int(value)) % max(1, int(parent_count)) == 0
        ]
        if not valid_last_values:
            raise ValueError("treemap value range cannot produce integer repeated-leaf averages")
        matrix[int(parent_count) - 1][leaf_index] = int(valid_last_values[int(rng.randrange(len(valid_last_values)))])
    return tuple(tuple(int(value) for value in row) for row in matrix)


def build_treemap_dataset(params: Mapping[str, object], *, instance_seed: int) -> TreemapDataset:
    """Sample a rectangular parent-child value grid for all treemap objectives.

    Every parent contains the same child-label set, and the generated values
    make repeated-label averages integral so public tasks never need to relax
    answer formatting during retries.
    """

    parent_min, parent_max = generation_range(
        params,
        min_key="treemap_parent_count_min",
        max_key="treemap_parent_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    leaf_min, leaf_max = generation_range(
        params,
        min_key="treemap_leaf_count_min",
        max_key="treemap_leaf_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    value_min, value_max = generation_range(
        params,
        min_key="treemap_value_min",
        max_key="treemap_value_max",
        fallback_min=16,
        fallback_max=88,
    )
    rng = spawn_rng(int(instance_seed), "charts.treemap.tree")
    theme = dict(_THEMES[int(rng.randrange(len(_THEMES)))])
    parent_count = int(rng.randint(int(parent_min), min(int(parent_max), len(theme["parents"]))))
    leaf_count = int(rng.randint(int(leaf_min), min(int(leaf_max), len(theme["leaves"]))))
    parent_labels = [str(item) for item in theme["parents"]]
    leaf_labels = [str(item) for item in theme["leaves"]]
    rng.shuffle(parent_labels)
    rng.shuffle(leaf_labels)
    parent_labels = parent_labels[:parent_count]
    leaf_labels = leaf_labels[:leaf_count]
    values = _build_values(
        parent_count=int(parent_count),
        leaf_count=int(leaf_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
    )
    leaves: list[TreemapLeaf] = []
    parents: list[TreemapParent] = []
    for parent_index, parent_label in enumerate(parent_labels):
        parent_id = f"parent_{parent_index}"
        base_color = _theme_color(parent_index, parent_count, instance_seed=int(instance_seed))
        leaf_ids: list[str] = []
        parent_total = 0
        for leaf_index, leaf_label in enumerate(leaf_labels):
            leaf_id = f"{parent_id}_leaf_{leaf_index}"
            value = int(values[parent_index][leaf_index])
            leaf_ids.append(leaf_id)
            parent_total += int(value)
            leaves.append(
                TreemapLeaf(
                    leaf_id=str(leaf_id),
                    parent_id=str(parent_id),
                    parent_label=str(parent_label),
                    label=str(leaf_label),
                    value=int(value),
                    color_rgb=lighten(base_color, 0.08 + 0.08 * (leaf_index % 4)),
                )
            )
        parents.append(
            TreemapParent(
                parent_id=str(parent_id),
                label=str(parent_label),
                leaf_ids=tuple(leaf_ids),
                value=int(parent_total),
                color_rgb=base_color,
            )
        )
    return TreemapDataset(
        title=str(theme["title"]),
        parent_axis=str(theme["parent_axis"]),
        leaf_axis=str(theme["leaf_axis"]),
        parents=tuple(parents),
        leaves=tuple(leaves),
        generation_ranges={
            "parent_count_range": [int(parent_min), int(parent_max)],
            "leaf_count_range": [int(leaf_min), int(leaf_max)],
            "value_range": [int(value_min), int(value_max)],
        },
    )


__all__ = ["build_treemap_dataset", "darken", "lighten"]
