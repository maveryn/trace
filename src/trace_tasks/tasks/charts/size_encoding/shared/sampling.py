"""Dataset sampling and semantic selection primitives for size-encoded charts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.label_assets import (
    resolve_chart_category_labels,
    resolve_chart_entity_labels,
    resolve_chart_panel_labels,
    validate_chart_label_namespaces,
)
from .defaults import GEN_DEFAULTS, resolve_int
from .state import SCENE_NAMESPACE, SizeEncodingDataset, SizeEncodingItem, SizeEncodingSelection


def items_by_category(items: Sequence[SizeEncodingItem]) -> dict[str, list[SizeEncodingItem]]:
    grouped: dict[str, list[SizeEncodingItem]] = {}
    for item in items:
        grouped.setdefault(str(item.category), []).append(item)
    return grouped


def choose_index(length: int, *, params: Mapping[str, Any], instance_seed: int, namespace: str) -> int:
    if int(length) <= 0:
        raise ValueError(f"empty support for {namespace}")
    if params.get("_sample_cursor") is not None:
        return abs(int(params["_sample_cursor"])) % int(length)
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(rng.randrange(0, int(length)))


def _extreme(items: Sequence[SizeEncodingItem], direction: str) -> tuple[SizeEncodingItem, int]:
    ordered = sorted(items, key=lambda item: (int(item.value), str(item.label)))
    if str(direction) == "largest":
        winner = ordered[-1]
        runner_up = ordered[-2]
    elif str(direction) == "smallest":
        winner = ordered[0]
        runner_up = ordered[1]
    else:
        raise ValueError(f"unsupported extremum direction: {direction}")
    return winner, abs(int(winner.value) - int(runner_up.value))


def _outside_extreme_count(*, items: Sequence[SizeEncodingItem], winner: SizeEncodingItem, direction: str) -> int:
    if str(direction) == "largest":
        return sum(1 for item in items if str(item.category) != str(winner.category) and int(item.value) > int(winner.value))
    if str(direction) == "smallest":
        return sum(1 for item in items if str(item.category) != str(winner.category) and int(item.value) < int(winner.value))
    raise ValueError(f"unsupported extremum direction: {direction}")


def _sample_categories(*, count: int, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.categories")
    values = resolve_chart_category_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    ).labels
    return tuple(str(value) for value in values)


def _sample_labels(*, count: int, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.item_labels")
    values = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=6,
        allow_spaces=False,
    ).labels
    return tuple(str(value) for value in values)


def _resolved_label_metadata(resolved: Any) -> dict[str, Any]:
    return {
        "label_variant": str(resolved.label_variant),
        "label_pool_kind": str(resolved.label_pool_kind),
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
        "label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def _panel_labels(
    panel_count: int,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    reserved_labels: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Sample panel labels while keeping them disjoint from category labels."""

    if int(panel_count) <= 1:
        collision_check = validate_chart_label_namespaces(
            panel_labels=("Overall",),
            other_label_groups={"category_labels": tuple(str(label) for label in reserved_labels)},
            context="size-encoding panel labels",
        )
        return ("Overall",), {"panel_label_collision_check": dict(collision_check)}
    resolved = resolve_chart_panel_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.panel_labels"),
        count=int(panel_count),
        min_chars=1,
        max_chars=10,
        allow_spaces=False,
        variant_weights=params.get(
            "panel_label_variant_weights",
            group_default(
                GEN_DEFAULTS,
                "panel_label_variant_weights",
                {
                    "named_compact": 1.0,
                    "technical_topics": 1.0,
                    "condition_labels": 0.75,
                    "temporal_sequence": 0.5,
                    "report_topics": 0.5,
                },
            ),
        ),
        reserved_labels=tuple(str(label) for label in reserved_labels),
    )
    collision_check = validate_chart_label_namespaces(
        panel_labels=resolved.labels,
        other_label_groups={"category_labels": tuple(str(label) for label in reserved_labels)},
        context="size-encoding panel labels",
    )
    return tuple(str(label) for label in resolved.labels), {
        "panel_label_resolution": _resolved_label_metadata(resolved),
        "panel_label_collision_check": dict(collision_check),
    }


def _sample_items(
    *,
    categories: Sequence[str],
    panels: Sequence[str],
    item_count_by_panel: Mapping[str, int],
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[SizeEncodingItem, ...]:
    """Sample labeled items with category assignments and hidden values."""

    value_min, value_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=12,
        fallback_max=99,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    total_items = int(sum(int(value) for value in item_count_by_panel.values()))
    labels = _sample_labels(count=int(total_items), instance_seed=int(instance_seed))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.items")
    items: list[SizeEncodingItem] = []
    label_index = 0
    for panel in panels:
        count = int(item_count_by_panel[str(panel)])
        panel_categories = list(categories)
        rng.shuffle(panel_categories)
        assigned_categories = [panel_categories[index % len(panel_categories)] for index in range(count)]
        rng.shuffle(assigned_categories)
        for local_index in range(count):
            items.append(
                SizeEncodingItem(
                    item_id=f"item_{len(items):02d}",
                    label=str(labels[label_index]),
                    category=str(assigned_categories[local_index]),
                    panel=str(panel),
                    value=int(rng.randint(int(value_min), int(value_max))),
                )
            )
            label_index += 1
    return tuple(items)


def build_size_encoding_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    attempt_index: int,
) -> SizeEncodingDataset:
    """Sample items, categories, and panels without choosing any public objective."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.dataset.{int(attempt_index)}")
    cat_min, cat_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="category_count_min",
        max_key="category_count_max",
        fallback_min=3,
        fallback_max=5,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    category_count = int(rng.randint(int(cat_min), int(cat_max)))
    categories = _sample_categories(count=int(category_count), instance_seed=int(instance_seed) + int(attempt_index))
    panel_label_meta: dict[str, Any] = {}

    if str(scene_variant) == "small_multiple_bubble_cloud":
        panel_min, panel_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="panel_count_min",
            max_key="panel_count_max",
            fallback_min=2,
            fallback_max=4,
            context=f"generation defaults for {SCENE_NAMESPACE}",
        )
        item_min, item_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="panel_item_count_min",
            max_key="panel_item_count_max",
            fallback_min=7,
            fallback_max=10,
            context=f"generation defaults for {SCENE_NAMESPACE}",
        )
        panel_count = int(rng.randint(int(panel_min), int(panel_max)))
        panels, panel_label_meta = _panel_labels(
            int(panel_count),
            params=params,
            instance_seed=int(instance_seed),
            reserved_labels=categories,
        )
        item_count_by_panel = {str(panel): int(rng.randint(int(item_min), int(item_max))) for panel in panels}
    else:
        item_min, item_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="item_count_min",
            max_key="item_count_max",
            fallback_min=18,
            fallback_max=30,
            context=f"generation defaults for {SCENE_NAMESPACE}",
        )
        panels = ("Overall",)
        item_count_by_panel = {"Overall": int(rng.randint(int(item_min), int(item_max)))}

    total_item_count_min = resolve_int(params, "total_item_count_min", 1)
    total_item_count_max = resolve_int(params, "total_item_count_max", 10_000)
    total_item_count = int(sum(int(value) for value in item_count_by_panel.values()))
    if int(total_item_count) < int(total_item_count_min):
        raise ValueError("total item count below minimum")
    if int(total_item_count) > int(total_item_count_max):
        raise ValueError("total item count above maximum")

    items = _sample_items(
        categories=categories,
        panels=panels,
        item_count_by_panel=item_count_by_panel,
        params=params,
        instance_seed=int(instance_seed) + (997 * int(attempt_index)),
    )
    return SizeEncodingDataset(
        items=tuple(items),
        categories=tuple(categories),
        panels=tuple(panels),
        trace={**dict(panel_label_meta), "item_count_by_panel": {str(key): int(value) for key, value in item_count_by_panel.items()}},
    )


def select_extreme_item_in_category(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> SizeEncodingSelection:
    """Select the size extremum inside a feasible category group."""

    winner_gap_min = resolve_int(params, "winner_gap_min", 8)
    filtered_gap_min = resolve_int(params, "filtered_item_winner_gap_min", int(winner_gap_min))
    filtered_gap_max = resolve_int(params, "filtered_item_winner_gap_max", 10_000)
    outside_extreme_min = resolve_int(params, "filtered_item_outside_extreme_min", 0)
    feasible: list[tuple[str, list[SizeEncodingItem], SizeEncodingItem, int, int]] = []
    for category, group in sorted(items_by_category(dataset.items).items()):
        if len(group) < 2:
            continue
        candidate_winner, candidate_gap = _extreme(group, str(direction))
        if int(candidate_gap) < int(filtered_gap_min) or int(candidate_gap) > int(filtered_gap_max):
            continue
        candidate_outside_extreme_count = _outside_extreme_count(
            items=dataset.items,
            winner=candidate_winner,
            direction=str(direction),
        )
        if int(candidate_outside_extreme_count) < int(outside_extreme_min):
            continue
        feasible.append((str(category), list(group), candidate_winner, int(candidate_gap), int(candidate_outside_extreme_count)))
    if not feasible:
        raise ValueError("no feasible filtered extremum category")
    category, group, winner, gap, outside_extreme_count = feasible[
        choose_index(
            len(feasible),
            params=params,
            instance_seed=instance_seed,
            namespace=f"{SCENE_NAMESPACE}.filtered_category",
        )
    ]
    return SizeEncodingSelection(
        answer=str(winner.label),
        annotation_item_ids=(str(winner.item_id),),
        category_label=str(category),
        panel_label="",
        reference_label="",
        direction=str(direction),
        trace={
            "winner_gap": int(gap),
            "candidate_count": int(len(group)),
            "outside_extreme_count": int(outside_extreme_count),
        },
    )


def enforce_global_extreme_item_gap(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> SizeEncodingDataset:
    """Make one item the unique global extremum with a controlled runner-up gap."""

    if len(dataset.items) < 2:
        raise ValueError("global item extremum requires at least two items")
    value_min, value_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=12,
        fallback_max=99,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    global_gap_min = resolve_int(params, "global_item_winner_gap_min", resolve_int(params, "winner_gap_min", 8))
    global_gap_max = min(
        resolve_int(params, "global_item_winner_gap_max", 10_000),
        int(value_max) - int(value_min),
    )
    if int(global_gap_min) > int(global_gap_max):
        raise ValueError("global item-extremum gap cannot fit value range")

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.global_item_extreme_gap.{direction}")
    winner_index = int(rng.randrange(0, len(dataset.items)))
    runner_offset = int(rng.randrange(1, len(dataset.items)))
    runner_index = int((winner_index + runner_offset) % len(dataset.items))
    gap = int(rng.randint(int(global_gap_min), int(global_gap_max)))
    if str(direction) == "largest":
        winner_value = int(value_max)
        runner_value = int(value_max) - int(gap)

        def adjusted_value(index: int, item: SizeEncodingItem) -> int:
            if index == winner_index:
                return winner_value
            if index == runner_index:
                return runner_value
            return min(int(item.value), int(runner_value))

    elif str(direction) == "smallest":
        winner_value = int(value_min)
        runner_value = int(value_min) + int(gap)

        def adjusted_value(index: int, item: SizeEncodingItem) -> int:
            if index == winner_index:
                return winner_value
            if index == runner_index:
                return runner_value
            return max(int(item.value), int(runner_value))

    else:
        raise ValueError(f"unsupported extremum direction: {direction}")

    adjusted_items = tuple(
        replace(item, value=int(adjusted_value(index, item)))
        for index, item in enumerate(dataset.items)
    )
    return SizeEncodingDataset(
        items=adjusted_items,
        categories=dataset.categories,
        panels=dataset.panels,
        trace={
            **dict(dataset.trace),
            "global_item_gap_forced": True,
            "global_item_gap_target": int(gap),
            "global_item_winner_item_id": str(dataset.items[winner_index].item_id),
            "global_item_runner_item_id": str(dataset.items[runner_index].item_id),
        },
    )


def select_global_extreme_item_category(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
) -> SizeEncodingSelection:
    """Select the category of the single globally extremal item."""

    global_gap_min = resolve_int(params, "global_item_winner_gap_min", resolve_int(params, "winner_gap_min", 8))
    global_gap_max = resolve_int(params, "global_item_winner_gap_max", 10_000)
    ordered = sorted(dataset.items, key=lambda item: (int(item.value), str(item.label)))
    if len(ordered) < 2:
        raise ValueError("global item extremum requires at least two items")
    if str(direction) == "largest":
        winner = ordered[-1]
        runner = ordered[-2]
    elif str(direction) == "smallest":
        winner = ordered[0]
        runner = ordered[1]
    else:
        raise ValueError(f"unsupported extremum direction: {direction}")
    gap = abs(int(winner.value) - int(runner.value))
    if int(gap) < int(global_gap_min):
        raise ValueError("global item-extremum gap too small")
    if int(gap) > int(global_gap_max):
        raise ValueError("global item-extremum gap too large")
    return SizeEncodingSelection(
        answer=str(winner.category),
        annotation_item_ids=(str(winner.item_id),),
        category_label=str(winner.category),
        panel_label="",
        reference_label="",
        direction=str(direction),
        trace={
            "winner_gap": int(gap),
            "winner_item_label": str(winner.label),
            "winner_item_value": int(winner.value),
            "winner_item_category": str(winner.category),
            "closest_distractor_label": str(runner.label),
            "closest_distractor_value": int(runner.value),
            "closest_distractor_category": str(runner.category),
        },
    )
