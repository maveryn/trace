"""Count same-category items larger or smaller than a reference item."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from ._lifecycle import package_size_encoding_plan as P, run_size_encoding_lifecycle as R
from .shared.annotations import item_bbox_set_map as ANN
from .shared.defaults import GEN_DEFAULTS, resolve_int, resolve_scene_variant as V
from .shared.sampling import (
    build_size_encoding_dataset as DSET,
    items_by_category,
)
from .shared.state import DOMAIN, SCENE_NAMESPACE, SINGLE_PANEL_SCENE_VARIANTS, SizeEncodingDataset, SizeEncodingSelection
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.registry import register_task

T = "task_charts__size_encoding__category_relative_size_count"
Q = {"larger_than_reference_in_category_count": "larger", "smaller_than_reference_in_category_count": "smaller"}
D = dict(
    item_count_min=20,
    item_count_max=30,
    total_item_count_min=20,
    total_item_count_max=30,
    relative_size_reference_gap_min=20,
    relative_size_answer_count_min=1,
    relative_size_answer_count_max=5,
)
PGM = "count(filter(items, category=target_category and encoded_value(item) comparison reference_item)); comparison={larger,smaller}; output=integer_value; annotation=bbox_set_map(reference_item,counted_items); scene=size_encoding; scope=category_relative_size_count"


def _force_same_category_reference_split(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
    seed: int,
) -> SizeEncodingDataset:
    value_min, value_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=12,
        fallback_max=99,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    gap_min = resolve_int(params, "relative_size_reference_gap_min", 20)
    if int(value_min) + (2 * int(gap_min)) > int(value_max):
        raise ValueError("relative size gap cannot fit value range")

    answer_min = resolve_int(params, "relative_size_answer_count_min", 1)
    answer_max = resolve_int(params, "relative_size_answer_count_max", 10_000)
    feasible_groups = [
        (str(category), list(group))
        for category, group in sorted(items_by_category(dataset.items).items())
        if len(group) >= max(3, int(answer_min) + 2)
        and min(len(group) - 2, int(answer_max)) >= int(answer_min)
    ]
    if not feasible_groups:
        raise ValueError("no feasible category for relative size count")

    rng = spawn_rng(int(seed), f"{SCENE_NAMESPACE}.relative_size_count.{direction}")
    target_category, candidate_items = feasible_groups[int(rng.randrange(0, len(feasible_groups)))]
    rng.shuffle(candidate_items)
    answer_count = int(rng.randint(int(answer_min), min(len(candidate_items) - 2, int(answer_max))))
    counted_items = tuple(candidate_items[:answer_count])
    reference_item = candidate_items[answer_count]
    non_counted_items = tuple(candidate_items[answer_count + 1 :])

    reference_value = int((int(value_min) + int(value_max)) // 2)
    low_max = int(reference_value) - int(gap_min)
    high_min = int(reference_value) + int(gap_min)
    if low_max < int(value_min) or high_min > int(value_max):
        raise ValueError("relative size reference value leaves no separated side range")

    value_by_id = {str(item.item_id): int(item.value) for item in dataset.items}
    if str(direction) == "larger":
        for item in counted_items:
            value_by_id[str(item.item_id)] = int(rng.randint(high_min, int(value_max)))
        for item in non_counted_items:
            value_by_id[str(item.item_id)] = int(rng.randint(int(value_min), low_max))
    elif str(direction) == "smaller":
        for item in counted_items:
            value_by_id[str(item.item_id)] = int(rng.randint(int(value_min), low_max))
        for item in non_counted_items:
            value_by_id[str(item.item_id)] = int(rng.randint(high_min, int(value_max)))
    else:
        raise ValueError(f"unsupported relative size direction: {direction}")
    value_by_id[str(reference_item.item_id)] = int(reference_value)

    return SizeEncodingDataset(
        items=tuple(replace(item, value=int(value_by_id[str(item.item_id)])) for item in dataset.items),
        categories=dataset.categories,
        panels=dataset.panels,
        trace={
            **dict(dataset.trace),
            "relative_size_gap_forced": True,
            "relative_size_reference_gap_min": int(gap_min),
            "relative_size_target_category": str(target_category),
            "relative_size_reference_item_id": str(reference_item.item_id),
            "relative_size_counted_item_ids": [str(item.item_id) for item in counted_items],
            "relative_size_answer_count": int(answer_count),
        },
    )


def _count_items_around_reference(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
) -> SizeEncodingSelection:
    target_category = str(dataset.trace.get("relative_size_target_category", ""))
    reference_item_id = str(dataset.trace.get("relative_size_reference_item_id", ""))
    reference = next((item for item in dataset.items if str(item.item_id) == reference_item_id), None)
    if reference is None or not target_category:
        raise ValueError("relative size reference metadata missing")

    same_category_items = [item for item in dataset.items if str(item.category) == str(target_category)]
    if str(direction) == "larger":
        counted = [item for item in same_category_items if int(item.value) > int(reference.value)]
    elif str(direction) == "smaller":
        counted = [item for item in same_category_items if int(item.value) < int(reference.value)]
    else:
        raise ValueError(f"unsupported relative size direction: {direction}")
    counted = sorted(counted, key=lambda item: str(item.item_id))
    if not counted:
        raise ValueError("relative size count cannot be empty")

    closest_gap = min(abs(int(item.value) - int(reference.value)) for item in counted)
    gap_min = resolve_int(params, "relative_size_reference_gap_min", 20)
    if int(closest_gap) < int(gap_min):
        raise ValueError("relative size counted item gap too small")

    return SizeEncodingSelection(
        answer=str(len(counted)),
        annotation_item_ids=(str(reference.item_id), *(str(item.item_id) for item in counted)),
        category_label=str(target_category),
        panel_label="",
        reference_label=str(reference.label),
        direction=str(direction),
        trace={
            "reference_item_id": str(reference.item_id),
            "reference_value": int(reference.value),
            "counted_item_ids": [str(item.item_id) for item in counted],
            "counted_labels": [str(item.label) for item in counted],
            "counted_values": [int(item.value) for item in counted],
            "answer_count": int(len(counted)),
            "closest_counted_gap": int(closest_gap),
            "comparison_direction": str(direction),
            "target_category_label": str(target_category),
        },
    )


def _build_plan(params, seed, query_id, _probs):
    direction = Q[str(query_id)]
    variant, variant_probs = V(params, instance_seed=seed, supported_variants=SINGLE_PANEL_SCENE_VARIANTS)
    dataset = DSET(scene_variant=variant, params=params, instance_seed=seed, attempt_index=int(params.get("_attempt_index", 0)))
    dataset = _force_same_category_reference_split(dataset, direction=direction, params=params, seed=seed)
    selection = _count_items_around_reference(dataset, direction=direction, params=params)
    return P(dataset=dataset, selection=selection, params=params, scene_variant=variant, scene_variant_probabilities=variant_probs, prompt_key=query_id, annotation_kind="reference_counted_bbox_set_map", question_format="size_encoded_relative_count", program_code=PGM, reasoning_load=0.56, answer_type="integer", answer_hint='set "answer" to the requested number of items as an integer')


def _bind_annotation(plan, rendered):
    return ANN(rendered, {"reference_item": (str(plan.selection.trace["reference_item_id"]),), "counted_items": tuple(plan.selection.trace["counted_item_ids"])})


@register_task
class ChartsSizeEncodingCategoryRelativeSizeCountTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = DOMAIN
    objective_contract = "category_relative_size_count"
    supported_query_ids = tuple(Q)
    default_query_id = "larger_than_reference_in_category_count"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan, bind_annotation=_bind_annotation)


__all__ = ["ChartsSizeEncodingCategoryRelativeSizeCountTask"]
