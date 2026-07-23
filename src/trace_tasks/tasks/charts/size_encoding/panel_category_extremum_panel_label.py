"""Select the panel containing a category's extremal size-encoded item."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.size_encoding._lifecycle import package_size_encoding_plan as P, run_size_encoding_lifecycle as R
from trace_tasks.tasks.charts.size_encoding.shared.annotations import item_bbox as B
from trace_tasks.tasks.charts.size_encoding.shared.defaults import GEN_DEFAULTS, resolve_int, resolve_scene_variant as V
from trace_tasks.tasks.charts.size_encoding.shared.sampling import (
    build_size_encoding_dataset as DSET,
)
from trace_tasks.tasks.charts.size_encoding.shared.state import DOMAIN, PANEL_SCENE_VARIANTS, SCENE_NAMESPACE, SizeEncodingDataset, SizeEncodingItem, SizeEncodingSelection
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.registry import register_task

T = "task_charts__size_encoding__panel_category_extremum_panel_label"
Q = {"largest_category_item_panel_label": "largest", "smallest_category_item_panel_label": "smallest"}
D = dict(
    panel_count_min=4,
    panel_count_max=4,
    panel_item_count_min=5,
    panel_item_count_max=7,
    total_item_count_min=20,
    total_item_count_max=28,
    panel_category_item_winner_gap_min=24,
    panel_category_item_winner_gap_max=36,
)
PGM = "select_label(panel(arg_extreme(filter(items, category=target_category), encoded_value(item), direction))); output=string_label; annotation=bbox(answer_item); scene=size_encoding; scope=panel_category_extremum_panel_label"


def _force_panel_category_extreme(
    dataset: SizeEncodingDataset,
    *,
    direction: str,
    params: Mapping[str, Any],
    seed: int,
) -> SizeEncodingDataset:
    if len(dataset.panels) < 2:
        raise ValueError("panel category extremum requires multiple panels")
    value_min, value_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=12,
        fallback_max=99,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    gap_min = resolve_int(params, "panel_category_item_winner_gap_min", resolve_int(params, "winner_gap_min", 8))
    gap_max = min(
        resolve_int(params, "panel_category_item_winner_gap_max", 10_000),
        int(value_max) - int(value_min),
    )
    if int(gap_min) > int(gap_max):
        raise ValueError("panel category item-extremum gap cannot fit value range")

    rng = spawn_rng(int(seed), f"{SCENE_NAMESPACE}.panel_category_extremum_panel.{direction}")
    categories = tuple(str(category) for category in dataset.categories)
    if not categories:
        raise ValueError("panel category extremum requires categories")
    target_category = categories[int(rng.randrange(0, len(categories)))]
    target_items = [item for item in dataset.items if str(item.category) == str(target_category)]
    if len(target_items) < 2:
        raise ValueError("panel category extremum requires at least two items in target category")

    winner = target_items[int(rng.randrange(0, len(target_items)))]
    runner_candidates = [item for item in target_items if str(item.item_id) != str(winner.item_id)]
    runner = runner_candidates[int(rng.randrange(0, len(runner_candidates)))]
    forced_gap = int(rng.randint(int(gap_min), int(gap_max)))

    def panel_extreme_value(item: SizeEncodingItem) -> int:
        if str(item.item_id) == str(winner.item_id):
            return int(value_max) if str(direction) == "largest" else int(value_min)
        if str(item.item_id) == str(runner.item_id):
            return int(value_max) - forced_gap if str(direction) == "largest" else int(value_min) + forced_gap
        if str(item.category) != str(target_category):
            return int(item.value)
        if str(direction) == "largest":
            return min(int(item.value), int(value_max) - forced_gap)
        if str(direction) == "smallest":
            return max(int(item.value), int(value_min) + forced_gap)
        raise ValueError(f"unsupported extremum direction: {direction}")

    return SizeEncodingDataset(
        items=tuple(replace(item, value=int(panel_extreme_value(item))) for item in dataset.items),
        categories=dataset.categories,
        panels=dataset.panels,
        trace={
            **dict(dataset.trace),
            "panel_category_gap_forced": True,
            "panel_category_gap_target": int(forced_gap),
            "panel_category_target_category": str(target_category),
            "panel_category_winner_item_id": str(winner.item_id),
            "panel_category_runner_item_id": str(runner.item_id),
        },
    )


def _select_answer_panel(dataset: SizeEncodingDataset, *, direction: str, params: Mapping[str, Any]) -> SizeEncodingSelection:
    target_category = str(dataset.trace.get("panel_category_target_category", ""))
    if not target_category:
        raise ValueError("panel category target missing from dataset trace")
    target_items = [item for item in dataset.items if str(item.category) == str(target_category)]
    if len(target_items) < 2:
        raise ValueError("panel category extremum requires at least two target-category items")

    ordered = sorted(target_items, key=lambda item: (int(item.value), str(item.label)))
    if str(direction) == "largest":
        winner, runner = ordered[-1], ordered[-2]
    elif str(direction) == "smallest":
        winner, runner = ordered[0], ordered[1]
    else:
        raise ValueError(f"unsupported extremum direction: {direction}")

    observed_gap = abs(int(winner.value) - int(runner.value))
    gap_min = resolve_int(params, "panel_category_item_winner_gap_min", resolve_int(params, "winner_gap_min", 8))
    gap_max = resolve_int(params, "panel_category_item_winner_gap_max", 10_000)
    if int(observed_gap) < int(gap_min):
        raise ValueError("panel category item-extremum gap too small")
    if int(observed_gap) > int(gap_max):
        raise ValueError("panel category item-extremum gap too large")

    return SizeEncodingSelection(
        answer=str(winner.panel),
        annotation_item_ids=(str(winner.item_id),),
        category_label=str(target_category),
        panel_label=str(winner.panel),
        reference_label="",
        direction=str(direction),
        trace={
            "winner_gap": int(observed_gap),
            "winner_item_label": str(winner.label),
            "winner_item_value": int(winner.value),
            "winner_item_category": str(winner.category),
            "winner_panel_label": str(winner.panel),
            "closest_distractor_label": str(runner.label),
            "closest_distractor_value": int(runner.value),
            "closest_distractor_panel": str(runner.panel),
            "target_category_label": str(target_category),
        },
    )


def _build_plan(params, seed, query_id, _probs):
    direction = Q[str(query_id)]
    variant, variant_probs = V(params, instance_seed=seed, supported_variants=PANEL_SCENE_VARIANTS)
    dataset = DSET(scene_variant=variant, params=params, instance_seed=seed, attempt_index=int(params.get("_attempt_index", 0)))
    dataset = _force_panel_category_extreme(dataset, direction=direction, params=params, seed=seed)
    selection = _select_answer_panel(dataset, direction=direction, params=params)
    return P(dataset=dataset, selection=selection, params=params, scene_variant=variant, scene_variant_probabilities=variant_probs, prompt_key=query_id, annotation_kind="answer_item_bbox", question_format="size_encoded_panel_label_comparison", program_code=PGM, reasoning_load=0.70)


def _bind_annotation(plan, rendered):
    box = B(rendered, plan.selection.annotation_item_ids[0])
    return "bbox", box, {"type": "bbox", "bbox": box, "pixel_bbox": box, "bbox_set": [box]}


@register_task
class ChartsSizeEncodingPanelCategoryExtremumPanelLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'ranking')
    domain = DOMAIN
    objective_contract = "panel_category_extremum_panel_label"
    supported_query_ids = tuple(Q)
    default_query_id = "largest_category_item_panel_label"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan, bind_annotation=_bind_annotation)


__all__ = ["ChartsSizeEncodingPanelCategoryExtremumPanelLabelTask"]
