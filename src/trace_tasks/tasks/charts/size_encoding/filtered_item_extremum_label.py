"""Select the extremal size-encoded item within one category."""

from __future__ import annotations

from trace_tasks.tasks.charts.size_encoding._lifecycle import package_size_encoding_plan as P, run_size_encoding_lifecycle as R
from trace_tasks.tasks.charts.size_encoding.shared.annotations import item_bbox as B
from trace_tasks.tasks.charts.size_encoding.shared.defaults import resolve_scene_variant as V
from trace_tasks.tasks.charts.size_encoding.shared.sampling import build_size_encoding_dataset as DSET, select_extreme_item_in_category as SEL
from trace_tasks.tasks.charts.size_encoding.shared.state import DOMAIN, SINGLE_PANEL_SCENE_VARIANTS
from trace_tasks.tasks.registry import register_task

T = "task_charts__size_encoding__filtered_item_extremum_label"
Q = {"largest_size_item_in_category_label": "largest", "smallest_size_item_in_category_label": "smallest"}
D = dict(filtered_item_winner_gap_min=24, filtered_item_winner_gap_max=36, filtered_item_outside_extreme_min=2)
PGM = "select_label(arg_extreme(filter(items, category=target_category), encoded_value(item), direction)); output=string_label; annotation=bbox(answer_item); scene=size_encoding; scope=filtered_item_extremum_label"


def _build_plan(params, seed, query_id, _probs):
    direction = Q[str(query_id)]
    variant, variant_probs = V(params, instance_seed=seed, supported_variants=SINGLE_PANEL_SCENE_VARIANTS)
    dataset = DSET(scene_variant=variant, params=params, instance_seed=seed, attempt_index=int(params.get("_attempt_index", 0)))
    selection = SEL(dataset, direction=direction, params=params, instance_seed=seed)
    return P(dataset=dataset, selection=selection, params=params, scene_variant=variant, scene_variant_probabilities=variant_probs, prompt_key=query_id, annotation_kind="answer_item_bbox", question_format="size_encoded_label_comparison", program_code=PGM, reasoning_load=0.44)


def _bind_annotation(plan, rendered):
    box = B(rendered, plan.selection.annotation_item_ids[0])
    return "bbox", box, {"type": "bbox", "bbox": box, "pixel_bbox": box, "bbox_set": [box]}


@register_task
class ChartsSizeEncodingFilteredItemExtremumLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'ranking')
    domain = DOMAIN
    objective_contract = "filtered_item_extremum_label"
    supported_query_ids = tuple(Q)
    default_query_id = "largest_size_item_in_category_label"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan, bind_annotation=_bind_annotation)


__all__ = ["ChartsSizeEncodingFilteredItemExtremumLabelTask"]
