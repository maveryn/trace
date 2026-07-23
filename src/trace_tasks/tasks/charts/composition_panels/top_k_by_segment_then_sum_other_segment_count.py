"""Select panels by one segment ranking and sum another segment count."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID as Q
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default

from ._lifecycle import package_composition_panels_plan as P, run_composition_panels_lifecycle as R
from .shared.defaults import GEN_DEFAULTS, resolve_scene_variant as V
from .shared.sampling import (
    balanced_int,
    build_base_panels,
    counts_for_panel,
    package_dataset,
    ranked_share_fixtures,
    sample_scene_frame,
    selected_panel_sum_selection,
    top_panels_by_share,
)
from .shared.state import DOMAIN


T = "task_charts__composition_panels__top_k_by_segment_then_sum_other_segment_count"
TASK_PARAM_DEFAULTS = {
    "panel_count_min": 4,
    "panel_count_max": 6,
    "segment_count_min": 4,
    "segment_count_max": 4,
    "total_values": [100, 200, 400, 500],
    "top_k_values": [2],
}
PGM = "sum(count(panel,target_segment) for panel in top_k(panels, share(panel,rank_segment), k)); output=integer_count; annotation=bbox_set(selected_panels); scene=composition_panels; scope=top_k_by_segment_then_sum_other_segment_count"
PROMPT_KEY = "top_k_by_segment_then_sum_other_segment_count"
ANNOTATION_HINT = 'set "annotation" to an array of [x0,y0,x1,y1] pixel boxes around the full selected panels included in the sum'
JSON_EXAMPLE = '{"annotation":[[120,90,380,520],[460,90,720,520]],"answer":140}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":140}'


def _build_plan(params, seed, _query_id, _probs):
    """Build the rank-then-sum objective plan for this public task."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    variant, variant_probs = V(task_params, instance_seed=seed)
    frame = sample_scene_frame(task_params, instance_seed=seed)
    rng = spawn_rng(int(seed), f"{T}.objective")
    rank_segment, target_segment = rng.sample(list(frame.segment_labels), 2)
    top_k_values = tuple(int(value) for value in task_params.get("top_k_values", group_default(GEN_DEFAULTS, "top_k_values", [2, 3])))
    feasible_k = [value for value in top_k_values if 1 < int(value) < len(frame.panel_labels)]
    top_k = balanced_int(feasible_k, params=task_params, instance_seed=seed, namespace=f"{T}.top_k")
    fixed_by_panel = ranked_share_fixtures(
        frame.panel_labels,
        segment=str(rank_segment),
        support_values=[18, 24, 30, 36, 42, 48],
        rng=rng,
    )
    ranked_labels = sorted(
        frame.panel_labels,
        key=lambda label: int(fixed_by_panel[str(label)][str(rank_segment)]),
        reverse=True,
    )
    for label in frame.panel_labels:
        fixed_by_panel[str(label)][str(target_segment)] = (
            int(rng.choice([20, 30])) if str(label) in {str(value) for value in ranked_labels[: int(top_k)]} else 20
        )
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    dataset = package_dataset(frame, panels)
    selected = top_panels_by_share(panels, segment=str(rank_segment), k=int(top_k))
    selected_counts = [counts_for_panel(panel)[str(target_segment)] for panel in selected]
    selection = selected_panel_sum_selection(
        selected_panels=selected,
        target_counts=selected_counts,
        trace={
            "rank_segment": str(rank_segment),
            "target_segment": str(target_segment),
            "top_k": int(top_k),
            "selected_panels": [str(panel.label) for panel in selected],
            "selected_target_counts": [int(value) for value in selected_counts],
            "calculation": "rank_panels_then_sum_target_counts",
        },
    )
    return P(
        dataset=dataset,
        selection=selection,
        params=task_params,
        scene_variant=variant,
        scene_variant_probabilities=variant_probs,
        prompt_key=PROMPT_KEY,
        annotation_hint_template=ANNOTATION_HINT,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        program_code=PGM,
        reasoning_load=0.82,
    )


@register_task
class ChartsCompositionPanelsTopKBySegmentThenSumOtherSegmentCountTask:
    task_id = T
    reasoning_operations = ('counting', 'ranking', 'aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "top_k_by_segment_then_sum_other_segment_count"
    supported_query_ids = (Q,)
    default_query_id = Q
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**TASK_PARAM_DEFAULTS, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsTopKBySegmentThenSumOtherSegmentCountTask"]
