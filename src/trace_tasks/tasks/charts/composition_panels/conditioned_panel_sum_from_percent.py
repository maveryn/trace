"""Compute a conditioned panel sum from percent composition values."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.composition_panels._lifecycle import (
    package_composition_panels_plan as P,
    run_composition_panels_lifecycle as R,
)
from trace_tasks.tasks.charts.composition_panels.shared.defaults import GEN_DEFAULTS, resolve_scene_variant as V
from trace_tasks.tasks.charts.composition_panels.shared.sampling import (
    balanced_int,
    build_base_panels,
    counts_for_panel,
    package_dataset,
    sample_scene_frame,
    selected_panel_sum_selection,
)
from trace_tasks.tasks.charts.composition_panels.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default


T = "task_charts__composition_panels__conditioned_panel_sum_from_percent"
TASK_PARAM_DEFAULTS = {
    "panel_count_min": 4,
    "panel_count_max": 6,
    "segment_count_min": 4,
    "segment_count_max": 4,
    "total_values": [100, 200, 400, 500],
    "condition_threshold_values": [30, 40],
}
PGM = "sum(count(panel,target_segment) for panel in filter(panels, share(panel,condition_segment) > threshold)); output=integer_value; annotation=bbox_set(selected_panels); scene=composition_panels; scope=conditioned_panel_sum_from_percent"
PROMPT_KEY = "conditioned_panel_sum_from_percent"
ANNOTATION_HINT = 'set "annotation" to an array of [x0,y0,x1,y1] pixel boxes around the full panels included in the sum'
JSON_EXAMPLE = '{"annotation":[[120,90,380,520],[460,90,720,520]],"answer":180}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":180}'


def _condition_share_fixtures(
    frame,
    *,
    condition_segment,
    target_segment,
    threshold,
    selected_count,
    rng,
):
    chosen = set(rng.sample(list(frame.panel_labels), int(selected_count)))
    fixtures = {}
    for label in frame.panel_labels:
        if str(label) in chosen:
            condition_value = int(rng.choice([int(threshold) + 10, int(threshold) + 15]))
        else:
            condition_value = int(rng.choice([10, 20, max(10, int(threshold) - 10)]))
        fixtures[str(label)] = {
            str(condition_segment): int(condition_value),
            str(target_segment): int(rng.choice([20, 30])),
        }
    return fixtures


def _build_plan(params, seed, _query_id, _probs):
    """Build the filter-then-sum objective plan for this public task."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    variant, variant_probs = V(task_params, instance_seed=seed)
    frame = sample_scene_frame(task_params, instance_seed=seed)
    rng = spawn_rng(int(seed), f"{T}.objective")
    condition_segment, target_segment = rng.sample(list(frame.segment_labels), 2)
    threshold_values = tuple(int(value) for value in task_params.get("condition_threshold_values", group_default(GEN_DEFAULTS, "condition_threshold_values", [32, 35, 38])))
    threshold = balanced_int(threshold_values, params=task_params, instance_seed=seed, namespace=f"{T}.threshold")
    selected_count = 2
    fixed_by_panel = _condition_share_fixtures(
        frame,
        condition_segment=str(condition_segment),
        target_segment=str(target_segment),
        threshold=int(threshold),
        selected_count=int(selected_count),
        rng=rng,
    )
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    dataset = package_dataset(frame, panels)
    selected = tuple(panel for panel in panels if int(panel.shares_by_segment[str(condition_segment)]) > int(threshold))
    selected_counts = [counts_for_panel(panel)[str(target_segment)] for panel in selected]
    selection = selected_panel_sum_selection(
        selected_panels=selected,
        target_counts=selected_counts,
        trace={
            "condition_segment": str(condition_segment),
            "target_segment": str(target_segment),
            "threshold": int(threshold),
            "selected_panels": [str(panel.label) for panel in selected],
            "selected_target_counts": [int(value) for value in selected_counts],
            "calculation": "filter_panels_by_percentage_then_sum_target_counts",
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
        reasoning_load=0.78,
    )


@register_task
class ChartsCompositionPanelsConditionedPanelSumFromPercentTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'comparison', 'aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "conditioned_panel_sum_from_percent"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**TASK_PARAM_DEFAULTS, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsConditionedPanelSumFromPercentTask"]
