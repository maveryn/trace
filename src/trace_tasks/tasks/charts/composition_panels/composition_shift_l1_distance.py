"""Compute the L1 composition shift between two panels."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.values import int_sum
from trace_tasks.tasks.charts.composition_panels._lifecycle import package_composition_panels_plan as P, run_composition_panels_lifecycle as R
from trace_tasks.tasks.charts.composition_panels.shared.defaults import resolve_scene_variant as V
from trace_tasks.tasks.charts.composition_panels.shared.sampling import balanced_int, build_base_panels, package_dataset, sample_scene_frame
from trace_tasks.tasks.charts.composition_panels.shared.state import AnnotationRole, DOMAIN, CompositionPanelsSelection
from trace_tasks.tasks.registry import register_task


T = "task_charts__composition_panels__composition_shift_l1_distance"
TASK_PARAM_DEFAULTS = {
    "panel_count_min": 4,
    "panel_count_max": 6,
    "segment_count_min": 4,
    "segment_count_max": 4,
}
PGM = "sum(abs(share(end_panel,segment)-share(start_panel,segment)) for segment in segments); output=integer_value; annotation=bbox_set(compared_panels); scene=composition_panels; scope=composition_shift_l1_distance"
PROMPT_KEY = "composition_shift_l1_distance"
ANNOTATION_HINT = 'set "annotation" to an array of two [x0,y0,x1,y1] pixel boxes around the full compared panels'
JSON_EXAMPLE = '{"annotation":[[120,90,380,520],[460,90,720,520]],"answer":50}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":50}'

SHIFT_SHARE_PRESETS = (
    ((40, 30, 20, 10), (30, 30, 30, 10)),
    ((40, 30, 20, 10), (25, 30, 35, 10)),
    ((45, 25, 20, 10), (25, 25, 40, 10)),
    ((45, 25, 20, 10), (20, 25, 20, 35)),
    ((50, 20, 20, 10), (20, 35, 35, 10)),
    ((55, 25, 10, 10), (20, 30, 35, 15)),
)


def _build_plan(params, seed, _query_id, _probs):
    """Build the two-panel L1 composition-shift objective plan."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    variant, variant_probs = V(task_params, instance_seed=seed)
    frame = sample_scene_frame(task_params, instance_seed=seed)
    rng = spawn_rng(int(seed), f"{T}.objective")
    panels = build_base_panels(frame=frame, instance_seed=seed)
    dataset = package_dataset(frame, panels)
    start_label, end_label = rng.sample(list(frame.panel_labels), 2)
    preset_index = balanced_int(
        range(len(SHIFT_SHARE_PRESETS)),
        params=task_params,
        instance_seed=seed,
        namespace=f"{T}.shift_answer",
    )
    start_shares, end_shares = SHIFT_SHARE_PRESETS[int(preset_index)]
    fixed_by_panel = {
        str(start_label): {
            str(segment): int(value)
            for segment, value in zip(frame.segment_labels, start_shares)
        },
        str(end_label): {
            str(segment): int(value)
            for segment, value in zip(frame.segment_labels, end_shares)
        },
    }
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    dataset = package_dataset(frame, panels)
    by_label = {str(panel.label): panel for panel in panels}
    start_panel = by_label[str(start_label)]
    end_panel = by_label[str(end_label)]
    changes = tuple(
        abs(int(end_panel.shares_by_segment[str(segment)]) - int(start_panel.shares_by_segment[str(segment)]))
        for segment in frame.segment_labels
    )
    roles = (
        AnnotationRole("compared_panel", str(start_panel.label)),
        AnnotationRole("compared_panel", str(end_panel.label)),
    )
    selection = CompositionPanelsSelection(
        answer_value=int_sum(changes),
        annotation_values=tuple(int(value) for value in changes),
        annotation_roles=roles,
        question_format="numeric_open",
        trace={
            "start_panel": str(start_panel.label),
            "end_panel": str(end_panel.label),
            "compared_panels": [str(start_panel.label), str(end_panel.label)],
            "segment_changes": [int(value) for value in changes],
            "calculation": "sum_absolute_percentage_point_changes",
        },
        annotation_type="bbox_set",
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
        reasoning_load=0.84,
    )


@register_task
class ChartsCompositionPanelsCompositionShiftL1DistanceTask:
    task_id = T
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "composition_shift_l1_distance"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**TASK_PARAM_DEFAULTS, **dict(params)}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsCompositionShiftL1DistanceTask"]
