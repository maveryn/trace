"""Select the panel whose derived segment count is nearest a target count."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.values import count_from_percent_share
from trace_tasks.tasks.charts.composition_panels._lifecycle import package_composition_panels_plan as P, run_composition_panels_lifecycle as R
from trace_tasks.tasks.charts.composition_panels.shared.defaults import resolve_scene_variant as V
from trace_tasks.tasks.charts.composition_panels.shared.sampling import (
    build_base_panels,
    counts_for_panel,
    fixed_shares_for_answer_panel,
    package_dataset,
    sample_scene_frame,
    select_unique_nearest_panel_value,
)
from trace_tasks.tasks.charts.composition_panels.shared.state import AnnotationRole, DOMAIN, CompositionPanelsSelection, PanelSpec
from trace_tasks.tasks.registry import register_task


T = "task_charts__composition_panels__segment_count_nearest_target_panel_label"
PGM = "select(panel where abs(count(panel,target_segment)-target_count) is minimum); output=string_label; annotation=bbox(answer_panel); scene=composition_panels; scope=segment_count_nearest_target_panel_label"
PROMPT_KEY = "segment_count_nearest_target_panel_label"
ANNOTATION_HINT = 'set "annotation" to one [x0,y0,x1,y1] pixel box around the full answer panel'
JSON_EXAMPLE = '{"annotation":[120,90,380,520],"answer":"Harbor"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"Harbor"}'
ANSWER_HINT = 'set "answer" to the exact visible panel label as a string'
MIN_NEAREST_MARGIN_COUNT = 180


def _segment_count_values(panels: tuple[PanelSpec, ...], *, target_segment: str) -> tuple[tuple[PanelSpec, int], ...]:
    """Return per-panel counts for one public task-selected segment."""

    return tuple(
        (panel, int(counts_for_panel(panel)[str(target_segment)]))
        for panel in panels
    )


def _farthest_share_for_target(panel: PanelSpec, *, target_count: int) -> int:
    candidates = (6, 10, 14, 50, 54, 58)
    return max(candidates, key=lambda share: abs(count_from_percent_share(int(panel.total), int(share)) - int(target_count)))


def _build_plan(params, seed, _query_id, _probs):
    """Build a nearest-target objective over all panel counts."""

    variant, variant_probs = V(params, instance_seed=seed)
    frame = sample_scene_frame(params, instance_seed=seed)
    rng = spawn_rng(int(seed), f"{T}.objective")
    target_segment = str(rng.choice(list(frame.segment_labels)))
    base_panels = build_base_panels(frame=frame, instance_seed=seed)
    answer_panel = rng.choice(list(base_panels))
    answer_share = int(rng.choice([26, 30, 34, 38]))
    answer_count = count_from_percent_share(int(answer_panel.total), int(answer_share))
    target_count = int(answer_count + int(rng.choice([-40, -20, 20, 40])))
    fixed_by_panel = fixed_shares_for_answer_panel(
        base_panels,
        answer_label=str(answer_panel.label),
        answer_shares={str(target_segment): int(answer_share)},
        distractor_shares={str(target_segment): 6},
    )
    fixed_by_panel = {
        str(panel.label): (
            dict(fixed_by_panel[str(panel.label)])
            if str(panel.label) == str(answer_panel.label)
            else {str(target_segment): _farthest_share_for_target(panel, target_count=int(target_count))}
        )
        for panel in base_panels
    }
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    answer_panel, nearest_count, nearest_distance, nearest_margin = select_unique_nearest_panel_value(
        _segment_count_values(panels, target_segment=str(target_segment)),
        target_value=int(target_count),
        min_margin=int(MIN_NEAREST_MARGIN_COUNT),
        error_label="segment count nearest target",
    )
    dataset = package_dataset(frame, panels)
    selection = CompositionPanelsSelection(
        answer_value=str(answer_panel.label),
        annotation_values=(int(nearest_count), int(target_count), int(nearest_distance)),
        annotation_roles=(AnnotationRole("answer_panel", str(answer_panel.label)),),
        question_format="string_label",
        trace={
            "target_segment": str(target_segment),
            "target_count": int(target_count),
            "answer_panel": str(answer_panel.label),
            "answer_count": int(nearest_count),
            "answer_distance_from_target": int(nearest_distance),
            "nearest_distractor_margin_count": int(nearest_margin),
            "answer_hint": ANSWER_HINT,
            "calculation": "derive_segment_counts_from_percent_and_total_then_select_nearest_target",
        },
        annotation_type="bbox",
    )
    return P(
        dataset=dataset,
        selection=selection,
        params=params,
        scene_variant=variant,
        scene_variant_probabilities=variant_probs,
        prompt_key=PROMPT_KEY,
        annotation_hint_template=ANNOTATION_HINT,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        program_code=PGM,
        reasoning_load=0.7,
    )


@register_task
class ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "segment_count_nearest_target_panel_label"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params=dict(params), max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask"]
