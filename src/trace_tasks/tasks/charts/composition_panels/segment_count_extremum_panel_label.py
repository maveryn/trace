"""Select the panel with the extreme derived segment count."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.composition_panels._lifecycle import package_composition_panels_plan as P, run_composition_panels_lifecycle as R
from trace_tasks.tasks.charts.composition_panels.shared.defaults import resolve_scene_variant as V
from trace_tasks.tasks.charts.composition_panels.shared.sampling import (
    build_base_panels,
    counts_for_panel,
    fixed_shares_for_answer_panel,
    package_dataset,
    sample_scene_frame,
    select_unique_extreme_panel_value,
)
from trace_tasks.tasks.charts.composition_panels.shared.state import AnnotationRole, DOMAIN, CompositionPanelsSelection, PanelSpec
from trace_tasks.tasks.registry import register_task


T = "task_charts__composition_panels__segment_count_extremum_panel_label"
LARGEST_QUERY_ID = "largest_count"
SMALLEST_QUERY_ID = "smallest_count"
QUERY_IDS = (LARGEST_QUERY_ID, SMALLEST_QUERY_ID)
PGM = "select(panel where count(panel,target_segment) is extremum(direction)); output=string_label; annotation=point_map(segment_percent,panel_total); scene=composition_panels; scope=segment_count_extremum_panel_label"
PROMPT_KEY = "segment_count_extremum_panel_label"
ANNOTATION_HINT = 'set "annotation" to an object with keys "segment_percent" and "panel_total"; put [x,y] pixel points on the answer panel\'s Segment "{target_segment}" percentage label and total-count text'
JSON_EXAMPLE = '{"annotation":{"segment_percent":[260,310],"panel_total":[245,110]},"answer":"Harbor"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"Harbor"}'
ANSWER_HINT = 'set "answer" to the exact visible panel label as a string'
MIN_EXTREMUM_GAP_COUNT = 30


def _segment_count_values(panels: tuple[PanelSpec, ...], *, target_segment: str) -> tuple[tuple[PanelSpec, int], ...]:
    """Return per-panel counts for one public task-selected segment."""

    return tuple(
        (panel, int(counts_for_panel(panel)[str(target_segment)]))
        for panel in panels
    )


def _extreme_panel(panels: tuple[PanelSpec, ...], *, target_segment: str, query_id: str) -> tuple[PanelSpec, int, int]:
    return select_unique_extreme_panel_value(
        _segment_count_values(panels, target_segment=str(target_segment)),
        select_largest=str(query_id) == LARGEST_QUERY_ID,
        min_margin=int(MIN_EXTREMUM_GAP_COUNT),
        error_label="segment count",
    )


def _build_plan(params, seed, selected_query_id, _probs):
    """Build a direct segment-count extremum objective over all panels."""

    variant, variant_probs = V(params, instance_seed=seed)
    frame = sample_scene_frame(params, instance_seed=seed)
    rng = spawn_rng(int(seed), f"{T}.objective")
    target_segment = str(rng.choice(list(frame.segment_labels)))
    base_panels = build_base_panels(frame=frame, instance_seed=seed)
    if str(selected_query_id) == LARGEST_QUERY_ID:
        answer_label = max(base_panels, key=lambda panel: int(panel.total)).label
        answer_share = 56
        distractor_share = 16
    else:
        answer_label = min(base_panels, key=lambda panel: int(panel.total)).label
        answer_share = 6
        distractor_share = 28
    fixed_by_panel = fixed_shares_for_answer_panel(
        base_panels,
        answer_label=str(answer_label),
        answer_shares={str(target_segment): int(answer_share)},
        distractor_shares={str(target_segment): int(distractor_share)},
    )
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    answer_panel, answer_count, answer_gap = _extreme_panel(
        panels,
        target_segment=str(target_segment),
        query_id=str(selected_query_id),
    )
    dataset = package_dataset(frame, panels)
    answer_share = int(answer_panel.shares_by_segment[str(target_segment)])
    selection = CompositionPanelsSelection(
        answer_value=str(answer_panel.label),
        annotation_values=(int(answer_share), int(answer_panel.total), int(answer_count)),
        annotation_roles=(
            AnnotationRole("segment_percent", str(answer_panel.label), str(target_segment), key="segment_percent"),
            AnnotationRole("panel_total", str(answer_panel.label), key="panel_total"),
        ),
        question_format="string_label",
        trace={
            "target_segment": str(target_segment),
            "extremum_direction": "largest" if str(selected_query_id) == LARGEST_QUERY_ID else "smallest",
            "extremum_word": "largest" if str(selected_query_id) == LARGEST_QUERY_ID else "smallest",
            "answer_panel": str(answer_panel.label),
            "answer_count": int(answer_count),
            "answer_segment_percent": int(answer_share),
            "answer_panel_total": int(answer_panel.total),
            "nearest_distractor_gap_count": int(answer_gap),
            "answer_hint": ANSWER_HINT,
            "calculation": "derive_segment_counts_from_percent_and_total_then_select_extremum",
        },
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
        reasoning_load=0.66,
    )


@register_task
class ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "segment_count_extremum_panel_label"
    supported_query_ids = QUERY_IDS
    default_query_id = LARGEST_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params=dict(params), max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask"]
