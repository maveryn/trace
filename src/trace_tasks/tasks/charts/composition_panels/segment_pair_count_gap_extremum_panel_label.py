"""Select the panel with the extreme count gap between two segments."""

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


T = "task_charts__composition_panels__segment_pair_count_gap_extremum_panel_label"
LARGEST_QUERY_ID = "largest_count_gap"
SMALLEST_QUERY_ID = "smallest_count_gap"
QUERY_IDS = (LARGEST_QUERY_ID, SMALLEST_QUERY_ID)
PGM = "select(panel where abs(count(panel,segment_a)-count(panel,segment_b)) is extremum(direction)); output=string_label; annotation=bbox(answer_panel); scene=composition_panels; scope=segment_pair_count_gap_extremum_panel_label"
PROMPT_KEY = "segment_pair_count_gap_extremum_panel_label"
ANNOTATION_HINT = 'set "annotation" to one [x0,y0,x1,y1] pixel box around the full answer panel'
JSON_EXAMPLE = '{"annotation":[120,90,380,520],"answer":"Harbor"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"Harbor"}'
ANSWER_HINT = 'set "answer" to the exact visible panel label as a string'
MIN_LARGEST_GAP_MARGIN_COUNT = 300
MIN_SMALLEST_GAP_MARGIN_COUNT = 250


def _panel_segment_count_gap(panel: PanelSpec, *, segment_a: str, segment_b: str) -> int:
    counts = counts_for_panel(panel)
    return abs(int(counts[str(segment_a)]) - int(counts[str(segment_b)]))


def _segment_pair_gap_values(
    panels: tuple[PanelSpec, ...],
    *,
    segment_a: str,
    segment_b: str,
) -> tuple[tuple[PanelSpec, int], ...]:
    """Return per-panel absolute count gaps for two public task-selected segments."""

    return tuple(
        (panel, _panel_segment_count_gap(panel, segment_a=str(segment_a), segment_b=str(segment_b)))
        for panel in panels
    )


def _extreme_gap_panel(
    panels: tuple[PanelSpec, ...],
    *,
    segment_a: str,
    segment_b: str,
    query_id: str,
) -> tuple[PanelSpec, int, int]:
    min_margin = MIN_LARGEST_GAP_MARGIN_COUNT if str(query_id) == LARGEST_QUERY_ID else MIN_SMALLEST_GAP_MARGIN_COUNT
    return select_unique_extreme_panel_value(
        _segment_pair_gap_values(panels, segment_a=str(segment_a), segment_b=str(segment_b)),
        select_largest=str(query_id) == LARGEST_QUERY_ID,
        min_margin=int(min_margin),
        error_label="segment pair count gap",
    )


def _build_plan(params, seed, selected_query_id, _probs):
    """Build a count-gap extremum objective over two visible segments."""

    variant, variant_probs = V(params, instance_seed=seed)
    frame = sample_scene_frame(params, instance_seed=seed)
    if len(frame.segment_labels) < 2:
        raise ValueError("segment_pair_count_gap_extremum_panel_label requires at least two segments")
    rng = spawn_rng(int(seed), f"{T}.objective")
    segment_a, segment_b = tuple(str(value) for value in rng.sample(list(frame.segment_labels), 2))
    base_panels = build_base_panels(frame=frame, instance_seed=seed)
    answer_panel = rng.choice(list(base_panels))
    if str(selected_query_id) == LARGEST_QUERY_ID:
        answer_shares = {str(segment_a): 54, str(segment_b): 6}
        distractor_shares = {str(segment_a): 26, str(segment_b): 22}
        extremum_word = "largest"
    else:
        answer_shares = {str(segment_a): 25, str(segment_b): 25}
        distractor_shares = {str(segment_a): 52, str(segment_b): 8}
        extremum_word = "smallest"
    fixed_by_panel = fixed_shares_for_answer_panel(
        base_panels,
        answer_label=str(answer_panel.label),
        answer_shares=answer_shares,
        distractor_shares=distractor_shares,
    )
    panels = build_base_panels(frame=frame, instance_seed=seed, fixed_by_panel=fixed_by_panel)
    answer_panel, answer_gap, gap_margin = _extreme_gap_panel(
        panels,
        segment_a=str(segment_a),
        segment_b=str(segment_b),
        query_id=str(selected_query_id),
    )
    counts = counts_for_panel(answer_panel)
    dataset = package_dataset(frame, panels)
    selection = CompositionPanelsSelection(
        answer_value=str(answer_panel.label),
        annotation_values=(int(answer_gap), int(counts[str(segment_a)]), int(counts[str(segment_b)])),
        annotation_roles=(AnnotationRole("answer_panel", str(answer_panel.label)),),
        question_format="string_label",
        trace={
            "segment_a": str(segment_a),
            "segment_b": str(segment_b),
            "extremum_direction": "largest" if str(selected_query_id) == LARGEST_QUERY_ID else "smallest",
            "extremum_word": str(extremum_word),
            "answer_panel": str(answer_panel.label),
            "answer_count_gap": int(answer_gap),
            "answer_segment_a_count": int(counts[str(segment_a)]),
            "answer_segment_b_count": int(counts[str(segment_b)]),
            "nearest_distractor_gap_margin_count": int(gap_margin),
            "answer_hint": ANSWER_HINT,
            "calculation": "derive_two_segment_counts_from_percent_and_total_then_select_extreme_absolute_count_difference",
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
        reasoning_load=0.68,
    )


@register_task
class ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "segment_pair_count_gap_extremum_panel_label"
    supported_query_ids = QUERY_IDS
    default_query_id = LARGEST_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params=dict(params), max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)


__all__ = ["ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask"]
