"""Sampling primitives for composition-panel charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import (
    resolve_chart_category_labels,
    resolve_chart_panel_labels,
    validate_chart_label_namespaces,
)
from trace_tasks.tasks.charts.shared.composition.values import (
    counts_from_percent_shares,
    int_sum,
    select_unique_extremum,
    select_unique_nearest,
)
from trace_tasks.tasks.charts.shared.labeled_chart_values import balanced_choice_from_values
from trace_tasks.tasks.charts.shared.labeled_chart_composition import sample_composition_with_sum
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import GEN_DEFAULTS, resolve_count_bounds
from .state import AnnotationRole, PanelSpec, SCENE_NAMESPACE, CompositionPanelsDataset, CompositionPanelsSelection, SUPPORTED_SCENE_VARIANTS


@dataclass(frozen=True)
class CompositionPanelsFrame:
    panel_labels: tuple[str, ...]
    segment_labels: tuple[str, ...]
    total_values: tuple[int, ...]
    trace: dict[str, Any]


def balanced_int(values: Sequence[int], *, params: Mapping[str, Any], instance_seed: int, namespace: str) -> int:
    return balanced_choice_from_values(
        [int(value) for value in values],
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def params_with_shifted_sample_cursor(params: Mapping[str, Any], *, divisor: int) -> dict[str, Any]:
    shifted = dict(params)
    if "_sample_cursor" not in shifted:
        return shifted
    shifted["_sample_cursor"] = abs(int(shifted["_sample_cursor"])) // max(1, int(divisor))
    return shifted


def sample_scene_frame(params: Mapping[str, Any], *, instance_seed: int) -> CompositionPanelsFrame:
    """Sample the reusable scene frame before any public objective is chosen.

    Key invariant: this function samples only panel/segment labels, counts, and
    shared scene ranges; public task files decide the objective-specific fixed
    values, selected panels, answer, and annotation roles.
    """

    panel_min, panel_max = resolve_count_bounds(
        params,
        min_key="panel_count_min",
        max_key="panel_count_max",
        fallback_min=6,
        fallback_max=9,
    )
    segment_min, segment_max = resolve_count_bounds(
        params,
        min_key="segment_count_min",
        max_key="segment_count_max",
        fallback_min=5,
        fallback_max=7,
    )
    count_params = params_with_shifted_sample_cursor(
        params,
        divisor=len(SUPPORTED_SCENE_VARIANTS),
    )
    panel_count = balanced_int(
        range(int(panel_min), int(panel_max) + 1),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_count",
    )
    segment_count = balanced_int(
        range(int(segment_min), int(segment_max) + 1),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.segment_count",
    )
    segment_labels = choose_segments(int(segment_count), instance_seed=int(instance_seed))
    panel_labels, panel_label_meta = choose_panel_labels(
        int(panel_count),
        segment_labels=segment_labels,
        params=params,
        instance_seed=int(instance_seed),
    )
    return CompositionPanelsFrame(
        panel_labels=tuple(panel_labels),
        segment_labels=tuple(segment_labels),
        total_values=sample_total_values(params),
        trace={
            "panel_count": int(panel_count),
            "panel_count_range": [int(panel_min), int(panel_max)],
            "segment_count": int(segment_count),
            "segment_count_range": [int(segment_min), int(segment_max)],
            **dict(panel_label_meta),
        },
    )


def choose_segments(segment_count: int, *, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.segment_labels")
    labels = resolve_chart_category_labels(
        rng,
        count=int(segment_count),
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


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


def choose_panel_labels(
    panel_count: int,
    *,
    segment_labels: Sequence[str],
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
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
                    "temporal_sequence": 0.6,
                    "named_compact": 1.0,
                    "report_topics": 0.75,
                    "technical_topics": 1.0,
                    "condition_labels": 0.5,
                },
            ),
        ),
        reserved_labels=tuple(str(label) for label in segment_labels),
    )
    collision_check = validate_chart_label_namespaces(
        panel_labels=resolved.labels,
        other_label_groups={"segment_labels": tuple(str(label) for label in segment_labels)},
        context="composition-panel labels",
    )
    return tuple(str(label) for label in resolved.labels), {
        "panel_label_resolution": _resolved_label_metadata(resolved),
        "panel_label_collision_check": dict(collision_check),
    }


def sample_total_values(params: Mapping[str, Any]) -> tuple[int, ...]:
    raw = params.get(
        "total_values",
        group_default(GEN_DEFAULTS, "total_values", [1000, 1200, 1400, 1600, 1800, 2000, 2400, 3000]),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("total_values must be a sequence")
    values = tuple(int(value) for value in raw)
    if not values or any(int(value) % 100 != 0 or int(value) <= 0 for value in values):
        raise ValueError("total_values must contain positive multiples of 100")
    return values


def choose_total(total_values: Sequence[int], *, instance_seed: int, namespace: str) -> int:
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(total_values[int(rng.randrange(0, len(total_values)))])


def sample_shares(
    *,
    segment_labels: Sequence[str],
    fixed: Mapping[str, int],
    instance_seed: int,
    namespace: str,
    value_min: int = 6,
    value_max: int = 58,
) -> dict[str, int]:
    fixed_values = {str(key): int(value) for key, value in fixed.items()}
    remaining_labels = [str(label) for label in segment_labels if str(label) not in fixed_values]
    fixed_sum = int(sum(fixed_values.values()))
    remaining_sum = int(100 - fixed_sum)
    if not remaining_labels:
        if int(remaining_sum) != 0:
            raise ValueError("fixed shares do not sum to 100")
        return dict(fixed_values)
    if int(remaining_sum) < len(remaining_labels) * int(value_min) or int(remaining_sum) > len(remaining_labels) * int(value_max):
        raise ValueError("fixed shares leave no feasible remainder")
    rng = spawn_rng(int(instance_seed), str(namespace))
    values = sample_composition_with_sum(
        rng,
        target_sum=int(remaining_sum),
        count=len(remaining_labels),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    rng.shuffle(values)
    shares = dict(fixed_values)
    shares.update({str(label): int(value) for label, value in zip(remaining_labels, values)})
    if int(sum(shares.values())) != 100:
        raise RuntimeError("percentage shares drifted from 100")
    return {str(label): int(shares[str(label)]) for label in segment_labels}


def build_base_panels(
    *,
    frame: CompositionPanelsFrame,
    instance_seed: int,
    fixed_by_panel: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[PanelSpec, ...]:
    panels: list[PanelSpec] = []
    fixed_by_panel = fixed_by_panel or {}
    for index, label in enumerate(frame.panel_labels):
        shares = sample_shares(
            segment_labels=frame.segment_labels,
            fixed=fixed_by_panel.get(str(label), {}),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.shares.{label}.{index}",
        )
        panels.append(
            PanelSpec(
                label=str(label),
                total=choose_total(
                    frame.total_values,
                    instance_seed=int(instance_seed),
                    namespace=f"{SCENE_NAMESPACE}.total.{label}.{index}",
                ),
                shares_by_segment=shares,
            )
        )
    return tuple(panels)


def fixed_shares_for_answer_panel(
    panels: Sequence[PanelSpec],
    *,
    answer_label: str,
    answer_shares: Mapping[str, int],
    distractor_shares: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Return fixed segment-share overrides for one answer panel and all distractors."""

    return {
        str(panel.label): dict(answer_shares if str(panel.label) == str(answer_label) else distractor_shares)
        for panel in panels
    }


def select_unique_extreme_panel_value(
    values: Sequence[tuple[PanelSpec, int]],
    *,
    select_largest: bool,
    min_margin: int,
    error_label: str,
) -> tuple[PanelSpec, int, int]:
    """Select a unique panel extremum from task-supplied numeric values."""

    selected = select_unique_extremum(
        values,
        select_largest=bool(select_largest),
        min_margin=int(min_margin),
        error_label=str(error_label),
        item_label="panels",
    )
    return selected.item, int(selected.value), int(selected.margin)


def select_unique_nearest_panel_value(
    values: Sequence[tuple[PanelSpec, int]],
    *,
    target_value: int,
    min_margin: int,
    error_label: str,
) -> tuple[PanelSpec, int, int, int]:
    """Select the panel whose task-supplied numeric value is uniquely nearest a target."""

    selected = select_unique_nearest(
        tuple(values),
        value_fn=lambda item: int(item[1]),
        target_value=int(target_value),
        min_margin=int(min_margin),
        error_label=str(error_label),
        item_label="panels",
    )
    answer_panel, _answer_value = selected.item
    return answer_panel, int(selected.value), int(selected.distance), int(selected.margin)


def ranked_share_fixtures(
    panel_labels: Sequence[str],
    *,
    segment: str,
    support_values: Sequence[int],
    rng: Any,
) -> dict[str, dict[str, int]]:
    shuffled = [int(value) for value in support_values]
    rng.shuffle(shuffled)
    return {str(label): {str(segment): int(shuffled[index])} for index, label in enumerate(panel_labels)}


def top_panels_by_share(panels: Sequence[PanelSpec], *, segment: str, k: int) -> tuple[PanelSpec, ...]:
    return tuple(sorted(panels, key=lambda panel: int(panel.shares_by_segment[str(segment)]), reverse=True)[: int(k)])


def counts_for_panel(panel: PanelSpec) -> dict[str, int]:
    return counts_from_percent_shares(int(panel.total), panel.shares_by_segment)


def selected_panel_sum_selection(
    *,
    selected_panels: Sequence[PanelSpec],
    target_counts: Sequence[int],
    trace: Mapping[str, Any],
) -> CompositionPanelsSelection:
    """Package a selected-panel aggregate while callers own the selection rule."""

    counts = tuple(int(value) for value in target_counts)
    return CompositionPanelsSelection(
        answer_value=int_sum(counts),
        annotation_values=counts,
        annotation_roles=tuple(AnnotationRole("selected_panel", str(panel.label)) for panel in selected_panels),
        question_format="numeric_open",
        trace={str(key): value for key, value in trace.items()},
        annotation_type="bbox_set",
    )


def package_dataset(frame: CompositionPanelsFrame, panels: Sequence[PanelSpec]) -> CompositionPanelsDataset:
    return CompositionPanelsDataset(
        panels=tuple(panels),
        segment_labels=tuple(frame.segment_labels),
        trace=dict(frame.trace),
    )
