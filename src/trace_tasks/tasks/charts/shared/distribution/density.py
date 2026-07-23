"""Density/violin dataset builders for distribution-style chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ..chart_scene_types import ViolinPlotSpec
from .config import (
    DensityQueryVariant,
    DistributionChartDefaults,
    _resolve_boxplot_value_bounds,
    _resolve_optional_nonnegative_int_bounds,
    _resolve_optional_positive_int_bounds,
    _resolve_violin_category_count_bounds,
)
from ..label_assets import sample_chart_labels
from ..labeled_chart_sampling import choose_mark_count

def _build_controlled_extreme_density_specs(
    *,
    labels: Sequence[str],
    candidate_modes: Sequence[int],
    query_id: DensityQueryVariant,
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    mode_window_size_bounds: Tuple[int, int],
    winner_gap_bounds: Tuple[int, int],
) -> Tuple[List[ViolinPlotSpec], Dict[str, Any]]:
    """Build a controlled unimodal set with one narrow extreme winner."""

    category_count = int(len(labels))
    min_window_size, max_window_size = mode_window_size_bounds
    if min_window_size < category_count:
        raise ValueError("mode_window_size_min must be >= category_count")
    if max_window_size < min_window_size:
        raise ValueError("mode_window_size_max must be >= mode_window_size_min")
    max_available_window = int(len(candidate_modes))
    if min_window_size > max_available_window:
        raise ValueError("mode window is larger than the available density support")

    feasible_window_sizes = [
        int(size)
        for size in range(int(min_window_size), min(int(max_window_size), max_available_window) + 1)
        if int(size) >= category_count
    ]
    if not feasible_window_sizes:
        raise ValueError("no feasible mode window sizes for controlled density generation")
    window_size = int(rng.choice(feasible_window_sizes))

    min_gap, max_gap = winner_gap_bounds
    max_feasible_gap = int(window_size) - int(category_count) + 1
    if max_feasible_gap < 1:
        raise ValueError("controlled density mode window leaves no room for a unique extreme winner")
    feasible_gaps = [
        int(gap)
        for gap in range(int(min_gap), int(max_gap) + 1)
        if 1 <= int(gap) <= int(max_feasible_gap)
    ]
    if not feasible_gaps:
        raise ValueError("winner gap bounds are infeasible for the requested controlled density window")
    winner_gap = int(rng.choice(feasible_gaps))

    low_bound = int(candidate_modes[0])
    high_bound = int(candidate_modes[-1])
    feasible_starts = list(range(int(low_bound), int(high_bound) - int(window_size) + 2))
    if not feasible_starts:
        raise ValueError("no feasible controlled density mode windows within candidate support")
    window_start = int(rng.choice(feasible_starts))
    window_modes = list(range(int(window_start), int(window_start) + int(window_size)))

    if str(query_id) == "highest_mode":
        winner_mode = int(window_modes[-1])
        runner_up_mode = int(winner_mode) - int(winner_gap)
        remaining_pool = [int(mode) for mode in window_modes if int(mode) < int(runner_up_mode)]
        chosen_remaining = rng.sample(remaining_pool, category_count - 2)
        chosen_modes = list(chosen_remaining) + [int(runner_up_mode), int(winner_mode)]
    elif str(query_id) == "lowest_mode":
        winner_mode = int(window_modes[0])
        runner_up_mode = int(winner_mode) + int(winner_gap)
        remaining_pool = [int(mode) for mode in window_modes if int(mode) > int(runner_up_mode)]
        chosen_remaining = rng.sample(remaining_pool, category_count - 2)
        chosen_modes = [int(winner_mode), int(runner_up_mode)] + list(chosen_remaining)
    else:
        raise ValueError(f"unsupported controlled extreme density query_id: {query_id}")

    rng.shuffle(chosen_modes)
    specs: List[ViolinPlotSpec] = []
    for label, mode in zip(labels, chosen_modes):
        support_padding_low = int(rng.randint(2, min(5, int(mode) - int(value_min))))
        support_padding_high = int(rng.randint(2, min(5, int(value_max) - int(mode))))
        specs.append(
            ViolinPlotSpec(
                label=str(label),
                support_min=int(mode) - int(support_padding_low),
                support_max=int(mode) + int(support_padding_high),
                mode_values=(int(mode),),
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
            )
        )

    return specs, {
        "generation_profile": "controlled_extreme",
        "mode_window_size": int(window_size),
        "extreme_winner_gap": int(winner_gap),
    }


def _build_controlled_bimodal_density_specs(
    *,
    labels: Sequence[str],
    candidate_modes: Sequence[int],
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    separation_bounds: Tuple[int, int],
    clearance_bounds: Tuple[int, int],
) -> Tuple[List[ViolinPlotSpec], Dict[str, Any]]:
    """Build a controlled bimodal set with a clearer bimodal witness."""

    category_count = int(len(labels))
    min_separation, max_separation = separation_bounds
    min_clearance, max_clearance = clearance_bounds
    feasible_pairs: List[Tuple[int, int, int, List[int]]] = []
    for clearance in range(int(min_clearance), int(max_clearance) + 1):
        for lower in candidate_modes:
            for separation in range(int(min_separation), int(max_separation) + 1):
                upper = int(lower) + int(separation)
                if int(upper) not in candidate_modes:
                    continue
                distractors = [
                    int(mode)
                    for mode in candidate_modes
                    if int(mode) not in {int(lower), int(upper)}
                    and abs(int(mode) - int(lower)) > int(clearance)
                    and abs(int(mode) - int(upper)) > int(clearance)
                ]
                if len(distractors) >= category_count - 1:
                    feasible_pairs.append((int(lower), int(upper), int(clearance), distractors))
    if not feasible_pairs:
        raise ValueError("no feasible controlled bimodal density pairs for the requested separation/clearance")

    lower, upper, distractor_clearance, distractor_pool = feasible_pairs[int(rng.randint(0, len(feasible_pairs) - 1))]
    chosen_distractors = rng.sample(distractor_pool, category_count - 1)
    specs: List[ViolinPlotSpec] = []
    for index, label in enumerate(labels):
        if int(index) == 0:
            support_min = max(int(value_min), int(lower) - int(rng.randint(1, 2)))
            support_max = min(int(value_max), int(upper) + int(rng.randint(1, 2)))
            specs.append(
                ViolinPlotSpec(
                    label=str(label),
                    support_min=int(support_min),
                    support_max=int(support_max),
                    mode_values=(int(lower), int(upper)),
                    fill_rgb=fill_rgb,
                    outline_rgb=outline_rgb,
                )
            )
        else:
            mode = int(chosen_distractors[int(index) - 1])
            support_padding_low = int(rng.randint(2, min(5, int(mode) - int(value_min))))
            support_padding_high = int(rng.randint(2, min(5, int(value_max) - int(mode))))
            specs.append(
                ViolinPlotSpec(
                    label=str(label),
                    support_min=int(mode) - int(support_padding_low),
                    support_max=int(mode) + int(support_padding_high),
                    mode_values=(int(mode),),
                    fill_rgb=fill_rgb,
                    outline_rgb=outline_rgb,
                )
            )
    rng.shuffle(specs)
    return specs, {
        "generation_profile": "controlled_bimodal",
        "bimodal_mode_separation": int(upper) - int(lower),
        "bimodal_distractor_clearance": int(distractor_clearance),
    }


def _build_controlled_support_span_density_specs(
    *,
    labels: Sequence[str],
    query_id: DensityQueryVariant,
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    support_span_bounds: Tuple[int, int],
    winner_gap_bounds: Tuple[int, int],
) -> Tuple[List[ViolinPlotSpec], Dict[str, Any]]:
    """Build a controlled violin set with one unique support-span extremum."""

    category_count = int(len(labels))
    min_span, max_span = support_span_bounds
    value_span = int(value_max) - int(value_min)
    candidate_spans = [
        int(span)
        for span in range(int(min_span), min(int(max_span), int(value_span)) + 1)
        if int(span) >= 4
    ]
    if len(candidate_spans) < int(category_count):
        raise ValueError("not enough feasible support spans for controlled violin spread generation")

    min_gap, max_gap = winner_gap_bounds
    feasible_choices: List[Tuple[int, int, int, List[int]]] = []
    for gap in range(int(min_gap), int(max_gap) + 1):
        if int(gap) < 1:
            continue
        for winner_span in candidate_spans:
            if str(query_id) == "widest_support":
                runner_span = int(winner_span) - int(gap)
                remaining_pool = [
                    int(span)
                    for span in candidate_spans
                    if int(span) < int(runner_span)
                ]
            elif str(query_id) == "narrowest_support":
                runner_span = int(winner_span) + int(gap)
                remaining_pool = [
                    int(span)
                    for span in candidate_spans
                    if int(span) > int(runner_span)
                ]
            else:
                raise ValueError(f"unsupported controlled support-span variant: {query_id}")
            if int(runner_span) in set(candidate_spans) and len(remaining_pool) >= int(category_count) - 2:
                feasible_choices.append((int(gap), int(winner_span), int(runner_span), list(remaining_pool)))
    if not feasible_choices:
        raise ValueError("support-span winner gap bounds are infeasible for the requested violin setup")

    winner_gap, winner_span, runner_span, remaining_pool = feasible_choices[
        int(rng.randint(0, len(feasible_choices) - 1))
    ]
    chosen_spans = [int(winner_span), int(runner_span)] + [
        int(span)
        for span in rng.sample(remaining_pool, int(category_count) - 2)
    ]
    rng.shuffle(chosen_spans)

    specs: List[ViolinPlotSpec] = []
    for label, support_span in zip(labels, chosen_spans):
        support_min = int(rng.randint(int(value_min), int(value_max) - int(support_span)))
        support_max = int(support_min) + int(support_span)
        inner_min = int(support_min) + 2
        inner_max = int(support_max) - 2
        mode = int(rng.randint(int(inner_min), int(inner_max))) if int(inner_min) <= int(inner_max) else int(
            round((int(support_min) + int(support_max)) / 2.0)
        )
        specs.append(
            ViolinPlotSpec(
                label=str(label),
                support_min=int(support_min),
                support_max=int(support_max),
                mode_values=(int(mode),),
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
            )
        )

    return specs, {
        "generation_profile": "controlled_support_span",
        "support_span_winner_gap": int(winner_gap),
        "support_span_bounds": [int(min_span), int(max_span)],
    }


def build_density_dataset_for_variant(
    *,
    query_id: DensityQueryVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    mark_style: Mapping[str, Any],
) -> Tuple[List[ViolinPlotSpec], str, List[int], Dict[str, Any]]:
    """Build one violin-backed density task dataset."""

    category_count_min, category_count_max = _resolve_violin_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    value_min, value_max = _resolve_boxplot_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    category_count = choose_mark_count(
        list(range(int(category_count_min), int(category_count_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:category_count:{str(query_id)}",
    )
    labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(query_id)}:{int(category_count)}",
        )
    )
    rng = spawn_rng(int(instance_seed), f"{task_id}.density.{str(query_id)}")
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])

    specs: List[ViolinPlotSpec] = []
    generation_meta: Dict[str, Any] = {"generation_profile": "baseline"}
    if str(query_id) in {"highest_mode", "lowest_mode"}:
        candidate_modes = list(range(int(value_min) + 3, int(value_max) - 2))
        if len(candidate_modes) < int(category_count):
            raise ValueError("density mode support is too small for requested category count")
        mode_window_size_bounds = _resolve_optional_positive_int_bounds(
            params,
            min_key="mode_window_size_min",
            max_key="mode_window_size_max",
            gen_defaults=gen_defaults,
        )
        winner_gap_bounds = _resolve_optional_positive_int_bounds(
            params,
            min_key="extreme_winner_gap_min",
            max_key="extreme_winner_gap_max",
            gen_defaults=gen_defaults,
        )
        if mode_window_size_bounds is not None or winner_gap_bounds is not None:
            if mode_window_size_bounds is None or winner_gap_bounds is None:
                raise ValueError(
                    "controlled extreme density generation requires both mode_window_size_* and "
                    "extreme_winner_gap_* bounds"
                )
            specs, generation_meta = _build_controlled_extreme_density_specs(
                labels=labels,
                candidate_modes=candidate_modes,
                query_id=str(query_id),
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
                mode_window_size_bounds=mode_window_size_bounds,
                winner_gap_bounds=winner_gap_bounds,
            )
        else:
            rng.shuffle(candidate_modes)
            chosen_modes = candidate_modes[: int(category_count)]
            rng.shuffle(chosen_modes)
            for label, mode in zip(labels, chosen_modes):
                support_padding_low = int(rng.randint(2, min(5, int(mode) - int(value_min))))
                support_padding_high = int(rng.randint(2, min(5, int(value_max) - int(mode))))
                specs.append(
                    ViolinPlotSpec(
                        label=str(label),
                        support_min=int(mode) - int(support_padding_low),
                        support_max=int(mode) + int(support_padding_high),
                        mode_values=(int(mode),),
                        fill_rgb=fill_rgb,
                        outline_rgb=outline_rgb,
                    )
                )
        if str(query_id) == "highest_mode":
            winning_spec = max(specs, key=lambda spec: int(spec.mode_values[0]))
        else:
            winning_spec = min(specs, key=lambda spec: int(spec.mode_values[0]))
        answer_label = str(winning_spec.label)
        annotation_values = [int(winning_spec.mode_values[0])]
    elif str(query_id) == "bimodal_label":
        candidate_modes = list(range(int(value_min) + 3, int(value_max) - 2))
        bimodal_separation_bounds = _resolve_optional_positive_int_bounds(
            params,
            min_key="bimodal_mode_separation_min",
            max_key="bimodal_mode_separation_max",
            gen_defaults=gen_defaults,
        )
        bimodal_clearance_bounds = _resolve_optional_nonnegative_int_bounds(
            params,
            min_key="bimodal_distractor_clearance_min",
            max_key="bimodal_distractor_clearance_max",
            gen_defaults=gen_defaults,
        )
        if bimodal_separation_bounds is not None or bimodal_clearance_bounds is not None:
            if bimodal_separation_bounds is None or bimodal_clearance_bounds is None:
                raise ValueError(
                    "controlled bimodal density generation requires both bimodal_mode_separation_* and "
                    "bimodal_distractor_clearance_* bounds"
                )
            specs, generation_meta = _build_controlled_bimodal_density_specs(
                labels=labels,
                candidate_modes=candidate_modes,
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
                separation_bounds=bimodal_separation_bounds,
                clearance_bounds=bimodal_clearance_bounds,
            )
        else:
            rng.shuffle(candidate_modes)
            unimodal_modes = candidate_modes[: int(category_count)]
            for index, (label, mode) in enumerate(zip(labels, unimodal_modes)):
                if int(index) == 0:
                    lower = int(rng.randint(int(value_min) + 2, int(value_max) - 7))
                    upper = int(rng.randint(int(lower) + 3, int(min(value_max - 2, lower + 6))))
                    support_min = max(int(value_min), int(lower) - int(rng.randint(1, 3)))
                    support_max = min(int(value_max), int(upper) + int(rng.randint(1, 3)))
                    specs.append(
                        ViolinPlotSpec(
                            label=str(label),
                            support_min=int(support_min),
                            support_max=int(support_max),
                            mode_values=(int(lower), int(upper)),
                            fill_rgb=fill_rgb,
                            outline_rgb=outline_rgb,
                        )
                    )
                else:
                    support_padding_low = int(rng.randint(2, min(5, int(mode) - int(value_min))))
                    support_padding_high = int(rng.randint(2, min(5, int(value_max) - int(mode))))
                    specs.append(
                        ViolinPlotSpec(
                            label=str(label),
                            support_min=int(mode) - int(support_padding_low),
                            support_max=int(mode) + int(support_padding_high),
                            mode_values=(int(mode),),
                            fill_rgb=fill_rgb,
                            outline_rgb=outline_rgb,
                        )
                    )
            rng.shuffle(specs)
        winning_spec = next(spec for spec in specs if len(spec.mode_values) == 2)
        answer_label = str(winning_spec.label)
        annotation_values = sorted(int(value) for value in winning_spec.mode_values)
    elif str(query_id) in {"widest_support", "narrowest_support"}:
        support_span_bounds = _resolve_optional_positive_int_bounds(
            params,
            min_key="support_span_min",
            max_key="support_span_max",
            gen_defaults=gen_defaults,
        )
        if support_span_bounds is None:
            support_span_bounds = (6, max(6, int(value_max) - int(value_min) - 2))
        support_winner_gap_bounds = _resolve_optional_positive_int_bounds(
            params,
            min_key="support_winner_gap_min",
            max_key="support_winner_gap_max",
            gen_defaults=gen_defaults,
        )
        if support_winner_gap_bounds is None:
            support_winner_gap_bounds = (2, 4)
        specs, generation_meta = _build_controlled_support_span_density_specs(
            labels=labels,
            query_id=str(query_id),
            value_min=int(value_min),
            value_max=int(value_max),
            rng=rng,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
            support_span_bounds=support_span_bounds,
            winner_gap_bounds=support_winner_gap_bounds,
        )
        if str(query_id) == "widest_support":
            winning_spec = max(specs, key=lambda spec: int(spec.support_max) - int(spec.support_min))
        else:
            winning_spec = min(specs, key=lambda spec: int(spec.support_max) - int(spec.support_min))
        answer_label = str(winning_spec.label)
        annotation_values = [int(winning_spec.support_min), int(winning_spec.support_max)]
    else:
        raise ValueError(f"unsupported density query_id: {query_id}")

    trace_extras = {
        "scene_variant": "violin",
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "value_range": [int(value_min), int(value_max)],
        **generation_meta,
        "answer_label": str(answer_label),
        "annotation_values": [int(value) for value in annotation_values],
        "support_by_label": {
            str(spec.label): {
                "support_min": int(spec.support_min),
                "support_max": int(spec.support_max),
                "support_span": int(spec.support_max) - int(spec.support_min),
                "mode_values": [int(value) for value in spec.mode_values],
                "bimodal": bool(len(spec.mode_values) == 2),
            }
            for spec in specs
        },
    }
    return specs, str(answer_label), [int(value) for value in annotation_values], trace_extras


__all__ = ["build_density_dataset_for_variant"]
