"""Sampling primitives for density-curve chart scenes."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import normalize_positive_weights, weighted_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.density_curve.shared.defaults import (
    DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E,
    GEN_DEFAULTS,
    RENDER_DEFAULTS,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
    SUPPORTED_CURVE_LINE_STYLES,
    SUPPORTED_DENSITY_FAMILIES,
    TRACE_SAFE_DENSITY_CURVE_PALETTE_RGB,
    gen_float,
    gen_int,
)
from trace_tasks.tasks.charts.density_curve.shared.metrics import (
    curve_metrics,
    density_values_for_family,
    integral,
    metric_values_by_label,
)
from trace_tasks.tasks.charts.density_curve.shared.state import (
    DensityCurve,
    DensityCurveDataset,
    DensityCurveObjectiveSpec,
    DensityCurveQuery,
    DensityCurveSceneSample,
    CurvePoint,
    RGB,
)
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_category_labels
from trace_tasks.tasks.shared.color_distance import color_distance, sample_color_palette_with_distance_constraints
from trace_tasks.tasks.shared.config_defaults import group_default


def resolve_curve_count(params: Mapping[str, Any], *, rng) -> tuple[int, dict[str, float]]:
    """Sample the number of visible density curves."""

    count_min = max(2, gen_int(params, "density_curve_count_min", 4))
    count_max = max(count_min, gen_int(params, "density_curve_count_max", 6))
    raw_weights = params.get(
        "density_curve_count_weights",
        group_default(GEN_DEFAULTS, "density_curve_count_weights", {}),
    )
    if isinstance(raw_weights, Mapping):
        weights = {
            str(count): float(raw_weights.get(str(count), raw_weights.get(int(count), 0.0)))
            for count in range(int(count_min), int(count_max) + 1)
        }
    else:
        weights = {}
    probabilities = normalize_positive_weights(
        weights,
        default_keys=tuple(str(count) for count in range(int(count_min), int(count_max) + 1)),
    )
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))
    return int(selected), dict(probabilities)


def resolve_density_family(params: Mapping[str, Any], *, rng) -> tuple[str, dict[str, float]]:
    """Sample a density family."""

    raw_weights = params.get("density_family_weights", group_default(GEN_DEFAULTS, "density_family_weights", {}))
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}
    probabilities = normalize_positive_weights(
        {str(key): float(value) for key, value in raw_weights.items() if str(key) in set(SUPPORTED_DENSITY_FAMILIES)},
        default_keys=SUPPORTED_DENSITY_FAMILIES,
    )
    return str(weighted_choice(rng, probabilities, sort_keys=True)), dict(probabilities)


def sample_palette(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    count: int,
    anchor_colors: Sequence[RGB],
) -> tuple[RGB, ...]:
    """Sample visually separated curve identity colors.

    Scene role: density curves often overlap, so this helper owns the curve
    color contract before objective binding. Key invariant: the default path
    returns a deterministic subset of the shared Trace visibility-safe palette
    with pairwise Lab separation high enough for every visible curve.
    """

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette")
    channel_min = int(params.get("density_curve_color_channel_min", group_default(RENDER_DEFAULTS, "mark_color_channel_min", 20)))
    channel_max = int(params.get("density_curve_color_channel_max", group_default(RENDER_DEFAULTS, "mark_color_channel_max", 210)))
    min_distance = max(
        float(DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E),
        float(params.get("density_curve_color_min_distance", group_default(RENDER_DEFAULTS, "mark_color_min_distance", DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E))),
    )
    distance_space = str(params.get("density_curve_color_distance_space", group_default(RENDER_DEFAULTS, "mark_color_distance_space", "lab")))
    palette_mode = str(params.get("density_curve_palette_mode", group_default(RENDER_DEFAULTS, "density_curve_palette_mode", "safe_qualitative")))
    if palette_mode == "safe_qualitative":
        del anchor_colors
        return _sample_trace_safe_curve_palette(
            rng,
            count=int(count),
            min_pairwise_distance=float(min_distance),
            distance_space=str(distance_space),
        )
    return sample_color_palette_with_distance_constraints(
        rng,
        palette_size=int(count),
        channel_min=int(channel_min),
        channel_max=int(channel_max),
        anchor_colors=tuple(anchor_colors),
        min_distance=float(min_distance),
        distance_space=str(distance_space),
    )


def _sample_trace_safe_curve_palette(
    rng,
    *,
    count: int,
    min_pairwise_distance: float,
    distance_space: str,
) -> tuple[RGB, ...]:
    """Choose one Trace-safe color subset satisfying the curve separation floor."""

    candidate_pool = tuple(tuple(int(channel) for channel in color) for color in TRACE_SAFE_DENSITY_CURVE_PALETTE_RGB)
    size = int(count)
    if size < 1:
        return tuple()
    if size > len(candidate_pool):
        raise ValueError("density_curve_count exceeds shared Trace safe color support")
    threshold = float(min_pairwise_distance)
    feasible: list[tuple[RGB, ...]] = []
    best_combo: tuple[RGB, ...] | None = None
    best_distance = -1.0
    for combo in combinations(candidate_pool, size):
        min_distance = _min_pairwise_color_distance(combo, distance_space=str(distance_space))
        if float(min_distance) > float(best_distance):
            best_distance = float(min_distance)
            best_combo = tuple(combo)
        if float(min_distance) >= float(threshold):
            feasible.append(tuple(combo))
    if not feasible:
        raise ValueError(
            f"no {size}-color Trace safe density-curve subset reaches pairwise {threshold:.1f}; "
            f"best={best_distance:.3f}"
        )
    selected = list(feasible[int(rng.randrange(len(feasible)))])
    rng.shuffle(selected)
    return tuple(tuple(int(channel) for channel in color) for color in selected)


def _min_pairwise_color_distance(colors: Sequence[RGB], *, distance_space: str) -> float:
    """Return the minimum pairwise distance in a color subset."""

    if len(colors) < 2:
        return float("inf")
    return min(
        float(color_distance(first, second, distance_space=str(distance_space)))
        for first, second in combinations(tuple(colors), 2)
    )


def sample_density_curve_scene(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    render_params: Any,
) -> DensityCurveSceneSample:
    """Sample the visual density-curve scene before objective binding."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.dataset")
    curve_count, curve_count_probabilities = resolve_curve_count(params, rng=rng)
    x_min = gen_float(params, "density_curve_x_min", 0.0)
    x_max = gen_float(params, "density_curve_x_max", 100.0)
    grid_size = max(81, gen_int(params, "density_curve_x_grid_size", 161))
    xs = tuple(float(x_min) + (float(x_max) - float(x_min)) * (float(index) / float(grid_size - 1)) for index in range(int(grid_size)))
    interval_width_min = gen_float(params, "density_curve_interval_width_min", 18.0)
    interval_width_max = max(interval_width_min, gen_float(params, "density_curve_interval_width_max", 34.0))
    interval_width = float(rng.uniform(float(interval_width_min), float(interval_width_max)))
    interval_start = float(rng.uniform(float(x_min) + 8.0, float(x_max) - float(interval_width) - 8.0))
    interval_end = float(interval_start + interval_width)
    reference_x = float(rng.uniform(float(x_min) + 14.0, float(x_max) - 14.0))
    labels_resolution = resolve_chart_category_labels(
        rng,
        count=int(curve_count),
        min_chars=int(params.get("density_curve_label_min_chars", group_default(GEN_DEFAULTS, "density_curve_label_min_chars", 3))),
        max_chars=int(params.get("density_curve_label_max_chars", group_default(GEN_DEFAULTS, "density_curve_label_max_chars", 10))),
        allow_spaces=bool(params.get("density_curve_label_allow_spaces", group_default(GEN_DEFAULTS, "density_curve_label_allow_spaces", False))),
    )
    labels = tuple(str(label) for label in labels_resolution.labels)
    palette = sample_palette(
        params,
        instance_seed=int(instance_seed),
        count=int(curve_count),
        anchor_colors=(
            tuple(render_params.plot_fill_rgb),
            tuple(render_params.text_color_rgb),
            tuple(render_params.grid_color_rgb),
        ),
    )
    styles_raw = params.get("density_curve_line_style_weights", group_default(GEN_DEFAULTS, "density_curve_line_style_weights", {}))
    style_probabilities = normalize_positive_weights(
        {str(key): float(value) for key, value in styles_raw.items()} if isinstance(styles_raw, Mapping) else {},
        default_keys=SUPPORTED_CURVE_LINE_STYLES,
    )

    curves: list[DensityCurve] = []
    family_probabilities: dict[str, float] | None = None
    y_max = 0.0
    for index, label in enumerate(labels):
        family, probabilities = resolve_density_family(params, rng=rng)
        family_probabilities = dict(probabilities)
        raw_values, component_count, _components = density_values_for_family(rng, family=str(family), xs=xs)
        total = max(1e-9, integral(xs, raw_values))
        ys = tuple(float(value) / total for value in raw_values)
        mean_x, mode_x, mode_y, interval_mass, density_at_x = curve_metrics(
            xs,
            ys,
            interval_start=float(interval_start),
            interval_end=float(interval_end),
            reference_x=float(reference_x),
        )
        y_max = max(float(y_max), max(float(value) for value in ys))
        style = str(weighted_choice(rng, style_probabilities, sort_keys=True))
        curves.append(
            DensityCurve(
                label=str(label),
                color_rgb=tuple(int(channel) for channel in palette[int(index) % len(palette)]),
                line_style=str(style),
                family=str(family),
                component_count=int(component_count),
                points=tuple(CurvePoint(x_value=float(x), y_value=float(y)) for x, y in zip(xs, ys)),
                mean_x=float(mean_x),
                mode_x=float(mode_x),
                mode_y=float(mode_y),
                interval_mass=float(interval_mass),
                density_at_x=float(density_at_x),
            )
        )

    return DensityCurveSceneSample(
        curves=tuple(curves),
        x_min=float(x_min),
        x_max=float(x_max),
        y_max=float(max(1e-9, y_max)),
        curve_count=int(curve_count),
        interval_start=float(interval_start),
        interval_end=float(interval_end),
        reference_x=float(reference_x),
        label_resolution=labels_resolution,
        curve_count_range=(gen_int(params, "density_curve_count_min", 4), gen_int(params, "density_curve_count_max", 6)),
        curve_count_probabilities=dict(curve_count_probabilities),
        density_family_probabilities=dict(family_probabilities or {}),
    )


def bind_density_curve_query(
    sample: DensityCurveSceneSample,
    *,
    prompt_key: str,
    objective: DensityCurveObjectiveSpec,
    answer_label: str,
    winner_gap: float,
) -> DensityCurveDataset:
    """Bind one objective query to a sampled density-curve scene."""

    metric_by_label = {
        "mean_x_by_label": {str(curve.label): round(float(curve.mean_x), 4) for curve in sample.curves},
        "mode_x_by_label": {str(curve.label): round(float(curve.mode_x), 4) for curve in sample.curves},
        "interval_mass_by_label": {str(curve.label): round(float(curve.interval_mass), 6) for curve in sample.curves},
        "density_at_x_by_label": {str(curve.label): round(float(curve.density_at_x), 8) for curve in sample.curves},
        "selected_metric_by_label": {
            str(label): round(float(value), 8)
            for label, value in metric_values_by_label(sample.curves, metric_name=objective.metric_name).items()
        },
    }
    query_trace = {
        "prompt_key": str(prompt_key),
        "metric_name": str(objective.metric_name),
        "direction": str(objective.direction),
        "answer": str(answer_label),
        "annotation_key": str(objective.annotation_key),
        "visible_role": str(objective.visible_role),
        "curve_count": int(sample.curve_count),
        "curve_count_range": [int(sample.curve_count_range[0]), int(sample.curve_count_range[1])],
        "curve_count_probabilities": dict(sample.curve_count_probabilities),
        "density_family_probabilities": dict(sample.density_family_probabilities),
        "winner_gap": round(float(winner_gap), 6),
        "x_range": [round(float(sample.x_min), 3), round(float(sample.x_max), 3)],
        "interval_start": round(float(sample.interval_start), 3),
        "interval_end": round(float(sample.interval_end), 3),
        "interval_label": f"{sample.interval_start:.0f}-{sample.interval_end:.0f}",
        "reference_x": round(float(sample.reference_x), 3),
        "reference_x_label": f"{sample.reference_x:.0f}",
        "labels": [str(curve.label) for curve in sample.curves],
        "families_by_label": {str(curve.label): str(curve.family) for curve in sample.curves},
        "component_count_by_label": {str(curve.label): int(curve.component_count) for curve in sample.curves},
        "line_style_by_label": {str(curve.label): str(curve.line_style) for curve in sample.curves},
        "color_rgb_by_label": {
            str(curve.label): [int(channel) for channel in curve.color_rgb]
            for curve in sample.curves
        },
        "min_curve_pairwise_lab_distance": round(
            float(_min_pairwise_color_distance(tuple(curve.color_rgb for curve in sample.curves), distance_space="lab")),
            3,
        ),
        "label_resolution": {
            key: list(value) if isinstance(value, tuple) else dict(value) if isinstance(value, Mapping) else value
            for key, value in dict(sample.label_resolution.__dict__).items()
        },
        **metric_by_label,
    }
    return DensityCurveDataset(
        curves=tuple(sample.curves),
        query=DensityCurveQuery(
            prompt_key=str(prompt_key),
            answer_label=str(answer_label),
            annotation_key=str(objective.annotation_key),
            visible_role=str(objective.visible_role),
            interval_start=float(sample.interval_start),
            interval_end=float(sample.interval_end),
            reference_x=float(sample.reference_x),
            trace=dict(query_trace),
        ),
        x_min=float(sample.x_min),
        x_max=float(sample.x_max),
        y_max=float(sample.y_max),
        curve_count=int(sample.curve_count),
        label_resolution=sample.label_resolution,
    )
