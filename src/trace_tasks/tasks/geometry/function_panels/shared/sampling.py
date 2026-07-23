"""Sampling and construction primitives for function-panel scenes."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import cycle
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import normalize_positive_weights, weighted_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index

from .defaults import label_pool
from .state import (
    DEFAULT_LABEL_POOL_6,
    DEFAULT_LABEL_POOL_9,
    GRID_MAX,
    GRID_MIN,
    INTERSECTION_CIRCLE_CIRCLE,
    INTERSECTION_LINE_CIRCLE,
    INTERSECTION_TANGENT,
    INTERSECTION_TWO_POINTS,
    RULE_FUNCTION_TEST,
    RULE_ONE_TO_ONE_TEST,
    RULE_RANGE_MATCH,
    RULE_SIGN_INTERVAL,
    RULE_X_AXIS_SYMMETRY,
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
    CircleSpec,
    IntersectionPanelSpec,
    IntersectionSelection,
    Point,
    PropertySelection,
    RelationSpec,
)


def format_number(value: float) -> str:
    """Format compact numeric labels for prompt slots and trace."""

    rounded = round(float(value))
    if abs(float(value) - float(rounded)) <= 1e-9:
        return str(int(rounded))
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def format_interval(interval: Sequence[float]) -> str:
    """Format a closed interval."""

    return f"[{format_number(float(interval[0]))}, {format_number(float(interval[1]))}]"


def resolve_property_selection(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    max_panel_count: int = 6,
) -> PropertySelection:
    """Resolve visible labels, selected label, and panel count."""

    pool = label_pool(params, max_count=max_panel_count, gen_defaults=gen_defaults)
    supported_counts = tuple(count for count in (4, 6) if int(count) <= len(pool))
    panel_count, panel_count_probs = resolve_geometry_option_count(
        params=params,
        gen_defaults=gen_defaults,
        field_name="panel_count",
        supported_counts=supported_counts,
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    visible = _visible_labels(pool, panel_count=int(panel_count))
    selected_label, selected_probs = _resolve_selected_label(
        params,
        instance_seed=int(instance_seed),
        label_pool=visible,
        namespace=f"{namespace}.selected_label",
        gen_defaults=gen_defaults,
    )
    return PropertySelection(
        selected_label=str(selected_label),
        label_pool=tuple(visible),
        label_probabilities=dict(selected_probs),
        panel_count_probabilities=dict(panel_count_probs),
    )


def resolve_intersection_selection(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> IntersectionSelection:
    """Resolve visible labels and panel count for intersection panels."""

    pool = label_pool(params, max_count=9, gen_defaults=gen_defaults)
    supported_counts = tuple(count for count in (4, 6, 9) if int(count) <= len(pool))
    panel_count, panel_count_probs = resolve_geometry_option_count(
        params=params,
        gen_defaults=gen_defaults,
        field_name="panel_count",
        supported_counts=supported_counts,
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    visible = _visible_labels(pool, panel_count=int(panel_count))
    selected_label, selected_probs = _resolve_selected_label(
        params,
        instance_seed=int(instance_seed),
        label_pool=visible,
        namespace=f"{namespace}.selected_label",
        gen_defaults=gen_defaults,
    )
    return IntersectionSelection(
        selected_label=str(selected_label),
        label_pool=tuple(visible),
        label_probabilities=dict(selected_probs),
        panel_count_probabilities=dict(panel_count_probs),
    )


def _resolve_selected_label(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    label_pool: Sequence[str],
    namespace: str,
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    label_set = tuple(str(label).upper() for label in label_pool)
    explicit = params.get("winner_label", params.get("answer_label"))
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in set(label_set):
            raise ValueError(f"unsupported selected label: {selected}")
        return selected, {label: (1.0 if label == selected else 0.0) for label in label_set}

    raw_weights = params.get("winner_label_weights", {label: 1.0 for label in label_set})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("winner_label_weights must be a mapping when provided")
    probabilities = normalize_positive_weights(
        {str(key).upper(): float(value) for key, value in raw_weights.items() if str(key).upper() in set(label_set)},
        default_keys=label_set,
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = str(weighted_choice(rng, probabilities, sort_keys=True)).upper()

    return selected, {str(key): float(value) for key, value in sorted(probabilities.items())}


def _visible_labels(label_pool: Sequence[str], *, panel_count: int) -> tuple[str, ...]:
    return tuple(str(label) for label in label_pool[: int(panel_count)])


def selected_interval(rule_kind: str, *, sign_kind: str | None = None) -> tuple[float, float] | None:
    """Return the prompt-facing interval for interval tasks."""

    if str(rule_kind) == RULE_SIGN_INTERVAL and str(sign_kind or "") in {SIGN_POSITIVE, SIGN_NEGATIVE}:
        return (-4.0, 4.0)
    return None


def build_property_relations(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selection: PropertySelection,
    rule_kind: str,
    sign_kind: str | None,
    namespace: str,
) -> dict[str, RelationSpec]:
    """Build one uniquely answered set of candidate relation panels."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.relations")
    target_interval = selected_interval(rule_kind, sign_kind=sign_kind)
    selection = replace(selection, target_interval=target_interval)
    for _ in range(100):
        if str(rule_kind) == RULE_FUNCTION_TEST:
            winner = _sample_function_winner(rng, _template_index(params, instance_seed=instance_seed, namespace=f"{namespace}.template", support_size=10))
            distractors = _sample_function_distractors(rng)
        elif str(rule_kind) == RULE_ONE_TO_ONE_TEST:
            winner = _sample_one_to_one_winner(rng, _template_index(params, instance_seed=instance_seed, namespace=f"{namespace}.template", support_size=9))
            distractors = _sample_one_to_one_distractors(rng)
        elif str(rule_kind) == RULE_RANGE_MATCH:
            bank = _sample_range_bank(rng)
            winner = bank[_template_index(params, instance_seed=instance_seed, namespace=f"{namespace}.template", support_size=len(bank))]
            distractors = [relation for relation in bank if relation.relation_id != winner.relation_id]
        elif str(rule_kind) == RULE_X_AXIS_SYMMETRY:
            winner = _sample_x_axis_symmetry_winner(
                rng,
                _template_index(params, instance_seed=instance_seed, namespace=f"{namespace}.template", support_size=4),
            )
            distractors = _sample_x_axis_symmetry_distractors(rng)
        elif str(rule_kind) == RULE_SIGN_INTERVAL:
            if target_interval is None:
                raise ValueError("sign interval requires a target interval")
            positive = str(sign_kind) == SIGN_POSITIVE
            winner = _sample_interval_sign_winner(
                rng,
                _template_index(params, instance_seed=instance_seed, namespace=f"{namespace}.template", support_size=4),
                interval=target_interval,
                positive=positive,
            )
            distractors = _sample_sign_distractors(rng, target_interval=target_interval, positive=positive)
        else:
            raise ValueError(f"unsupported panel rule: {rule_kind}")
        rng.shuffle(distractors)
        relations = _assign_relations(selection, winner=winner, distractors=distractors)
        if _matches_property(selection.selected_label, relations, rule_kind=rule_kind, sign_kind=sign_kind, interval=target_interval) == [selection.selected_label]:
            return relations
    raise RuntimeError("failed to construct a unique function-panel answer")


def _assign_relations(selection: PropertySelection, *, winner: RelationSpec, distractors: Sequence[RelationSpec]) -> dict[str, RelationSpec]:
    result: dict[str, RelationSpec] = {}
    stream = iter(distractors)
    for label in selection.label_pool:
        result[str(label)] = winner if str(label) == str(selection.selected_label) else next(stream)
    return result


def _template_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str, support_size: int) -> int:
    index = resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=str(namespace))
    return int(index) % max(1, int(support_size))


def _line(relation_id: str, start: Point, end: Point, *, one_to_one: bool = True) -> RelationSpec:
    domain = (min(float(start[0]), float(end[0])), max(float(start[0]), float(end[0])))
    range_ = (min(float(start[1]), float(end[1])), max(float(start[1]), float(end[1])))
    is_function = abs(float(start[0]) - float(end[0])) > 1e-9
    return RelationSpec(
        relation_id=str(relation_id),
        draw_kind="polyline",
        points=(start, end),
        domain=domain,
        range=range_,
        is_function=bool(is_function),
        is_one_to_one=bool(is_function and one_to_one and abs(float(start[1]) - float(end[1])) > 1e-9),
    )


def _polyline(relation_id: str, points: Sequence[Point], *, is_function: bool, is_one_to_one: bool) -> RelationSpec:
    domain = (min(float(point[0]) for point in points), max(float(point[0]) for point in points))
    range_ = (min(float(point[1]) for point in points), max(float(point[1]) for point in points))
    return RelationSpec(
        relation_id=str(relation_id),
        draw_kind="polyline",
        points=tuple((float(x), float(y)) for x, y in points),
        domain=domain,
        range=range_,
        is_function=bool(is_function),
        is_one_to_one=bool(is_one_to_one),
    )


def _ellipse(relation_id: str, center: Point, radii: Point) -> RelationSpec:
    cx, cy = float(center[0]), float(center[1])
    rx, ry = abs(float(radii[0])), abs(float(radii[1]))
    return RelationSpec(
        relation_id=str(relation_id),
        draw_kind="ellipse",
        center=(cx, cy),
        radii=(rx, ry),
        domain=(cx - rx, cx + rx),
        range=(cy - ry, cy + ry),
        is_function=False,
        is_one_to_one=False,
    )


def _tag_symmetry(relation: RelationSpec, *axes: str) -> RelationSpec:
    ordered = tuple(axis for axis in ("x_axis", "y_axis", "origin") if axis in {str(value) for value in axes})
    return replace(relation, symmetry_axes=ordered)


def _parabola_points(*, orientation: str, offset: Point = (0.0, 0.0), scale: float = 0.22, sample_count: int = 7) -> tuple[Point, ...]:
    count = max(2, int(sample_count))
    values = tuple(-4.0 + (8.0 * float(index) / float(count - 1)) for index in range(count))
    ox, oy = float(offset[0]), float(offset[1])
    if str(orientation) == "vertical_up":
        return tuple((x + ox, (float(scale) * x * x) - 3.0 + oy) for x in values)
    if str(orientation) == "vertical_down":
        return tuple((x + ox, 3.0 - (float(scale) * x * x) + oy) for x in values)
    if str(orientation) == "sideways_right":
        return tuple(((float(scale) * y * y) - 3.0 + ox, y + oy) for y in values)
    if str(orientation) == "sideways_left":
        return tuple((3.0 - (float(scale) * y * y) + ox, y + oy) for y in values)
    raise ValueError(f"unsupported parabola orientation: {orientation}")


def _parabola(relation_id: str, *, orientation: str, offset: Point = (0.0, 0.0), scale: float = 0.22) -> RelationSpec:
    is_function = str(orientation) in {"vertical_up", "vertical_down"}
    return _polyline(
        str(relation_id),
        _parabola_points(orientation=str(orientation), offset=offset, scale=float(scale)),
        is_function=is_function,
        is_one_to_one=False,
    )


def _mirrored_zigzag(relation_id: str, points_upper: Sequence[Point]) -> RelationSpec:
    upper = tuple((float(x), abs(float(y))) for x, y in points_upper)
    lower = tuple((float(x), -abs(float(y))) for x, y in reversed(points_upper))
    return _tag_symmetry(
        _polyline(str(relation_id), (*upper, *lower), is_function=False, is_one_to_one=False),
        "x_axis",
    )


def _shifted_mirrored_zigzag(relation_id: str, points_upper: Sequence[Point], *, y_offset: float) -> RelationSpec:
    base = _mirrored_zigzag(str(relation_id), points_upper)
    shifted = tuple((float(x), float(y) + float(y_offset)) for x, y in base.points)
    return _polyline(str(relation_id), shifted, is_function=False, is_one_to_one=False)


def _monotone_zigzag(relation_id: str, *, ascending: bool = True) -> RelationSpec:
    if bool(ascending):
        points = ((-4.0, -3.6), (-2.4, -2.2), (-1.2, -1.4), (0.5, 0.2), (2.0, 1.4), (4.0, 3.4))
    else:
        points = ((-4.0, 3.4), (-2.0, 1.5), (-0.5, 0.3), (1.2, -1.2), (2.6, -2.1), (4.0, -3.5))
    return _polyline(str(relation_id), points, is_function=True, is_one_to_one=True)


def _axis_symmetric_zigzag(relation_id: str, *, axis: str) -> RelationSpec:
    if str(axis) == "y_axis":
        points = ((-3.6, 2.8), (-2.0, -1.4), (-0.6, 1.5), (0.6, 1.5), (2.0, -1.4), (3.6, 2.8))
    elif str(axis) == "origin":
        points = ((-4.0, -3.0), (-2.2, -1.4), (-0.5, -0.4), (0.5, 0.4), (2.2, 1.4), (4.0, 3.0))
    else:
        raise ValueError(f"unsupported zigzag symmetry axis: {axis}")
    return _tag_symmetry(_polyline(str(relation_id), points, is_function=True, is_one_to_one=False), str(axis))


def _relation_shape_family(relation: RelationSpec) -> str:
    relation_id = str(relation.relation_id)
    if "parabola" in relation_id:
        return "parabola"
    if "zigzag" in relation_id:
        return "zigzag"
    if relation.draw_kind == "ellipse":
        return "ellipse"
    if "line" in relation_id or len(relation.points) == 2:
        return "line"
    return "polyline"


def _sample_function_winner(rng, template_index: int) -> RelationSpec:
    if int(template_index) == 7:
        return _parabola("target_vertical_parabola_function", orientation=str(rng.choice(("vertical_up", "vertical_down"))))
    if int(template_index) == 8:
        return _polyline("target_u_zigzag_function", ((-4.0, 2.7), (-2.2, -0.8), (0.0, -2.5), (2.2, -0.8), (4.0, 2.7)), is_function=True, is_one_to_one=False)
    if int(template_index) == 9:
        return _monotone_zigzag("target_monotone_zigzag_function", ascending=bool(rng.randrange(2)))

    endpoints = (
        ((-4.0, -3.0), (4.0, 3.0)),
        ((-4.0, 3.0), (4.0, -3.0)),
        ((-4.0, -2.0), (-1.0, 2.0), (2.0, -1.0), (4.0, 3.0)),
        ((-4.0, 1.0), (-2.0, -2.0), (0.0, -3.0), (2.0, -2.0), (4.0, 1.0)),
        ((-4.0, -3.5), (-2.0, -2.0), (0.0, -0.5), (2.0, 1.4), (4.0, 3.4)),
        ((-4.0, 3.3), (-2.0, 1.5), (0.0, 0.0), (2.0, -1.6), (4.0, -3.4)),
        ((-4.0, 0.0), (-2.0, 2.0), (0.0, 0.0), (2.0, -2.0), (4.0, 0.0)),
    )
    if int(template_index) < 0 or int(template_index) >= len(endpoints):
        template_index = int(rng.randrange(len(endpoints)))
    points = tuple(
        (float(x), float(y + rng.choice((-0.4, 0.0, 0.4))))
        for x, y in endpoints[int(template_index)]
    )
    return _polyline(
        "target_function_relation",
        points,
        is_function=True,
        is_one_to_one=int(template_index) in {0, 1, 4, 5},
    )


def _sample_one_to_one_winner(rng, template_index: int) -> RelationSpec:
    if int(template_index) in {6, 7}:
        return _monotone_zigzag("target_monotone_zigzag_one_to_one", ascending=int(template_index) == 6)
    if int(template_index) == 8:
        return _polyline("target_piecewise_linear_one_to_one", ((-4.0, -3.4), (-2.5, -2.5), (-1.0, -0.8), (1.2, 0.6), (2.8, 2.1), (4.0, 3.2)), is_function=True, is_one_to_one=True)

    x0 = float(rng.choice((-4.0, -3.5, -3.0)))
    x1 = float(rng.choice((3.0, 3.5, 4.0)))
    ascending = int(template_index) % 2 == 0
    if ascending:
        points = ((x0, float(rng.choice((-4.0, -3.0, -2.0)))), (-1.0, -0.5), (1.0, 1.2), (x1, float(rng.choice((2.6, 3.3, 4.0)))))
    else:
        points = ((x0, float(rng.choice((2.6, 3.3, 4.0)))), (-1.0, 0.8), (1.0, -1.2), (x1, float(rng.choice((-4.0, -3.0, -2.0)))))
    return _polyline("target_injective_relation", points, is_function=True, is_one_to_one=True)


def _sample_function_distractors(rng) -> list[RelationSpec]:
    return [
        _ellipse("closed_curve_not_function", (0.0, 0.0), (2.6, 1.8)),
        _line("vertical_segment_not_function", (float(rng.choice((-2, -1, 1, 2))), -4.0), (float(rng.choice((-2, -1, 1, 2))), 4.0), one_to_one=False),
        _parabola("sideways_parabola_not_function", orientation=str(rng.choice(("sideways_right", "sideways_left")))),
        _polyline("sideways_v_not_function", ((3.0, -4.0), (-1.0, 0.0), (3.0, 4.0)), is_function=False, is_one_to_one=False),
        _ellipse("offset_ellipse_not_function", (1.0, -0.5), (2.2, 2.8)),
        _mirrored_zigzag("mirrored_zigzag_not_function", ((-3.8, 2.8), (-2.0, 1.1), (-0.5, 2.4), (1.4, 0.9), (3.7, 2.9))),
    ]


def _sample_one_to_one_distractors(rng) -> list[RelationSpec]:
    return [
        _polyline("v_shape_not_injective", ((-4.0, 3.0), (0.0, -3.0), (4.0, 3.0)), is_function=True, is_one_to_one=False),
        _parabola("vertical_parabola_not_injective", orientation=str(rng.choice(("vertical_up", "vertical_down")))),
        _polyline("flat_segment_not_injective", ((-4.0, 1.0), (4.0, 1.0)), is_function=True, is_one_to_one=False),
        _polyline("turning_relation_not_injective", ((-4.0, -2.0), (-1.0, 3.0), (2.0, -1.0), (4.0, 2.0)), is_function=True, is_one_to_one=False),
        _parabola("sideways_parabola_not_function_for_one_to_one", orientation=str(rng.choice(("sideways_right", "sideways_left")))),
        _polyline("wide_u_not_injective", ((-4.0, 2.0), (-2.0, -2.0), (0.0, -3.0), (2.0, -2.0), (4.0, 2.0)), is_function=True, is_one_to_one=False),
    ]


def _sample_range_bank(rng) -> tuple[RelationSpec, ...]:
    candidates = (
        _line("range_line_a", (-4.0, -4.0), (4.0, 1.0)),
        _line("range_line_b", (-4.0, -2.0), (4.0, 4.0)),
        _polyline("range_polyline_c", ((-4.0, 3.0), (-1.0, -3.0), (4.0, 2.0)), is_function=True, is_one_to_one=False),
        _parabola("range_vertical_parabola_d", orientation="vertical_up", scale=0.18),
        _mirrored_zigzag("range_mirrored_zigzag_e", ((-3.8, 2.7), (-2.0, 1.0), (0.0, 2.4), (1.8, 1.1), (3.8, 2.9))),
        _ellipse("range_ellipse_f", (1.0, -0.5), (2.5, 2.2)),
    )
    jitter = float(rng.choice((-0.3, 0.0, 0.3)))
    return tuple(_shift_range(relation, jitter=float(index % 3) * 0.1 + jitter) for index, relation in enumerate(candidates))


def _shift_range(relation: RelationSpec, *, jitter: float) -> RelationSpec:
    if relation.points:
        shifted = tuple((float(x), max(float(GRID_MIN), min(float(GRID_MAX), float(y) + float(jitter)))) for x, y in relation.points)
        return replace(relation, points=shifted, range=(min(y for _x, y in shifted), max(y for _x, y in shifted)))
    if relation.center is not None:
        cy = max(float(GRID_MIN) + 2.5, min(float(GRID_MAX) - 2.5, float(relation.center[1]) + float(jitter)))
        if relation.radii is None:
            return relation
        return replace(relation, center=(float(relation.center[0]), cy), range=(cy - float(relation.radii[1]), cy + float(relation.radii[1])))
    return relation


def _sample_x_axis_symmetry_winner(rng, template_index: int) -> RelationSpec:
    templates = (
        lambda: _tag_symmetry(_ellipse("target_x_axis_symmetric_ellipse", (0.0, 0.0), (float(rng.choice((2.6, 3.0, 3.4))), float(rng.choice((1.2, 1.6, 2.0))))), "x_axis"),
        lambda: _tag_symmetry(_parabola("target_x_axis_sideways_parabola", orientation=str(rng.choice(("sideways_right", "sideways_left")))), "x_axis"),
        lambda: _mirrored_zigzag("target_x_axis_mirrored_zigzag", ((-3.8, 2.7), (-2.0, 1.0), (-0.2, 2.4), (1.6, 1.1), (3.8, 2.9))),
        lambda: _mirrored_zigzag("target_x_axis_angular_curve", ((-3.5, 1.4), (-2.1, 3.0), (-0.2, 1.0), (1.7, 2.5), (3.6, 1.5))),
    )
    if int(template_index) < 0 or int(template_index) >= len(templates):
        template_index = int(rng.randrange(len(templates)))
    return templates[int(template_index)]()


def _sample_x_axis_symmetry_distractors(rng) -> list[RelationSpec]:
    return [
        _tag_symmetry(_ellipse("y_axis_symmetric_distractor", (0.0, 1.2), (1.4, 2.2)), "y_axis"),
        _tag_symmetry(_parabola("y_axis_parabola_distractor", orientation=str(rng.choice(("vertical_up", "vertical_down")))), "y_axis"),
        _axis_symmetric_zigzag("origin_symmetric_zigzag_distractor", axis="origin"),
        _polyline("asymmetric_curve_a", ((-4.0, -2.0), (-1.0, 2.6), (2.0, -0.7), (4.0, 1.5)), is_function=True, is_one_to_one=False),
        _shifted_mirrored_zigzag("shifted_mirrored_zigzag_distractor", ((-3.6, 2.4), (-1.8, 0.9), (0.0, 2.2), (1.7, 1.0), (3.6, 2.5)), y_offset=float(rng.choice((-0.9, 0.9)))),
    ]


def _interval_sign_relation(rng, relation_id: str, *, interval: tuple[float, float], positive: bool) -> RelationSpec:
    x0, x3 = float(interval[0]), float(interval[1])
    x1 = x0 + ((x3 - x0) / 3.0)
    x2 = x0 + (2.0 * (x3 - x0) / 3.0)
    candidates = (1.2, 2.0, 2.8, 3.6) if bool(positive) else (-1.2, -2.0, -2.8, -3.6)
    values = [float(value) for value in rng.sample(list(candidates), 4)]
    return _polyline(str(relation_id), ((x0, values[0]), (x1, values[1]), (x2, values[2]), (x3, values[3])), is_function=True, is_one_to_one=False)


def _sample_interval_sign_winner(rng, template_index: int, *, interval: tuple[float, float], positive: bool) -> RelationSpec:
    if int(template_index) == 1:
        sign = 1.0 if bool(positive) else -1.0
        points = tuple((x, sign * y) for x, y in ((-4.0, 1.6), (-2.0, 3.4), (0.0, 2.1), (2.0, 3.0), (4.0, 1.4)))
        return _polyline("target_interval_sign_zigzag_relation", points, is_function=True, is_one_to_one=False)
    if int(template_index) == 2:
        sign = 1.0 if bool(positive) else -1.0
        points = tuple((x, sign * y) for x, y in ((-4.0, 3.5), (-2.0, 1.8), (0.0, 1.2), (2.0, 1.8), (4.0, 3.5)))
        return _polyline("target_interval_sign_parabola_like_relation", points, is_function=True, is_one_to_one=False)
    if int(template_index) == 3:
        sign = 1.0 if bool(positive) else -1.0
        points = tuple((x, sign * y) for x, y in ((-4.0, 1.2), (-2.6, 1.9), (-1.0, 2.7), (1.0, 2.0), (2.8, 3.2), (4.0, 2.4)))
        return _polyline("target_interval_sign_monotone_broken_relation", points, is_function=True, is_one_to_one=False)
    return _interval_sign_relation(rng, "target_interval_sign_relation", interval=interval, positive=positive)


def _sample_sign_distractors(rng, *, target_interval: tuple[float, float], positive: bool) -> list[RelationSpec]:
    return [
        _sample_interval_sign_winner(rng, 1, interval=target_interval, positive=not bool(positive)),
        _interval_sign_relation(rng, "opposite_sign_relation_b", interval=target_interval, positive=not bool(positive)),
        _polyline("sign_crossing_relation_a", ((-4.0, 3.0), (-1.0, 1.0), (1.0, -1.0), (4.0, -3.0)), is_function=True, is_one_to_one=True),
        _polyline("sign_crossing_relation_b", ((-4.0, -3.0), (-1.0, -1.0), (1.0, 1.0), (4.0, 3.0)), is_function=True, is_one_to_one=True),
        _polyline("sign_mixed_relation", ((-4.0, -2.0), (-2.0, 2.0), (0.0, -1.5), (2.0, 1.5), (4.0, -2.0)), is_function=True, is_one_to_one=False),
    ]


def _points_over_interval(relation: RelationSpec, interval: tuple[float, float]) -> list[Point]:
    lo, hi = min(float(interval[0]), float(interval[1])), max(float(interval[0]), float(interval[1]))
    return sorted((point for point in relation_points(relation) if lo - 1e-6 <= float(point[0]) <= hi + 1e-6), key=lambda point: (float(point[0]), float(point[1])))


def _matches_property(
    selected_label: str,
    relations_by_label: Mapping[str, RelationSpec],
    *,
    rule_kind: str,
    sign_kind: str | None,
    interval: tuple[float, float] | None,
) -> list[str]:
    if str(rule_kind) == RULE_FUNCTION_TEST:
        return [label for label, relation in relations_by_label.items() if bool(relation.is_function)]
    if str(rule_kind) == RULE_ONE_TO_ONE_TEST:
        return [label for label, relation in relations_by_label.items() if bool(relation.is_function and relation.is_one_to_one)]
    if str(rule_kind) == RULE_RANGE_MATCH:
        target = relations_by_label[str(selected_label)].range
        return [label for label, relation in relations_by_label.items() if tuple(relation.range) == tuple(target)]
    if str(rule_kind) == RULE_X_AXIS_SYMMETRY:
        return [label for label, relation in relations_by_label.items() if "x_axis" in set(relation.symmetry_axes)]
    if str(rule_kind) == RULE_SIGN_INTERVAL:
        if interval is None:
            return []
        positive = str(sign_kind) == SIGN_POSITIVE
        return [
            label
            for label, relation in relations_by_label.items()
            if bool(relation.is_function)
            and bool(_points_over_interval(relation, interval))
            and all((float(point[1]) > 1e-6 if positive else float(point[1]) < -1e-6) for point in _points_over_interval(relation, interval))
        ]
    return []


def relation_points(relation: RelationSpec) -> list[Point]:
    """Return sample points used for rendering and trace."""

    if relation.points:
        return list(relation.points)
    return []


def relation_trace_payload(relation: RelationSpec) -> dict[str, Any]:
    """Serialize one relation for trace metadata."""

    payload: dict[str, Any] = {
        "relation_id": str(relation.relation_id),
        "shape_family": _relation_shape_family(relation),
        "draw_kind": str(relation.draw_kind),
        "domain": [float(relation.domain[0]), float(relation.domain[1])],
        "range": [float(relation.range[0]), float(relation.range[1])],
        "domain_text": format_interval(relation.domain),
        "range_text": format_interval(relation.range),
        "is_function": bool(relation.is_function),
        "is_one_to_one": bool(relation.is_one_to_one),
        "symmetry_axes": [str(axis) for axis in relation.symmetry_axes],
        "symmetric_about_x_axis": "x_axis" in set(relation.symmetry_axes),
        "symmetric_about_y_axis": "y_axis" in set(relation.symmetry_axes),
        "symmetric_about_origin": "origin" in set(relation.symmetry_axes),
    }
    if relation.points:
        payload["points"] = [[float(x), float(y)] for x, y in relation.points]
    if relation.center is not None:
        payload["center"] = [float(relation.center[0]), float(relation.center[1])]
    if relation.radii is not None:
        payload["radii"] = [float(relation.radii[0]), float(relation.radii[1])]
    return payload


def build_intersection_panels(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selection: IntersectionSelection,
    pair_kind: str,
    relation_class: str,
    namespace: str,
) -> dict[str, IntersectionPanelSpec]:
    """Build one uniquely answered set of primitive-pair panels."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.pairs")
    for _ in range(100):
        if str(pair_kind) == INTERSECTION_LINE_CIRCLE and str(relation_class) == INTERSECTION_TANGENT:
            winner = _line_circle_tangent_panel()
            distractors = [_line_circle_secant_panel(), _line_circle_disjoint_panel(), _circle_circle_two_panel(), _circle_circle_tangent_panel(), _circle_circle_disjoint_panel()]
        elif str(pair_kind) == INTERSECTION_LINE_CIRCLE and str(relation_class) == INTERSECTION_TWO_POINTS:
            winner = _line_circle_secant_panel()
            distractors = [_line_circle_tangent_panel(), _line_circle_disjoint_panel(), _circle_circle_two_panel(), _circle_circle_tangent_panel(), _circle_circle_disjoint_panel()]
        elif str(pair_kind) == INTERSECTION_CIRCLE_CIRCLE and str(relation_class) == INTERSECTION_TWO_POINTS:
            winner = _circle_circle_two_panel()
            distractors = [_line_circle_tangent_panel(), _line_circle_secant_panel(), _line_circle_disjoint_panel(), _circle_circle_tangent_panel(), _circle_circle_disjoint_panel()]
        else:
            raise ValueError("unsupported intersection panel target")
        rng.shuffle(distractors)
        panels = _assign_panels(selection, winner=winner, distractors=distractors)
        if _matches_intersection(selection.selected_label, panels, pair_kind=pair_kind, relation_class=relation_class) == [selection.selected_label]:
            return panels
    raise RuntimeError("failed to construct a unique intersection-panel answer")


def _assign_panels(selection: IntersectionSelection, *, winner: IntersectionPanelSpec, distractors: Sequence[IntersectionPanelSpec]) -> dict[str, IntersectionPanelSpec]:
    result: dict[str, IntersectionPanelSpec] = {}
    distractor_list = tuple(distractors)
    if not distractor_list:
        raise ValueError("at least one intersection distractor is required")
    distractors_by_label = cycle(distractor_list)
    for label in selection.label_pool:
        if str(label) == str(selection.selected_label):
            result[str(label)] = winner
            continue
        result[str(label)] = next(distractors_by_label)
    return result


def _matches_intersection(
    selected_label: str,
    panels_by_label: Mapping[str, IntersectionPanelSpec],
    *,
    pair_kind: str,
    relation_class: str,
) -> list[str]:
    if str(pair_kind) == INTERSECTION_LINE_CIRCLE and str(relation_class) == INTERSECTION_TANGENT:
        return [label for label, panel in panels_by_label.items() if panel.pair_kind == INTERSECTION_LINE_CIRCLE and panel.relation_class == INTERSECTION_TANGENT]
    if str(pair_kind) == INTERSECTION_LINE_CIRCLE and str(relation_class) == INTERSECTION_TWO_POINTS:
        return [label for label, panel in panels_by_label.items() if panel.pair_kind == INTERSECTION_LINE_CIRCLE and len(panel.intersection_points) == 2]
    if str(pair_kind) == INTERSECTION_CIRCLE_CIRCLE and str(relation_class) == INTERSECTION_TWO_POINTS:
        return [label for label, panel in panels_by_label.items() if panel.pair_kind == INTERSECTION_CIRCLE_CIRCLE and len(panel.intersection_points) == 2]
    return []


def _quadrants(points: Sequence[Point]) -> tuple[str, ...]:
    result = []
    for x, y in points:
        if float(x) >= 0 and float(y) >= 0:
            result.append("QI")
        elif float(x) < 0 <= float(y):
            result.append("QII")
        elif float(x) < 0 and float(y) < 0:
            result.append("QIII")
        else:
            result.append("QIV")
    return tuple(result)


def _line_circle_tangent_panel() -> IntersectionPanelSpec:
    circle = ((0.0, 0.0), 2.4)
    point = (0.0, 2.4)
    return IntersectionPanelSpec("line_circle_tangent", INTERSECTION_LINE_CIRCLE, INTERSECTION_TANGENT, (((-4.0, 2.4), (4.0, 2.4)),), (circle,), (point,), _quadrants((point,)))


def _line_circle_secant_panel() -> IntersectionPanelSpec:
    circle = ((0.0, 0.0), 2.5)
    points = ((-2.5, 0.0), (2.5, 0.0))
    return IntersectionPanelSpec("line_circle_two_points", INTERSECTION_LINE_CIRCLE, INTERSECTION_TWO_POINTS, (((-4.0, 0.0), (4.0, 0.0)),), (circle,), points, _quadrants(points))


def _line_circle_disjoint_panel() -> IntersectionPanelSpec:
    circle = ((0.0, 0.0), 2.0)
    return IntersectionPanelSpec("line_circle_disjoint", INTERSECTION_LINE_CIRCLE, "disjoint", (((-4.0, 3.4), (4.0, 3.4)),), (circle,), (), ())


def _circle_circle_two_panel() -> IntersectionPanelSpec:
    left = ((-1.2, 0.0), 2.3)
    right = ((1.2, 0.0), 2.3)
    y = math.sqrt((2.3 * 2.3) - (1.2 * 1.2))
    points = ((0.0, -float(y)), (0.0, float(y)))
    return IntersectionPanelSpec("circle_circle_two_points", INTERSECTION_CIRCLE_CIRCLE, INTERSECTION_TWO_POINTS, (), (left, right), points, _quadrants(points))


def _circle_circle_tangent_panel() -> IntersectionPanelSpec:
    left = ((-1.6, 0.0), 1.6)
    right = ((1.6, 0.0), 1.6)
    point = (0.0, 0.0)
    return IntersectionPanelSpec("circle_circle_tangent", INTERSECTION_CIRCLE_CIRCLE, INTERSECTION_TANGENT, (), (left, right), (point,), _quadrants((point,)))


def _circle_circle_disjoint_panel() -> IntersectionPanelSpec:
    return IntersectionPanelSpec("circle_circle_disjoint", INTERSECTION_CIRCLE_CIRCLE, "disjoint", (), (((-2.5, 0.0), 1.1), ((2.5, 0.0), 1.1)), (), ())


def panel_trace_payload(panel: IntersectionPanelSpec) -> dict[str, Any]:
    """Serialize one intersection panel for trace metadata."""

    return {
        "pair_id": str(panel.pair_id),
        "pair_kind": str(panel.pair_kind),
        "relation_class": str(panel.relation_class),
        "intersection_count": int(len(panel.intersection_points)),
        "intersection_points": [[float(x), float(y)] for x, y in panel.intersection_points],
        "intersection_quadrants": [str(value) for value in panel.intersection_quadrants],
        "line_segments": [
            [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]]
            for start, end in panel.line_segments
        ],
        "circles": [
            {"center": [float(center[0]), float(center[1])], "radius": float(radius)}
            for center, radius in panel.circles
        ],
    }
