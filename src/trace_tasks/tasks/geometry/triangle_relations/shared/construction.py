"""Construction pools for triangle-relations analytical diagrams."""

from __future__ import annotations

import math
from functools import cache
from typing import Any, Callable, Iterable, Mapping

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1
from trace_tasks.tasks.geometry.shared.vector2d import mid

from .state import (
    AngleLabel,
    RightAngleMark,
    SegmentLabel,
    TickGroup,
    TriangleRelationsCase,
)

DEGREE_SYMBOL = chr(176)


def _answer_key(value: int | float) -> str:
    return str(fmt_measure(float(value)))


def _trim_answer_variants(
    cases: Iterable[TriangleRelationsCase],
    *,
    max_answers: int = 240,
    max_variants_per_answer: int = 3,
) -> tuple[TriangleRelationsCase, ...]:
    """Keep a broad answer support without carrying every equivalent variant."""

    by_answer: dict[str, list[TriangleRelationsCase]] = {}
    for case in cases:
        if not _case_has_readable_labeled_geometry(case):
            continue
        if not _case_has_no_visible_answer_collision(case):
            continue
        bucket = by_answer.setdefault(_answer_key(case.answer), [])
        if len(bucket) < int(max_variants_per_answer):
            bucket.append(case)

    answer_keys = sorted(by_answer, key=lambda key: (float(key), key))
    if len(answer_keys) > int(max_answers):
        step = (len(answer_keys) - 1) / float(int(max_answers) - 1)
        selected_keys: list[str] = []
        selected: set[str] = set()
        for index in range(int(max_answers)):
            key = answer_keys[round(index * step)]
            if key in selected:
                continue
            selected.add(key)
            selected_keys.append(key)
        answer_keys = selected_keys
    trimmed: list[TriangleRelationsCase] = []
    for key in answer_keys:
        trimmed.extend(by_answer[key])
    return tuple(trimmed)


def _is_valid_triangle(a: float, b: float, c: float) -> bool:
    return (a + b > c) and (a + c > b) and (b + c > a)


def _segment_length(
    vertices: Mapping[str, tuple[float, float]], segment: tuple[str, str]
) -> float:
    start = vertices[str(segment[0])]
    end = vertices[str(segment[1])]
    return math.hypot(float(start[0]) - float(end[0]), float(start[1]) - float(end[1]))


def _case_has_readable_labeled_geometry(case: TriangleRelationsCase) -> bool:
    """Reject constructions where faithful proportions make labeled support unreadable."""

    if any(
        _segment_length(case.vertices, label.segment) < 52.0
        for label in case.segment_labels
    ):
        return False
    if (
        str(case.formula_family).startswith("angle_bisector")
        and _angle_bisector_split_fraction(case) < 0.2
    ):
        return False
    return True


def _case_has_no_visible_answer_collision(case: TriangleRelationsCase) -> bool:
    """Reject cases where a non-target visible label directly equals the answer."""

    answer_key = _answer_key(case.answer)
    target_segment = (
        tuple(case.target_segment) if case.target_segment is not None else None
    )
    for label in case.segment_labels:
        if target_segment is not None and tuple(label.segment) == target_segment:
            continue
        if _visible_numeric_label_key(label.text) == answer_key:
            return False
    for label in case.angle_labels:
        if _visible_numeric_label_key(label.text) == answer_key:
            return False
    return True


def _visible_numeric_label_key(text: object) -> str | None:
    """Return the normalized value of a plain visible numeric label, if any."""

    raw = str(text).strip().replace(DEGREE_SYMBOL, "")
    if not raw:
        return None
    if any(token in raw for token in ("?", "x", "θ", "=")):
        return None
    try:
        return _answer_key(float(raw))
    except ValueError:
        return None


def _angle_bisector_split_fraction(case: TriangleRelationsCase) -> float:
    values = dict(case.trace_values)
    bd = values.get("BD")
    dc = values.get("DC", values.get("derived_DC"))
    if dc is None and "DC_expression" in values:
        dc = _expression_value(values["DC_expression"], values.get("x"))
    if bd is None or dc is None:
        return 1.0
    base = float(bd) + float(dc)
    if base <= 0.0:
        return 0.0
    return min(float(bd), float(dc)) / base


def _expression_value(expression: object, x_value: object) -> int | None:
    text = str(expression).strip()
    try:
        x_int = int(x_value)
    except (TypeError, ValueError):
        return None
    if text == "x":
        return x_int
    if text.startswith("x+"):
        try:
            return x_int + int(text[2:])
        except ValueError:
            return None
    if text.startswith("x-"):
        try:
            return x_int - int(text[2:])
        except ValueError:
            return None
    return None


@cache
def _pythagorean_triples(
    *, max_leg: int = 180, max_hypotenuse: int = 260
) -> tuple[tuple[int, int, int], ...]:
    """Return oriented integer right-triangle triples within a compact render range."""

    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for a in range(3, int(max_leg) + 1):
        for b in range(a, int(max_leg) + 1):
            c_squared = a * a + b * b
            c = math.isqrt(c_squared)
            if c * c != c_squared or c > int(max_hypotenuse):
                continue
            for oriented in ((a, b, c), (b, a, c)):
                if oriented not in seen:
                    seen.add(oriented)
                    triples.append(oriented)
    return tuple(sorted(triples, key=lambda item: (item[2], item[0], item[1])))


def _case(
    *,
    case_kind: str,
    answer: int | float,
    answer_type: str,
    formula_family: str,
    formula_text: str,
    reasoning_steps: int,
    vertices: Mapping[str, tuple[float, float]],
    edges: Iterable[tuple[str, str]],
    polygons: Iterable[tuple[str, ...]],
    segment_labels: Iterable[SegmentLabel],
    target_segment: tuple[str, str] | None = None,
    target_point: str | None = None,
    point_annotation_labels: Iterable[str] = (),
    point_mark_labels: Iterable[str] = (),
    angle_labels: Iterable[AngleLabel] = (),
    right_angles: Iterable[RightAngleMark] = (),
    tick_groups: Iterable[TickGroup] = (),
    filled_polygons: Iterable[tuple[str, ...]] = (),
    trace_values: Mapping[str, Any] | None = None,
    answer_rounding: str = "integer",
) -> TriangleRelationsCase:
    """Normalize one resolved construction into the scene state dataclass."""

    return TriangleRelationsCase(
        case_kind=str(case_kind),
        answer=(
            int(answer)
            if str(answer_type) == "integer"
            else float(round1(float(answer)))
        ),
        answer_type=str(answer_type),
        answer_rounding=str(answer_rounding),
        formula_family=str(formula_family),
        formula_text=str(formula_text),
        reasoning_steps=int(reasoning_steps),
        vertices={
            str(key): (float(value[0]), float(value[1]))
            for key, value in vertices.items()
        },
        edges=tuple((str(a), str(b)) for a, b in edges),
        polygons=tuple(tuple(str(label) for label in polygon) for polygon in polygons),
        segment_labels=tuple(segment_labels),
        target_segment=target_segment,
        target_point=target_point,
        point_annotation_labels=tuple(str(label) for label in point_annotation_labels),
        point_mark_labels=tuple(str(label) for label in point_mark_labels),
        angle_labels=tuple(angle_labels),
        right_angles=tuple(right_angles),
        tick_groups=tuple(tick_groups),
        filled_polygons=tuple(
            tuple(str(label) for label in polygon) for polygon in filled_polygons
        ),
        trace_values=dict(trace_values or {}),
    )


def _triangle_scale_points(split_ratio: float) -> dict[str, tuple[float, float]]:
    a, b, c = (360.0, 110.0), (135.0, 430.0), (605.0, 430.0)
    d = (a[0] + (b[0] - a[0]) * split_ratio, a[1] + (b[1] - a[1]) * split_ratio)
    e = (a[0] + (c[0] - a[0]) * split_ratio, a[1] + (c[1] - a[1]) * split_ratio)
    return {"A": a, "B": b, "C": c, "D": d, "E": e}


def _nested_triangle_case(
    *,
    case_kind: str,
    answer: int,
    labels: tuple[SegmentLabel, ...],
    target_segment: tuple[str, str],
    split_ratio: float,
    formula_family: str,
    formula_text: str,
    trace_values: Mapping[str, Any],
) -> TriangleRelationsCase:
    return _case(
        case_kind=case_kind,
        answer=int(answer),
        answer_type="integer",
        formula_family=formula_family,
        formula_text=formula_text,
        reasoning_steps=3,
        vertices=_triangle_scale_points(float(split_ratio)),
        edges=(("A", "B"), ("A", "C"), ("B", "C"), ("D", "E")),
        polygons=(("A", "B", "C"),),
        segment_labels=labels,
        target_segment=target_segment,
        tick_groups=(TickGroup((("D", "E"), ("B", "C")), kind="parallel"),),
        trace_values=trace_values,
    )


@cache
def similar_side_cases() -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for ad in range(2, 18):
        for db in range(2, 20):
            for ae in range(2, 48):
                if (ae * db) % ad != 0:
                    continue
                ec = (ae * db) // ad
                if not 2 <= ec <= 180:
                    continue
                ratio = ad / float(ad + db)
                cases.append(
                    _nested_triangle_case(
                        case_kind="nested_similarity_side",
                        answer=ec,
                        labels=(
                            SegmentLabel(("A", "D"), str(ad), -30.0, "AD"),
                            SegmentLabel(("D", "B"), str(db), -30.0, "DB"),
                            SegmentLabel(("A", "E"), str(ae), 30.0, "AE"),
                            SegmentLabel(("E", "C"), "EC=?", 30.0, "EC"),
                        ),
                        target_segment=("E", "C"),
                        split_ratio=ratio,
                        formula_family="similar_triangles_side_transfer",
                        formula_text="EC = AE * DB / AD",
                        trace_values={"AD": ad, "DB": db, "AE": ae, "EC": ec},
                    )
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


@cache
def parallel_section_cross_cases() -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for ad in range(2, 20):
        for db in range(2, 22):
            ab = ad + db
            for bc in range(10, 180):
                if (bc * ad) % ab != 0:
                    continue
                de = (bc * ad) // ab
                if not 3 <= de <= 140:
                    continue
                cases.append(
                    _nested_triangle_case(
                        case_kind="parallel_cross_section",
                        answer=de,
                        labels=(
                            SegmentLabel(("A", "D"), str(ad), -30.0, "AD"),
                            SegmentLabel(("D", "B"), str(db), -30.0, "DB"),
                            SegmentLabel(("B", "C"), str(bc), 34.0, "BC"),
                            SegmentLabel(("D", "E"), "DE=?", -30.0, "DE"),
                        ),
                        target_segment=("D", "E"),
                        split_ratio=ad / float(ab),
                        formula_family="parallel_section_scale",
                        formula_text="DE = BC * AD / (AD + DB)",
                        trace_values={"AD": ad, "DB": db, "AB": ab, "BC": bc, "DE": de},
                    )
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


@cache
def parallel_section_base_cases() -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for ad in range(2, 20):
        for db in range(2, 22):
            ab = ad + db
            for de in range(3, 110):
                if (de * ab) % ad != 0:
                    continue
                bc = (de * ab) // ad
                if not 10 <= bc <= 220:
                    continue
                cases.append(
                    _nested_triangle_case(
                        case_kind="parallel_base_scale",
                        answer=bc,
                        labels=(
                            SegmentLabel(("A", "D"), str(ad), -30.0, "AD"),
                            SegmentLabel(("D", "B"), str(db), -30.0, "DB"),
                            SegmentLabel(("D", "E"), str(de), -30.0, "DE"),
                            SegmentLabel(("B", "C"), "BC=?", 34.0, "BC"),
                        ),
                        target_segment=("B", "C"),
                        split_ratio=ad / float(ab),
                        formula_family="parallel_section_scale",
                        formula_text="BC = DE * (AD + DB) / AD",
                        trace_values={"AD": ad, "DB": db, "AB": ab, "DE": de, "BC": bc},
                    )
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


@cache
def chained_rectangle_diagonal_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build two-step Pythagorean cases over one split rectangle diagram."""

    max_visible_diagonal = 180
    triples_by_height: dict[int, list[tuple[int, int]]] = {}
    for width, height, diagonal in _pythagorean_triples(
        max_leg=320, max_hypotenuse=480
    ):
        if int(diagonal) > max_visible_diagonal:
            continue
        triples_by_height.setdefault(height, []).append((width, diagonal))
    cases: list[TriangleRelationsCase] = []
    for height, entries in sorted(triples_by_height.items()):
        for left_w, left_diag in entries:
            for total_w, target_diag in entries:
                right_w = int(total_w) - int(left_w)
                if not 3 <= right_w <= 280:
                    continue
                if not 8 <= total_w <= 300:
                    continue
                if int(target_diag) > max_visible_diagonal:
                    continue
                _append_chained_rectangle_case(
                    cases,
                    left_w=int(left_w),
                    height=int(height),
                    left_diag=int(left_diag),
                    right_w=int(right_w),
                    target_diag=int(target_diag),
                )
    return _trim_answer_variants(cases, max_answers=260, max_variants_per_answer=3)


def _append_chained_rectangle_case(
    cases: list[TriangleRelationsCase],
    *,
    left_w: int,
    height: int,
    left_diag: int,
    right_w: int,
    target_diag: int,
) -> None:
    """Append one split-rectangle case after integer-triple selection."""

    total_w = left_w + right_w
    scale = min(24.0, 440.0 / total_w, 280.0 / height)
    a = (150.0, 420.0)
    e = (a[0] + left_w * scale, a[1])
    b = (a[0] + total_w * scale, a[1])
    d = (a[0], a[1] - height * scale)
    f = (e[0], d[1])
    c = (b[0], d[1])
    cases.append(
        _case(
            case_kind="split_rectangle_diagonal",
            answer=target_diag,
            answer_type="integer",
            formula_family="two_step_pythagorean_length",
            formula_text="height from left diagonal, then target diagonal from total width",
            reasoning_steps=4,
            vertices={"A": a, "B": b, "C": c, "D": d, "E": e, "F": f},
            edges=(
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("D", "A"),
                ("E", "F"),
                ("D", "E"),
                ("D", "B"),
            ),
            polygons=(("A", "B", "C", "D"),),
            segment_labels=(
                SegmentLabel(("A", "E"), str(left_w), 28.0, "AE"),
                SegmentLabel(("E", "B"), str(right_w), 28.0, "EB"),
                SegmentLabel(("D", "E"), str(left_diag), -30.0, "DE"),
                SegmentLabel(("D", "B"), "DB=?", -30.0, "DB"),
            ),
            target_segment=("D", "B"),
            right_angles=(RightAngleMark("A", "B", "D"),),
            filled_polygons=(("A", "B", "C", "D"),),
            trace_values={
                "AE": left_w,
                "EB": right_w,
                "DE": left_diag,
                "derived_AD": height,
                "AB": total_w,
                "DB": target_diag,
            },
        )
    )


@cache
def rectangle_triangle_shared_height_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build cases where a rectangle diagonal exposes the adjacent triangle height."""

    triples_by_height: dict[int, list[tuple[int, int]]] = {}
    for width, height, diagonal in _pythagorean_triples(
        max_leg=160, max_hypotenuse=240
    ):
        triples_by_height.setdefault(height, []).append((width, diagonal))
    cases: list[TriangleRelationsCase] = []
    for height, entries in sorted(triples_by_height.items()):
        for rect_w, rect_diag in entries:
            for base, hyp in entries:
                if not 4 <= rect_w <= 140 or not 4 <= base <= 160:
                    continue
                _append_rectangle_triangle_shared_height_case(
                    cases,
                    rect_w=int(rect_w),
                    height=int(height),
                    rect_diag=int(rect_diag),
                    base=int(base),
                    hyp=int(hyp),
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


def _append_rectangle_triangle_shared_height_case(
    cases: list[TriangleRelationsCase],
    *,
    rect_w: int,
    height: int,
    rect_diag: int,
    base: int,
    hyp: int,
) -> None:
    """Append one shared-height Pythagorean case."""

    scale = min(22.0, 220.0 / rect_w, 260.0 / base, 275.0 / height)
    b = (335.0, 420.0)
    a = (b[0] - rect_w * scale, b[1])
    c = (b[0] + base * scale, b[1])
    d = (b[0], b[1] - height * scale)
    e = (a[0], d[1])
    cases.append(
        _case(
            case_kind="rectangle_triangle_shared_height",
            answer=hyp,
            answer_type="integer",
            formula_family="two_step_pythagorean_length",
            formula_text="shared height from rectangle diagonal, then triangle hypotenuse",
            reasoning_steps=4,
            vertices={"A": a, "B": b, "C": c, "D": d, "E": e},
            edges=(
                ("A", "B"),
                ("B", "D"),
                ("D", "E"),
                ("E", "A"),
                ("B", "C"),
                ("C", "D"),
                ("A", "D"),
            ),
            polygons=(("A", "B", "D", "E"), ("B", "C", "D")),
            segment_labels=(
                SegmentLabel(("A", "B"), str(rect_w), 28.0, "AB"),
                SegmentLabel(("A", "D"), str(rect_diag), -30.0, "AD"),
                SegmentLabel(("B", "C"), str(base), 28.0, "BC"),
                SegmentLabel(("C", "D"), "CD=?", -32.0, "CD"),
            ),
            target_segment=("C", "D"),
            right_angles=(RightAngleMark("B", "A", "D"),),
            filled_polygons=(("A", "B", "D", "E"), ("B", "C", "D")),
            trace_values={
                "AB": rect_w,
                "AD": rect_diag,
                "BD": height,
                "BC": base,
                "CD": hyp,
            },
        )
    )


def _bisector_points(
    ab: int, ac: int, bd: int, dc: int
) -> dict[str, tuple[float, float]]:
    bc = bd + dc
    scale = min(28.0, 430.0 / bc, 310.0 / max(ab, ac))
    base_width = bc * scale
    x_from_b = (((ab * ab) + (bc * bc) - (ac * ac)) / (2.0 * bc)) * scale
    height = math.sqrt(max(64.0, (ab * scale) ** 2 - x_from_b**2))
    left = 360.0 - base_width / 2.0
    base_y = 430.0
    return {
        "A": (left + x_from_b, base_y - height),
        "B": (left, base_y),
        "C": (left + base_width, base_y),
        "D": (left + bd * scale, base_y),
    }


def _bisector_case(
    *,
    case_kind: str,
    answer: int,
    ab: int,
    ac: int,
    bd: int,
    dc: int,
    labels: tuple[SegmentLabel, ...],
    target: tuple[str, str],
    trace_values: Mapping[str, Any],
) -> TriangleRelationsCase:
    return _case(
        case_kind=case_kind,
        answer=answer,
        answer_type="integer",
        formula_family="angle_bisector_theorem",
        formula_text="AB / AC = BD / DC",
        reasoning_steps=3,
        vertices=_bisector_points(ab, ac, bd, dc),
        edges=(("A", "B"), ("A", "C"), ("B", "C"), ("A", "D")),
        polygons=(("A", "B", "C"),),
        segment_labels=labels,
        target_segment=target,
        angle_labels=(
            AngleLabel("A", "B", "D", "", 34.0),
            AngleLabel("A", "D", "C", "", 34.0),
        ),
        tick_groups=(
            TickGroup((("A", "B"), ("A", "C")), count=0, kind="angle_bisector"),
        ),
        trace_values=trace_values,
    )


@cache
def angle_bisector_split_cases() -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for dc in range(4, 181):
        for ratio_left in range(2, 13):
            for ratio_right in range(2, 15):
                if math.gcd(ratio_left, ratio_right) != 1:
                    continue
                numerator = dc * ratio_left
                if numerator % ratio_right != 0:
                    continue
                bd = numerator // ratio_right
                side_scale = max(
                    2, math.ceil((bd + dc + 2) / float(ratio_left + ratio_right))
                )
                ab = ratio_left * side_scale
                ac = ratio_right * side_scale
                if not _is_valid_triangle(ab, ac, bd + dc):
                    continue
                cases.append(
                    _bisector_case(
                        case_kind="bisector_split",
                        answer=dc,
                        ab=ab,
                        ac=ac,
                        bd=bd,
                        dc=dc,
                        labels=(
                            SegmentLabel(
                                ("A", "B"),
                                str(ab),
                                -32.0,
                                "AB",
                                placement="side_readout",
                            ),
                            SegmentLabel(
                                ("A", "C"),
                                str(ac),
                                32.0,
                                "AC",
                                placement="side_readout",
                            ),
                            SegmentLabel(
                                ("B", "D"),
                                str(bd),
                                28.0,
                                "BD",
                                placement="side_readout",
                            ),
                            SegmentLabel(("D", "C"), "DC=?", 28.0, "DC"),
                        ),
                        target=("D", "C"),
                        trace_values={"AB": ab, "AC": ac, "BD": bd, "DC": dc},
                    )
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


def _angle_bisector_base_dimensions(
    answer: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return valid dimensions for base-length targets with the given base."""

    dimensions: list[tuple[int, int, int, int]] = []
    for ratio_left in range(2, 13):
        for ratio_right in range(2, 15):
            if math.gcd(ratio_left, ratio_right) != 1:
                continue
            ratio_sum = ratio_left + ratio_right
            if answer % ratio_sum != 0:
                continue
            multiplier = answer // ratio_sum
            bd = ratio_left * multiplier
            dc = ratio_right * multiplier
            side_scale = max(2, math.ceil((answer + 2) / float(ratio_sum)))
            ab = ratio_left * side_scale
            ac = ratio_right * side_scale
            if _is_valid_triangle(ab, ac, answer):
                dimensions.append((ab, ac, bd, dc))
    return tuple(dimensions)


@cache
def angle_bisector_base_cases() -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for bc in range(8, 241):
        for ab, ac, bd, dc in _angle_bisector_base_dimensions(bc):
            cases.append(
                _bisector_case(
                    case_kind="bisector_base",
                    answer=bc,
                    ab=ab,
                    ac=ac,
                    bd=bd,
                    dc=dc,
                    labels=(
                        SegmentLabel(("A", "B"), str(ab), -32.0, "AB"),
                        SegmentLabel(("A", "C"), str(ac), 32.0, "AC"),
                        SegmentLabel(("B", "D"), str(bd), 28.0, "BD"),
                        SegmentLabel(("B", "C"), "BC=?", 44.0, "BC"),
                    ),
                    target=("B", "C"),
                    trace_values={
                        "AB": ab,
                        "AC": ac,
                        "BD": bd,
                        "derived_DC": dc,
                        "BC": bc,
                    },
                )
            )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


@cache
def centroid_vertex_cases() -> tuple[TriangleRelationsCase, ...]:
    return tuple(
        _centroid_case(given=value, target_whole=False) for value in range(4, 124)
    )


@cache
def centroid_whole_cases() -> tuple[TriangleRelationsCase, ...]:
    return tuple(
        _centroid_case(given=value, target_whole=True) for value in range(8, 248, 2)
    )


def _centroid_case(*, given: int, target_whole: bool) -> TriangleRelationsCase:
    a, b, c = (360.0, 110.0), (140.0, 430.0), (590.0, 430.0)
    d = mid(b, c)
    g = (a[0] + (d[0] - a[0]) * (2.0 / 3.0), a[1] + (d[1] - a[1]) * (2.0 / 3.0))
    if target_whole:
        answer = (3 * given) // 2
        labels = (
            SegmentLabel(("A", "G"), str(given), -30.0, "AG"),
            SegmentLabel(("A", "D"), "AD=?", 30.0, "AD"),
        )
        target = ("A", "D")
        values = {"AG": given, "GD": given // 2, "AD": answer}
        kind = "centroid_whole_median"
    else:
        answer = 2 * given
        labels = (
            SegmentLabel(("G", "D"), str(given), 30.0, "GD"),
            SegmentLabel(("A", "G"), "AG=?", -30.0, "AG"),
        )
        target = ("A", "G")
        values = {"GD": given, "AG": answer, "AD": answer + given}
        kind = "centroid_vertex_segment"
    return _case(
        case_kind=kind,
        answer=answer,
        answer_type="integer",
        formula_family="centroid_median_ratio",
        formula_text="AG:GD = 2:1 and AD = AG + GD",
        reasoning_steps=2,
        vertices={"A": a, "B": b, "C": c, "D": d, "G": g},
        edges=(("A", "B"), ("A", "C"), ("B", "C"), ("A", "D")),
        polygons=(("A", "B", "C"),),
        segment_labels=labels,
        target_segment=target,
        point_mark_labels=("G",),
        tick_groups=(TickGroup((("B", "D"), ("D", "C")), count=1),),
        trace_values={"D_midpoint_of_BC": True, "G_is_centroid": True, **values},
    )


def _right_triangle_case(
    *,
    case_kind: str,
    adjacent: float,
    opposite: float,
    hypotenuse: float,
    theta: float,
    visible_sides: tuple[str, ...],
    target_side: str | None,
    formula_family: str,
    formula_text: str,
    answer: float,
    context: str = "triangle",
) -> TriangleRelationsCase:
    """Construct one right-triangle trigonometry diagram with one target witness."""

    scale = min(420.0 / max(adjacent, 1.0), 270.0 / max(opposite, 1.0))
    a = (140.0, 430.0)
    c = (140.0 + adjacent * scale, 430.0)
    b = (c[0], 430.0 - opposite * scale)
    side_segments = {
        "adjacent": ("A", "C"),
        "opposite": ("B", "C"),
        "hypotenuse": ("A", "B"),
    }
    labels: list[SegmentLabel] = []
    for side in visible_sides:
        segment_name = "".join(side_segments[side])
        value = {"adjacent": adjacent, "opposite": opposite, "hypotenuse": hypotenuse}[
            side
        ]
        labels.append(
            SegmentLabel(side_segments[side], fmt_measure(value), 32.0, segment_name)
        )
    if target_side is not None:
        segment_name = "".join(side_segments[target_side])
        labels.append(
            SegmentLabel(
                side_segments[target_side], f"{segment_name}=?", -34.0, segment_name
            )
        )
    angle_text = "?" if target_side is None else f"θ={fmt_measure(theta)}°"
    answer_type = "number"
    rounding = "one_decimal"
    annotation_mode = "segment" if target_side is not None else "point"
    trace_values = {
        "adjacent": round1(adjacent),
        "opposite": round1(opposite),
        "hypotenuse": round1(hypotenuse),
        "theta_degrees": round1(theta),
        "visible_sides": list(visible_sides),
        "target_side": target_side,
        "context_style": context,
        "annotation_mode": annotation_mode,
    }
    return _case(
        case_kind=case_kind,
        answer=answer,
        answer_type=answer_type,
        answer_rounding=rounding,
        formula_family=formula_family,
        formula_text=formula_text,
        reasoning_steps=1,
        vertices={"A": a, "B": b, "C": c},
        edges=(("A", "C"), ("C", "B"), ("A", "B")),
        polygons=(("A", "B", "C"),),
        segment_labels=tuple(labels),
        target_segment=side_segments[target_side] if target_side is not None else None,
        target_point="A" if target_side is None else None,
        angle_labels=(AngleLabel("A", "B", "C", angle_text, 54.0, "theta"),),
        right_angles=(RightAngleMark("C", "A", "B"),),
        filled_polygons=(("A", "B", "C"),),
        trace_values=trace_values,
    )


def _missing_side_cases(
    kind: str,
    builder: Callable[
        [float, float],
        tuple[float, float, float, float, tuple[str, ...], str, str, str],
    ],
) -> tuple[TriangleRelationsCase, ...]:
    cases: list[TriangleRelationsCase] = []
    for theta in range(15, 76, 2):
        for known in range(5, 91):
            adjacent, opposite, hypotenuse, answer, visible, target, family, formula = (
                builder(float(theta), float(known))
            )
            cases.append(
                _right_triangle_case(
                    case_kind=kind,
                    adjacent=adjacent,
                    opposite=opposite,
                    hypotenuse=hypotenuse,
                    theta=float(theta),
                    visible_sides=visible,
                    target_side=target,
                    formula_family=family,
                    formula_text=formula,
                    answer=round1(answer),
                )
            )
    return _trim_answer_variants(cases, max_answers=520, max_variants_per_answer=2)


@cache
def height_from_angle_ground_cases() -> tuple[TriangleRelationsCase, ...]:
    return _missing_side_cases(
        "height_from_tangent",
        lambda theta, known: (
            known,
            known * math.tan(math.radians(theta)),
            math.hypot(known, known * math.tan(math.radians(theta))),
            known * math.tan(math.radians(theta)),
            ("adjacent",),
            "opposite",
            "right_triangle_tangent",
            "height = ground * tan(theta)",
        ),
    )


@cache
def ground_from_angle_height_cases() -> tuple[TriangleRelationsCase, ...]:
    return _missing_side_cases(
        "ground_from_tangent",
        lambda theta, known: (
            known / math.tan(math.radians(theta)),
            known,
            math.hypot(known / math.tan(math.radians(theta)), known),
            known / math.tan(math.radians(theta)),
            ("opposite",),
            "adjacent",
            "right_triangle_tangent",
            "ground = height / tan(theta)",
        ),
    )


@cache
def hypotenuse_from_angle_height_cases() -> tuple[TriangleRelationsCase, ...]:
    return _missing_side_cases(
        "hypotenuse_from_sine",
        lambda theta, known: (
            known / math.tan(math.radians(theta)),
            known,
            known / math.sin(math.radians(theta)),
            known / math.sin(math.radians(theta)),
            ("opposite",),
            "hypotenuse",
            "right_triangle_sine",
            "hypotenuse = height / sin(theta)",
        ),
    )


@cache
def height_from_angle_hypotenuse_cases() -> tuple[TriangleRelationsCase, ...]:
    return _missing_side_cases(
        "height_from_sine",
        lambda theta, known: (
            known * math.cos(math.radians(theta)),
            known * math.sin(math.radians(theta)),
            known,
            known * math.sin(math.radians(theta)),
            ("hypotenuse",),
            "opposite",
            "right_triangle_sine",
            "height = hypotenuse * sin(theta)",
        ),
    )


@cache
def ground_from_angle_hypotenuse_cases() -> tuple[TriangleRelationsCase, ...]:
    return _missing_side_cases(
        "ground_from_cosine",
        lambda theta, known: (
            known * math.cos(math.radians(theta)),
            known * math.sin(math.radians(theta)),
            known,
            known * math.cos(math.radians(theta)),
            ("hypotenuse",),
            "adjacent",
            "right_triangle_cosine",
            "ground = hypotenuse * cos(theta)",
        ),
    )


@cache
def angle_bisector_variable_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build angle-bisector equation cases while keeping A/B/C/D witnesses stable."""

    cases: list[TriangleRelationsCase] = []
    for answer in range(3, 181):
        for offset in range(1, 11):
            dc = answer + offset
            for ratio_left in range(2, 13):
                for ratio_right in range(2, 15):
                    if math.gcd(ratio_left, ratio_right) != 1:
                        continue
                    numerator = dc * ratio_left
                    if numerator % ratio_right != 0:
                        continue
                    bd = numerator // ratio_right
                    side_scale = max(
                        2, math.ceil((bd + dc + 2) / float(ratio_left + ratio_right))
                    )
                    ab = ratio_left * side_scale
                    ac = ratio_right * side_scale
                    if not _is_valid_triangle(ab, ac, bd + dc):
                        continue
                    dc_label = f"x+{offset}"
                    labels = (
                        SegmentLabel(("A", "B"), str(ab), -32.0, "AB"),
                        SegmentLabel(("A", "C"), str(ac), 32.0, "AC"),
                        SegmentLabel(("B", "D"), str(bd), 28.0, "BD"),
                        SegmentLabel(("D", "C"), str(dc_label), 28.0, "DC"),
                    )
                    trace = {
                        "AB": ab,
                        "AC": ac,
                        "BD": bd,
                        "DC_expression": dc_label,
                        "x": answer,
                    }
                    cases.append(
                        _angle_bisector_variable_case(
                            "split_ratio_variable",
                            answer,
                            ab,
                            ac,
                            bd,
                            dc,
                            labels,
                            trace,
                        )
                    )

        for offset in range(1, 11):
            ab = answer + offset
            for bd in range(3, 28):
                for dc in range(3, 32):
                    if (ab * dc) % bd != 0:
                        continue
                    ac = (ab * dc) // bd
                    if not 4 <= ac <= 220:
                        continue
                    if not _is_valid_triangle(ab, ac, bd + dc):
                        continue
                    ab_label = f"x+{offset}"
                    labels = (
                        SegmentLabel(("A", "B"), str(ab_label), -32.0, "AB"),
                        SegmentLabel(("A", "C"), str(ac), 32.0, "AC"),
                        SegmentLabel(("B", "D"), str(bd), 28.0, "BD"),
                        SegmentLabel(("D", "C"), str(dc), 28.0, "DC"),
                    )
                    trace = {
                        "AB_expression": ab_label,
                        "AC": ac,
                        "BD": bd,
                        "DC": dc,
                        "x": answer,
                    }
                    cases.append(
                        _angle_bisector_variable_case(
                            "adjacent_ratio_variable",
                            answer,
                            ab,
                            ac,
                            bd,
                            dc,
                            labels,
                            trace,
                        )
                    )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


def _angle_bisector_variable_case(
    family: str,
    answer: int,
    ab: int,
    ac: int,
    bd: int,
    dc: int,
    labels: tuple[SegmentLabel, ...],
    trace: Mapping[str, Any],
) -> TriangleRelationsCase:
    return _case(
        case_kind=str(family),
        answer=int(answer),
        answer_type="integer",
        formula_family="angle_bisector_theorem_variable",
        formula_text="AB / AC = BD / DC, solve for x",
        reasoning_steps=3,
        vertices=_bisector_points(int(ab), int(ac), int(bd), int(dc)),
        edges=(("A", "B"), ("A", "C"), ("B", "C"), ("A", "D")),
        polygons=(("A", "B", "C"),),
        segment_labels=labels,
        point_annotation_labels=("A", "B", "C", "D"),
        angle_labels=(
            AngleLabel("A", "B", "D", "", 34.0),
            AngleLabel("A", "D", "C", "", 34.0),
        ),
        tick_groups=(
            TickGroup((("A", "B"), ("A", "C")), count=0, kind="angle_bisector"),
        ),
        trace_values={**trace, "internal_case_family": str(family)},
    )


@cache
def split_triangle_angle_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build split-triangle angle-chase cases under the triangle-relations scene."""

    vertices = {
        "A": (150.0, 155.0),
        "B": (360.0, 160.0),
        "C": (610.0, 155.0),
        "D": (365.0, 430.0),
    }
    cases: list[TriangleRelationsCase] = []
    for answer in range(20, 136):
        for angle_a in range(25, 116):
            angle_b = 180 - int(answer) - int(angle_a)
            if not 20 <= angle_b <= 120:
                continue
            labels = (
                AngleLabel("A", "B", "D", f"{angle_a}{DEGREE_SYMBOL}", 48.0, "given_A"),
                AngleLabel("B", "A", "D", f"{angle_b}{DEGREE_SYMBOL}", 48.0, "given_B"),
                AngleLabel("D", "A", "B", "?", 58.0, "target_angle"),
            )
            trace = {
                "angle_a": int(angle_a),
                "angle_b": int(angle_b),
                "answer": int(answer),
            }
            cases.append(
                _split_triangle_angle_case(
                    vertices,
                    "single_cevian_triangle_angle_sum",
                    answer,
                    labels,
                    trace,
                    1,
                )
            )

        for angle_a in range(25, 96):
            for angle_d in range(25, 96):
                angle_c = 180 - int(angle_a) - int(angle_d) - int(answer)
                if not 20 <= angle_c <= 95:
                    continue
                left_missing = 180 - int(angle_a) - int(angle_d)
                labels = (
                    AngleLabel(
                        "A", "B", "D", f"{angle_a}{DEGREE_SYMBOL}", 48.0, "given_A"
                    ),
                    AngleLabel(
                        "D", "A", "B", f"{angle_d}{DEGREE_SYMBOL}", 58.0, "given_D_left"
                    ),
                    AngleLabel(
                        "C", "B", "D", f"{angle_c}{DEGREE_SYMBOL}", 48.0, "given_C"
                    ),
                    AngleLabel("D", "B", "C", "?", 58.0, "target_angle"),
                )
                trace = {
                    "angle_a": int(angle_a),
                    "angle_d_left": int(angle_d),
                    "angle_c": int(angle_c),
                    "left_triangle_missing_angle": int(left_missing),
                    "straight_angle_supplement": 180 - int(left_missing),
                    "answer": int(answer),
                }
                cases.append(
                    _split_triangle_angle_case(
                        vertices,
                        "two_step_adjacent_triangle_angle_sum",
                        answer,
                        labels,
                        trace,
                        3,
                    )
                )
    return _trim_answer_variants(cases, max_answers=220, max_variants_per_answer=3)


def _split_triangle_angle_case(
    vertices: Mapping[str, tuple[float, float]],
    family: str,
    answer: int,
    labels: tuple[AngleLabel, ...],
    trace: Mapping[str, Any],
    reasoning_steps: int,
) -> TriangleRelationsCase:
    """Append-compatible constructor for one split-triangle angle chase."""

    return _case(
        case_kind=str(family),
        answer=int(answer),
        answer_type="integer",
        formula_family="split_triangle_angle_sum",
        formula_text="triangle angle sum and, when needed, straight-angle supplement",
        reasoning_steps=int(reasoning_steps),
        vertices=vertices,
        edges=(("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("B", "D")),
        polygons=(("A", "B", "D"), ("B", "C", "D")),
        segment_labels=(),
        point_annotation_labels=("A", "B", "C", "D"),
        angle_labels=tuple(labels),
        filled_polygons=(("A", "B", "D"), ("B", "C", "D")),
        trace_values={**dict(trace), "internal_construction_family": str(family)},
    )


@cache
def split_triangle_trig_side_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build split right-triangle trigonometry cases with a shared altitude."""

    def _num(value: float | int) -> str:
        return str(fmt_measure(float(value)))

    cases: list[TriangleRelationsCase] = []
    standard_angles = (30, 45, 60)
    vertices = {
        "A": (365.0, 135.0),
        "B": (145.0, 430.0),
        "C": (625.0, 430.0),
        "D": (365.0, 430.0),
    }

    def add_case(
        *,
        family: str,
        answer: float,
        target: tuple[str, str],
        labels: tuple[SegmentLabel, ...],
        angles: tuple[AngleLabel, ...],
        trace: Mapping[str, Any],
        ticks: tuple[TickGroup, ...] = (),
    ) -> None:
        cases.append(
            _case(
                case_kind=str(family),
                answer=round1(float(answer)),
                answer_type="number",
                answer_rounding="one_decimal",
                formula_family="shared_altitude_right_triangle_trig",
                formula_text="use one right triangle to derive the shared altitude, then solve the target side",
                reasoning_steps=3,
                vertices=vertices,
                edges=(("A", "B"), ("B", "D"), ("D", "C"), ("C", "A"), ("A", "D")),
                polygons=(("A", "B", "D"), ("A", "D", "C")),
                segment_labels=labels,
                target_segment=target,
                point_annotation_labels=("A", "B", "C", "D"),
                angle_labels=angles,
                right_angles=(RightAngleMark("D", "A", "C"),),
                tick_groups=ticks,
                filled_polygons=(("A", "B", "D"), ("A", "D", "C")),
                trace_values={
                    **dict(trace),
                    "internal_construction_family": str(family),
                    "answer": round1(float(answer)),
                },
            )
        )

    for known_base in range(6, 31):
        for left_angle in standard_angles:
            for right_angle in standard_angles:
                if (
                    len(
                        [
                            case
                            for case in cases
                            if case.case_kind == "shared_altitude_two_angles_side"
                        ]
                    )
                    >= 80
                ):
                    break
                altitude = float(known_base) * math.tan(math.radians(float(left_angle)))
                answer = altitude / math.sin(math.radians(float(right_angle)))
                if not 6.0 <= answer <= 80.0:
                    continue
                add_case(
                    family="shared_altitude_two_angles_side",
                    answer=answer,
                    target=("A", "C"),
                    labels=(
                        SegmentLabel(("B", "D"), _num(known_base), 32.0, "BD"),
                        SegmentLabel(("A", "C"), "AC=?", 36.0, "AC"),
                    ),
                    angles=(
                        AngleLabel(
                            "B",
                            "A",
                            "D",
                            f"{left_angle}{DEGREE_SYMBOL}",
                            52.0,
                            "angle_B",
                        ),
                        AngleLabel(
                            "C",
                            "D",
                            "A",
                            f"{right_angle}{DEGREE_SYMBOL}",
                            52.0,
                            "angle_C",
                        ),
                    ),
                    trace={
                        "known_segment": "BD",
                        "known_value": int(known_base),
                        "left_angle": int(left_angle),
                        "right_angle": int(right_angle),
                        "altitude": round(float(altitude), 4),
                        "target_side": "AC",
                    },
                )
            if (
                len(
                    [
                        case
                        for case in cases
                        if case.case_kind == "shared_altitude_two_angles_side"
                    ]
                )
                >= 80
            ):
                break
        if (
            len(
                [
                    case
                    for case in cases
                    if case.case_kind == "shared_altitude_two_angles_side"
                ]
            )
            >= 80
        ):
            break

    for known_side in range(8, 41):
        for left_angle in standard_angles:
            for right_angle in standard_angles:
                if (
                    len(
                        [
                            case
                            for case in cases
                            if case.case_kind == "shared_altitude_side_then_hypotenuse"
                        ]
                    )
                    >= 80
                ):
                    break
                altitude = float(known_side) * math.sin(math.radians(float(left_angle)))
                answer = altitude / math.sin(math.radians(float(right_angle)))
                if not 5.0 <= answer <= 70.0:
                    continue
                add_case(
                    family="shared_altitude_side_then_hypotenuse",
                    answer=answer,
                    target=("A", "C"),
                    labels=(
                        SegmentLabel(("A", "B"), _num(known_side), -36.0, "AB"),
                        SegmentLabel(("A", "C"), "AC=?", 36.0, "AC"),
                    ),
                    angles=(
                        AngleLabel(
                            "B",
                            "A",
                            "D",
                            f"{left_angle}{DEGREE_SYMBOL}",
                            52.0,
                            "angle_B",
                        ),
                        AngleLabel(
                            "C",
                            "D",
                            "A",
                            f"{right_angle}{DEGREE_SYMBOL}",
                            52.0,
                            "angle_C",
                        ),
                    ),
                    trace={
                        "known_segment": "AB",
                        "known_value": int(known_side),
                        "left_angle": int(left_angle),
                        "right_angle": int(right_angle),
                        "altitude": round(float(altitude), 4),
                        "target_side": "AC",
                    },
                )
            if (
                len(
                    [
                        case
                        for case in cases
                        if case.case_kind == "shared_altitude_side_then_hypotenuse"
                    ]
                )
                >= 80
            ):
                break
        if (
            len(
                [
                    case
                    for case in cases
                    if case.case_kind == "shared_altitude_side_then_hypotenuse"
                ]
            )
            >= 80
        ):
            break

    for half_base in range(4, 31):
        for base_angle in standard_angles:
            if (
                len(
                    [
                        case
                        for case in cases
                        if case.case_kind == "isosceles_altitude_trig_side"
                    ]
                )
                >= 80
            ):
                break
            answer = float(half_base) / math.cos(math.radians(float(base_angle)))
            if not 6.0 <= answer <= 75.0:
                continue
            add_case(
                family="isosceles_altitude_trig_side",
                answer=answer,
                target=("A", "B"),
                labels=(
                    SegmentLabel(("B", "D"), _num(half_base), 32.0, "BD"),
                    SegmentLabel(("A", "B"), "AB=?", -36.0, "AB"),
                ),
                angles=(
                    AngleLabel(
                        "B", "A", "D", f"{base_angle}{DEGREE_SYMBOL}", 52.0, "angle_B"
                    ),
                ),
                ticks=(
                    TickGroup((("A", "B"), ("A", "C")), count=1),
                    TickGroup((("B", "D"), ("D", "C")), count=2),
                ),
                trace={
                    "known_segment": "BD",
                    "known_value": int(half_base),
                    "left_angle": int(base_angle),
                    "right_angle": int(base_angle),
                    "target_side": "AB",
                },
            )
        if (
            len(
                [
                    case
                    for case in cases
                    if case.case_kind == "isosceles_altitude_trig_side"
                ]
            )
            >= 80
        ):
            break
    return _trim_answer_variants(cases, max_answers=240, max_variants_per_answer=3)


@cache
def _right_triangle_altitude_rows() -> tuple[tuple[int, int, int, int, int], ...]:
    """Generate integer altitude-theorem rows from Pythagorean parameter pairs."""

    rows: set[tuple[int, int, int, int, int]] = set()
    for m, n, r in _pythagorean_triples(max_leg=70, max_hypotenuse=130):
        if m > 50 or n > 50:
            continue
        for scale_factor in range(1, 21):
            left_projection = m * m * scale_factor
            right_projection = n * n * scale_factor
            altitude = m * n * scale_factor
            left_leg = m * r * scale_factor
            right_leg = n * r * scale_factor
            if left_projection + right_projection > 6000:
                continue
            rows.add((left_projection, right_projection, altitude, left_leg, right_leg))
    return tuple(
        sorted(rows, key=lambda item: (item[2], item[0], item[1], item[3], item[4]))
    )


@cache
def _right_triangle_altitude_cases(mode: str) -> tuple[TriangleRelationsCase, ...]:
    """Build right-triangle altitude theorem cases for one semantic mode."""
    target_segments = {
        "altitude": ("A", "D"),
        "left_projection": ("B", "D"),
        "right_projection": ("D", "C"),
        "left_leg": ("A", "B"),
        "right_leg": ("A", "C"),
    }
    cases: list[TriangleRelationsCase] = []
    for (
        left_projection,
        right_projection,
        altitude,
        left_leg,
        right_leg,
    ) in _right_triangle_altitude_rows():
        hypotenuse = int(left_projection) + int(right_projection)
        scale = 430.0 / float(hypotenuse)
        base_x = 130.0
        base_y = 420.0
        foot_x = base_x + float(left_projection) * scale
        vertices = {
            "B": (base_x, base_y),
            "D": (foot_x, base_y),
            "C": (base_x + float(hypotenuse) * scale, base_y),
            "A": (foot_x, base_y - float(altitude) * scale),
        }
        branch_cases: list[
            tuple[str, int, tuple[SegmentLabel, ...], Mapping[str, Any]]
        ] = []
        if mode == "altitude_from_two_projections":
            branch_cases.append(
                (
                    "altitude",
                    int(altitude),
                    (
                        SegmentLabel(("B", "D"), str(left_projection), 30.0, "BD"),
                        SegmentLabel(("D", "C"), str(right_projection), 30.0, "DC"),
                        SegmentLabel(("A", "D"), "AD=?", -34.0, "AD"),
                    ),
                    {"relation": "altitude_geometric_mean_from_split_hypotenuse"},
                )
            )
        elif mode == "projection_from_altitude_and_projection":
            branch_cases.extend(
                (
                    (
                        "right_projection",
                        int(right_projection),
                        (
                            SegmentLabel(("B", "D"), str(left_projection), 30.0, "BD"),
                            SegmentLabel(("D", "C"), "DC=?", 30.0, "DC"),
                            SegmentLabel(("A", "D"), str(altitude), -34.0, "AD"),
                        ),
                        {"relation": "projection_from_altitude_and_other_projection"},
                    ),
                    (
                        "left_projection",
                        int(left_projection),
                        (
                            SegmentLabel(("B", "D"), "BD=?", 30.0, "BD"),
                            SegmentLabel(("D", "C"), str(right_projection), 30.0, "DC"),
                            SegmentLabel(("A", "D"), str(altitude), -34.0, "AD"),
                        ),
                        {"relation": "projection_from_altitude_and_other_projection"},
                    ),
                )
            )
        elif (
            mode == "leg_projection_relation"
            and left_leg is not None
            and right_leg is not None
        ):
            branch_cases.extend(
                (
                    (
                        "left_leg",
                        int(left_leg),
                        (
                            SegmentLabel(("B", "C"), str(hypotenuse), 48.0, "BC"),
                            SegmentLabel(("B", "D"), str(left_projection), 30.0, "BD"),
                            SegmentLabel(("A", "B"), "AB=?", -34.0, "AB"),
                        ),
                        {"relation": "leg_geometric_mean_from_hypotenuse_projection"},
                    ),
                    (
                        "right_leg",
                        int(right_leg),
                        (
                            SegmentLabel(("B", "C"), str(hypotenuse), 48.0, "BC"),
                            SegmentLabel(("D", "C"), str(right_projection), 30.0, "DC"),
                            SegmentLabel(("A", "C"), "AC=?", 34.0, "AC"),
                        ),
                        {"relation": "leg_geometric_mean_from_hypotenuse_projection"},
                    ),
                )
            )
        elif (
            mode == "projection_leg_relation"
            and left_leg is not None
            and right_leg is not None
        ):
            branch_cases.extend(
                (
                    (
                        "left_projection",
                        int(left_projection),
                        (
                            SegmentLabel(("B", "C"), str(hypotenuse), 48.0, "BC"),
                            SegmentLabel(("A", "B"), str(left_leg), -34.0, "AB"),
                            SegmentLabel(("B", "D"), "BD=?", 30.0, "BD"),
                        ),
                        {"relation": "projection_length_from_leg_hypotenuse"},
                    ),
                    (
                        "right_projection",
                        int(right_projection),
                        (
                            SegmentLabel(("B", "C"), str(hypotenuse), 48.0, "BC"),
                            SegmentLabel(("A", "C"), str(right_leg), 34.0, "AC"),
                            SegmentLabel(("D", "C"), "DC=?", 30.0, "DC"),
                        ),
                        {"relation": "projection_length_from_leg_hypotenuse"},
                    ),
                )
            )
        for target_role, answer, labels, relation_trace in branch_cases:
            cases.append(
                _case(
                    case_kind=f"right_triangle_altitude_{target_role}",
                    answer=int(answer),
                    answer_type="integer",
                    formula_family="right_triangle_altitude_projection",
                    formula_text="AD^2 = BD * DC and leg^2 = hypotenuse * adjacent_projection",
                    reasoning_steps=2,
                    vertices=vertices,
                    edges=(("A", "B"), ("A", "C"), ("B", "C"), ("A", "D")),
                    polygons=(("A", "B", "C"),),
                    segment_labels=labels,
                    target_segment=target_segments[str(target_role)],
                    point_annotation_labels=("A", "B", "C", "D"),
                    right_angles=(
                        RightAngleMark("A", "B", "C"),
                        RightAngleMark("D", "A", "C"),
                    ),
                    filled_polygons=(("A", "B", "C"),),
                    trace_values={
                        "left_projection": int(left_projection),
                        "right_projection": int(right_projection),
                        "altitude": int(altitude),
                        "hypotenuse": int(hypotenuse),
                        "left_leg": None if left_leg is None else int(left_leg),
                        "right_leg": None if right_leg is None else int(right_leg),
                        "target_role": str(target_role),
                        "target_name": "".join(target_segments[str(target_role)]),
                        **dict(relation_trace),
                    },
                )
            )
    return _trim_answer_variants(cases, max_answers=240, max_variants_per_answer=3)


@cache
def altitude_from_two_projections_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build cases where the altitude is the geometric mean of two projections."""

    return _right_triangle_altitude_cases("altitude_from_two_projections")


@cache
def projection_from_altitude_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build cases where one projection is unknown from altitude and the other projection."""

    return _right_triangle_altitude_cases("projection_from_altitude_and_projection")


@cache
def leg_from_projection_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build cases where one leg is unknown from hypotenuse and adjacent projection."""

    return _right_triangle_altitude_cases("leg_projection_relation")


@cache
def projection_from_leg_cases() -> tuple[TriangleRelationsCase, ...]:
    """Build cases where one projection is unknown from a leg and the hypotenuse."""

    return _right_triangle_altitude_cases("projection_leg_relation")


def case_trace_values(case: TriangleRelationsCase) -> dict[str, Any]:
    """Return JSON-safe trace values for one resolved construction case."""

    return {
        "internal_case_kind": str(case.case_kind),
        "formula_family": str(case.formula_family),
        "formula": str(case.formula_text),
        "reasoning_steps": int(case.reasoning_steps),
        **dict(case.trace_values),
    }


__all__ = [
    "angle_bisector_base_cases",
    "angle_bisector_split_cases",
    "angle_bisector_variable_cases",
    "altitude_from_two_projections_cases",
    "case_trace_values",
    "centroid_vertex_cases",
    "centroid_whole_cases",
    "chained_rectangle_diagonal_cases",
    "ground_from_angle_height_cases",
    "ground_from_angle_hypotenuse_cases",
    "height_from_angle_ground_cases",
    "height_from_angle_hypotenuse_cases",
    "hypotenuse_from_angle_height_cases",
    "leg_from_projection_cases",
    "parallel_section_base_cases",
    "parallel_section_cross_cases",
    "rectangle_triangle_shared_height_cases",
    "projection_from_altitude_cases",
    "projection_from_leg_cases",
    "similar_side_cases",
    "split_triangle_angle_cases",
    "split_triangle_trig_side_cases",
]
