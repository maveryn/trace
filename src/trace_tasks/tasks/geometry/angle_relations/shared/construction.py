"""Objective-neutral diagram construction for angle-relations tasks."""

from __future__ import annotations

import math

from .spatial_primitives import (
    RenderContext,
    _add,
    _angle_label_center,
    _draw_angle_arc,
    _angle_annotation_point,
    _bbox_from_points,
    _draw_angle_label,
    _draw_point_labels,
    _draw_polygon,
    _draw_polyline,
    _draw_text,
    _offset_points,
    _sub,
    _triangle_from_base_angles,
    _unit,
)
from .relations import (
    exterior_angle_value,
    format_angle_expression,
    format_degrees,
    format_linear_expression,
    linear_expression_value,
    supplement_angle_value,
)
from .state import (
    ANGLE_ABC,
    ANGLE_AEF,
    ANGLE_BAC,
    ANGLE_BCD,
    ANGLE_CFE,
    POINT_A,
    POINT_B,
    POINT_C,
    POINT_D,
    AngleRelationCase,
    RenderedAngleRelationScene,
)

POINT_P = "P"
POINT_Q = "Q"
POINT_R = "R"


def _angle_local_label_center(
    vertex: tuple[float, float],
    primary_arm: tuple[float, float],
    secondary_arm: tuple[float, float],
    *,
    primary_scale: float,
    secondary_scale: float,
) -> tuple[float, float]:
    """Place a label in an angle using local ray directions."""

    primary = _unit(_sub(primary_arm, vertex))
    secondary = _unit(_sub(secondary_arm, vertex))
    return _add(_add(vertex, primary, float(primary_scale)), secondary, float(secondary_scale))


def _line_local_label_offset(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    along_scale: float,
    normal_scale: float,
) -> tuple[float, float]:
    """Return a label offset using a rendered line's local axes."""

    direction = _unit(_sub(end, start))
    normal = (-direction[1], direction[0])
    return (
        (direction[0] * float(along_scale)) + (normal[0] * float(normal_scale)),
        (direction[1] * float(along_scale)) + (normal[1] * float(normal_scale)),
    )


def _draw_parallel_arrow_marks(ctx: RenderContext, segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]) -> tuple[float, float, float, float]:
    """Draw arrow-style parallel markers on each parallel segment."""

    marker_points: list[tuple[float, float]] = []
    for start, end in segments:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        length = max(1e-9, math.hypot(dx, dy))
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        center = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
        half_len = 20.0
        head_len = 10.0
        head_half_width = 6.0
        tail = (center[0] - ux * half_len, center[1] - uy * half_len)
        tip = (center[0] + ux * half_len, center[1] + uy * half_len)
        head_left = (
            tip[0] - ux * head_len + nx * head_half_width,
            tip[1] - uy * head_len + ny * head_half_width,
        )
        head_right = (
            tip[0] - ux * head_len - nx * head_half_width,
            tip[1] - uy * head_len - ny * head_half_width,
        )
        width = max(2, int(ctx.line_width) - 1)
        _draw_polyline(ctx, [tail, tip], fill=ctx.accent_color, width=width)
        _draw_polyline(ctx, [head_left, tip, head_right], fill=ctx.accent_color, width=width)
        marker_points.extend([tail, tip, head_left, head_right])
    return _bbox_from_points(marker_points, width=ctx.width, height=ctx.height, pad=5.0)


def make_triangle_exterior_case(given_a: int, answer_b: int) -> AngleRelationCase:
    """Construct a reusable triangle exterior-angle scene case."""

    exterior = exterior_angle_value(given_a, answer_b)
    right_interior = 180 - int(exterior)

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render triangle/exterior geometry without choosing task identity."""

        a, b, c = _triangle_from_base_angles(
            left_angle=int(given_a),
            right_angle=int(right_interior),
            canvas_width=float(ctx.width),
            canvas_height=float(ctx.height),
        )
        d = (min(ctx.width - 70.0, c[0] + 95.0), c[1])
        a, b, c, d = _offset_points(ctx, (a, b, c, d))
        _draw_polygon(ctx, [a, b, c])
        _draw_polyline(ctx, [c, d])
        labels = _draw_point_labels(ctx, {"A": a, "B": b, "C": c, "D": d})
        target_arc, target_label = _draw_angle_label(ctx, "?", b, a, c, radius=66.0)
        given_arc, label_a = _draw_angle_label(ctx, format_degrees(given_a), a, c, b, radius=66.0)
        exterior_arc, label_ext = _draw_angle_label(ctx, format_degrees(exterior), c, b, d, radius=72.0)
        annotation_points = {
            ANGLE_ABC: _angle_annotation_point(b, a, c, label_radius=66.0),
            ANGLE_BAC: _angle_annotation_point(a, c, b, label_radius=66.0),
            ANGLE_BCD: _angle_annotation_point(c, b, d, label_radius=72.0),
        }
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(answer_b),
            annotation_bboxes=(target_arc, given_arc, exterior_arc),
            annotation_roles=(ANGLE_ABC, ANGLE_BAC, ANGLE_BCD),
            scene_entities=(
                {"type": "triangle", "points": {"A": a, "B": b, "C": c}},
                {"type": "extension_ray", "points": {"C": c, "D": d}},
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {ANGLE_ABC: target_arc, ANGLE_BAC: given_arc, ANGLE_BCD: exterior_arc},
                "angle_label_bboxes": {ANGLE_ABC: target_label, ANGLE_BAC: label_a, ANGLE_BCD: label_ext},
            },
            witness={
                "given_angle_A": int(given_a),
                "given_exterior_angle_BCD": int(exterior),
                "answer_angle_ABC": int(answer_b),
            },
            reasoning_steps=2,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(answer_b), build=build)


def make_parallel_supplement_case(given: int, *, parallel_line_count: int = 2) -> AngleRelationCase:
    """Construct a reusable parallel-line supplement scene case."""

    answer = supplement_angle_value(given)
    line_count = int(parallel_line_count)
    if line_count not in (2, 3):
        raise ValueError("parallel supplement line count must be 2 or 3")

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render parallel-line geometry without choosing task identity."""

        top_l, top_r = (135.0, 170.0), (585.0, 170.0)
        bot_l, bot_r = (105.0, 385.0), (615.0, 385.0)
        mid_l, mid_r = (120.0, 278.0), (600.0, 278.0)
        x_top = 300.0
        vertical_gap = float(bot_l[1] - top_l[1])
        target_theta = math.radians(float(answer))
        tan_theta = math.tan(target_theta)
        if abs(tan_theta) < 0.25:
            tan_theta = 0.25 if tan_theta >= 0.0 else -0.25
        x_bot = x_top + (vertical_gap / tan_theta)
        p = (x_top, top_l[1])
        q = (x_bot, bot_l[1])
        top_l, top_r, mid_l, mid_r, bot_l, bot_r, p, q = _offset_points(
            ctx,
            (top_l, top_r, mid_l, mid_r, bot_l, bot_r, p, q),
        )
        _draw_polyline(ctx, [top_l, top_r])
        if line_count == 3:
            _draw_polyline(ctx, [mid_l, mid_r])
        _draw_polyline(ctx, [bot_l, bot_r])
        _draw_polyline(ctx, [p, q])
        mark_segments = ((top_l, top_r), (mid_l, mid_r), (bot_l, bot_r)) if line_count == 3 else ((top_l, top_r), (bot_l, bot_r))
        mark_bbox = _draw_parallel_arrow_marks(ctx, mark_segments)
        labels = _draw_point_labels(ctx, {"A": top_l, "B": top_r, "C": bot_l, "D": bot_r, "E": p, "F": q})
        given_arc, given_bbox = _draw_angle_label(ctx, format_degrees(given), p, top_l, q, radius=64.0)
        target_arc, target_bbox = _draw_angle_label(ctx, "?", q, bot_l, p, radius=64.0)
        annotation_points = {
            ANGLE_CFE: _angle_annotation_point(q, bot_l, p, label_radius=64.0),
            ANGLE_AEF: _angle_annotation_point(p, top_l, q, label_radius=64.0),
        }
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(answer),
            annotation_bboxes=(target_arc, given_arc),
            annotation_roles=(ANGLE_CFE, ANGLE_AEF),
            scene_entities=(
                {
                    "type": "parallel_lines",
                    "line_count": int(line_count),
                    "segments": {
                        "AB": (top_l, top_r),
                        **({"middle_unlabeled": (mid_l, mid_r)} if line_count == 3 else {}),
                        "CD": (bot_l, bot_r),
                    },
                },
                {"type": "transversal", "segment": (p, q), "points": {"E": p, "F": q}},
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {ANGLE_CFE: target_arc, ANGLE_AEF: given_arc},
                "angle_label_bboxes": {ANGLE_CFE: target_bbox, ANGLE_AEF: given_bbox},
                "parallel_marks_bbox": mark_bbox,
            },
            witness={
                "given_angle_AEF": int(given),
                "parallel_lines": ["AB", "CD"],
                "parallel_line_count": int(line_count),
                "answer_angle_CFE": int(answer),
            },
            reasoning_steps=2,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(answer), build=build)


def make_parallel_algebraic_case(
    *,
    x_value: int,
    first_coeff: int,
    first_const: int,
    second_coeff: int,
    second_const: int,
) -> AngleRelationCase:
    """Construct one algebraic same-side angle case on three parallel lines."""

    first_angle = linear_expression_value(first_coeff, first_const, x_value)
    second_angle = linear_expression_value(second_coeff, second_const, x_value)
    if int(first_angle) + int(second_angle) != 180:
        raise ValueError("parallel algebraic expressions must be supplementary")
    if not 35 <= int(first_angle) <= 80:
        raise ValueError("first expression angle must be the acute angle")
    if not 100 <= int(second_angle) <= 145:
        raise ValueError("second expression angle must be the obtuse target angle")

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render the algebraic parallel-line case without task identity."""

        top_l, top_r = (80.0, 116.0), (640.0, 116.0)
        mid_l, mid_r = (80.0, 286.0), (640.0, 286.0)
        bot_l, bot_r = (80.0, 456.0), (640.0, 456.0)
        p_x = 190.0
        vertical_gap = float(bot_l[1] - top_l[1])
        r_x = p_x + (vertical_gap / max(0.25, math.tan(math.radians(float(first_angle)))))
        p = (p_x, top_l[1])
        r = (r_x, bot_l[1])
        mid_t = (mid_l[1] - top_l[1]) / (bot_l[1] - top_l[1])
        q = (p[0] + ((r[0] - p[0]) * mid_t), mid_l[1])
        top_l, top_r, mid_l, mid_r, bot_l, bot_r, p, q, r = _offset_points(
            ctx,
            (top_l, top_r, mid_l, mid_r, bot_l, bot_r, p, q, r),
        )
        for segment in ((top_l, top_r), (mid_l, mid_r), (bot_l, bot_r)):
            _draw_polyline(ctx, [segment[0], segment[1]])
        _draw_polyline(ctx, [p, r])
        mark_bbox = _draw_parallel_arrow_marks(ctx, ((top_l, top_r), (mid_l, mid_r), (bot_l, bot_r)))
        labels = _draw_point_labels(
            ctx,
            {
                "A": top_l,
                "B": top_r,
                "C": mid_l,
                "D": mid_r,
                "E": bot_l,
                "F": bot_r,
                "P": p,
                "Q": q,
                "R": r,
            },
            offsets={
                "P": (0.0, -26.0),
                "Q": _line_local_label_offset(mid_l, mid_r, along_scale=-24.0, normal_scale=24.0),
                "R": (0.0, 24.0),
            },
        )
        first_expr = format_angle_expression(first_coeff, first_const)
        second_expr = format_angle_expression(second_coeff, second_const)
        first_arc = _draw_angle_arc(ctx, p, top_r, q, radius=46.0)
        second_arc = _draw_angle_arc(ctx, q, mid_r, p, radius=58.0)
        first_bbox = _draw_text(
            ctx,
            first_expr,
            _angle_local_label_center(p, top_r, q, primary_scale=92.0, secondary_scale=34.0),
        )
        second_bbox = _draw_text(
            ctx,
            second_expr,
            _angle_local_label_center(q, mid_r, p, primary_scale=88.0, secondary_scale=74.0),
        )
        target_arc, target_bbox = _draw_angle_label(ctx, "?", r, bot_r, q, radius=86.0)
        annotation_points = {"BPQ": p, "DQP": q, "FRQ": r}
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(second_angle),
            annotation_bboxes=(first_arc, second_arc, target_arc),
            annotation_roles=("BPQ", "DQP", "FRQ"),
            scene_entities=(
                {
                    "type": "parallel_lines",
                    "line_count": 3,
                    "segments": {"top": (top_l, top_r), "middle": (mid_l, mid_r), "bottom": (bot_l, bot_r)},
                },
                {
                    "type": "transversal",
                    "segment": (p, r),
                    "points": {"P": p, "Q": q, "R": r},
                },
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {"BPQ": first_arc, "DQP": second_arc, "FRQ": target_arc},
                "angle_label_bboxes": {"BPQ": first_bbox, "DQP": second_bbox, "FRQ": target_bbox},
                "parallel_marks_bbox": mark_bbox,
                "intersections": {"P": p, "Q": q, "R": r},
            },
            witness={
                "parallel_line_count": 3,
                "transversal_count": 1,
                "relation_id": "same_side_supplementary_expression_pair",
                "expression_angle_names": ["BPQ", "DQP"],
                "expression_angle_values": [int(first_angle), int(second_angle)],
                "target_angle_name": "FRQ",
                "target_angle_measure": int(second_angle),
                "x": int(x_value),
                "first_expression": format_linear_expression(first_coeff, first_const),
                "second_expression": format_linear_expression(second_coeff, second_const),
                "equation": "BPQ + DQP = 180; FRQ = DQP",
            },
            reasoning_steps=3,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(second_angle), build=build)


def make_parallel_transversal_triangle_case(left_angle: int, right_angle: int) -> AngleRelationCase:
    """Construct an opposite-triangle angle case between parallel lines."""

    answer = 180 - int(left_angle) - int(right_angle)
    if answer <= 0:
        raise ValueError("parallel transversal triangle angle must be positive")

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render support base angles on the triangle opposite the target angle."""

        top_y = 142.0
        target_y = 286.0
        bottom_y = 430.0
        target_x = ctx.width / 2.0
        q = (float(target_x), float(target_y))
        lower_height = float(bottom_y - target_y)
        top_scale = float(target_y - top_y) / max(1.0, lower_height)
        left_dx = lower_height / math.tan(math.radians(float(left_angle)))
        right_dx = lower_height / math.tan(math.radians(float(right_angle)))
        s = (float(target_x - left_dx), float(bottom_y))
        t = (float(target_x + right_dx), float(bottom_y))
        p = _add(q, _sub(q, t), top_scale)
        r = _add(q, _sub(q, s), top_scale)
        top_l, top_r = (88.0, top_y), (ctx.width - 88.0, top_y)
        bot_l, bot_r = (88.0, bottom_y), (ctx.width - 88.0, bottom_y)
        top_l, top_r, bot_l, bot_r, p, q, r, s, t = _offset_points(
            ctx,
            (top_l, top_r, bot_l, bot_r, p, q, r, s, t),
        )
        _draw_polyline(ctx, [top_l, top_r])
        _draw_polyline(ctx, [bot_l, bot_r])
        _draw_polyline(ctx, [p, t])
        _draw_polyline(ctx, [r, s])
        mark_bbox = _draw_parallel_arrow_marks(ctx, ((top_l, top_r), (bot_l, bot_r)))
        labels = _draw_point_labels(
            ctx,
            {"P": p, "Q": q, "R": r, "S": s, "T": t},
            offsets={"P": (0.0, -30.0), "R": (0.0, -30.0)},
        )
        left_arc, left_bbox = _draw_angle_label(
            ctx,
            format_degrees(left_angle),
            s,
            t,
            q,
            radius=64.0,
        )
        right_arc, right_bbox = _draw_angle_label(
            ctx,
            format_degrees(right_angle),
            t,
            q,
            s,
            radius=64.0,
        )
        target_arc, target_bbox = _draw_angle_label(ctx, "?", q, p, r, radius=68.0)
        annotation_points = {POINT_P: p, POINT_Q: q, POINT_R: r, "S": s, "T": t}
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(answer),
            annotation_bboxes=(target_arc, left_arc, right_arc),
            annotation_roles=(POINT_P, POINT_Q, POINT_R, "S", "T"),
            scene_entities=(
                {"type": "parallel_lines", "line_count": 2, "segments": {"top": (top_l, top_r), "bottom": (bot_l, bot_r)}},
                {
                    "type": "transversal_pair",
                    "segments": {"left": (p, t), "right": (r, s)},
                    "points": {"P": p, "Q": q, "R": r, "S": s, "T": t},
                },
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {
                    "target_angle": target_arc,
                    "left_base_support_angle": left_arc,
                    "right_base_support_angle": right_arc,
                },
                "angle_label_bboxes": {
                    "target_angle": target_bbox,
                    "left_base_support_angle": left_bbox,
                    "right_base_support_angle": right_bbox,
                },
                "parallel_marks_bbox": mark_bbox,
                "intersections": {"P": p, "Q": q, "R": r, "S": s, "T": t},
            },
            witness={
                "parallel_line_count": 2,
                "transversal_count": 2,
                "relation_id": "parallel_transversal_lower_triangle_sum_then_vertical_angle",
                "displayed_lower_triangle_base_angles": [int(left_angle), int(right_angle)],
                "answer_angle_PQR": int(answer),
                "equation": (
                    "target_angle = vertical_opposite_angle = "
                    "180 - left_lower_triangle_base_angle - right_lower_triangle_base_angle"
                ),
            },
            reasoning_steps=3,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(answer), build=build)


def make_algebraic_single_extension_case(
    given_angle_a: int,
    x_value: int,
    target_coeff: int,
    target_const: int,
    exterior_coeff: int,
) -> AngleRelationCase:
    """Construct a reusable single-extension algebraic angle case."""

    answer_angle_b = linear_expression_value(target_coeff, target_const, x_value)
    exterior_c = exterior_angle_value(given_angle_a, answer_angle_b)
    exterior_const = int(exterior_c) - (int(exterior_coeff) * int(x_value))
    angle_c = 180 - int(given_angle_a) - int(answer_angle_b)

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render single-extension algebraic geometry for task-owned solving."""

        a, b, c = _triangle_from_base_angles(
            left_angle=int(given_angle_a),
            right_angle=int(angle_c),
            width=380.0,
            canvas_width=float(ctx.width),
            canvas_height=float(ctx.height),
        )
        d = (min(ctx.width - 54.0, c[0] + 92.0), c[1])
        a, b, c, d = _offset_points(ctx, (a, b, c, d))
        _draw_polygon(ctx, [a, b, c])
        _draw_polyline(ctx, [c, d])
        labels = _draw_point_labels(ctx, {"A": a, "B": b, "C": c, "D": d})
        target_expr = format_linear_expression(target_coeff, target_const)
        exterior_expr = format_linear_expression(exterior_coeff, exterior_const)
        target_arc = _draw_angle_arc(ctx, b, a, c, radius=42.0)
        given_arc, given_bbox = _draw_angle_label(ctx, format_degrees(given_angle_a), a, c, b, radius=64.0)
        exterior_arc = _draw_angle_arc(ctx, c, b, d, radius=52.0)
        target_bbox = _draw_text(ctx, target_expr, _angle_label_center(b, a, c, radius=66.0))
        exterior_bbox = _draw_text(ctx, exterior_expr, _angle_label_center(c, b, d, radius=76.0))
        annotation_points = {POINT_A: a, POINT_B: b, POINT_C: c, POINT_D: d}
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(answer_angle_b),
            annotation_bboxes=(target_arc, given_arc, exterior_arc),
            annotation_roles=(POINT_A, POINT_B, POINT_C, POINT_D),
            scene_entities=(
                {"type": "triangle", "points": {"A": a, "B": b, "C": c}},
                {"type": "extension_ray", "points": {"C": c, "D": d}},
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {ANGLE_ABC: target_arc, ANGLE_BAC: given_arc, ANGLE_BCD: exterior_arc},
                "angle_label_bboxes": {ANGLE_ABC: target_bbox, ANGLE_BAC: given_bbox, ANGLE_BCD: exterior_bbox},
            },
            witness={
                "equation": "exterior_BCD = angle_BAC + angle_ABC",
                "angle_BAC": int(given_angle_a),
                "x": int(x_value),
                "expression_angle_ABC": format_linear_expression(target_coeff, target_const),
                "expression_exterior_BCD": format_linear_expression(exterior_coeff, exterior_const),
                "answer_angle_ABC": int(answer_angle_b),
            },
            reasoning_steps=3,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(answer_angle_b), build=build)


def make_algebraic_double_extension_case(
    given_angle_a: int,
    x_value: int,
    target_coeff: int,
    target_const: int,
    exterior_coeff: int,
) -> AngleRelationCase:
    """Construct a reusable double-extension algebraic angle case."""

    answer_angle_b = linear_expression_value(target_coeff, target_const, x_value)
    exterior_c = exterior_angle_value(given_angle_a, answer_angle_b)
    exterior_const = int(exterior_c) - (int(exterior_coeff) * int(x_value))
    angle_c = 180 - int(given_angle_a) - int(answer_angle_b)

    def build(ctx: RenderContext) -> RenderedAngleRelationScene:
        """Render double-extension algebraic geometry for task-owned solving."""

        a, b, c = _triangle_from_base_angles(
            left_angle=int(given_angle_a),
            right_angle=int(angle_c),
            width=380.0,
            canvas_width=float(ctx.width),
            canvas_height=float(ctx.height),
        )
        d = (min(ctx.width - 54.0, c[0] + 92.0), c[1])
        e = (max(54.0, a[0] - 92.0), a[1])
        a, b, c, d, e = _offset_points(ctx, (a, b, c, d, e))
        _draw_polyline(ctx, [e, a])
        _draw_polygon(ctx, [a, b, c])
        _draw_polyline(ctx, [c, d])
        labels = _draw_point_labels(ctx, {"E": e, "A": a, "B": b, "C": c, "D": d})
        target_expr = format_linear_expression(target_coeff, target_const)
        exterior_expr = format_linear_expression(exterior_coeff, exterior_const)
        target_arc = _draw_angle_arc(ctx, b, a, c, radius=42.0)
        given_arc, given_bbox = _draw_angle_label(ctx, format_degrees(given_angle_a), a, c, b, radius=64.0)
        exterior_c_arc = _draw_angle_arc(ctx, c, b, d, radius=52.0)
        target_bbox = _draw_text(ctx, target_expr, _angle_label_center(b, a, c, radius=66.0))
        exterior_c_bbox = _draw_text(ctx, exterior_expr, _angle_label_center(c, b, d, radius=76.0))
        annotation_points = {POINT_A: a, POINT_B: b, POINT_C: c, POINT_D: d}
        return RenderedAngleRelationScene(
            image=ctx.image,
            answer=int(answer_angle_b),
            annotation_bboxes=(target_arc, given_arc, exterior_c_arc),
            annotation_roles=(POINT_A, POINT_B, POINT_C, POINT_D),
            scene_entities=(
                {"type": "triangle", "points": {"A": a, "B": b, "C": c}},
                {"type": "extension_ray", "points": {"A": a, "E": e}},
                {"type": "extension_ray", "points": {"C": c, "D": d}},
            ),
            render_map={
                "point_label_bboxes": labels,
                "angle_arc_bboxes": {ANGLE_ABC: target_arc, ANGLE_BAC: given_arc, ANGLE_BCD: exterior_c_arc},
                "angle_label_bboxes": {ANGLE_ABC: target_bbox, ANGLE_BAC: given_bbox, ANGLE_BCD: exterior_c_bbox},
            },
            witness={
                "equation": "exterior_BCD = angle_BAC + angle_ABC",
                "angle_BAC": int(given_angle_a),
                "x": int(x_value),
                "expression_angle_ABC": format_linear_expression(target_coeff, target_const),
                "expression_exterior_BCD": format_linear_expression(exterior_coeff, exterior_const),
                "answer_angle_ABC": int(answer_angle_b),
            },
            reasoning_steps=3,
            annotation_keyed_points=annotation_points,
        )

    return AngleRelationCase(answer=int(answer_angle_b), build=build)


__all__ = [
    "make_algebraic_double_extension_case",
    "make_algebraic_single_extension_case",
    "make_parallel_algebraic_case",
    "make_parallel_supplement_case",
    "make_parallel_transversal_triangle_case",
    "make_triangle_exterior_case",
]
