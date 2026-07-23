"""Scene-private prompt and trace assembly for graph-paper public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isclose, sqrt
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    bbox_set_artifacts,
    point_set_artifacts,
    scalar_bbox_artifacts,
    scalar_point_artifacts,
    scalar_segment_artifacts,
)
from .shared.construction import (
    pi_expression,
    polygon_area,
    polygon_perimeter,
)
from .shared.defaults import int_default, split_defaults_for
from .shared.prompts import prompt_defaults as resolve_prompt_defaults
from .shared.prompts import render_prompt_artifacts
from .shared.rendering import (
    angle_points,
    draw_angle,
    draw_ellipse_or_circle,
    draw_measurement_guide,
    draw_polygon,
    draw_segment,
    draw_vertex_labels,
    make_context,
    object_color,
    random_center_for_radii,
    random_shift_points,
    render_metadata,
    slot_centers,
)
from .shared.sampling import (
    choose_from_seed,
    count_target,
    label_subset,
    make_class_sequence,
    reduced_slope,
    resolve_count,
    rng_for,
    unique_metric_values,
)
from .shared.state import GraphObject, GraphPaperContext, Point, POINT_LABELS, PromptPlan, SCENE_ID


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class GraphPaperComponents:
    """Prompt, image, annotation, and trace sections before final TaskOutput."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_type: str
    annotation_value: Any
    image: Any
    trace_payload: Mapping[str, Any]


@dataclass(frozen=True)
class GraphPaperTaskPlan:
    """Public-file semantic plan consumed by scene-private lifecycle plumbing."""

    builder: Callable[
        [Mapping[str, Any], "GraphPaperTaskPlan"],
        tuple[GraphPaperComponents, TypedValue],
    ]
    prompt_key: str
    salt: str
    default_branch: str = "single"
    value_param: str = ""
    target_field: str = ""
    prompt_keys_by_branch: Mapping[str, str] = field(default_factory=dict)
    role_by_branch: Mapping[str, str] = field(default_factory=dict)
    target_class_by_branch: Mapping[str, str] = field(default_factory=dict)
    target_text_by_branch: Mapping[str, str] = field(default_factory=dict)

    def prompt_key_for(self, branch_name: str) -> str:
        """Return the public-file prompt key for a selected branch."""

        if self.prompt_keys_by_branch:
            return str(self.prompt_keys_by_branch[str(branch_name)])
        return str(self.prompt_key)

    def role_for(self, branch_name: str) -> str:
        """Return a non-public lifecycle role for a selected branch."""

        if self.role_by_branch:
            return str(self.role_by_branch[str(branch_name)])
        return str(branch_name)

    def target_class_for(self, branch_name: str, fallback: str) -> str:
        """Return the semantic target class owned by the selected branch."""

        if self.target_class_by_branch:
            return str(self.target_class_by_branch[str(branch_name)])
        return str(fallback)

    def target_text_for(self, branch_name: str, fallback: str) -> str:
        """Return the prompt-facing target phrase for the selected branch."""

        if self.target_text_by_branch:
            return str(self.target_text_by_branch[str(branch_name)])
        return str(fallback)


@dataclass(frozen=True)
class _CountCandidate:
    """One count-task visual candidate before it mutates the canvas."""

    kind: str
    class_name: str
    graph_points: tuple[Point, ...]
    radius_x: float = 0.0
    radius_y: float = 0.0
    metric_value: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)


def graph_paper_prompt_plan(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    answer_hint: str,
    annotation_hint: str,
    json_example: str,
    json_example_answer_only: str,
    shape_text: str = "",
    metric_text: str = "",
    target_text: str = "",
) -> PromptPlan:
    """Create one prompt plan from config defaults and task-owned dynamic slots."""

    bundle_id, scene_key, task_key = resolve_prompt_defaults(prompt_defaults)
    return PromptPlan(
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=str(task_key),
        prompt_key=str(prompt_key),
        answer_hint=str(answer_hint),
        annotation_hint=str(annotation_hint),
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        shape_text=str(shape_text),
        metric_text=str(metric_text),
        target_text=str(target_text),
    )


def object_trace(entity: GraphObject) -> dict[str, Any]:
    """Serialize one graph object for scene trace metadata."""

    payload: dict[str, Any] = {
        "label": str(entity.label),
        "kind": str(entity.kind),
        "class_name": str(entity.class_name),
        "bbox_px": [round(float(value), 3) for value in entity.bbox_px],
        "points_px": [
            [round(float(coord), 3) for coord in point] for point in entity.points_px
        ],
        "metric_value": round(float(entity.metric_value), 6),
    }
    if entity.graph_points:
        payload["graph_points"] = [
            [round(float(coord), 3) for coord in point] for point in entity.graph_points
        ]
    if entity.extra:
        payload["extra"] = dict(entity.extra)
    return payload


def build_graph_paper_components(
    *,
    ctx: GraphPaperContext,
    prompt_plan: PromptPlan,
    instance_seed: int,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    task_name: str,
    answer_type: str,
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    objects: Sequence[GraphObject],
    prompt_key: str,
    program_code: str,
    scene_kind: str,
    semantic_args: Mapping[str, Any],
    task_params: Mapping[str, Any],
    render_map: Mapping[str, Any] | None = None,
) -> GraphPaperComponents:
    """Assemble common graph-paper prompt and trace payload sections."""

    prompt_artifacts = render_prompt_artifacts(
        plan=prompt_plan, instance_seed=int(instance_seed)
    )
    probabilities = {
        str(key): float(value) for key, value in branch_probabilities.items()
    }
    safe_semantic_args = _json_safe(dict(semantic_args))
    safe_task_params = _json_safe(
        {
            str(key): value
            for key, value in task_params.items()
            if not str(key).startswith("_")
        }
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_name),
        params={
            "scene_id": SCENE_ID,
            "query_id_probabilities": dict(probabilities),
            "prompt_key": str(prompt_key),
            "program_code": str(program_code),
            **safe_semantic_args,
        },
    )
    render_spec = _json_safe(render_metadata(ctx))
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "scene_kind": str(scene_kind),
            "entities": [_json_safe(object_trace(entity)) for entity in objects],
            "relations": {
                "answer_value": answer_value,
                "answer_type": str(answer_type),
                "annotation_type": str(annotation_type),
                "program_code": str(program_code),
                **safe_semantic_args,
            },
        },
        "query_spec": query_spec,
        "render_spec": render_spec,
        "render_map": _json_safe(dict(render_map or {})),
        "projected_annotation": _json_safe(dict(projected_annotation)),
        "witness_symbolic": _json_safe(dict(witness_symbolic)),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "query_id_probabilities": dict(probabilities),
            "prompt_key": str(prompt_key),
            "program_code": str(program_code),
            "answer_type": str(answer_type),
            "answer_value": answer_value,
            "annotation_type": str(annotation_type),
            "task_params": safe_task_params,
            **safe_semantic_args,
        },
    }
    return GraphPaperComponents(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_type=str(annotation_type),
        annotation_value=annotation_value,
        image=ctx.image,
        trace_payload=trace_payload,
    )


def _final_output(
    *,
    components: GraphPaperComponents,
    answer_gt: TypedValue,
    branch_name: str,
) -> TaskOutput:
    """Build the final task output once a public entrypoint has selected semantics."""

    return TaskOutput(
        prompt=str(components.prompt),
        answer_gt=answer_gt,
        annotation_gt=TypedValue(
            type=str(components.annotation_type), value=components.annotation_value
        ),
        image=components.image,
        image_id="img_0",
        trace_payload=dict(components.trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_name),
        prompt_variants=dict(components.prompt_variants),
    )


def _select_branch(
    task: Any, instance_seed: int, params: Mapping[str, Any], default_branch: str
):
    """Resolve the public branch using the task-owned supported branch tuple."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(default_branch),
        task_id=str(task.task_id),
    )


def _task_defaults(
    task: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve scene defaults for the public entrypoint."""

    return split_defaults_for(str(task.task_id))


def _make_prompt(
    prompt_defaults: Mapping[str, Any],
    *,
    prompt_key: str,
    answer_hint: str,
    annotation_hint: str,
    json_example: str,
    json_example_answer_only: str,
    shape_text: str = "",
    metric_text: str = "",
    target_text: str = "",
) -> PromptPlan:
    """Bind task-specific prompt slots without embedding prompt templates in source."""

    return graph_paper_prompt_plan(
        prompt_defaults=prompt_defaults,
        prompt_key=str(prompt_key),
        answer_hint=str(answer_hint),
        annotation_hint=str(annotation_hint),
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        shape_text=str(shape_text),
        metric_text=str(metric_text),
        target_text=str(target_text),
    )


def run_graph_paper_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: GraphPaperTaskPlan,
) -> TaskOutput:
    """Run one graph-paper task from the semantic plan owned by the public file."""

    del max_attempts
    branch_name, branch_probabilities, task_params = _select_branch(
        task,
        int(instance_seed),
        params,
        str(plan.default_branch),
    )
    generation_defaults, rendering_defaults, prompt_defaults = _task_defaults(task)
    context = {
        "instance_seed": int(instance_seed),
        "branch_name": str(branch_name),
        "branch_probabilities": dict(branch_probabilities),
        "task_params": dict(task_params),
        "generation_defaults": generation_defaults,
        "rendering_defaults": rendering_defaults,
        "prompt_defaults": prompt_defaults,
    }
    components, answer_gt = plan.builder(context, plan)
    return _final_output(
        components=components, answer_gt=answer_gt, branch_name=str(branch_name)
    )


def _new_context(
    context: Mapping[str, Any], salt: str
) -> tuple[Any, GraphPaperContext]:
    """Create deterministic RNG and graph-paper context for one semantic role."""

    rng = rng_for(int(context["instance_seed"]), str(salt))
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=dict(context["task_params"]),
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )
    return rng, ctx


def _component_payload(
    context: Mapping[str, Any],
    *,
    ctx: GraphPaperContext,
    prompt_plan: PromptPlan,
    answer_type: str,
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    objects: Sequence[GraphObject],
    prompt_key: str,
    program_code: str,
    scene_kind: str,
    semantic_args: Mapping[str, Any],
) -> tuple[GraphPaperComponents, TypedValue]:
    """Create components plus the typed answer for one graph-paper objective."""

    components = build_graph_paper_components(
        ctx=ctx,
        prompt_plan=prompt_plan,
        instance_seed=int(context["instance_seed"]),
        branch_name=str(context["branch_name"]),
        branch_probabilities=context["branch_probabilities"],
        task_name="",
        answer_type=str(answer_type),
        answer_value=answer_value,
        annotation_type=str(annotation_type),
        annotation_value=annotation_value,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        objects=objects,
        prompt_key=str(prompt_key),
        program_code=str(program_code),
        scene_kind=str(scene_kind),
        semantic_args=semantic_args,
        task_params=context["task_params"],
    )
    return components, TypedValue(type=str(answer_type), value=answer_value)


def _integer_shift_points(
    ctx: GraphPaperContext,
    points: Sequence[Point],
    rng: Any,
    *,
    margin_units: float = 1.0,
) -> tuple[Point, ...]:
    """Translate integer graph points by an integer in-bounds offset."""

    return random_shift_points(
        ctx,
        points,
        rng,
        margin_units=float(margin_units),
        step=1.0,
    )


def _corner_rectangle_points(width: int, height: int) -> tuple[Point, ...]:
    """Return integer-lattice rectangle vertices from the origin corner."""

    return (
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    )


def _corner_right_triangle_points(width: int, height: int) -> tuple[Point, ...]:
    """Return integer-lattice right-triangle vertices from the origin corner."""

    return ((0.0, 0.0), (float(width), 0.0), (0.0, float(height)))


def _corner_triangle_points(
    base_width: int,
    apex_x: int,
    height: int,
) -> tuple[Point, ...]:
    """Return a non-rotated lattice triangle with a horizontal base."""

    return ((0.0, 0.0), (float(base_width), 0.0), (float(apex_x), float(height)))


def _corner_parallelogram_points(
    base_width: int,
    side_dx: int,
    side_dy: int,
) -> tuple[Point, ...]:
    """Return integer-lattice slanted parallelogram vertices from the origin."""

    return (
        (0.0, 0.0),
        (float(base_width), 0.0),
        (float(base_width + side_dx), float(side_dy)),
        (float(side_dx), float(side_dy)),
    )


def _translated_corner_shape(
    *,
    left: int,
    bottom: int,
    width: int,
    height: int,
    shape_kind: str,
) -> tuple[Point, ...]:
    """Return an integer-lattice rectangle or right triangle near a slot."""

    base = (
        _corner_right_triangle_points(width, height)
        if str(shape_kind) == "right_triangle"
        else _corner_rectangle_points(width, height)
    )
    return tuple(
        (float(point[0]) + float(left), float(point[1]) + float(bottom))
        for point in base
    )


def _slot_anchor(
    ctx: GraphPaperContext,
    center: Point,
    *,
    width: int,
    height: int,
    margin_units: float = 1.0,
) -> tuple[int, int]:
    """Snap a slot center to an integer-lattice lower-left anchor in bounds."""

    limit = int(float(ctx.graph_half_range) - float(margin_units))
    lower = -limit
    left = int(round(float(center[0]) - (float(width) / 2.0)))
    bottom = int(round(float(center[1]) - (float(height) / 2.0)))
    left = max(lower, min(limit - int(width), left))
    bottom = max(lower, min(limit - int(height), bottom))
    return left, bottom


def _normalize_points_to_origin(points: Sequence[Point]) -> tuple[Point, ...]:
    """Translate graph points so the lower-left bbox corner is at the origin."""

    min_x = min(float(point[0]) for point in points)
    min_y = min(float(point[1]) for point in points)
    return tuple((float(point[0]) - min_x, float(point[1]) - min_y) for point in points)


def _integer_perimeter_triangle_candidates() -> tuple[tuple[Point, Point, Point], ...]:
    """Return small lattice triangles whose side lengths and perimeter are integer."""

    candidates: dict[
        tuple[tuple[Point, ...], tuple[int, int, int]], tuple[int, tuple[Point, ...]]
    ] = {}
    grid_points = [(x, y) for x in range(0, 9) for y in range(0, 9)]
    for first_index, first in enumerate(grid_points):
        for second_index in range(first_index + 1, len(grid_points)):
            second = grid_points[second_index]
            for third in grid_points[second_index + 1 :]:
                raw_points = (first, second, third)
                area2 = abs(
                    (second[0] - first[0]) * (third[1] - first[1])
                    - (second[1] - first[1]) * (third[0] - first[0])
                )
                if area2 == 0:
                    continue
                sides = []
                for start, end in ((first, second), (second, third), (third, first)):
                    length = sqrt(
                        float(start[0] - end[0]) ** 2 + float(start[1] - end[1]) ** 2
                    )
                    if not isclose(length, round(length), abs_tol=1e-9):
                        break
                    sides.append(int(round(length)))
                else:
                    ordered = _normalize_points_to_origin(raw_points)
                    width = max(point[0] for point in ordered)
                    height = max(point[1] for point in ordered)
                    if width <= 8.0 and height <= 8.0:
                        key = (
                            tuple(sorted(ordered)),
                            tuple(sorted(sides)),
                        )
                        candidates[key] = (int(sum(sides)), ordered)
    return tuple(
        ordered
        for _key, (_perimeter, ordered) in sorted(
            candidates.items(), key=lambda item: (item[1][0], item[0][1], item[0][0])
        )
    )


_INTEGER_PERIMETER_TRIANGLES: tuple[tuple[Point, Point, Point], ...] = (
    _integer_perimeter_triangle_candidates()
)


def _normalize_polygon_measurement_shape(value: Any) -> str:
    """Return the public shape family for single-polygon measurement tasks."""

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"right_triangle", "right_triangle_3_4_5"}:
        return "triangle"
    if text in {"rectangle", "triangle", "parallelogram"}:
        return str(text)
    return "rectangle"


def _is_right_triangle_lattice(base_width: int, apex_x: int, height: int) -> bool:
    """Return whether the sampled horizontal-base triangle is right angled."""

    base = int(base_width)
    x = int(apex_x)
    h = int(height)
    if x <= 0 or x >= base:
        return True
    return int(h * h) == int(x * (base - x))


def _sample_area_triangle_points(
    rng: Any, task_params: Mapping[str, Any]
) -> tuple[Point, ...]:
    """Sample a non-right lattice triangle with integer area."""

    explicit_shape_params = {
        "base_width",
        "apex_x",
        "height",
    } <= set(str(key) for key in task_params)
    for _ in range(50):
        base_width = int(task_params.get("base_width", rng.randint(4, 8)))
        height = int(task_params.get("height", rng.randint(3, 6)))
        apex_x = int(task_params.get("apex_x", rng.randint(1, max(1, base_width - 1))))
        if (
            0 < apex_x < base_width
            and (base_width * height) % 2 == 0
            and not _is_right_triangle_lattice(base_width, apex_x, height)
        ):
            return _corner_triangle_points(base_width, apex_x, height)
        if explicit_shape_params:
            break
    return _corner_triangle_points(6, 2, 4)


def _build_line_slope_value(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build and bind one segment slope measurement from grid rise/run."""

    task_params = dict(context["task_params"])
    rng, ctx = _new_context(context, plan.salt)
    dx = int(task_params.get("dx", rng.choice([2, 3, 4, 5, 6])))
    dy = int(task_params.get("dy", rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])))
    if dy == 0:
        dy = 2
    reduced_dy, reduced_dx = reduced_slope(dy, dx)
    start, end = _integer_shift_points(
        ctx,
        ((0.0, 0.0), (float(dx), float(dy))),
        rng,
        margin_units=1.2,
    )
    segment = draw_segment(ctx, "", start, end, color=ctx.accent_color)
    annotation_value, projected = scalar_segment_artifacts(
        segment.points_px[0], segment.points_px[1]
    )
    answer_value = round(float(dy) / float(dx), 1)
    if answer_value == -0.0:
        answer_value = 0.0
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the slope as a number rounded to one decimal place',
        annotation_hint='set "annotation" to the line segment as [[x0,y0],[x1,y1]] in pixels',
        json_example='{"annotation":[[210,420],[460,250]],"answer":-1.5}',
        json_example_answer_only='{"answer":-1.5}',
        target_text="the line segment",
        metric_text="slope",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="number",
        answer_value=float(answer_value),
        annotation_type="segment",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "segment": [start, end],
            "reduced_slope": [int(reduced_dy), int(reduced_dx)],
        },
        objects=(segment,),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="single_segment.slope_value",
        scene_kind="geometry_graph_paper_single_segment",
        semantic_args={"dx": int(dx), "dy": int(dy), "answer_rounding": "one_decimal"},
    )


def _build_circle_circumference_value(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build the exact circle circumference objective."""

    task_params = dict(context["task_params"])
    rng, ctx = _new_context(context, plan.salt)
    radius = int(task_params.get("radius", rng.randint(2, 5)))
    answer_value = pi_expression(2 * radius)
    center = random_center_for_radii(
        ctx, radius, radius, rng, margin_units=1.2, step=1.0
    )
    radius_endpoint = (float(center[0]) + float(radius), float(center[1]))
    circle = draw_ellipse_or_circle(
        ctx,
        "",
        center,
        radius,
        radius,
        class_name="circle",
        color=ctx.accent_color,
        filled=False,
    )
    draw_measurement_guide(ctx, center, radius_endpoint)
    annotation_value, projected = scalar_bbox_artifacts(circle.bbox_px)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the exact circumference using π',
        annotation_hint='set "annotation" to the circle bounding box [x0,y0,x1,y1] in pixels',
        json_example='{"annotation":[180,180,540,540],"answer":"8π"}',
        json_example_answer_only='{"answer":"8π"}',
        target_text="the circle",
        metric_text="circumference",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="string",
        answer_value=str(answer_value),
        annotation_type="bbox",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "center": center,
            "radius_endpoint": radius_endpoint,
            "radius_units": int(radius),
        },
        objects=(circle,),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="circle.radius_to_circumference_exact_pi",
        scene_kind="geometry_graph_paper_single_circle",
        semantic_args={"radius_units": int(radius)},
    )


def _build_ellipse_area_value(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build the exact ellipse area objective."""

    task_params = dict(context["task_params"])
    rng, ctx = _new_context(context, plan.salt)
    if "radius_x" in task_params and "radius_y" in task_params:
        radius_x = int(task_params["radius_x"])
        radius_y = int(task_params["radius_y"])
    else:
        candidate_pairs = [
            (radius_x, radius_y)
            for radius_x in range(2, 7)
            for radius_y in range(1, 6)
            if radius_x != radius_y
        ]
        candidate_products = sorted(
            {radius_x * radius_y for radius_x, radius_y in candidate_pairs}
        )
        target_product = int(rng.choice(candidate_products))
        radius_x, radius_y = rng.choice(
            [(rx, ry) for rx, ry in candidate_pairs if rx * ry == target_product]
        )
    answer_value = pi_expression(radius_x * radius_y)
    center = random_center_for_radii(
        ctx, radius_x, radius_y, rng, margin_units=1.2, step=1.0
    )
    major_axis = (
        (float(center[0]) - float(radius_x), float(center[1])),
        (float(center[0]) + float(radius_x), float(center[1])),
    )
    minor_axis = (
        (float(center[0]), float(center[1]) - float(radius_y)),
        (float(center[0]), float(center[1]) + float(radius_y)),
    )
    ellipse = draw_ellipse_or_circle(
        ctx,
        "",
        center,
        radius_x,
        radius_y,
        class_name="ellipse",
        color=ctx.accent_color,
        filled=False,
    )
    annotation_value, projected = scalar_bbox_artifacts(ellipse.bbox_px)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the exact area using π',
        annotation_hint='set "annotation" to the ellipse bounding box [x0,y0,x1,y1] in pixels',
        json_example='{"annotation":[160,230,560,500],"answer":"12π"}',
        json_example_answer_only='{"answer":"12π"}',
        target_text="the ellipse",
        metric_text="area",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="string",
        answer_value=str(answer_value),
        annotation_type="bbox",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "center": center,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "radius_x_units": int(radius_x),
            "radius_y_units": int(radius_y),
        },
        objects=(ellipse,),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="ellipse.radii_to_area_exact_pi",
        scene_kind="geometry_graph_paper_single_ellipse",
        semantic_args={
            "radius_x_units": int(radius_x),
            "radius_y_units": int(radius_y),
        },
    )


def _single_polygon_points(
    context: Mapping[str, Any], *, salt: str, perimeter_mode: bool = False
):
    """Sample one graph-paper polygon for a measurement task."""

    task_params = dict(context["task_params"])
    rng = rng_for(int(context["instance_seed"]), str(salt))
    shapes = ("rectangle", "triangle", "parallelogram")
    shape_kind = _normalize_polygon_measurement_shape(
        task_params.get(
            "shape_kind",
            choose_from_seed(
                shapes, instance_seed=int(context["instance_seed"]), salt=str(salt)
            ),
        )
    )
    if shape_kind == "parallelogram":
        base_width = int(task_params.get("base_width", rng.randint(3, 6)))
        side_dx, side_dy = (3, 4) if int(rng.randrange(0, 2)) == 0 else (4, 3)
        if "side_dx" in task_params and "side_dy" in task_params:
            side_dx = int(task_params["side_dx"])
            side_dy = int(task_params["side_dy"])
        return (
            _corner_parallelogram_points(base_width, side_dx, side_dy),
            "parallelogram",
        )
    width = int(task_params.get("width", rng.randint(3, 6)))
    height = int(task_params.get("height", rng.randint(3, 6)))
    if shape_kind == "triangle":
        if perimeter_mode:
            return rng.choice(_INTEGER_PERIMETER_TRIANGLES), "triangle"
        return _sample_area_triangle_points(rng, task_params), "triangle"
    return _corner_rectangle_points(width, height), "rectangle"


def _build_polygon_area_value(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build and bind one lattice polygon area measurement objective."""

    rng, ctx = _new_context(context, plan.salt)
    points, shape_text = _single_polygon_points(context, salt="area_shape")
    points = _integer_shift_points(ctx, points, rng, margin_units=1.2)
    answer_value = int(round(polygon_area(points)))
    polygon = draw_polygon(
        ctx,
        "",
        points,
        class_name=shape_text.replace(" ", "_"),
        color=ctx.accent_color,
        filled=False,
    )
    annotation_value, projected = point_set_artifacts(polygon.points_px)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the integer area in square grid units',
        annotation_hint='set "annotation" to the polygon vertices as pixel points',
        json_example='{"annotation":[[250,420],[460,420],[250,270]],"answer":12}',
        json_example_answer_only='{"answer":12}',
        shape_text=shape_text,
        target_text="the polygon",
        metric_text="area",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="point_set",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={"shape_kind": shape_text, "vertices": points},
        objects=(polygon,),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="lattice_polygon.area_value",
        scene_kind="geometry_graph_paper_single_polygon",
        semantic_args={"shape_kind": shape_text},
    )


def _build_polygon_perimeter_value(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build and bind one lattice polygon perimeter measurement objective."""

    rng, ctx = _new_context(context, plan.salt)
    points, shape_text = _single_polygon_points(
        context, salt="perim_shape", perimeter_mode=True
    )
    points = _integer_shift_points(ctx, points, rng, margin_units=1.2)
    answer_value = int(round(polygon_perimeter(points)))
    polygon = draw_polygon(
        ctx,
        "",
        points,
        class_name=shape_text.replace(" ", "_"),
        color=ctx.accent_color,
        filled=False,
    )
    annotation_value, projected = point_set_artifacts(polygon.points_px)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the integer perimeter in grid units',
        annotation_hint='set "annotation" to the polygon vertices as pixel points',
        json_example='{"annotation":[[250,420],[460,420],[250,270]],"answer":12}',
        json_example_answer_only='{"answer":12}',
        shape_text=shape_text,
        target_text="the polygon",
        metric_text="perimeter",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="point_set",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={"shape_kind": shape_text, "vertices": points},
        objects=(polygon,),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="lattice_polygon.perimeter_value",
        scene_kind="geometry_graph_paper_single_polygon",
        semantic_args={"shape_kind": shape_text},
    )


def _right_angle_base_polygon(corner_count: int) -> tuple[Point, ...]:
    """Return a rectilinear lattice polygon with the requested corner count."""

    if int(corner_count) == 4:
        return ((0, 0), (8, 0), (8, 6), (0, 6))
    if int(corner_count) == 6:
        return ((0, 0), (8, 0), (8, 4), (5, 4), (5, 7), (0, 7))
    if int(corner_count) == 8:
        return ((0, 0), (9, 0), (9, 3), (6, 3), (6, 6), (3, 6), (3, 9), (0, 9))
    raise ValueError(f"unsupported base corner count: {corner_count}")


def _chamfer_right_angle_corners(
    points: Sequence[Point], *, keep_indices: set[int], chamfer_units: float = 1.0
) -> tuple[Point, ...]:
    """Replace selected rectilinear corners with short diagonal chamfers.

    Retained corners remain right angles. Chamfered corners become two
    non-right-angle vertices, which lets the task control the final right-angle
    count while still drawing one simple lattice polygon.
    """

    pts = tuple((float(x), float(y)) for x, y in points)
    resolved: list[Point] = []
    for index, vertex in enumerate(pts):
        if index in keep_indices:
            resolved.append(vertex)
            continue
        previous = pts[(index - 1) % len(pts)]
        following = pts[(index + 1) % len(pts)]
        prev_vector = (previous[0] - vertex[0], previous[1] - vertex[1])
        next_vector = (following[0] - vertex[0], following[1] - vertex[1])
        prev_length = max(abs(prev_vector[0]), abs(prev_vector[1]))
        next_length = max(abs(next_vector[0]), abs(next_vector[1]))
        if prev_length <= chamfer_units or next_length <= chamfer_units:
            raise ValueError("base polygon edge is too short for chamfer")
        prev_unit = (
            0.0 if prev_vector[0] == 0 else prev_vector[0] / prev_length,
            0.0 if prev_vector[1] == 0 else prev_vector[1] / prev_length,
        )
        next_unit = (
            0.0 if next_vector[0] == 0 else next_vector[0] / next_length,
            0.0 if next_vector[1] == 0 else next_vector[1] / next_length,
        )
        resolved.append(
            (
                vertex[0] + prev_unit[0] * float(chamfer_units),
                vertex[1] + prev_unit[1] * float(chamfer_units),
            )
        )
        resolved.append(
            (
                vertex[0] + next_unit[0] * float(chamfer_units),
                vertex[1] + next_unit[1] * float(chamfer_units),
            )
        )
    return tuple(resolved)


def _right_angle_count_polygon_points(
    rng: Any,
    *,
    target_count: int,
    vertex_count: int | None = None,
) -> tuple[Point, ...]:
    """Construct one polygon with exactly the requested right-angle vertices."""

    target = int(target_count)
    recipes = [
        (base_count, base_count - target)
        for base_count in (4, 6, 8)
        if 0 <= base_count - target <= base_count
        and 6 <= (base_count + (base_count - target)) <= 12
    ]
    if vertex_count is not None:
        recipes = [
            recipe for recipe in recipes if recipe[0] + recipe[1] == int(vertex_count)
        ]
    if not recipes:
        raise ValueError("no right-angle polygon recipe matches requested support")

    base_count, _chamfer_count = uniform_choice(rng, tuple(recipes))
    base_points = _right_angle_base_polygon(int(base_count))
    corner_indices = list(range(int(base_count)))
    rng.shuffle(corner_indices)
    keep_indices = set(corner_indices[:target])
    points = _chamfer_right_angle_corners(
        base_points,
        keep_indices=keep_indices,
        chamfer_units=1.0,
    )
    points = _lattice_transform_points(points, rng)
    right_indices = _right_angle_vertex_indices(points)
    if len(points) < 6 or len(points) > 12 or len(right_indices) != target:
        raise ValueError("right-angle polygon construction invariant failed")
    return tuple(points)


def _build_right_angle_vertex_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build one polygon and count its right-angle vertices."""

    task_params = dict(context["task_params"])
    generation_defaults = context["generation_defaults"]
    rng, ctx = _new_context(context, plan.salt)
    if "target_count" in task_params:
        target_count = int(task_params["target_count"])
    else:
        count_min = int_default(
            task_params, generation_defaults, "right_angle_count_min", 1
        )
        count_max = int_default(
            task_params, generation_defaults, "right_angle_count_max", 5
        )
        target_count = int(
            uniform_choice(rng, tuple(range(int(count_min), int(count_max) + 1)))
        )
    target_count = max(1, min(5, int(target_count)))
    vertex_count = (
        int(task_params["vertex_count"]) if "vertex_count" in task_params else None
    )
    points = _right_angle_count_polygon_points(
        rng,
        target_count=int(target_count),
        vertex_count=vertex_count,
    )
    points = _integer_shift_points(ctx, points, rng, margin_units=1.2)
    right_indices = _right_angle_vertex_indices(points)
    polygon = draw_polygon(
        ctx,
        "",
        points,
        class_name="right_angle_polygon",
        color=ctx.accent_color,
        filled=False,
    )
    vertex_label_points = draw_vertex_labels(ctx, points)
    right_points_px = [polygon.points_px[index] for index in right_indices]
    right_labels = [POINT_LABELS[index] for index in right_indices]
    annotation_value, projected = point_set_artifacts(right_points_px)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        answer_hint='set "answer" to the integer count',
        annotation_hint=(
            'set "annotation" to pixel points at every polygon vertex whose two '
            "adjacent sides form a right angle"
        ),
        json_example='{"annotation":[[250,420],[460,420]],"answer":2}',
        json_example_answer_only='{"answer":2}',
        target_text="right-angle vertices",
        metric_text="count",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="integer",
        answer_value=len(right_indices),
        annotation_type="point_set",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "vertices": points,
            "right_angle_vertex_indices": list(right_indices),
            "right_angle_vertex_labels": list(right_labels),
            "right_angle_vertices": [points[index] for index in right_indices],
            "vertex_label_points_px": dict(vertex_label_points),
            "vertex_count": len(points),
        },
        objects=(
            replace(
                polygon,
                metric_value=float(len(right_indices)),
                extra={
                    **dict(polygon.extra),
                    "vertex_labels": list(POINT_LABELS[: len(points)]),
                    "right_angle_vertex_labels": list(right_labels),
                },
            ),
        ),
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="single_lattice_polygon.right_angle_vertex_count",
        scene_kind="geometry_graph_paper_single_polygon",
        semantic_args={
            "target_count": int(target_count),
            "vertex_count": len(points),
        },
    )


def _build_angle_extremum_label(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render labeled angles and bind the selected extremum annotation."""

    rng, ctx = _new_context(context, plan.salt)
    object_count = min(
        6,
        resolve_count(
            context["task_params"], context["generation_defaults"], fallback=6
        ),
    )
    labels = label_subset(object_count)
    values = unique_metric_values(rng, count=object_count, low=35, high=150)
    objects = []
    for index, (label, value, center) in enumerate(
        zip(
            labels,
            values,
            slot_centers(ctx, object_count, rng=rng, footprint_units=2.0),
            strict=True,
        )
    ):
        obj = draw_angle(
            ctx,
            label,
            angle_points(center, float(value), radius=2.35),
            color=object_color(ctx, index),
        )
        objects.append(replace(obj, metric_value=float(value)))
    winner = (
        max(objects, key=lambda item: item.metric_value)
        if plan.role_for(str(context["branch_name"])) == "max"
        else min(objects, key=lambda item: item.metric_value)
    )
    annotation_value, projected = scalar_point_artifacts(winner.points_px[1])
    branch_name = str(context["branch_name"])
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(branch_name),
        answer_hint='set "answer" to the selected angle label as a capital letter shown in the image',
        annotation_hint='set "annotation" to the selected angle vertex point [x,y] in pixels',
        json_example='{"annotation":[330,320],"answer":"B"}',
        json_example_answer_only='{"answer":"B"}',
        target_text=f"{branch_name} angle",
        metric_text="angle measure",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="option_letter",
        answer_value=str(winner.label),
        annotation_type="point",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "selected_label": str(winner.label),
            "angle_degrees": int(winner.metric_value),
        },
        objects=tuple(objects),
        prompt_key=plan.prompt_key_for(branch_name),
        program_code="labeled_angles.extremum_label",
        scene_kind="geometry_graph_paper_labeled_angles",
        semantic_args={"extremum": branch_name, "object_count": int(object_count)},
    )


def _build_length_extremum_label(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build the labeled-segment length extremum objective."""

    rng, ctx = _new_context(context, plan.salt)
    object_count = min(
        6,
        resolve_count(
            context["task_params"], context["generation_defaults"], fallback=6
        ),
    )
    labels = label_subset(object_count)
    vector_candidates = [
        (2, 0),
        (3, 0),
        (4, 0),
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 1),
        (1, 2),
        (3, 1),
        (1, 3),
        (2, 2),
        (3, 2),
        (2, 3),
        (4, 1),
        (1, 4),
        (4, 2),
        (2, 4),
    ]
    rng.shuffle(vector_candidates)
    values: list[tuple[int, int, int]] = []
    used_squares: set[int] = set()
    for dx, dy in vector_candidates:
        square = int(dx * dx + dy * dy)
        if square in used_squares:
            continue
        used_squares.add(square)
        values.append((int(dx), int(dy), square))
        if len(values) == object_count:
            break
    if values and not any(dx != 0 and dy != 0 for dx, dy, _square in values):
        for dx, dy in vector_candidates:
            square = int(dx * dx + dy * dy)
            if (
                dx != 0
                and dy != 0
                and square not in {value[2] for value in values[:-1]}
            ):
                values[-1] = (int(dx), int(dy), square)
                break
    objects = []
    for index, (label, (dx, dy, square)) in enumerate(zip(labels, values, strict=True)):
        start, end = _integer_shift_points(
            ctx,
            ((0.0, 0.0), (float(dx), float(dy))),
            rng,
            margin_units=1.2,
        )
        obj = draw_segment(ctx, label, start, end, color=object_color(ctx, index))
        objects.append(
            replace(
                obj,
                metric_value=float(square),
                extra={"dx": int(dx), "dy": int(dy), "length_squared": int(square)},
            )
        )
    winner = (
        max(objects, key=lambda item: item.metric_value)
        if plan.role_for(str(context["branch_name"])) == "max"
        else min(objects, key=lambda item: item.metric_value)
    )
    annotation_value, projected = scalar_segment_artifacts(
        winner.points_px[0], winner.points_px[1]
    )
    branch_name = str(context["branch_name"])
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=plan.prompt_key_for(branch_name),
        answer_hint='set "answer" to the selected segment label as a capital letter shown in the image',
        annotation_hint='set "annotation" to the selected segment as [[x0,y0],[x1,y1]] in pixels',
        json_example='{"annotation":[[210,420],[460,420]],"answer":"B"}',
        json_example_answer_only='{"answer":"B"}',
        target_text=f"{branch_name} segment",
        metric_text="length",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="option_letter",
        answer_value=str(winner.label),
        annotation_type="segment",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "selected_label": str(winner.label),
            "relative_length_squared": int(winner.metric_value),
        },
        objects=tuple(objects),
        prompt_key=plan.prompt_key_for(branch_name),
        program_code="labeled_segments.length_extremum_label",
        scene_kind="geometry_graph_paper_labeled_segments",
        semantic_args={"extremum": branch_name, "object_count": int(object_count)},
    )


def _shape_extremum_objects(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan, metric: str
):
    """Sample and render same-family shape objects for area/perimeter ranking."""

    rng, ctx = _new_context(context, plan.salt)
    object_count = min(
        6,
        resolve_count(
            context["task_params"], context["generation_defaults"], fallback=6
        ),
    )
    labels = list(label_subset(object_count))
    rng.shuffle(labels)
    shape_kind = str(
        dict(context["task_params"]).get(
            "shape_kind",
            choose_from_seed(
                ("rectangle", "right_triangle"),
                instance_seed=int(context["instance_seed"]),
                salt=f"{metric}_extremum_shape",
            ),
        )
    )
    objects = []
    used_values: set[int] = set()
    dimension_choices = [2, 3, 4] if str(metric) == "area" else [1, 2, 3, 4]
    max_dimension = max(dimension_choices)
    dimension_candidates: list[tuple[int, int, int]] = []
    seen_candidate_values: set[int] = set()
    size_pairs = [
        (int(width), int(height))
        for width in dimension_choices
        for height in dimension_choices
    ]
    rng.shuffle(size_pairs)
    for width, height in size_pairs:
        points = _translated_corner_shape(
            left=0,
            bottom=0,
            width=width,
            height=height,
            shape_kind=shape_kind,
        )
        value = polygon_area(points) if metric == "area" else polygon_perimeter(points)
        encoded = int(round(value * 100))
        if encoded in seen_candidate_values:
            continue
        seen_candidate_values.add(encoded)
        dimension_candidates.append((width, height, encoded))
    if len(dimension_candidates) < object_count:
        raise ValueError(f"not enough unique {shape_kind} values for {metric}")

    selected_dimensions = dimension_candidates[:object_count]
    selected_dimensions.sort(key=lambda item: item[0] * item[1], reverse=True)
    limit = int(float(ctx.graph_half_range))
    left_slots = [
        -limit,
        -limit + max_dimension + 1,
        limit - max_dimension,
    ]
    bottom_slots = [-limit, limit - max_dimension]
    placement_slots = [(left, bottom) for bottom in bottom_slots for left in left_slots]
    rng.shuffle(placement_slots)
    if len(placement_slots) < object_count:
        raise ValueError("not enough non-overlapping graph-paper placement slots")

    for index, (label, (width, height, expected_encoded), (slot_left, slot_bottom)) in enumerate(
        zip(labels, selected_dimensions, placement_slots, strict=True)
    ):
        points = _translated_corner_shape(
            left=int(slot_left),
            bottom=int(slot_bottom),
            width=width,
            height=height,
            shape_kind=shape_kind,
        )
        value = polygon_area(points) if metric == "area" else polygon_perimeter(points)
        encoded = int(round(value * 100))
        if encoded != expected_encoded or encoded in used_values:
            raise ValueError("shape extremum metric candidate changed unexpectedly")
        used_values.add(encoded)
        obj = draw_polygon(
            ctx,
            label,
            points,
            class_name=shape_kind,
            color=object_color(ctx, index),
            filled=False,
        )
        objects.append(replace(obj, metric_value=float(encoded)))
    return ctx, shape_kind, object_count, tuple(objects)


def _build_shape_extremum(
    context: Mapping[str, Any], *, plan: GraphPaperTaskPlan, metric: str
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build a labeled-shape area or perimeter extremum objective."""

    ctx, shape_kind, object_count, objects = _shape_extremum_objects(
        context, plan, metric
    )
    winner = (
        max(objects, key=lambda item: item.metric_value)
        if plan.role_for(str(context["branch_name"])) == "max"
        else min(objects, key=lambda item: item.metric_value)
    )
    annotation_value, projected = scalar_bbox_artifacts(winner.bbox_px)
    branch_name = str(context["branch_name"])
    prompt_key = plan.prompt_key_for(branch_name)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=prompt_key,
        answer_hint='set "answer" to the selected shape label as a capital letter shown in the image',
        annotation_hint='set "annotation" to the selected shape bounding box [x0,y0,x1,y1] in pixels',
        json_example='{"annotation":[180,180,320,310],"answer":"B"}',
        json_example_answer_only='{"answer":"B"}',
        shape_text=shape_kind.replace("_", " "),
        target_text=f"{branch_name} {metric}",
        metric_text=metric,
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="option_letter",
        answer_value=str(winner.label),
        annotation_type="bbox",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "selected_label": str(winner.label),
            f"relative_{metric}": int(winner.metric_value),
        },
        objects=objects,
        prompt_key=prompt_key,
        program_code=f"labeled_shapes.{metric}_extremum_label",
        scene_kind="geometry_graph_paper_labeled_shapes",
        semantic_args={
            "extremum": branch_name,
            "shape_kind": shape_kind,
            "object_count": int(object_count),
        },
    )


def _build_area_extremum_label(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build the labeled-shape area extremum objective."""

    return _build_shape_extremum(context, plan=plan, metric="area")


def _build_perimeter_extremum_label(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build the labeled-shape perimeter extremum objective."""

    return _build_shape_extremum(context, plan=plan, metric="perimeter")


ANGLE_CLASSES = ("acute", "right", "obtuse")
ANGLE_VALUE_BY_CLASS = {"acute": 45, "right": 90, "obtuse": 125}

TRIANGLE_CLASSES = ("equilateral", "right", "scalene", "non_equilateral_isosceles")
QUADRILATERAL_CLASSES = (
    "square",
    "non_square_rectangle",
    "non_square_rhombus",
    "slanted_parallelogram",
)
SHAPE_CLASSES = (
    "triangle",
    "quadrilateral",
    "pentagon",
    "hexagon",
    "circle",
    "ellipse",
)
CONVEXITY_CLASSES = ("convex", "concave")


def _squared_distance(first: Point, second: Point) -> float:
    """Return squared distance between two graph points."""

    return (float(first[0]) - float(second[0])) ** 2 + (
        float(first[1]) - float(second[1])
    ) ** 2


def _polygon_side_squares(points: Sequence[Point]) -> tuple[float, ...]:
    """Return squared side lengths for consecutive polygon vertices."""

    pts = tuple(points)
    return tuple(
        _squared_distance(pts[index], pts[(index + 1) % len(pts)])
        for index in range(len(pts))
    )


def _all_close(values: Sequence[float], *, tolerance: float = 1e-3) -> bool:
    """Return whether all values are equal within graph-space tolerance."""

    if not values:
        return False
    reference = float(values[0])
    return all(
        isclose(float(value), reference, abs_tol=float(tolerance)) for value in values
    )


def _distinct_value_count(values: Sequence[float], *, tolerance: float = 1e-3) -> int:
    """Count value groups after tolerance-based equality."""

    groups: list[float] = []
    for value in values:
        numeric = float(value)
        if not any(
            isclose(numeric, group, abs_tol=float(tolerance)) for group in groups
        ):
            groups.append(numeric)
    return len(groups)


def _dot_at(vertex: Point, first: Point, second: Point) -> float:
    """Return dot product for angle first-vertex-second."""

    return (float(first[0]) - float(vertex[0])) * (
        float(second[0]) - float(vertex[0])
    ) + (float(first[1]) - float(vertex[1])) * (float(second[1]) - float(vertex[1]))


def _has_right_angle(points: Sequence[Point], *, tolerance: float = 1e-3) -> bool:
    """Return whether any polygon vertex is right angled."""

    return bool(_right_angle_vertex_indices(points, tolerance=float(tolerance)))


def _right_angle_vertex_indices(
    points: Sequence[Point], *, tolerance: float = 1e-3
) -> tuple[int, ...]:
    """Return indices of vertices whose adjacent sides form a right angle."""

    pts = tuple(points)
    right_indices: list[int] = []
    for index, vertex in enumerate(pts):
        previous = pts[(index - 1) % len(pts)]
        following = pts[(index + 1) % len(pts)]
        if isclose(_dot_at(vertex, previous, following), 0.0, abs_tol=float(tolerance)):
            right_indices.append(int(index))
    return tuple(right_indices)


def _parallel_vectors(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
    *,
    tolerance: float = 1e-3,
) -> bool:
    """Return whether two graph-space vectors are parallel."""

    first_dx = float(first_end[0]) - float(first_start[0])
    first_dy = float(first_end[1]) - float(first_start[1])
    second_dx = float(second_end[0]) - float(second_start[0])
    second_dy = float(second_end[1]) - float(second_start[1])
    return isclose(
        (first_dx * second_dy) - (first_dy * second_dx),
        0.0,
        abs_tol=float(tolerance),
    )


def _is_parallelogram(points: Sequence[Point]) -> bool:
    """Return whether a four-point polygon is a parallelogram."""

    pts = tuple(points)
    if len(pts) != 4:
        return False
    return _parallel_vectors(pts[0], pts[1], pts[3], pts[2]) and _parallel_vectors(
        pts[1], pts[2], pts[0], pts[3]
    )


def _is_rectangle(points: Sequence[Point]) -> bool:
    """Return whether a four-point polygon is a rectangle, regardless of rotation."""

    return (
        len(tuple(points)) == 4
        and _is_parallelogram(points)
        and _has_right_angle(points)
    )


def _is_rhombus(points: Sequence[Point]) -> bool:
    """Return whether a four-point polygon has four equal sides."""

    pts = tuple(points)
    return len(pts) == 4 and _all_close(_polygon_side_squares(pts))


def _is_square(points: Sequence[Point]) -> bool:
    """Return whether a four-point polygon is a square, regardless of rotation."""

    return _is_rectangle(points) and _is_rhombus(points)


def _triangle_matches_target(points: Sequence[Point], target_class: str) -> bool:
    """Match triangle predicates using standard geometry definitions."""

    sides = _polygon_side_squares(tuple(points))
    distinct_side_count = _distinct_value_count(sides)
    if str(target_class) == "equilateral":
        return _all_close(sides)
    if str(target_class) == "right":
        return _has_right_angle(points)
    if str(target_class) == "scalene":
        return distinct_side_count == 3
    if str(target_class) == "non_equilateral_isosceles":
        return distinct_side_count == 2 and not _all_close(sides)
    return False


def _quadrilateral_matches_target(points: Sequence[Point], target_class: str) -> bool:
    """Match quadrilateral predicates using standard geometry definitions."""

    if str(target_class) == "square":
        return _is_square(points)
    if str(target_class) == "non_square_rectangle":
        return _is_rectangle(points) and not _is_square(points)
    if str(target_class) == "non_square_rhombus":
        return _is_rhombus(points) and not _is_square(points)
    if str(target_class) == "slanted_parallelogram":
        return _is_parallelogram(points) and not _is_rectangle(points)
    return False


def _triangle_class_name_matches_target(class_name: str, target_class: str) -> bool:
    """Return whether one triangle construction class satisfies a target predicate."""

    if str(target_class) == "scalene":
        return str(class_name) in {"right", "scalene"}
    return str(class_name) == str(target_class)


def _quadrilateral_class_name_matches_target(
    class_name: str, target_class: str
) -> bool:
    """Return whether one quadrilateral construction class satisfies a target predicate."""

    if str(target_class) == "slanted_parallelogram":
        return str(class_name) in {"non_square_rhombus", "slanted_parallelogram"}
    return str(class_name) == str(target_class)


def _object_class_matches(entity: GraphObject, target_class: str) -> bool:
    """Fallback exact class matcher for count objectives."""

    return str(entity.class_name) == str(target_class)


def _random_choice(rng: Any | None, values: Sequence[Any]) -> Any:
    """Choose from values with a deterministic first-item fallback."""

    options = tuple(values)
    if not options:
        raise ValueError("cannot choose from an empty sequence")
    if rng is None:
        return options[0]
    return options[int(rng.randrange(0, len(options)))]


def _lattice_transform_points(
    points: Sequence[Point],
    rng: Any | None,
    *,
    allow_swap: bool = True,
    allow_mirror: bool = True,
) -> tuple[Point, ...]:
    """Apply integer-grid-preserving orientation changes."""

    transformed = [(float(x), float(y)) for x, y in points]
    if allow_swap and rng is not None and int(rng.randrange(0, 2)) == 1:
        transformed = [(y, x) for x, y in transformed]
    if allow_mirror and rng is not None and int(rng.randrange(0, 2)) == 1:
        transformed = [(-x, y) for x, y in transformed]
    if allow_mirror and rng is not None and int(rng.randrange(0, 2)) == 1:
        transformed = [(x, -y) for x, y in transformed]
    return _normalize_points_to_origin(transformed)


def _anchor_points_near_center(points: Sequence[Point], center: Point) -> tuple[Point, ...]:
    """Place origin-normalized lattice points near a slot center."""

    normalized = _normalize_points_to_origin(points)
    min_x = min(float(point[0]) for point in normalized)
    max_x = max(float(point[0]) for point in normalized)
    min_y = min(float(point[1]) for point in normalized)
    max_y = max(float(point[1]) for point in normalized)
    width = max_x - min_x
    height = max_y - min_y
    left = int(round(float(center[0]) - (width / 2.0)))
    bottom = int(round(float(center[1]) - (height / 2.0)))
    return tuple((float(x) + float(left), float(y) + float(bottom)) for x, y in normalized)


def _integer_center_near_slot(
    ctx: GraphPaperContext,
    center: Point,
    radius_x: int,
    radius_y: int,
    *,
    margin_units: float = 1.0,
) -> Point:
    """Snap an ellipse or circle center near a slot while keeping axes on grid points."""

    limit = int(float(ctx.graph_half_range) - float(margin_units))
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    x = max(-limit + int(radius_x), min(limit - int(radius_x), x))
    y = max(-limit + int(radius_y), min(limit - int(radius_y), y))
    return (float(x), float(y))


def _graph_bbox_units(
    points: Sequence[Point], *, pad_units: float = 0.25
) -> tuple[float, float, float, float]:
    """Return a padded graph-unit bounding box."""

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (
        min(xs) - float(pad_units),
        min(ys) - float(pad_units),
        max(xs) + float(pad_units),
        max(ys) + float(pad_units),
    )


def _graph_bboxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two graph-unit boxes overlap."""

    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _count_candidate_graph_bbox(
    candidate: _CountCandidate, *, pad_units: float = 0.35
) -> tuple[float, float, float, float]:
    """Return a padded graph-unit bbox for a not-yet-drawn count object."""

    if candidate.kind in {"circle", "ellipse"}:
        center = candidate.graph_points[0]
        return (
            float(center[0]) - float(candidate.radius_x) - float(pad_units),
            float(center[1]) - float(candidate.radius_y) - float(pad_units),
            float(center[0]) + float(candidate.radius_x) + float(pad_units),
            float(center[1]) + float(candidate.radius_y) + float(pad_units),
        )
    return _graph_bbox_units(candidate.graph_points, pad_units=float(pad_units))


def _graph_bbox_in_bounds(
    ctx: GraphPaperContext, bbox: tuple[float, float, float, float]
) -> bool:
    """Return whether a padded graph-unit bbox stays inside the visible grid."""

    limit = float(ctx.graph_half_range)
    return (
        float(bbox[0]) >= -limit
        and float(bbox[1]) >= -limit
        and float(bbox[2]) <= limit
        and float(bbox[3]) <= limit
    )


def _draw_count_candidate(
    ctx: GraphPaperContext, candidate: _CountCandidate, *, object_index: int
) -> GraphObject:
    """Draw one accepted count-task candidate exactly once."""

    color = object_color(ctx, int(object_index))
    if candidate.kind == "angle":
        obj = draw_angle(ctx, "", candidate.graph_points, color=color)
    elif candidate.kind == "polygon":
        obj = draw_polygon(
            ctx,
            "",
            candidate.graph_points,
            class_name=candidate.class_name,
            color=color,
            filled=False,
        )
    elif candidate.kind in {"circle", "ellipse"}:
        obj = draw_ellipse_or_circle(
            ctx,
            "",
            candidate.graph_points[0],
            float(candidate.radius_x),
            float(candidate.radius_y),
            class_name=candidate.kind,
            color=color,
            filled=False,
        )
    else:
        raise ValueError(f"unsupported count candidate kind: {candidate.kind}")
    return replace(
        obj,
        class_name=str(candidate.class_name),
        metric_value=float(candidate.metric_value),
        extra={**dict(obj.extra), **dict(candidate.extra)},
    )


def _place_count_candidates(
    ctx: GraphPaperContext,
    *,
    class_sequence: Sequence[str],
    rng: Any,
    build_candidate: Callable[[str, Point], _CountCandidate],
    pad_units: float,
    footprint_units: float,
) -> tuple[GraphObject, ...]:
    """Place count objects using actual graph bboxes instead of slot centers."""

    classes = tuple(str(value) for value in class_sequence)
    object_count = len(classes)
    center_pool = slot_centers(
        ctx,
        max(72, int(object_count) * 18),
        rng=rng,
        footprint_units=float(footprint_units),
    )
    candidate_lists: list[
        list[tuple[_CountCandidate, tuple[float, float, float, float]]]
    ] = []
    for cls_name in classes:
        center_candidates = list(center_pool)
        rng.shuffle(center_candidates)
        candidates: list[tuple[_CountCandidate, tuple[float, float, float, float]]] = []
        for center in center_candidates:
            candidate = build_candidate(str(cls_name), center)
            raw_bbox = _count_candidate_graph_bbox(candidate, pad_units=0.0)
            if not _graph_bbox_in_bounds(ctx, raw_bbox):
                continue
            padded_bbox = _count_candidate_graph_bbox(
                candidate, pad_units=float(pad_units)
            )
            candidates.append((candidate, padded_bbox))
        if not candidates:
            raise ValueError("could not build in-bounds count object candidates")
        candidate_lists.append(candidates)

    placements: list[tuple[_CountCandidate, tuple[float, float, float, float]] | None] = [
        None
    ] * object_count

    def search(placed: set[int], accepted_boxes: list[tuple[float, float, float, float]]) -> bool:
        if len(placed) == object_count:
            return True

        best_index = -1
        best_options: list[
            tuple[_CountCandidate, tuple[float, float, float, float]]
        ] | None = None
        for index in range(object_count):
            if index in placed:
                continue
            options = [
                (candidate, bbox)
                for candidate, bbox in candidate_lists[index]
                if not any(
                    _graph_bboxes_overlap(bbox, accepted)
                    for accepted in accepted_boxes
                )
            ]
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_index = index
                best_options = options

        assert best_options is not None
        for candidate, bbox in best_options:
            placements[best_index] = (candidate, bbox)
            if search(placed | {best_index}, [*accepted_boxes, bbox]):
                return True
            placements[best_index] = None
        return False

    if not search(set(), []):
        raise ValueError("could not place non-overlapping count objects")

    objects: list[GraphObject] = []
    for index, placement in enumerate(placements):
        if placement is None:
            raise ValueError("missing count object placement")
        candidate, _bbox = placement
        objects.append(_draw_count_candidate(ctx, candidate, object_index=index))
    return tuple(objects)


def _lattice_convex_polygon_points(
    center: Point, side_count: int, rng: Any | None
) -> tuple[Point, ...]:
    """Return an irregular convex polygon with integer-lattice vertices."""

    templates: dict[int, tuple[tuple[Point, ...], ...]] = {
        5: (
            ((0, 0), (2, 0), (3, 1), (2, 3), (0, 2)),
            ((0, 1), (2, 0), (4, 1), (3, 3), (1, 4)),
            ((0, 0), (3, 0), (4, 1), (3, 3), (1, 2)),
        ),
        6: (
            ((0, 1), (1, 0), (3, 0), (4, 1), (3, 3), (1, 3)),
            ((0, 0), (2, 0), (4, 1), (3, 3), (2, 4), (0, 3)),
            ((0, 1), (1, 0), (3, 0), (4, 2), (3, 4), (1, 3)),
        ),
        7: (
            ((0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 4), (0, 3)),
            ((0, 2), (1, 0), (3, 0), (4, 1), (4, 3), (2, 4), (1, 4)),
            ((0, 1), (1, 0), (3, 1), (4, 2), (3, 4), (1, 4), (0, 3)),
        ),
    }
    sides = max(5, min(7, int(side_count)))
    template = tuple(_random_choice(rng, templates[sides]))
    transformed = _lattice_transform_points(template, rng)
    return _anchor_points_near_center(transformed, center)


def _lattice_concave_polygon_points(
    center: Point, side_count: int, rng: Any | None
) -> tuple[Point, ...]:
    """Return a concave polygon with integer-lattice vertices."""

    templates: dict[int, tuple[tuple[Point, ...], ...]] = {
        5: (
            ((0, 0), (4, 0), (2, 1), (4, 3), (0, 3)),
            ((0, 0), (3, 0), (3, 3), (1, 2), (0, 4)),
            ((0, 1), (4, 0), (3, 2), (4, 4), (0, 3)),
        ),
        6: (
            ((0, 0), (4, 0), (4, 3), (2, 2), (1, 4), (0, 3)),
            ((0, 0), (3, 0), (4, 2), (2, 2), (3, 4), (0, 4)),
            ((0, 1), (2, 0), (4, 1), (3, 3), (4, 5), (0, 4)),
        ),
        7: (
            ((0, 1), (1, 0), (4, 1), (3, 2), (4, 4), (2, 3), (0, 4)),
            ((0, 0), (3, 0), (4, 2), (2, 2), (3, 4), (1, 4), (0, 3)),
            ((0, 1), (1, 0), (3, 0), (4, 2), (2, 2), (3, 4), (0, 4)),
        ),
    }
    sides = max(5, min(7, int(side_count)))
    template = tuple(_random_choice(rng, templates[sides]))
    transformed = _lattice_transform_points(template, rng)
    return _anchor_points_near_center(transformed, center)


def _triangle_points(
    center: Point, class_name: str, rng: Any | None = None
) -> tuple[Point, ...]:
    """Return a compact triangle for classification-count scenes."""

    if class_name == "equilateral":
        base = float(_random_choice(rng, (2, 3, 4)))
        height = float(sqrt(3.0) * base / 2.0)
        local = ((0.0, 0.0), (base, 0.0), (base / 2.0, height))
        if rng is not None and int(rng.randrange(0, 2)) == 1:
            local = tuple((x, -y) for x, y in local)
        return _anchor_points_near_center(local, center)

    templates_by_class: dict[str, tuple[tuple[Point, ...], ...]] = {
        "non_equilateral_isosceles": (
            ((0, 0), (4, 0), (2, 1)),
            ((0, 0), (4, 0), (2, 2)),
            ((0, 0), (6, 0), (3, 2)),
            ((0, 0), (6, 0), (3, 4)),
        ),
        "right": (
            ((0, 0), (3, 0), (0, 2)),
            ((0, 0), (4, 0), (0, 3)),
            ((0, 0), (2, 0), (0, 5)),
            ((0, 0), (5, 0), (0, 2)),
        ),
        "scalene": (
            ((0, 0), (4, 0), (1, 3)),
            ((0, 0), (3, 0), (1, 4)),
            ((0, 0), (5, 0), (2, 3)),
            ((0, 0), (4, 0), (3, 2)),
            ((0, 0), (5, 0), (1, 2)),
        ),
    }
    templates = templates_by_class.get(str(class_name), templates_by_class["scalene"])
    for _ in range(30):
        template = tuple(_random_choice(rng, templates))
        points = _anchor_points_near_center(
            _lattice_transform_points(template, rng), center
        )
        if _triangle_matches_target(points, str(class_name)):
            return points
    fallback = _anchor_points_near_center(templates[0], center)
    if not _triangle_matches_target(fallback, str(class_name)):
        raise ValueError(f"triangle template does not match {class_name}")
    return fallback


def _quadrilateral_points(
    center: Point, class_name: str, rng: Any | None = None
) -> tuple[Point, ...]:
    """Return a compact quadrilateral for classification-count scenes."""

    templates_by_class: dict[str, tuple[tuple[Point, ...], ...]] = {
        "square": (
            ((0, 0), (2, 0), (2, 2), (0, 2)),
            ((0, 0), (3, 0), (3, 3), (0, 3)),
            ((1, 0), (2, 1), (1, 2), (0, 1)),
            ((2, 0), (4, 2), (2, 4), (0, 2)),
        ),
        "non_square_rectangle": (
            ((0, 0), (3, 0), (3, 2), (0, 2)),
            ((0, 0), (4, 0), (4, 2), (0, 2)),
            ((0, 0), (5, 0), (5, 2), (0, 2)),
        ),
        "non_square_rhombus": (
            ((0, 0), (2, 1), (3, 3), (1, 2)),
            ((0, 0), (3, 1), (4, 4), (1, 3)),
            ((0, 0), (1, 2), (3, 3), (2, 1)),
            ((0, 0), (3, 2), (5, 5), (2, 3)),
            ((0, 0), (4, 1), (5, 5), (1, 4)),
        ),
        "slanted_parallelogram": (
            ((0, 0), (3, 0), (4, 2), (1, 2)),
            ((0, 0), (4, 0), (5, 2), (1, 2)),
            ((0, 0), (3, 0), (5, 2), (2, 2)),
            ((0, 0), (4, 0), (6, 3), (2, 3)),
        ),
    }
    templates = templates_by_class.get(
        str(class_name), templates_by_class["slanted_parallelogram"]
    )
    for _ in range(30):
        template = tuple(_random_choice(rng, templates))
        points = _anchor_points_near_center(
            _lattice_transform_points(template, rng), center
        )
        if _quadrilateral_matches_target(points, str(class_name)):
            return points
    fallback = _anchor_points_near_center(templates[0], center)
    if not _quadrilateral_matches_target(fallback, str(class_name)):
        raise ValueError(f"quadrilateral template does not match {class_name}")
    return fallback


def _count_setup(
    context: Mapping[str, Any],
    plan: GraphPaperTaskPlan,
    *,
    salt: str,
    target_field: str,
    classes: Sequence[str],
    class_matches_target: Callable[[str, str], bool] | None = None,
    distractor_classes_override: Sequence[str] | None = None,
):
    """Resolve the target class, target count, and shuffled class sequence."""

    task_params = dict(context["task_params"])
    rng = rng_for(int(context["instance_seed"]), str(salt))
    object_count = min(
        6, resolve_count(task_params, context["generation_defaults"], fallback=6)
    )
    branch_name = str(context["branch_name"])
    fallback_target_class = str(
        task_params.get(
            str(target_field),
            choose_from_seed(
                tuple(str(value) for value in classes),
                instance_seed=int(context["instance_seed"]),
                salt=str(salt),
            ),
        )
    )
    target_class = plan.target_class_for(branch_name, fallback_target_class)
    target_total = max(
        1,
        min(
            5,
            count_target(
                task_params,
                context["generation_defaults"],
                object_count=object_count,
                instance_seed=int(context["instance_seed"]),
                salt=str(salt),
            ),
        ),
    )
    matches_target = class_matches_target or (
        lambda class_name, target: str(class_name) == str(target)
    )
    if distractor_classes_override is None:
        distractor_classes = [
            str(value)
            for value in classes
            if not matches_target(str(value), str(target_class))
        ]
    else:
        distractor_classes = [
            str(value)
            for value in distractor_classes_override
            if not matches_target(str(value), str(target_class))
        ]
    class_sequence = make_class_sequence(
        target_class=target_class,
        distractor_classes=distractor_classes,
        object_count=object_count,
        target_count=target_total,
        rng=rng,
    )
    return rng, object_count, target_class, target_total, class_sequence


def _count_components(
    context: Mapping[str, Any],
    *,
    ctx: GraphPaperContext,
    objects: Sequence[GraphObject],
    target_class: str,
    target_text: str,
    target_total: int,
    prompt_key: str,
    program_code: str,
    scene_kind: str,
    object_count: int,
    noun: str,
    match_predicate: Callable[[GraphObject], bool] | None = None,
) -> tuple[GraphPaperComponents, TypedValue]:
    """Build shared count-task components once the public class family is rendered."""

    resolved_predicate = match_predicate or (
        lambda entity: _object_class_matches(entity, str(target_class))
    )
    matching_objects = [obj for obj in objects if resolved_predicate(obj)]
    actual_target_total = len(matching_objects)
    matching = [obj.bbox_px for obj in matching_objects]
    annotation_value, projected = bbox_set_artifacts(matching)
    prompt_plan = _make_prompt(
        context["prompt_defaults"],
        prompt_key=str(prompt_key),
        answer_hint='set "answer" to the integer count',
        annotation_hint=f'set "annotation" to bounding boxes for every matching {noun}',
        json_example='{"annotation":[[120,120,210,210],[340,270,430,360]],"answer":2}',
        json_example_answer_only='{"answer":2}',
        target_text=str(target_text),
        metric_text="count",
    )
    return _component_payload(
        context,
        ctx=ctx,
        prompt_plan=prompt_plan,
        answer_type="integer",
        answer_value=int(actual_target_total),
        annotation_type="bbox_set",
        annotation_value=annotation_value,
        projected_annotation=projected,
        witness_symbolic={
            "target_class": str(target_class),
            "target_text": str(target_text),
            "planned_target_count": int(target_total),
            "matching_count": int(actual_target_total),
        },
        objects=tuple(objects),
        prompt_key=str(prompt_key),
        program_code=str(program_code),
        scene_kind=str(scene_kind),
        semantic_args={
            "target_class": str(target_class),
            "target_text": str(target_text),
            "object_count": int(object_count),
            "planned_target_count": int(target_total),
            "matching_count": int(actual_target_total),
        },
    )


def _build_angle_type_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render angle classes and bind matching-class count annotation."""

    rng, object_count, target_class, target_total, class_sequence = _count_setup(
        context,
        plan,
        salt=plan.salt,
        target_field=plan.target_field,
        classes=ANGLE_CLASSES,
    )
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=context["task_params"],
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )
    def build_angle_candidate(cls_name: str, center: Point) -> _CountCandidate:
        value = float(ANGLE_VALUE_BY_CLASS[str(cls_name)])
        return _CountCandidate(
            kind="angle",
            class_name=str(cls_name),
            graph_points=angle_points(center, value, radius=2.15),
            metric_value=value,
        )

    objects = _place_count_candidates(
        ctx,
        class_sequence=class_sequence,
        rng=rng,
        build_candidate=build_angle_candidate,
        pad_units=0.25,
        footprint_units=1.4,
    )
    return _count_components(
        context,
        ctx=ctx,
        objects=tuple(objects),
        target_class=target_class,
        target_text=plan.target_text_for(
            str(context["branch_name"]), f"{target_class} angles"
        ),
        target_total=target_total,
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="angle_set.class_count",
        scene_kind="geometry_graph_paper_angle_set",
        object_count=object_count,
        noun="angle",
    )


def _build_triangle_type_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render triangle classes and bind matching-class count annotation."""

    rng, object_count, target_class, target_total, class_sequence = _count_setup(
        context,
        plan,
        salt=plan.salt,
        target_field=plan.target_field,
        classes=TRIANGLE_CLASSES,
        class_matches_target=_triangle_class_name_matches_target,
        distractor_classes_override=(
            tuple(value for value in TRIANGLE_CLASSES if value != "equilateral")
            if plan.target_class_for(str(context["branch_name"]), "") != "equilateral"
            else None
        ),
    )
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=context["task_params"],
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )
    def build_triangle_candidate(cls_name: str, center: Point) -> _CountCandidate:
        return _CountCandidate(
            kind="polygon",
            class_name=str(cls_name),
            graph_points=_triangle_points(center, cls_name, rng),
        )

    objects = _place_count_candidates(
        ctx,
        class_sequence=class_sequence,
        rng=rng,
        build_candidate=build_triangle_candidate,
        pad_units=0.35,
        footprint_units=1.3,
    )
    return _count_components(
        context,
        ctx=ctx,
        objects=tuple(objects),
        target_class=target_class,
        target_text=plan.target_text_for(
            str(context["branch_name"]), f"{target_class.replace('_', ' ')} triangles"
        ),
        target_total=target_total,
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="triangle_set.class_count",
        scene_kind="geometry_graph_paper_triangle_set",
        object_count=object_count,
        noun="triangle",
        match_predicate=lambda entity: _triangle_matches_target(
            entity.graph_points, str(target_class)
        ),
    )


def _build_quadrilateral_type_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render quadrilateral classes and bind matching-class count annotation."""

    rng, object_count, target_class, target_total, class_sequence = _count_setup(
        context,
        plan,
        salt=plan.salt,
        target_field=plan.target_field,
        classes=QUADRILATERAL_CLASSES,
        class_matches_target=_quadrilateral_class_name_matches_target,
    )
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=context["task_params"],
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )
    def build_quadrilateral_candidate(cls_name: str, center: Point) -> _CountCandidate:
        return _CountCandidate(
            kind="polygon",
            class_name=str(cls_name),
            graph_points=_quadrilateral_points(center, cls_name, rng),
        )

    objects = _place_count_candidates(
        ctx,
        class_sequence=class_sequence,
        rng=rng,
        build_candidate=build_quadrilateral_candidate,
        pad_units=0.35,
        footprint_units=1.3,
    )
    return _count_components(
        context,
        ctx=ctx,
        objects=tuple(objects),
        target_class=target_class,
        target_text=plan.target_text_for(
            str(context["branch_name"]),
            f"{target_class.replace('_', ' ')} quadrilaterals",
        ),
        target_total=target_total,
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="quadrilateral_set.class_count",
        scene_kind="geometry_graph_paper_quadrilateral_set",
        object_count=object_count,
        noun="quadrilateral",
        match_predicate=lambda entity: _quadrilateral_matches_target(
            entity.graph_points, str(target_class)
        ),
    )


def _build_shape_type_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render mixed shape classes and bind matching-class count annotation."""

    rng, object_count, target_class, target_total, class_sequence = _count_setup(
        context,
        plan,
        salt=plan.salt,
        target_field=plan.target_field,
        classes=SHAPE_CLASSES,
    )
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=context["task_params"],
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )

    def build_shape_candidate(cls_name: str, center: Point) -> _CountCandidate:
        """Build one mixed-shape count candidate before bbox placement.

        The invariant is that the returned class name stays at the prompt-facing
        shape family level while subtype/radius details remain trace metadata.
        """

        if cls_name == "triangle":
            triangle_class = _random_choice(
                rng,
                (
                    "right",
                    "scalene",
                    "non_equilateral_isosceles",
                ),
            )
            return _CountCandidate(
                kind="polygon",
                class_name="triangle",
                graph_points=_triangle_points(center, str(triangle_class), rng),
                extra={"shape_variant": str(triangle_class)},
            )
        if cls_name == "quadrilateral":
            quadrilateral_class = _random_choice(
                rng,
                (
                    "square",
                    "non_square_rectangle",
                    "non_square_rhombus",
                    "slanted_parallelogram",
                ),
            )
            return _CountCandidate(
                kind="polygon",
                class_name="quadrilateral",
                graph_points=_quadrilateral_points(center, str(quadrilateral_class), rng),
                extra={"shape_variant": str(quadrilateral_class)},
            )
        if cls_name == "pentagon":
            return _CountCandidate(
                kind="polygon",
                class_name="pentagon",
                graph_points=_lattice_convex_polygon_points(center, 5, rng),
                extra={"side_count": 5},
            )
        if cls_name == "hexagon":
            return _CountCandidate(
                kind="polygon",
                class_name="hexagon",
                graph_points=_lattice_convex_polygon_points(center, 6, rng),
                extra={"side_count": 6},
            )
        if cls_name == "circle":
            radius = int(_random_choice(rng, (1, 2)))
            ellipse_center = _integer_center_near_slot(ctx, center, radius, radius)
            return _CountCandidate(
                kind="circle",
                class_name="circle",
                graph_points=(ellipse_center,),
                radius_x=float(radius),
                radius_y=float(radius),
            )
        radius_x, radius_y = _random_choice(rng, ((2, 1), (3, 1), (1, 2), (1, 3)))
        ellipse_center = _integer_center_near_slot(ctx, center, radius_x, radius_y)
        return _CountCandidate(
            kind="ellipse",
            class_name="ellipse",
            graph_points=(ellipse_center,),
            radius_x=float(radius_x),
            radius_y=float(radius_y),
        )

    objects = _place_count_candidates(
        ctx,
        class_sequence=class_sequence,
        rng=rng,
        build_candidate=build_shape_candidate,
        pad_units=0.35,
        footprint_units=1.3,
    )
    return _count_components(
        context,
        ctx=ctx,
        objects=tuple(objects),
        target_class=target_class,
        target_text=plan.target_text_for(
            str(context["branch_name"]), f"{target_class.replace('_', ' ')} shapes"
        ),
        target_total=target_total,
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="shape_set.class_count",
        scene_kind="geometry_graph_paper_mixed_shape_set",
        object_count=object_count,
        noun="shape",
    )


def _build_polygon_convexity_count(
    context: Mapping[str, Any], plan: GraphPaperTaskPlan
) -> tuple[GraphPaperComponents, TypedValue]:
    """Render convexity classes and bind matching-polygon count annotation."""

    rng, object_count, target_class, target_total, class_sequence = _count_setup(
        context,
        plan,
        salt=plan.salt,
        target_field=plan.target_field,
        classes=CONVEXITY_CLASSES,
    )
    ctx = make_context(
        instance_seed=int(context["instance_seed"]),
        params=context["task_params"],
        defaults=context["rendering_defaults"],
        theme_index=rng.randrange(0, 3),
    )
    def build_convexity_candidate(cls_name: str, center: Point) -> _CountCandidate:
        side_count = int(rng.randint(5, 7))
        points = (
            _lattice_convex_polygon_points(center, side_count, rng)
            if cls_name == "convex"
            else _lattice_concave_polygon_points(center, side_count, rng)
        )
        return _CountCandidate(
            kind="polygon",
            class_name=str(cls_name),
            graph_points=points,
            extra={"side_count": int(side_count)},
        )

    objects = _place_count_candidates(
        ctx,
        class_sequence=class_sequence,
        rng=rng,
        build_candidate=build_convexity_candidate,
        pad_units=0.35,
        footprint_units=1.3,
    )
    return _count_components(
        context,
        ctx=ctx,
        objects=tuple(objects),
        target_class=target_class,
        target_text=plan.target_text_for(
            str(context["branch_name"]), f"{target_class} polygons"
        ),
        target_total=target_total,
        prompt_key=plan.prompt_key_for(str(context["branch_name"])),
        program_code="polygon_set.convexity_count",
        scene_kind="geometry_graph_paper_polygon_set",
        object_count=object_count,
        noun="polygon",
    )


__all__ = [
    "GraphPaperComponents",
    "GraphPaperTaskPlan",
    "build_graph_paper_components",
    "graph_paper_prompt_plan",
    "object_trace",
    "run_graph_paper_entry",
]
