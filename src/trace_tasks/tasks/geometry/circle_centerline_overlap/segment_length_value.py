"""Find a missing centerline segment in an overlapping-circle chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

from .shared.annotations import segment_annotation
from .shared.construction import (
    boundary_names,
    boundary_segment_answer_support,
    center_distance_length,
    center_distance_answer_support,
    segment_length,
    select_boundary_pair,
    select_boundary_segment_overlap_case,
    select_boundary_target_role,
    select_circle_count,
    select_center_distance_overlap_case,
    select_label_mode,
)
from .shared.prompts import circle_centerline_prompt_artifacts
from .shared.rendering import (
    create_centerline_overlap_render_context,
    render_centerline_overlap_scene,
)
from .shared.state import (
    BOUNDARY_PAIRS,
    BOUNDARY_TARGET_ROLES,
    SCENE_ID,
    CenterlineOverlapDiagramSpec,
    CircleOverlapCase,
    RenderedCenterlineOverlapScene,
)

TASK_ID = "task_geometry__circle_centerline_overlap__segment_length_value"
QUERY_ID_CENTER_DISTANCE = "center_distance_from_overlap"
QUERY_ID_BOUNDARY_SEGMENT = "boundary_segment_from_overlap"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID_CENTER_DISTANCE,
    QUERY_ID_BOUNDARY_SEGMENT,
)
_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(_SCENE_DEFAULTS, task_id=TASK_ID)
)


@dataclass(frozen=True)
class _ResolvedSegmentProblem:
    """Task-owned answer, prompt, annotation, and trace facts for one branch."""

    diagram_spec: CenterlineOverlapDiagramSpec
    answer: int
    target_name: str
    case: CircleOverlapCase
    label_mode: str
    boundary_pair: str
    boundary_target_role: str
    known_segment_name: str
    known_segment_value: int
    case_probabilities: dict[str, float]
    label_mode_probabilities: dict[str, float]
    boundary_pair_probabilities: dict[str, float]
    boundary_target_role_probabilities: dict[str, float]
    circle_count: int
    circle_count_probabilities: dict[str, float]
    answer_support_probabilities: dict[str, float]


def _center_distance_problem(
    *,
    case: CircleOverlapCase,
    label_mode: str,
    case_probabilities: Mapping[str, float],
    label_mode_probabilities: Mapping[str, float],
    circle_count_probabilities: Mapping[str, float],
) -> _ResolvedSegmentProblem:
    """Bind the full center-distance branch and its minimal witness roles."""

    right_center = "C" if int(case.circle_count) == 3 else "B"
    target_name = f"A{right_center}"
    answer = int(center_distance_length(case))
    target_points = ("A", right_center)
    diagram_spec = CenterlineOverlapDiagramSpec(
        case=case,
        label_mode=str(label_mode),
        target_name=target_name,
        known_segment_name="",
        known_segment_value=0,
        target_segment_points=target_points,
        known_segment_points=("", ""),
        show_overlap_dimensions=True,
        annotation_roles=target_points,
    )
    return _ResolvedSegmentProblem(
        diagram_spec=diagram_spec,
        answer=int(answer),
        target_name=target_name,
        case=case,
        label_mode=str(label_mode),
        boundary_pair="",
        boundary_target_role="",
        known_segment_name="",
        known_segment_value=0,
        case_probabilities=dict(case_probabilities),
        label_mode_probabilities=dict(label_mode_probabilities),
        boundary_pair_probabilities={value: 0.0 for value in BOUNDARY_PAIRS},
        boundary_target_role_probabilities={
            value: 0.0 for value in BOUNDARY_TARGET_ROLES
        },
        circle_count=int(case.circle_count),
        circle_count_probabilities=dict(circle_count_probabilities),
        answer_support_probabilities=center_distance_answer_support(answer),
    )


def _boundary_segment_problem(
    *,
    case: CircleOverlapCase,
    label_mode: str,
    boundary_pair: str,
    boundary_target_role: str,
    case_probabilities: Mapping[str, float],
    label_mode_probabilities: Mapping[str, float],
    boundary_pair_probabilities: Mapping[str, float],
    boundary_target_role_probabilities: Mapping[str, float],
    circle_count_probabilities: Mapping[str, float],
) -> _ResolvedSegmentProblem:
    """Bind one boundary segment branch and its paired known segment."""

    target_name, known_segment_name, target_points, known_points = boundary_names(
        boundary_pair, boundary_target_role
    )
    answer = segment_length(case, boundary_pair, boundary_target_role)
    opposite_role = (
        "left_boundary_to_right_center"
        if str(boundary_target_role) == "left_center_to_right_boundary"
        else "left_center_to_right_boundary"
    )
    known_segment_value = segment_length(case, boundary_pair, opposite_role)
    annotation_roles = tuple(dict.fromkeys(target_points))
    diagram_spec = CenterlineOverlapDiagramSpec(
        case=case,
        label_mode=str(label_mode),
        target_name=str(target_name),
        known_segment_name=str(known_segment_name),
        known_segment_value=int(known_segment_value),
        target_segment_points=tuple(target_points),
        known_segment_points=tuple(known_points),
        show_overlap_dimensions=False,
        annotation_roles=annotation_roles,
    )
    return _ResolvedSegmentProblem(
        diagram_spec=diagram_spec,
        answer=int(answer),
        target_name=str(target_name),
        case=case,
        label_mode=str(label_mode),
        boundary_pair=str(boundary_pair),
        boundary_target_role=str(boundary_target_role),
        known_segment_name=str(known_segment_name),
        known_segment_value=int(known_segment_value),
        case_probabilities=dict(case_probabilities),
        label_mode_probabilities=dict(label_mode_probabilities),
        boundary_pair_probabilities=dict(boundary_pair_probabilities),
        boundary_target_role_probabilities=dict(boundary_target_role_probabilities),
        circle_count=int(case.circle_count),
        circle_count_probabilities=dict(circle_count_probabilities),
        answer_support_probabilities=boundary_segment_answer_support(answer),
    )


def _trace_payload(
    *,
    rendered: RenderedCenterlineOverlapScene,
    image_size: tuple[int, int],
    render_context: Any,
    noise_meta: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    problem: _ResolvedSegmentProblem,
    annotation_value: list[list[float]],
) -> dict[str, Any]:
    """Serialize task-owned answer and annotation facts into trace metadata."""

    case = problem.case
    query_params = {
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
        "overlap_case_key": str(case.key),
        "overlap_case_probabilities": dict(problem.case_probabilities),
        "circle_count": int(problem.circle_count),
        "circle_count_probabilities": dict(problem.circle_count_probabilities),
        "label_mode": str(problem.label_mode),
        "label_mode_probabilities": dict(problem.label_mode_probabilities),
        "boundary_pair": str(problem.boundary_pair),
        "boundary_pair_probabilities": dict(problem.boundary_pair_probabilities),
        "boundary_target_role": str(problem.boundary_target_role),
        "boundary_target_role_probabilities": dict(
            problem.boundary_target_role_probabilities
        ),
        "answer_support_probabilities": dict(problem.answer_support_probabilities),
        "target_name": str(problem.target_name),
        "known_segment_name": str(problem.known_segment_name),
        "known_segment_value": int(problem.known_segment_value),
        "answer_value": int(problem.answer),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID

    return {
        "scene_ir": {
            "domain": "geometry",
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "type": "circle_centerline_overlap_chain",
                "adjacent_overlaps_only": True,
                "circle_count": int(problem.circle_count),
                "distance_ab": int(case.distance_ab),
                "distance_bc": int(case.distance_bc),
                "distance_ac": int(case.distance_ac),
                "overlap_ab": int(case.overlap_ab),
                "overlap_bc": int(case.overlap_bc),
                "target_name": str(problem.target_name),
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas": {
                "width": int(image_size[0]),
                "height": int(image_size[1]),
            },
            "single_object_scene_rotation": render_context.scene_transform.metadata(),
            "style": {
                "technical_diagram": dict(render_context.diagram_style_meta),
                "background": dict(render_context.background_meta),
                "font_bold": False,
                "label_stroke_width": int(render_context.label_stroke_width),
                "post_image_noise": dict(noise_meta),
            },
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(
                    prompt_artifacts.prompt_variant_active_key
                ),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "radius_a": int(case.radius_a),
            "radius_b": int(case.radius_b),
            "radius_c": int(case.radius_c),
            "diameter_a": int(case.radius_a * 2),
            "diameter_b": int(case.radius_b * 2),
            "diameter_c": int(case.radius_c * 2),
            "distance_ab": int(case.distance_ab),
            "distance_bc": int(case.distance_bc),
            "distance_ac": int(case.distance_ac),
            "overlap_ab": int(case.overlap_ab),
            "overlap_bc": int(case.overlap_bc),
            "circle_count": int(problem.circle_count),
            "label_mode": str(problem.label_mode),
            "boundary_pair": str(problem.boundary_pair),
            "boundary_target_role": str(problem.boundary_target_role),
            "target_name": str(problem.target_name),
            "known_segment_name": str(problem.known_segment_name),
            "known_segment_value": int(problem.known_segment_value),
            "answer": int(problem.answer),
            "annotation_roles": list(rendered.annotation_roles),
        },
        "witness_symbolic": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "formula_family": "circle_centerline_overlap_segment_length",
            "circle_count": int(problem.circle_count),
            "radii": {
                "A": int(case.radius_a),
                "B": int(case.radius_b),
                **({"C": int(case.radius_c)} if int(problem.circle_count) == 3 else {}),
            },
            "distances": {
                "AB": int(case.distance_ab),
                **(
                    {"BC": int(case.distance_bc), "AC": int(case.distance_ac)}
                    if int(problem.circle_count) == 3
                    else {}
                ),
            },
            "overlaps": {
                "PQ": int(case.overlap_ab),
                **({"RS": int(case.overlap_bc)} if int(problem.circle_count) == 3 else {}),
            },
            "target_name": str(problem.target_name),
            "known_segment_name": str(problem.known_segment_name),
            "known_segment_value": int(problem.known_segment_value),
            "answer_value": int(problem.answer),
        },
        "projected_annotation": {
            "type": "segment",
            "segment": list(annotation_value),
            "pixel_segment": list(annotation_value),
        },
    }


@register_task
class GeometryCircleCenterlineOverlapSegmentLengthValueTask:
    """Find a missing centerline segment in an overlapping-circle chain."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Select a segment-measure query and bind its answer/annotation locally."""
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID_CENTER_DISTANCE,
            task_id=TASK_ID,
        )
        label_mode, label_mode_probabilities = select_label_mode(
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.{selected_query}.label_mode",
        )
        circle_count, circle_count_probabilities = select_circle_count(
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.{selected_query}.circle_count",
        )
        if str(selected_query) == QUERY_ID_CENTER_DISTANCE:
            case, case_probabilities = select_center_distance_overlap_case(
                circle_count=int(circle_count),
                instance_seed=int(instance_seed),
                params=task_params,
                namespace=f"{TASK_ID}.{selected_query}.overlap_case",
            )
            problem = _center_distance_problem(
                case=case,
                label_mode=str(label_mode),
                case_probabilities=case_probabilities,
                label_mode_probabilities=label_mode_probabilities,
                circle_count_probabilities=circle_count_probabilities,
            )
        elif str(selected_query) == QUERY_ID_BOUNDARY_SEGMENT:
            boundary_pair, boundary_pair_probabilities = select_boundary_pair(
                circle_count=int(circle_count),
                params=task_params,
                instance_seed=int(instance_seed),
                namespace=f"{TASK_ID}.{selected_query}.boundary_pair",
            )
            boundary_target_role, boundary_target_role_probabilities = (
                select_boundary_target_role(
                    params=task_params,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}.{selected_query}.boundary_target_role",
                )
            )
            case, case_probabilities = select_boundary_segment_overlap_case(
                boundary_pair=str(boundary_pair),
                boundary_target_role=str(boundary_target_role),
                circle_count=int(circle_count),
                instance_seed=int(instance_seed),
                params=task_params,
                namespace=f"{TASK_ID}.{selected_query}.overlap_case",
            )
            problem = _boundary_segment_problem(
                case=case,
                label_mode=str(label_mode),
                boundary_pair=str(boundary_pair),
                boundary_target_role=str(boundary_target_role),
                case_probabilities=case_probabilities,
                label_mode_probabilities=label_mode_probabilities,
                boundary_pair_probabilities=boundary_pair_probabilities,
                boundary_target_role_probabilities=boundary_target_role_probabilities,
                circle_count_probabilities=circle_count_probabilities,
            )
        else:
            raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")
        rendered: RenderedCenterlineOverlapScene | None = None
        render_context: Any | None = None
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_params = dict(task_params)
            attempt_params["_render_attempt"] = int(attempt)
            try:
                render_context = create_centerline_overlap_render_context(
                    instance_seed=int(instance_seed) + int(attempt),
                    params=attempt_params,
                    rendering_defaults=_RENDER_DEFAULTS,
                )
                rendered = render_centerline_overlap_scene(
                    render_context,
                    problem.diagram_spec,
                    instance_seed=int(instance_seed) + int(attempt),
                    render_namespace=f"{TASK_ID}.{selected_query}.render.scene",
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if rendered is None or render_context is None:
            raise RuntimeError(f"failed to generate {TASK_ID}") from last_error
        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        _prompt_defaults, prompt_artifacts = circle_centerline_prompt_artifacts(
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_query_key=str(selected_query),
            target_name=str(problem.target_name),
            label_mode=str(problem.label_mode),
            circle_count=int(problem.circle_count),
            answer_value=int(problem.answer),
            instance_seed=int(instance_seed),
        )
        annotation_value = segment_annotation(
            rendered,
            problem.diagram_spec.target_segment_points,
        )
        trace_payload = _trace_payload(
            rendered=rendered,
            image_size=image.size,
            render_context=render_context,
            noise_meta=dict(noise_meta),
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            prompt_artifacts=prompt_artifacts,
            problem=problem,
            annotation_value=annotation_value,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(problem.answer)),
            annotation_gt=TypedValue(type="segment", value=list(annotation_value)),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


_segment_length = segment_length
__all__ = [
    "BOUNDARY_PAIRS",
    "BOUNDARY_TARGET_ROLES",
    "GeometryCircleCenterlineOverlapSegmentLengthValueTask",
    "QUERY_ID_BOUNDARY_SEGMENT",
    "QUERY_ID_CENTER_DISTANCE",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "_segment_length",
]
