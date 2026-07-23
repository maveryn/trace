"""Compute cuboid surface area from three orthographic perimeter views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import required_group_defaults, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .shared.rendering import NOISE_DEFAULTS, render_cuboid_views_scene
from .shared.state import CuboidDimensions, RenderedCuboidViewsScene

DOMAIN = "geometry"
SCENE_ID = "cuboid_views"
TASK_ID = "task_geometry__cuboid_views__cuboid_projection_surface_area_value"
PROMPT_BUNDLE_ID = "geometry_cuboid_orthographic_views_v1"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
FORMULA_SCHEMA = "surface_area_from_orthographic_views"
FORMULA_TEXT = "recover L, W, H from the three view perimeters, then SA = 2(LW + LH + WH)"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)

_CUBOID_CASES: Tuple[Tuple[int, int, int], ...] = (
    (3, 4, 5),
    (3, 4, 7),
    (3, 4, 8),
    (3, 5, 7),
    (3, 5, 8),
    (3, 6, 8),
    (4, 5, 8),
    (4, 6, 7),
    (4, 5, 9),
    (3, 7, 9),
    (4, 6, 9),
    (3, 8, 9),
    (4, 5, 12),
    (4, 6, 11),
    (4, 7, 10),
    (4, 8, 9),
    (4, 6, 12),
    (6, 7, 8),
    (4, 7, 11),
    (5, 6, 11),
    (4, 7, 12),
    (5, 6, 12),
    (4, 9, 10),
    (5, 8, 10),
    (6, 7, 10),
    (5, 7, 12),
    (5, 8, 11),
    (7, 8, 9),
    (4, 10, 11),
    (6, 9, 10),
    (6, 9, 11),
    (6, 7, 14),
    (5, 10, 12),
    (6, 10, 11),
    (6, 7, 15),
    (7, 9, 11),
    (5, 9, 14),
    (8, 9, 10),
    (7, 8, 13),
    (8, 9, 11),
    (5, 11, 13),
    (6, 9, 14),
    (7, 9, 13),
    (7, 10, 12),
    (5, 10, 15),
    (8, 9, 12),
    (7, 8, 15),
    (7, 9, 14),
    (8, 10, 12),
    (5, 12, 14),
    (6, 11, 14),
    (6, 12, 13),
    (8, 9, 14),
    (8, 11, 12),
    (9, 10, 13),
    (6, 13, 14),
    (8, 10, 15),
    (9, 10, 14),
    (6, 13, 15),
    (7, 13, 14),
    (8, 11, 15),
    (9, 10, 15),
    (9, 11, 14),
    (9, 12, 13),
    (6, 14, 15),
    (8, 13, 14),
    (11, 12, 13),
    (8, 14, 15),
    (11, 12, 14),
    (9, 14, 15),
    (10, 13, 15),
    (11, 13, 15),
)


@dataclass(frozen=True)
class _ResolvedProblem:
    """Task-owned sampling and answer binding for the surface-area objective."""

    query_id: str
    query_probabilities: Dict[str, float]
    dimensions: CuboidDimensions
    answer: int
    support_probabilities: Dict[str, float]


def _validate_query_id(params: Mapping[str, Any]) -> tuple[str, Dict[str, float]]:
    explicit = params.get("query_id")
    if explicit is not None and str(explicit) not in set(SUPPORTED_QUERY_IDS):
        raise ValueError(f"unsupported query_id for {TASK_ID}: {explicit}")
    return QUERY_ID, {QUERY_ID: 1.0}


def _resolve_dimensions(*, instance_seed: int, params: Mapping[str, Any]) -> CuboidDimensions:
    explicit_dimensions = params.get("dimensions")
    if explicit_dimensions is not None:
        values = tuple(int(value) for value in explicit_dimensions)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError("dimensions must contain length, width, and height")
        length, width, height = values
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.case")
        length, width, height = uniform_choice(rng, _CUBOID_CASES)
        length = int(params.get("length_units", length))
        width = int(params.get("width_units", width))
        height = int(params.get("height_units", height))
    if length <= 0 or width <= 0 or height <= 0:
        raise ValueError("cuboid length, width, and height must be positive")
    return CuboidDimensions(length=int(length), width=int(width), height=int(height))


def _resolve_problem(*, instance_seed: int, params: Mapping[str, Any]) -> _ResolvedProblem:
    """Resolve the single public query and choose one dimensions case."""

    query_id, query_probabilities = _validate_query_id(params)
    dimensions = _resolve_dimensions(instance_seed=int(instance_seed), params=params)
    support_values = tuple(CuboidDimensions(*case).surface_area for case in _CUBOID_CASES)
    answer = int(dimensions.surface_area)
    return _ResolvedProblem(
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
        dimensions=dimensions,
        answer=int(answer),
        support_probabilities=geometry_selected_probability_map(
            tuple(sorted({int(value) for value in support_values} | {int(answer)})),
            int(answer),
            key_fn=lambda value: str(int(value)),
            is_selected=lambda value, selected: int(value) == int(selected),
        ),
    )


def _prompt_artifacts(
    *,
    prompt_defaults_all: Mapping[str, Any],
    problem: _ResolvedProblem,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render prompt variants using only external prompt prose."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_all,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context=f"prompt defaults for {TASK_ID}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(problem.query_id),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
        preferred_mode=str(params.get("prompt_mode", "answer_and_annotation")),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _symbolic_witness(problem: _ResolvedProblem) -> Dict[str, Any]:
    dimensions = problem.dimensions
    return {
        "formula_family": FORMULA_SCHEMA,
        "formula_schema": FORMULA_SCHEMA,
        "length": int(dimensions.length),
        "width": int(dimensions.width),
        "height": int(dimensions.height),
        "top_view_perimeter": int(dimensions.top_view_perimeter),
        "front_view_perimeter": int(dimensions.front_view_perimeter),
        "right_view_perimeter": int(dimensions.right_view_perimeter),
        "volume": int(dimensions.volume),
        "surface_area": int(dimensions.surface_area),
        "formula": FORMULA_TEXT,
        "answer_value": int(problem.answer),
    }


def _trace_payload(
    *,
    problem: _ResolvedProblem,
    rendered: RenderedCuboidViewsScene,
    prompt_artifacts: Any,
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: Sequence[int],
    annotation_bbox_map: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    """Build verifier payload from the same problem and render trace."""

    witness = _symbolic_witness(problem)
    query_params = {
        "scene_id": SCENE_ID,
        "scene_variant": "three_view_cuboid_projection",
        "query_id": str(problem.query_id),
        "query_id_probabilities": dict(problem.query_probabilities),
        "formula_schema": FORMULA_SCHEMA,
        "target_support_probabilities": dict(problem.support_probabilities),
        **dict(witness),
    }
    return {
        "scene_ir": {
            "scene_kind": "geometry_cuboid_orthographic_views",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(problem.query_id),
                "scene_variant": "three_view_cuboid_projection",
                "formula_schema": FORMULA_SCHEMA,
                "answer_value": int(problem.answer),
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "query_spec": {
            "scene_id": SCENE_ID,
            "query_id": str(problem.query_id),
            "template_id": PROMPT_BUNDLE_ID,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_params),
        },
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": "three_view_cuboid_projection",
            "query_id": str(problem.query_id),
            "query_id_probabilities": dict(problem.query_probabilities),
            "answer_type": "integer",
            "answer_value": int(problem.answer),
            "answer_rounding": "integer",
            "annotation_roles": list(rendered.annotation_roles),
            "reasoning_steps": 2,
            **dict(witness),
        },
        "witness_symbolic": {
            "type": "cuboid_orthographic_formula",
            "scene_id": SCENE_ID,
            "query_id": str(problem.query_id),
            "answer_value": int(problem.answer),
            "source_witness_type": "bbox_map",
            "original_annotation_value": list(rendered.annotation_roles),
            **dict(witness),
        },
        "projected_annotation": {
            "type": "bbox_map",
            "bbox_map": dict(annotation_bbox_map),
            "pixel_bbox_map": dict(annotation_bbox_map),
        },
    }


@register_task
class GeometryCuboidProjectionSurfaceAreaValueTask:
    """Compute cuboid surface area from orthographic-view perimeters."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    reasoning_kind = "surface_area"

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one surface-area instance; task file owns output binding."""

        generation_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
            _SCENE_DEFAULTS,
            task_id=TASK_ID,
        )
        _ = generation_defaults
        problem = _resolve_problem(instance_seed=int(instance_seed), params=params)
        last_error: Exception | None = None
        rendered: RenderedCuboidViewsScene | None = None
        render_meta: Dict[str, Any] | None = None
        for _attempt in range(max(1, int(max_attempts))):
            try:
                rendered_attempt, render_meta_attempt = render_cuboid_views_scene(
                    dimensions=problem.dimensions,
                    instance_seed=int(instance_seed),
                    params=params,
                    render_defaults=render_defaults,
                )
                rendered = rendered_attempt
                render_meta = dict(render_meta_attempt)
                break
            except Exception as exc:
                last_error = exc
        if rendered is None or render_meta is None:
            raise RuntimeError(f"failed to generate {TASK_ID}") from last_error

        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=NOISE_DEFAULTS,
        )
        annotation_bbox_map = {
            str(role): bbox_to_list(bbox)
            for role, bbox in rendered.annotation_bboxes.items()
        }
        prompt_artifacts = _prompt_artifacts(
            prompt_defaults_all=prompt_defaults,
            problem=problem,
            params=params,
            instance_seed=int(instance_seed),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(problem.answer)),
            annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_bbox_map)),
            image=image,
            image_id="img0",
            trace_payload=_trace_payload(
                problem=problem,
                rendered=rendered,
                prompt_artifacts=prompt_artifacts,
                render_meta=render_meta,
                noise_meta=noise_meta,
                image_size=image.size,
                annotation_bbox_map=annotation_bbox_map,
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(problem.query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryCuboidProjectionSurfaceAreaValueTask",
    "QUERY_ID",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
