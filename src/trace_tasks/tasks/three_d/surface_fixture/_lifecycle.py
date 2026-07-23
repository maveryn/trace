"""Scene-private lifecycle orchestration for surface-fixture public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.scene_config import get_domain_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.three_d.shared.canvas import render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import _resolve_render_params

from .shared.annotations import bbox_set_annotation_for_elements_with_metadata
from .shared.prompts import build_prompt_artifacts, dynamic_slots_for_surface
from .shared.rendering import render_surface_fixture
from .shared.sampling import ResolvedSurfaceFixtureAxes, resolve_scene_and_element
from .shared.state import SCENE_ID


ObjectivePreparer = Callable[
    [
        int,
        Mapping[str, Any],
        ResolvedSurfaceFixtureAxes,
        Mapping[str, float],
        str,
    ],
    "SurfaceFixturePlan",
]


@dataclass(frozen=True)
class SurfaceFixturePlan:
    """Task-owned objective data for one surface-fixture instance."""

    dataset: Mapping[str, Any]
    answer_gt: TypedValue
    target_element_ids: tuple[str, ...]
    answer_value_probabilities: Mapping[str, float]
    object_description: str
    objective_params: Mapping[str, Any]
    execution_extra: Mapping[str, Any] | None = None


_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _attempt_seed(instance_seed: int, *, public_name: str, attempt_index: int) -> int:
    if int(attempt_index) == 0:
        return int(instance_seed)
    return int(spawn_rng(int(instance_seed), f"{public_name}.attempt_seed.{attempt_index}").randrange(1, 2**62))


def _surface_trace_params(
    *,
    axes: ResolvedSurfaceFixtureAxes,
    plan: SurfaceFixturePlan,
    branch_probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    dataset = dict(plan.dataset)
    params: Dict[str, Any] = {
        "query_id_probabilities": dict(branch_probabilities),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "target_element_type": str(dataset["target_element_type"]),
        "target_element_type_probabilities": dict(axes.element_type_probabilities),
        "answer_value": int(dataset["answer_value"]),
        "answer_value_probabilities": dict(plan.answer_value_probabilities),
    }
    params.update(dict(plan.objective_params))
    return params


def _build_surface_trace_payload(
    *,
    public_name: str,
    selected_branch: str,
    axes: ResolvedSurfaceFixtureAxes,
    plan: SurfaceFixturePlan,
    rendered: Any,
    annotation_artifacts: AnnotationArtifacts,
    annotation_bboxes_by_element_id: Mapping[str, Sequence[float]],
    annotation_bbox_normalization: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    render_params: Any,
    image: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble common trace sections after task-specific answer binding."""

    dataset = dict(plan.dataset)
    target_ids = [str(element_id) for element_id in plan.target_element_ids]
    target_raw_bboxes = {
        str(element_id): list(rendered.element_bboxes_px[str(element_id)])
        for element_id in target_ids
    }
    target_bboxes = {
        str(element_id): list(annotation_bboxes_by_element_id[str(element_id)])
        for element_id in target_ids
    }
    target_centers = {
        str(element_id): list(rendered.element_centers_px[str(element_id)])
        for element_id in target_ids
    }
    solver_trace = dict(dataset["solver_trace"])
    solver_trace.update(
        {
            "answer_value": int(dataset["answer_value"]),
            "target_element_ids": list(target_ids),
            "target_element_bboxes_px": dict(target_bboxes),
            "target_element_raw_bboxes_px": dict(target_raw_bboxes),
            "target_element_centers_px": dict(target_centers),
        }
    )
    execution_trace = {
        "query_id": str(selected_branch),
        "scene_variant": str(axes.scene_variant),
        "answer_value": int(dataset["answer_value"]),
        "target_element_type": str(dataset["target_element_type"]),
        "target_element_name": str(dataset["target_element_name"]),
        "target_element_plural": str(dataset["target_element_plural"]),
        "target_element_ids": list(target_ids),
        "target_element_bboxes_px": dict(target_bboxes),
        "target_element_raw_bboxes_px": dict(target_raw_bboxes),
        "target_element_centers_px": dict(target_centers),
        "layout_rows": int(dataset["layout_rows"]),
        "layout_columns": int(dataset["layout_columns"]),
        "layout_style": str(dataset["layout_style"]),
        "surface_cells": [dict(cell) for cell in dataset["surface_cells"]],
        "question_format": str(selected_branch),
        "solver_trace": dict(solver_trace),
    }
    if "layout_family" in dataset:
        execution_trace["layout_family"] = str(dataset["layout_family"])
        execution_trace["layout_family_probabilities"] = dict(dataset.get("layout_family_probabilities", {}))
        execution_trace["layout_style_probabilities"] = dict(dataset.get("layout_style_probabilities", {}))
    if plan.execution_extra:
        execution_trace.update(dict(plan.execution_extra))
    scene_relations = {
        "scene_variant": str(axes.scene_variant),
        "fixture_display_name": str(dataset["fixture_display_name"]),
        "target_element_type": str(dataset["target_element_type"]),
        "target_element_name": str(dataset["target_element_name"]),
        "target_element_plural": str(dataset["target_element_plural"]),
        "answer_value": int(dataset["answer_value"]),
        "target_element_ids": list(target_ids),
        "layout_rows": int(dataset["layout_rows"]),
        "layout_columns": int(dataset["layout_columns"]),
        "layout_style": str(dataset["layout_style"]),
    }
    if "layout_family" in dataset:
        scene_relations["layout_family"] = str(dataset["layout_family"])
    return {
        "scene_ir": {
            "scene_kind": f"three_d_surface_fixture_{public_name.rsplit('__', 1)[-1]}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(scene_relations),
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "scene_canvas_preset": str(render_params.canvas_preset),
            "scene_canvas_width": int(render_params.canvas_width),
            "scene_canvas_height": int(render_params.canvas_height),
            "scene_canvas_policy": str(render_params.canvas_policy),
            **render_params_canvas_metadata(render_params),
            "final_canvas_width": int(image.width),
            "final_canvas_height": int(image.height),
            "final_canvas_pixels": int(image.width) * int(image.height),
            "coord_space": "pixel",
            "scene_variant": str(axes.scene_variant),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "projection_model": "synthetic_perspective_panel_v0",
            "surface_world_corners": [list(point) for point in dataset["surface_world_corners"]],
            "annotation_bbox_normalization": dict(annotation_bbox_normalization),
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "fixture_bbox_px": list(rendered.fixture_bbox_px),
            "element_bboxes_px": dict(rendered.element_bboxes_px),
            "element_centers_px": dict(rendered.element_centers_px),
            "target_element_bboxes_px": dict(target_bboxes),
            "target_element_raw_bboxes_px": dict(target_raw_bboxes),
            "target_element_centers_px": dict(target_centers),
            "annotation_raw_bboxes_px": [list(target_raw_bboxes[element_id]) for element_id in target_ids],
            "annotation_bboxes_px": [list(target_bboxes[element_id]) for element_id in target_ids],
            "annotation_bbox_normalization": dict(annotation_bbox_normalization),
        },
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "counted_surface_element_set",
            "element_ids": list(target_ids),
            "target_element_type": str(dataset["target_element_type"]),
            "answer_value": int(dataset["answer_value"]),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def run_surface_fixture_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key: str,
    supported_branches: Sequence[str],
    default_branch: str,
    supported_scenes: Sequence[str],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common query selection, rendering, prompt, annotation, and output assembly."""

    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_scene_and_element(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
        supported_scenes=tuple(str(scene) for scene in supported_scenes),
    )
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            plan = prepare_objective(int(attempt_seed), clean_params, axes, branch_probabilities, selected_branch)
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_surface_fixture(background, dataset=plan.dataset, render_params=render_params)
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts, annotation_bboxes_by_element_id, annotation_bbox_normalization = (
                bbox_set_annotation_for_elements_with_metadata(rendered, plan.target_element_ids)
            )
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_surface(
                    plan.dataset,
                    object_description=str(plan.object_description),
                ),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_surface_trace_params(
                    axes=axes,
                    plan=plan,
                    branch_probabilities=branch_probabilities,
                ),
            )
            trace_payload = _build_surface_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                annotation_bboxes_by_element_id=annotation_bboxes_by_element_id,
                annotation_bbox_normalization=annotation_bbox_normalization,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid surface fixture after {max_attempts} attempts: {last_error}")


__all__ = [
    "ResolvedSurfaceFixtureAxes",
    "SurfaceFixturePlan",
    "run_surface_fixture_lifecycle",
]
