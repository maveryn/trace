"""Folded-paper angle value task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import (
    probability_map,
    resolve_task_query_id_param,
    strip_query_id_params,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import paper_fold_bbox_annotation
from .shared.construction import fold_answer_cases, fold_answer_support, fold_geometry
from .shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_DEFAULTS,
    SCENE_ID,
    SCENE_KIND,
    SCENE_VARIANT,
)
from .shared.prompts import paper_fold_prompt_artifacts
from .shared.rendering import make_render_context, render_paper_fold_scene
from .shared.state import FoldAnglePlan, RenderedPaperFoldScene
from ..shared.measurement_rendering import round1
from ..shared.annotation_values import PixelAnnotationArtifacts

TASK_ID = "task_geometry__paper_fold__paper_fold_angle_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
PROMPT_TASK_KEY = "paper_fold_angle_value"
OBJECT_DESCRIPTION = (
    "a folded paper corner with dashed original edges, a crease, a folded edge, and angle labels"
)
ANNOTATION_KEYS = ("target_angle_cue", "given_angle_label")


def _resolve_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> FoldAnglePlan:
    """Bind the single fold-angle objective to one valid numeric construction."""

    del gen_defaults
    answer_cases = fold_answer_cases()
    if not answer_cases:
        raise ValueError("paper fold answer support must not be empty")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.single.answer_support")
    selected_support_answer, selected_cases = uniform_choice(rng, answer_cases)
    rng = spawn_rng(
        int(instance_seed),
        f"{TASK_ID}.single.answer_case.{selected_support_answer:.1f}",
    )
    default_height, default_offset = uniform_choice(rng, tuple(selected_cases))
    height_units = float(params.get("height_units", default_height))
    folded_offset_units = float(params.get("folded_offset_units", default_offset))
    if not (2.0 < folded_offset_units < height_units):
        raise ValueError("paper fold requires 2 < folded_offset_units < height_units")

    geometry = fold_geometry(height_units, folded_offset_units)
    answer = round1(geometry.half_angle_degrees)
    support_values = sorted({float(value) for value in fold_answer_support()})
    support_probabilities = probability_map(tuple(f"{float(value):.1f}" for value in support_values))
    total_angle = round1(2.0 * answer)
    given_angle = round1(180.0 - total_angle)
    return FoldAnglePlan(
        answer=float(answer),
        geometry=geometry,
        params={
            "height_units": float(height_units),
            "folded_offset_units": float(folded_offset_units),
            "upper_segment_units": float(round1(geometry.upper_segment_units)),
            "lower_segment_units": float(round1(geometry.lower_segment_units)),
            "half_angle_degrees": float(round1(geometry.half_angle_degrees)),
            "total_angle_degrees": float(total_angle),
            "given_angle_degrees": float(given_angle),
            "known_angle_degrees": float(given_angle),
            "target_role": "half_angle_x",
            "answer_value": float(answer),
            "answer_support": [f"{float(value):.1f}" for value in support_values],
            "answer_support_size": int(len(support_values)),
            "selected_support_answer": float(selected_support_answer),
            "formula_family": "fold_bisector_with_straight_angle",
            "reasoning_steps": 2,
        },
        support_probabilities=dict(support_probabilities),
    )


def _trace_payload(
    *,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    plan: FoldAnglePlan,
    rendered: RenderedPaperFoldScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    prompt_artifacts: PromptTraceArtifacts,
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build the verifier trace after answer and annotation are bound."""

    annotation_roles = [str(role) for role in annotation_artifacts.value.keys()]
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
        "target_support_probabilities": dict(plan.support_probabilities),
        **dict(plan.params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    return {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_query),
                "scene_variant": SCENE_VARIANT,
                "answer_value": float(rendered.answer),
                "annotation_roles": list(annotation_roles),
                **dict(rendered.witness),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "answer_type": "number",
            "answer_value": float(rendered.answer),
            "answer_rounding": "one_decimal",
            "annotation_roles": list(annotation_roles),
            "reasoning_steps": int(rendered.reasoning_steps),
            **dict(rendered.witness),
        },
        "witness_symbolic": {
            "type": "paper_fold_angle_bisector",
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_value": float(rendered.answer),
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **dict(rendered.witness),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


@register_task
class GeometryPaperFoldAngleValueTask:
    """Task-owned folded-paper angle objective."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'formula_evaluation')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Run the full public task lifecycle after binding the single angle objective.

        The public file owns query validation, answer/annotation binding, prompt slots,
        trace payload construction, and final TaskOutput assembly; shared helpers only
        provide identity-free fold construction, rendering, prompt, and annotation utilities.
        """

        selected_query = resolve_task_query_id_param(
            params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id="single",
            task_id=TASK_ID,
        )
        task_params = strip_query_id_params(params)
        query_probabilities = {
            query: (1.0 if query == str(selected_query) else 0.0)
            for query in SUPPORTED_QUERY_IDS
        }
        gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
            SCENE_DEFAULTS,
            task_id=TASK_ID,
        )
        plan = _resolve_plan(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=gen_defaults,
        )

        rendered: RenderedPaperFoldScene | None = None
        render_meta: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                ctx, render_meta_attempt = make_render_context(
                    instance_seed=int(instance_seed) + int(attempt),
                    params=task_params,
                    render_defaults=render_defaults,
                )
                rendered = render_paper_fold_scene(ctx, plan)
                render_meta = dict(render_meta_attempt)
                render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
                break
            except Exception as exc:
                last_error = exc
        if rendered is None or render_meta is None:
            raise RuntimeError(f"failed to render {TASK_ID}") from last_error

        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        annotation_artifacts = paper_fold_bbox_annotation(
            rendered.annotation_bboxes,
            roles=ANNOTATION_KEYS,
        )
        _prompt_defaults, prompt_artifacts = paper_fold_prompt_artifacts(
            prompt_defaults=prompt_defaults,
            prompt_task_key=PROMPT_TASK_KEY,
            object_description=OBJECT_DESCRIPTION,
            annotation_keys=ANNOTATION_KEYS,
            answer=float(rendered.answer),
            instance_seed=int(instance_seed),
        )
        trace_payload = _trace_payload(
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            plan=plan,
            rendered=rendered,
            annotation_artifacts=annotation_artifacts,
            prompt_artifacts=prompt_artifacts,
            render_meta=render_meta,
            noise_meta=noise_meta,
            image_size=(int(image.size[0]), int(image.size[1])),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="number", value=float(rendered.answer)),
            annotation_gt=TypedValue(
                type=annotation_artifacts.annotation_type,
                value=annotation_artifacts.value,
            ),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryPaperFoldAngleValueTask"]
