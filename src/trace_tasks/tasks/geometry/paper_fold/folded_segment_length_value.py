"""Folded-paper side length task."""

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

from .shared.annotations import paper_fold_segment_annotation
from .shared.construction import (
    fold_segment_answer_cases,
    fold_segment_answer_support,
    fold_segment_geometry,
)
from .shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_DEFAULTS,
    SCENE_ID,
    SCENE_KIND,
)
from .shared.prompts import paper_fold_segment_prompt_artifacts
from .shared.rendering import make_render_context, render_paper_fold_segment_scene
from .shared.state import FoldSegmentCase, FoldSegmentPlan, RenderedPaperFoldScene
from ..shared.annotation_values import PixelAnnotationArtifacts

TASK_ID = "task_geometry__paper_fold__folded_segment_length_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
PROMPT_TASK_KEY = "folded_segment_length_value"
OBJECT_DESCRIPTION = (
    "a folded paper corner with dashed original edges, a crease, a folded flap, and side-length labels"
)
TARGET_SEGMENTS = ("EP", "FP")
TASK_SCENE_VARIANT = "corner_fold_side_correspondence"


def _case_from_dimensions(*, leg_ae: int, leg_af: int, target_segment: str) -> FoldSegmentCase:
    """Bind one target folded segment to an integer right-triangle construction."""

    geometry = fold_segment_geometry(int(leg_ae), int(leg_af))
    if str(target_segment) == "EP":
        return FoldSegmentCase(
            leg_ae=int(geometry.leg_ae),
            leg_af=int(geometry.leg_af),
            crease_ef=int(geometry.crease_ef),
            target_segment="EP",
            known_leg_segment="AF",
            target_answer=int(geometry.leg_ae),
        )
    if str(target_segment) == "FP":
        return FoldSegmentCase(
            leg_ae=int(geometry.leg_ae),
            leg_af=int(geometry.leg_af),
            crease_ef=int(geometry.crease_ef),
            target_segment="FP",
            known_leg_segment="AE",
            target_answer=int(geometry.leg_af),
        )
    raise ValueError(f"unsupported target_segment for {TASK_ID}: {target_segment!r}")


def _resolve_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> FoldSegmentPlan:
    """Bind the folded-segment objective to one answer-balanced Pythagorean case."""

    del gen_defaults
    support_values = sorted({int(value) for value in fold_segment_answer_support()})
    support_probabilities = probability_map(tuple(str(value) for value in support_values))
    explicit_leg_ae = params.get("leg_ae")
    explicit_leg_af = params.get("leg_af")
    explicit_target = params.get("target_segment")
    if explicit_leg_ae is not None or explicit_leg_af is not None:
        if explicit_leg_ae is None or explicit_leg_af is None:
            raise ValueError("paper fold segment overrides require both leg_ae and leg_af")
        target_segment = str(explicit_target or "FP")
        case = _case_from_dimensions(
            leg_ae=int(explicit_leg_ae),
            leg_af=int(explicit_leg_af),
            target_segment=target_segment,
        )
        selected_support_answer = int(case.target_answer)
    else:
        answer_cases = fold_segment_answer_cases()
        if not answer_cases:
            raise ValueError("paper fold segment answer support must not be empty")
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.single.answer_support")
        selected_support_answer, selected_cases = uniform_choice(rng, answer_cases)
        filtered_cases = tuple(
            case
            for case in selected_cases
            if explicit_target is None or str(case.target_segment) == str(explicit_target)
        )
        if not filtered_cases:
            raise ValueError(f"target_segment={explicit_target!r} has no compatible support cases")
        rng = spawn_rng(
            int(instance_seed),
            f"{TASK_ID}.single.answer_case.{int(selected_support_answer)}",
        )
        case = uniform_choice(rng, filtered_cases)

    if str(case.target_segment) not in TARGET_SEGMENTS:
        raise ValueError(f"unsupported target_segment for {TASK_ID}: {case.target_segment!r}")
    geometry = fold_segment_geometry(int(case.leg_ae), int(case.leg_af))
    answer = int(case.target_answer)
    pythagorean_unknown = "AE" if str(case.target_segment) == "EP" else "AF"
    return FoldSegmentPlan(
        answer=int(answer),
        geometry=geometry,
        case=case,
        params={
            "leg_ae": int(geometry.leg_ae),
            "leg_af": int(geometry.leg_af),
            "crease_ef": int(geometry.crease_ef),
            "known_leg_segment": str(case.known_leg_segment),
            "target_segment": str(case.target_segment),
            "pythagorean_unknown_original_segment": str(pythagorean_unknown),
            "answer_value": int(answer),
            "answer_support": [str(value) for value in support_values],
            "answer_support_size": int(len(support_values)),
            "selected_support_answer": int(selected_support_answer),
            "formula_family": "pythagorean_leg_then_fold_correspondence",
            "reasoning_steps": 2,
        },
        support_probabilities=dict(support_probabilities),
    )


def _trace_payload(
    *,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    plan: FoldSegmentPlan,
    rendered: RenderedPaperFoldScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    prompt_artifacts: PromptTraceArtifacts,
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build verifier trace after integer answer and segment annotation are bound."""

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
                "scene_variant": TASK_SCENE_VARIANT,
                "answer_value": int(rendered.answer),
                "annotation_type": annotation_artifacts.annotation_type,
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
            "scene_variant": TASK_SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "answer_type": "integer",
            "answer_value": int(rendered.answer),
            "annotation_type": annotation_artifacts.annotation_type,
            "reasoning_steps": int(rendered.reasoning_steps),
            **dict(rendered.witness),
        },
        "witness_symbolic": {
            "type": "paper_fold_folded_segment_length",
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_value": int(rendered.answer),
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **dict(rendered.witness),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


def _render_segment_scene(
    *,
    plan: FoldSegmentPlan,
    instance_seed: int,
    task_params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
) -> tuple[RenderedPaperFoldScene, dict[str, Any]]:
    """Render this objective with retry while preserving one plan-bound answer."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            ctx, render_meta_attempt = make_render_context(
                instance_seed=int(instance_seed) + int(attempt),
                params=task_params,
                render_defaults=render_defaults,
            )
            rendered = render_paper_fold_segment_scene(ctx, plan)
            render_meta = dict(render_meta_attempt)
            render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            return rendered, render_meta
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to render {TASK_ID}") from last_error


@register_task
class GeometryPaperFoldedSegmentLengthValueTask:
    """Task-owned folded-paper side-length objective."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'formula_evaluation')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Own the segment objective lifecycle from query binding to TaskOutput."""

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
        rendered, render_meta = _render_segment_scene(
            plan=plan,
            instance_seed=int(instance_seed),
            task_params=task_params,
            render_defaults=render_defaults,
            max_attempts=int(max_attempts),
        )
        if rendered.annotation_segment is None:
            raise RuntimeError(f"{TASK_ID} rendered without target segment annotation")

        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        annotation_artifacts = paper_fold_segment_annotation(rendered.annotation_segment)
        _prompt_defaults, prompt_artifacts = paper_fold_segment_prompt_artifacts(
            prompt_defaults=prompt_defaults,
            prompt_task_key=PROMPT_TASK_KEY,
            object_description=OBJECT_DESCRIPTION,
            target_segment=str(plan.case.target_segment),
            answer=int(rendered.answer),
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
            answer_gt=TypedValue(type="integer", value=int(rendered.answer)),
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


__all__ = ["GeometryPaperFoldedSegmentLengthValueTask"]
