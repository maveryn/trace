"""Neutral lifecycle helpers for Raven-matrix public tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_query_spec,
)

from .shared.annotations import selected_option_bbox_annotation
from .shared.defaults import resolve_axes, resolve_render_params
from .shared.prompts import build_prompt
from .shared.rendering import render_raven_scene
from .shared.state import SCENE_ID, RavenAxes, RenderedRavenScene


_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class RavenSceneArtifacts:
    """Rendered scene, prompt, and annotation artifacts for one public task."""

    image: Image.Image
    prompt: str
    prompt_variants: dict[str, str]
    prompt_meta: dict[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    rendered_scene: RenderedRavenScene
    render_params: Any
    background_meta: dict[str, Any]
    scene_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    annotation_artifacts: AnnotationArtifacts


DatasetFactory = Callable[[int, Mapping[str, Any], RavenAxes], Mapping[str, Any]]
TaskFieldFactory = Callable[[RavenAxes, Mapping[str, Any]], Mapping[str, Any]]


def _json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    return value


def run_raven_matrix_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    namespace_base: str,
    prompt_task_key: str,
    prompt_query_key: str,
    dataset_factory: DatasetFactory,
    task_field_factory: TaskFieldFactory,
    question_format: str,
    view_family: str,
) -> TaskOutput:
    """Run common scene plumbing around task-owned Raven objective hooks."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            axes = resolve_axes(
                instance_seed=int(attempt_seed),
                params=params,
                generation_defaults=generation_defaults,
                namespace=str(namespace_base),
            )
            dataset = dict(dataset_factory(int(attempt_seed), params, axes))
            dataset.update(
                {
                    "scene_variant": str(axes.scene_variant),
                    "matrix_size": 3,
                    "target_cell_id": "cell_2_2",
                    "target_row_index": 2,
                    "target_col_index": 2,
                    "cell_count": 9,
                    "visible_matrix_cell_count": 8,
                }
            )
            artifacts = prepare_raven_scene(
                instance_seed=int(attempt_seed),
                params=params,
                rendering_defaults=rendering_defaults,
                prompt_defaults=prompt_defaults,
                dataset=dataset,
                scene_variant=str(axes.scene_variant),
                prompt_task_key=str(prompt_task_key),
                prompt_query_key=str(prompt_query_key),
                namespace_base=str(namespace_base),
            )
            answer_value = str(dataset["answer_option_label"])
            answer_gt = TypedValue(type="option_letter", value=str(answer_value))
            task_fields = {
                "query_id": str(selected_branch),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "option_count": int(axes.option_count),
                "answer_option_index_probabilities": dict(
                    axes.answer_option_probabilities
                ),
                "max_attempts": int(max_attempts),
                **dict(task_field_factory(axes, dataset)),
            }
            trace_payload = build_trace_payload(
                dataset=dataset,
                rendered_scene=artifacts.rendered_scene,
                render_params=artifacts.render_params,
                prompt_meta=artifacts.prompt_meta,
                task_fields=task_fields,
                background_meta=artifacts.background_meta,
                scene_style_meta=artifacts.scene_style_meta,
                post_noise_meta=artifacts.post_noise_meta,
                projected_annotation=artifacts.annotation_artifacts.projected_annotation,
                question_format=str(question_format),
                view_family=str(view_family),
            )
            trace_payload["query_spec"] = build_prompt_query_spec(
                prompt_artifacts=artifacts.prompt_artifacts,
                query_id=str(selected_branch),
                params=task_fields,
            )
            return TaskOutput(
                prompt=artifacts.prompt,
                prompt_variants=artifacts.prompt_variants,
                answer_gt=answer_gt,
                annotation_gt=artifacts.annotation_artifacts.annotation_gt,
                image=artifacts.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Raven-matrix task failed")


def prepare_raven_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    dataset: Mapping[str, Any],
    scene_variant: str,
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
) -> RavenSceneArtifacts:
    """Render one Raven scene and derive prompt/annotation artifacts."""

    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace_base)}.background",
    )
    render_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        cell_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        unknown_cell_fill_rgb=tuple(int(value) for value in scene_style.step_fill_rgb),
        option_panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        option_symbol_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        border_color_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        accent_color_rgb=tuple(int(value) for value in scene_style.mark_rgb),
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_raven_scene(
        background,
        scene_variant=str(scene_variant),
        matrix_rows=list(dataset["matrix_rows"]),
        option_specs=list(dataset["option_specs"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_meta, prompt_artifacts = build_prompt(
        prompt_defaults,
        scene_variant=str(scene_variant),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        instance_seed=int(instance_seed),
    )
    annotation_artifacts = selected_option_bbox_annotation(
        rendered_scene.option_cell_bbox_map,
        str(dataset["correct_option_panel_id"]),
    )
    return RavenSceneArtifacts(
        image=image,
        prompt=prompt,
        prompt_variants=prompt_variants,
        prompt_meta=prompt_meta,
        prompt_artifacts=prompt_artifacts,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        scene_style_meta=dict(scene_style_meta),
        post_noise_meta=dict(post_noise_meta),
        annotation_artifacts=annotation_artifacts,
    )


def build_trace_payload(
    *,
    dataset: Mapping[str, Any],
    rendered_scene: RenderedRavenScene,
    render_params: Any,
    prompt_meta: Mapping[str, Any],
    task_fields: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    question_format: str,
    view_family: str,
) -> dict[str, Any]:
    """Build scene/render/execution trace fields for one Raven task."""

    option_slot_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.option_panel_bbox_map.items()
    }
    option_cell_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.option_cell_bbox_map.items()
    }
    matrix_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.matrix_cell_bbox_map.items()
    }
    return {
        "scene_ir": {
            "scene_kind": f"puzzle_raven_{dataset.get('scene_variant', '')}",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.get("scene_variant", "")),
                "answer_option_label": str(dataset["answer_option_label"]),
                "correct_option_panel_id": str(dataset["correct_option_panel_id"]),
                "target_cell_id": str(dataset["target_cell_id"]),
                "view_family": str(view_family),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.get("scene_variant", "")),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [
                round(float(value), 3) for value in rendered_scene.scene_bbox_px
            ],
            "text_style": {
                "value_font_size_px": int(render_params.value_font_size_px),
                "option_label_font_size_px": int(render_params.option_label_font_size_px),
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": [
                    round(float(value), 3) for value in rendered_scene.scene_bbox_px
                ],
                "matrix_cell_bboxes_px": dict(matrix_bboxes),
                "option_panel_bboxes_px": dict(option_slot_bboxes),
                "option_cell_bboxes_px": dict(option_cell_bboxes),
                "annotation_source": "option_cell_bboxes_px",
            },
            render_params.unit_size_jitter,
        ),
        "execution_trace": {
            **dict(task_fields),
            "scene_variant": str(dataset.get("scene_variant", "")),
            "question_format": str(question_format),
            "view_family": str(view_family),
            "target_cell_id": str(dataset["target_cell_id"]),
            "target_row_index": int(dataset["target_row_index"]),
            "target_col_index": int(dataset["target_col_index"]),
            "matrix_size": int(dataset["matrix_size"]),
            "cell_count": int(dataset["cell_count"]),
            "visible_matrix_cell_count": int(dataset["visible_matrix_cell_count"]),
            "matrix_rows": [
                [dict(cell) for cell in row] for row in dataset["matrix_rows"]
            ],
            "matrix_panel_specs": [
                [dict(spec) for spec in row] for row in dataset["matrix_panel_specs"]
            ],
            "answer_panel_spec": dict(dataset["answer_panel_spec"]),
            "answer_option_label": str(dataset["answer_option_label"]),
            "correct_option_index": int(dataset["correct_option_index"]),
            "correct_option_panel_id": str(dataset["correct_option_panel_id"]),
            "option_count": int(dataset["option_count"]),
            "option_specs": [dict(option) for option in dataset["option_specs"]],
            "solver_trace": dict(dataset["solver_trace"]),
            "supporting_option_panel_ids": [str(dataset["correct_option_panel_id"])],
        },
        "witness_symbolic": {
            "type": str(projected_annotation.get("type", "")),
            "value": projected_annotation.get("bbox", projected_annotation.get("value")),
        },
        "projected_annotation": dict(projected_annotation),
        "prompt_spec": {
            "bundle_id": str(prompt_meta["bundle_id"]),
            "active": dict(prompt_meta["prompt_variant"]),
        },
        "variant_payload": {
            str(key): _json_ready(value)
            for key, value in dict(dataset).items()
            if str(key) not in {"option_specs", "solver_trace"}
        },
    }


__all__ = [
    "RavenSceneArtifacts",
    "build_trace_payload",
    "prepare_raven_scene",
    "run_raven_matrix_task",
]
