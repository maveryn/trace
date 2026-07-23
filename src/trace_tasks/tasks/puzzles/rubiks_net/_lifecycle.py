"""Neutral lifecycle helpers for Rubik cube-net public tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

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
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_query_spec,
)

from .shared.annotations import selected_option_bbox_annotation
from .shared.defaults import resolve_axes, resolve_render_params
from .shared.prompts import build_prompt
from .shared.rendering import render_rubiks_scene
from .shared.rules import signature_for_trace, state_for_trace, state_signature
from .shared.state import SCENE_ID, RenderedRubiksScene, RubiksAxes

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class RubiksSceneArtifacts:
    """Rendered scene, prompt, and annotation artifacts for one public task."""

    image: Image.Image
    prompt: str
    prompt_variants: dict[str, str]
    prompt_meta: dict[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    rendered_scene: RenderedRubiksScene
    render_params: Any
    background_meta: dict[str, Any]
    scene_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    annotation_artifacts: AnnotationArtifacts


DatasetFactory = Callable[[Any, int, Mapping[str, Any], RubiksAxes], Mapping[str, Any]]
DatasetFactoryFactory = Callable[[str], DatasetFactory]
TaskFieldFactory = Callable[[RubiksAxes, Mapping[str, Any]], Mapping[str, Any]]


def _json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    return value


def _option_specs_for_trace(option_specs) -> list[dict[str, Any]]:
    """Serialize option specs without tuple-key state maps."""

    serialized = []
    for option in option_specs:
        item = {
            str(key): value
            for key, value in dict(option).items()
            if str(key) != "state"
        }
        if "state" in option:
            item["state"] = state_for_trace(option["state"])
            item["state_signature"] = signature_for_trace(
                state_signature(option["state"])
            )
        serialized.append(item)
    return serialized


def _solver_trace_for_trace(solver_trace: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize solver trace values into canonical JSON-compatible structures."""

    serialized: dict[str, Any] = {}
    for key, value in solver_trace.items():
        if str(key).endswith("state_signature"):
            serialized[str(key)] = signature_for_trace(value)
        else:
            serialized[str(key)] = _json_ready(value)
    return serialized


def run_public_rubiks_task(
    *,
    public_identity: str,
    supported_branches: tuple[str, ...],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    namespace_base: str,
    prompt_task_key: str,
    prompt_query_key: str | None,
    object_description_key: str,
    task_field_factory: TaskFieldFactory,
    question_format: str | None,
    dataset_factory: DatasetFactory | None = None,
    dataset_factory_factory: DatasetFactoryFactory | None = None,
) -> TaskOutput:
    """Select the public branch and then run the neutral Rubik scene lifecycle."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(item) for item in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_identity),
        namespace=f"{namespace_base}.branch",
    )
    selected_factory = (
        dataset_factory_factory(str(selected_branch))
        if dataset_factory_factory is not None
        else dataset_factory
    )
    if selected_factory is None:
        raise ValueError("Rubik public lifecycle requires a dataset factory")
    rendered_prompt_key = str(prompt_query_key or selected_branch)
    rendered_question_format = str(question_format or selected_branch)
    return run_rubiks_task(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=max(1, int(max_attempts)),
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        namespace_base=str(namespace_base),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=rendered_prompt_key,
        object_description_key=str(object_description_key),
        dataset_factory=selected_factory,
        task_field_factory=task_field_factory,
        question_format=rendered_question_format,
    )


def run_rubiks_task(
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
    object_description_key: str,
    dataset_factory: DatasetFactory,
    task_field_factory: TaskFieldFactory,
    question_format: str,
) -> TaskOutput:
    """Run common scene plumbing around task-owned Rubik objective hooks."""

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
            rng = spawn_rng(int(attempt_seed), f"{namespace_base}.dataset")
            dataset = dict(dataset_factory(rng, int(attempt_seed), params, axes))
            dataset.update(
                {
                    "scene_variant": str(axes.scene_variant),
                    "view_family": SCENE_ID,
                }
            )
            artifacts = prepare_rubiks_scene(
                instance_seed=int(attempt_seed),
                params=params,
                rendering_defaults=rendering_defaults,
                prompt_defaults=prompt_defaults,
                dataset=dataset,
                scene_variant=str(axes.scene_variant),
                prompt_task_key=str(prompt_task_key),
                prompt_query_key=str(prompt_query_key),
                object_description_key=str(object_description_key),
                namespace_base=str(namespace_base),
            )
            answer_value = str(dataset["answer_option_label"])
            answer_gt = TypedValue(type="option_letter", value=str(answer_value))
            task_fields = {
                "query_id": str(selected_branch),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
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
    raise RuntimeError("Rubik cube-net task failed")


def prepare_rubiks_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    dataset: Mapping[str, Any],
    scene_variant: str,
    prompt_task_key: str,
    prompt_query_key: str,
    object_description_key: str,
    namespace_base: str,
) -> RubiksSceneArtifacts:
    """Render one Rubik scene and derive prompt/annotation artifacts."""

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
        net_panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        option_panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        target_swatch_panel_fill_rgb=tuple(
            int(value) for value in scene_style.panel_fill_rgb
        ),
        sticker_outline_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        border_color_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        coordinate_fill_rgb=tuple(int(value) for value in scene_style.step_fill_rgb),
        coordinate_grid_rgb=tuple(int(value) for value in scene_style.grid_rgb),
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_rubiks_scene(
        background,
        dataset=dataset,
        scene_variant=str(scene_variant),
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
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        object_description_key=str(object_description_key),
        dataset=dataset,
        instance_seed=int(instance_seed),
    )
    annotation_artifacts = selected_option_bbox_annotation(
        rendered_scene.option_panel_bbox_map,
        f"option_{dataset['answer_option_label']}",
    )
    return RubiksSceneArtifacts(
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
    rendered_scene: RenderedRubiksScene,
    render_params: Any,
    prompt_meta: Mapping[str, Any],
    task_fields: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    question_format: str,
) -> dict[str, Any]:
    """Build scene/render/execution trace fields for one Rubik task."""

    option_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.option_panel_bbox_map.items()
    }
    sticker_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.sticker_bbox_map.items()
    }
    candidate_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.candidate_net_bbox_map.items()
    }
    correct_option_panel_id = f"option_{dataset['answer_option_label']}"
    execution_trace = {
        **dict(task_fields),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.get("scene_variant", "")),
        "face_color_names": dict(dataset["face_color_names"]),
        "color_map": dict(dataset["color_map"]),
        "scramble_sequence": [
            str(item) for item in dataset.get("scramble_sequence", [])
        ],
        "query_sequence": [str(item) for item in dataset.get("query_sequence", [])],
        "base_sequence": [str(item) for item in dataset.get("base_sequence", [])],
        "move_sequence_text": str(dataset.get("move_sequence_text", "")),
        "base_sequence_text": str(dataset.get("base_sequence_text", "")),
        "start_state": state_for_trace(dataset["start_state"]),
        "final_state": state_for_trace(dataset["final_state"]),
        "target_face": str(dataset.get("target_face", "")),
        "target_face_name": str(dataset.get("target_face_name", "")),
        "target_row": int(dataset.get("target_row", -1)),
        "target_col": int(dataset.get("target_col", -1)),
        "target_sticker_id": str(dataset.get("target_sticker_id", "")),
        "answer_color_name": str(dataset.get("answer_color_name", "")),
        "answer_color_rgb": list(dataset.get("answer_color_rgb", [])),
        "target_color_name": str(dataset.get("target_color_name", "")),
        "target_color_rgb": list(dataset.get("target_color_rgb", [])),
        "counted_sticker_ids": [
            str(item) for item in dataset.get("counted_sticker_ids", [])
        ],
        "answer_count": int(dataset.get("answer_count", -1)),
        "option_specs": _option_specs_for_trace(dataset["option_specs"]),
        "option_count": int(dataset["option_count"]),
        "answer_option_label": str(dataset["answer_option_label"]),
        "supporting_option_panel_ids": [str(correct_option_panel_id)],
        "question_format": str(question_format),
        "view_family": SCENE_ID,
        "render_mode": str(dataset.get("render_mode", "")),
        "rubiks_operation": str(dataset.get("rubiks_operation", "")),
        "result_mode": str(dataset.get("result_mode", "")),
        "solver_trace": _solver_trace_for_trace(dataset["solver_trace"]),
    }
    return {
        "scene_ir": {
            "scene_kind": "puzzle_rubiks_cube_net",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.get("scene_variant", "")),
                "answer_option_label": str(dataset["answer_option_label"]),
                "correct_option_panel_id": str(correct_option_panel_id),
                "view_family": SCENE_ID,
            },
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.get("scene_variant", "")),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "net_panel_bbox_px": list(rendered_scene.net_panel_bbox_px),
            "net_bbox_px": list(rendered_scene.net_bbox_px),
            "target_swatch_bbox_px": (
                list(rendered_scene.target_swatch_bbox_px)
                if rendered_scene.target_swatch_bbox_px is not None
                else None
            ),
            "unit_size_jitter": dict(render_params.unit_size_jitter or {}),
        },
        "render_map": with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "net_panel_bbox_px": list(rendered_scene.net_panel_bbox_px),
                "net_bbox_px": list(rendered_scene.net_bbox_px),
                "sticker_bboxes_px": dict(sticker_bboxes),
                "option_panel_bboxes_px": dict(option_bboxes),
                "candidate_net_bboxes_px": dict(candidate_bboxes),
                "annotation_source": "option_panel_bboxes_px",
            },
            render_params.unit_size_jitter or {},
        ),
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": str(projected_annotation.get("type", "")),
            "value": projected_annotation.get(
                "bbox", projected_annotation.get("value")
            ),
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
    "RubiksSceneArtifacts",
    "build_trace_payload",
    "prepare_rubiks_scene",
    "run_public_rubiks_task",
    "run_rubiks_task",
]
