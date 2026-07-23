"""Scene-private lifecycle for public toggle-grid tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import resolve_render_params
from .shared.output import build_trace_payload, json_ready
from .shared.prompts import (
    object_description_for_scene_variant,
    render_toggle_prompt_artifacts,
)
from .shared.rendering import render_toggle_repair_scene, render_toggle_result_scene
from .shared.state import DOMAIN, SCENE_ID, ToggleDataset

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)
_AttemptBuilder = Callable[[int, Mapping[str, Any], int], TaskOutput]
_SampleBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Any], ToggleDataset]
_RenderBuilder = Callable[..., Any]
_OutputBinder = Callable[
    [ToggleDataset, Mapping[str, Any], str, Mapping[str, float]], "ToggleBinding"
]


@dataclass(frozen=True)
class ToggleBinding:
    """Task-owned answer, annotation, and trace fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    semantic_params: Mapping[str, Any]
    execution_fields: Mapping[str, Any]


class ToggleGridSceneTask:
    """Neutral lifecycle; public task files provide objective callbacks."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)
    prompt_task_key: str
    prompt_query_key: str
    namespace: str
    sample_builder: _SampleBuilder
    render_builder: _RenderBuilder
    output_binder: _OutputBinder

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one toggle-grid task instance."""

        return retry_toggle_generation(
            build_case=self._build_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )

    def _build_case(
        self,
        instance_seed: int,
        params: Mapping[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Run common scene plumbing around task-owned callbacks."""

        _ = int(max_attempts)
        generation_defaults, rendering_defaults, prompt_defaults = (
            load_toggle_task_defaults(str(self.task_id))
        )
        return run_toggle_single_query_case(
            task_id=str(self.task_id),
            supported_query_ids=tuple(str(item) for item in self.supported_query_ids),
            namespace=str(self.namespace),
            prompt_task_key=str(self.prompt_task_key),
            prompt_query_key=str(self.prompt_query_key),
            instance_seed=int(instance_seed),
            params=params,
            generation_defaults=generation_defaults,
            rendering_defaults=rendering_defaults,
            prompt_defaults=prompt_defaults,
            sample_builder=self.sample_builder,
            render_builder=self.render_builder,
            output_binder=self.output_binder,
        )


def load_toggle_task_defaults(
    task_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one public task."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(task_id),
    )


def retry_toggle_generation(
    *,
    build_case: _AttemptBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Retry construction when semantic constraints reject a generated case."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return build_case(
                int(instance_seed) + int(attempt_index),
                params,
                int(max_attempts),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("toggle-grid generation failed without a captured error")


def prepare_toggle_visual_case(
    *,
    dataset: ToggleDataset,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    render_builder: _RenderBuilder,
    namespace: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve style, render image, and build prompt artifacts."""

    render_params = resolve_render_params(params, rendering_defaults)
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_builder(
        background,
        dataset=dataset,
        style=scene_style,
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    object_description = object_description_for_scene_variant(
        prompt_defaults,
        str(dataset.scene_variant),
    )
    prompt_artifacts = render_toggle_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        object_description=str(object_description),
        instance_seed=int(instance_seed),
    )
    return {
        "image": image,
        "rendered_scene": rendered_scene,
        "render_params": render_params,
        "scene_style_meta": dict(scene_style_meta),
        "background_meta": dict(background_meta),
        "post_noise_meta": dict(post_noise_meta),
        "prompt_defaults": dict(prompt_defaults),
        "prompt_artifacts": prompt_artifacts,
    }


def run_toggle_single_query_case(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    namespace: str,
    prompt_task_key: str,
    prompt_query_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    sample_builder: _SampleBuilder,
    render_builder: _RenderBuilder,
    output_binder: _OutputBinder,
) -> TaskOutput:
    """Run the single-branch toggle task lifecycle."""

    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{namespace}.branch",
    )
    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported toggle-grid query_id: {selected_query}")
    rng = spawn_rng(int(instance_seed), f"{namespace}.sample")
    dataset = sample_builder(task_params, generation_defaults, rng)
    visual = prepare_toggle_visual_case(
        dataset=dataset,
        params=task_params,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        render_builder=render_builder,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    binding = output_binder(
        dataset,
        visual,
        str(selected_query),
        dict(branch_probabilities),
    )
    return build_toggle_task_output(
        dataset=dataset,
        visual=visual,
        public_query_id=str(selected_query),
        branch_probabilities=dict(branch_probabilities),
        answer_gt=binding.answer_gt,
        annotation_gt=binding.annotation_gt,
        projected_annotation=binding.projected_annotation,
        witness_symbolic=binding.witness_symbolic,
        prompt_query_key=str(prompt_query_key),
        semantic_params=binding.semantic_params,
        execution_fields=binding.execution_fields,
    )


def build_toggle_task_output(
    *,
    dataset: ToggleDataset,
    visual: Mapping[str, Any],
    public_query_id: str,
    branch_probabilities: Mapping[str, float],
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    prompt_query_key: str,
    semantic_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
) -> TaskOutput:
    """Assemble TaskOutput from task-owned answer and annotation binding."""

    rendered_scene = visual["rendered_scene"]
    render_params = visual["render_params"]
    prompt_artifacts = visual["prompt_artifacts"]
    query_params = {
        "query_id": str(public_query_id),
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "grid_rows": int(dataset.rows),
        "grid_cols": int(dataset.cols),
        "target_answer_support": list(dataset.target_answer_support),
        **dict(semantic_params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params=query_params,
    )
    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "start_grid_bbox_px": list(rendered_scene.start_grid_bbox_px),
        "target_grid_bbox_px": (
            list(rendered_scene.target_grid_bbox_px)
            if rendered_scene.target_grid_bbox_px is not None
            else None
        ),
        "start_cell_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.start_cell_bbox_map.items()
        },
        "target_cell_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.target_cell_bbox_map.items()
        },
        "option_panel_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.option_panel_bbox_map.items()
        },
        "annotation_source": "task_selected_bbox",
    }
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": "puzzle_toggle_grid",
            "relations": {
                "selected_branch": str(public_query_id),
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "answer_value": answer_gt.value,
            },
        },
        query_spec=query_spec,
        render_spec={
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "scene_style": dict(visual["scene_style_meta"]),
            "background_style": dict(visual["background_meta"]),
            "post_image_noise": dict(visual["post_noise_meta"]),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "layout": "toggle_grid_with_visual_options",
        },
        render_map=render_map,
        execution_trace={
            **dict(query_params),
            "start_state": [
                [int(value) for value in row] for row in dataset.start_state
            ],
            "target_state": [
                [int(value) for value in row] for row in dataset.target_state
            ],
            "pressed_cells": [
                [int(row), int(col)] for row, col in dataset.pressed_cells
            ],
            "result_options": [json_ready(option) for option in dataset.result_options],
            "switch_options": [json_ready(option) for option in dataset.switch_options],
            "answer_value": answer_gt.value,
            "toggle_rule": (
                "pressing a switch toggles that cell and its orthogonal neighbors"
            ),
            **dict(execution_fields),
        },
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_gt.to_dict(),
        prompt_defaults=visual["prompt_defaults"],
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=visual["image"],
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "ToggleBinding",
    "ToggleGridSceneTask",
    "load_toggle_task_defaults",
    "retry_toggle_generation",
    "run_toggle_single_query_case",
]
