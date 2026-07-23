"""Scene-private lifecycle for public voxel-cube puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

from .shared.defaults import resolve_render_params, sample_voxel_palette
from .shared.output import build_trace_payload, json_ready
from .shared.prompts import render_voxel_prompt_artifacts
from .shared.state import DOMAIN, SCENE_ID, RenderedVoxelScene, VoxelDataset

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.12)
_AttemptBuilder = Callable[[int, Mapping[str, Any], int], TaskOutput]
_SampleBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Any], VoxelDataset]
_RenderBuilder = Callable[..., RenderedVoxelScene]
_PromptQueryResolver = Callable[[VoxelDataset], str]
_OutputBinder = Callable[
    [VoxelDataset, Mapping[str, Any], str, Mapping[str, float]], "VoxelBinding"
]


@dataclass(frozen=True)
class VoxelBinding:
    """Task-owned answer, annotation, and trace fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    semantic_params: Mapping[str, Any]
    execution_fields: Mapping[str, Any]


class VoxelCubeSceneTask:
    """Neutral lifecycle; public task files provide objective callbacks."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)
    prompt_task_key: str
    namespace: str
    sample_builder: _SampleBuilder
    render_builder: _RenderBuilder
    prompt_query_key_resolver: _PromptQueryResolver
    output_binder: _OutputBinder

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one voxel-cube task instance."""

        return retry_voxel_generation(
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

        attempt_limit = int(max_attempts)
        generation_defaults, rendering_defaults, prompt_defaults = (
            load_voxel_task_defaults(str(self.task_id))
        )
        return run_voxel_single_query_case(
            task_id=str(self.task_id),
            supported_query_ids=tuple(str(item) for item in self.supported_query_ids),
            namespace=str(self.namespace),
            prompt_task_key=str(self.prompt_task_key),
            instance_seed=int(instance_seed),
            params=params,
            generation_defaults=generation_defaults,
            rendering_defaults=rendering_defaults,
            prompt_defaults=prompt_defaults,
            sample_builder=self.sample_builder,
            render_builder=self.render_builder,
            prompt_query_key_resolver=self.prompt_query_key_resolver,
            output_binder=self.output_binder,
            attempt_limit=attempt_limit,
        )


def load_voxel_task_defaults(
    task_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one public task."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(task_id),
    )


def retry_voxel_generation(
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
    raise RuntimeError("voxel-cube generation failed without a captured error")


def run_voxel_single_query_case(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    namespace: str,
    prompt_task_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    sample_builder: _SampleBuilder,
    render_builder: _RenderBuilder,
    prompt_query_key_resolver: _PromptQueryResolver,
    output_binder: _OutputBinder,
    attempt_limit: int,
) -> TaskOutput:
    """Run the single-public-query voxel task lifecycle."""

    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{namespace}.branch",
    )
    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported voxel-cube query_id: {selected_query}")
    rng = spawn_rng(int(instance_seed), f"{namespace}.sample")
    dataset = sample_builder(task_params, generation_defaults, rng)
    prompt_query_key = str(prompt_query_key_resolver(dataset))
    visual = prepare_voxel_visual_case(
        dataset=dataset,
        params=task_params,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=prompt_query_key,
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
    return build_voxel_task_output(
        task_id=str(task_id),
        dataset=dataset,
        visual=visual,
        public_query_id=str(selected_query),
        branch_probabilities=dict(branch_probabilities),
        prompt_query_key=prompt_query_key,
        answer_gt=binding.answer_gt,
        annotation_gt=binding.annotation_gt,
        projected_annotation=binding.projected_annotation,
        witness_symbolic=binding.witness_symbolic,
        semantic_params=binding.semantic_params,
        execution_fields=binding.execution_fields,
        attempt_limit=int(attempt_limit),
    )


def prepare_voxel_visual_case(
    *,
    dataset: VoxelDataset,
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
    render_params = replace(
        render_params,
        palette=sample_voxel_palette(spawn_rng(int(instance_seed), f"{namespace}.palette")),
    )
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
    prompt_artifacts = render_voxel_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dataset.semantic_params),
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


def build_voxel_task_output(
    *,
    task_id: str,
    dataset: VoxelDataset,
    visual: Mapping[str, Any],
    public_query_id: str,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    semantic_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    attempt_limit: int,
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
        "target_answer_support": list(dataset.answer_support),
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
        "stack_bbox_px": (
            list(rendered_scene.stack_bbox_px)
            if rendered_scene.stack_bbox_px is not None
            else None
        ),
        "reference_stack_bbox_px": (
            list(rendered_scene.reference_stack_bbox_px)
            if rendered_scene.reference_stack_bbox_px is not None
            else None
        ),
        "changed_stack_bbox_px": (
            list(rendered_scene.changed_stack_bbox_px)
            if rendered_scene.changed_stack_bbox_px is not None
            else None
        ),
        "projection_cell_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.projection_cell_bbox_map.items()
        },
        "option_panel_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.option_panel_bbox_map.items()
        },
    }
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": "puzzle_voxel_cube",
            "scene_id": SCENE_ID,
            "stack": json_ready(dataset.stack),
            "answer_value": answer_gt.value,
        },
        query_spec=query_spec,
        render_spec={
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_style": dict(visual["scene_style_meta"]),
            "background_style": dict(visual["background_meta"]),
            "post_image_noise": dict(visual["post_noise_meta"]),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "layout": "isometric_voxel_stack_with_projection_panels",
            "voxel_palette": {
                "palette_id": str(render_params.palette.palette_id),
                "cube_top_rgb": list(render_params.palette.cube_top_rgb),
                "cube_left_rgb": list(render_params.palette.cube_left_rgb),
                "cube_right_rgb": list(render_params.palette.cube_right_rgb),
                "cube_edge_rgb": list(render_params.palette.cube_edge_rgb),
                "projection_fill_rgb": list(render_params.palette.projection_fill_rgb),
                "projection_empty_rgb": list(render_params.palette.projection_empty_rgb),
            },
        },
        render_map=render_map,
        execution_trace={
            **dict(query_params),
            "dataset": json_ready(dataset),
            "task_id": str(task_id),
            "max_attempts": int(attempt_limit),
            "answer_value": answer_gt.value,
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
