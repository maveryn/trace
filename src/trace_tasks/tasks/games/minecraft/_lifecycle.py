"""Private lifecycle plumbing for Minecraft-like games public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style

from .shared.annotations import minecraft_bbox_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_minecraft_common_trace_params, build_minecraft_trace_payload
from .shared.prompts import MinecraftPromptSlots, build_minecraft_prompt_artifacts
from .shared.rendering import render_minecraft_block_world_scene, resolve_minecraft_render_params
from .shared.sampling import MinecraftAxes
from .shared.state import MinecraftSceneSample, RenderedMinecraftScene


AttemptBuilder = Callable[[Any, MinecraftAxes], "MinecraftAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Mapping[str, Any]],
    "MinecraftObjectivePlan",
]


@dataclass(frozen=True)
class MinecraftAttemptResult:
    """Task-owned sample plus answer, prompt, trace, and annotation hooks."""

    answer_gt: TypedValue
    sample: MinecraftSceneSample
    prompt_slots: MinecraftPromptSlots
    annotation_entity_ids: tuple[str, ...]
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MinecraftObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    axes: MinecraftAxes
    attempt_namespace: str
    construct_attempt: AttemptBuilder


def minecraft_integer_attempt(
    *,
    sample: MinecraftSceneSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: list[Any],
    example_answer: int,
    counted_resource_kind: str = "",
    target_stack_height: int = 0,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MinecraftAttemptResult:
    """Package a task-owned integer answer with bbox-set annotation witnesses."""

    return MinecraftAttemptResult(
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        sample=sample,
        prompt_slots=MinecraftPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=list(example_annotation),
            example_answer=int(example_answer),
            counted_resource_kind=str(counted_resource_kind),
            target_stack_height=int(target_stack_height),
        ),
        annotation_entity_ids=tuple(str(entity_id) for entity_id in sample.annotation_entity_ids),
        extra_query_params=dict(extra_query_params or {}),
        execution_extra={"answer": int(sample.answer)},
    )


@dataclass(frozen=True)
class MinecraftLifecycleResult:
    """Rendered prompt/image/annotation payload returned to public task files."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


@lru_cache(maxsize=None)
def minecraft_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache scene defaults for one public Minecraft task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def run_minecraft_lifecycle(
    *,
    public_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    supported_branches: tuple[str, ...],
    default_branch: str,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run query selection, retry, render, prompt, annotation, and output assembly."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_id),
        namespace=f"{namespace}.query",
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_branch),
        dict(branch_probabilities),
        gen_defaults,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, objective.axes)
        except ValueError:
            continue
        lifecycle = render_minecraft_lifecycle(
            domain=str(domain),
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            task_params=task_params,
            axes=objective.axes,
            attempt=attempt,
            prompt_defaults=prompt_defaults,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        return TaskOutput(
            prompt=str(lifecycle.prompt),
            prompt_variants=dict(lifecycle.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=lifecycle.annotation_artifacts.annotation_gt,
            image=lifecycle.image,
            image_id="img0",
            trace_payload=dict(lifecycle.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )
    raise RuntimeError(f"{public_id} failed to generate after {max_attempts} attempts")


def render_minecraft_lifecycle(
    *,
    domain: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: MinecraftAxes,
    attempt: MinecraftAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> MinecraftLifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene, background_meta, panel_style_meta = _render_scene(
        sample=attempt.sample,
        axes=axes,
        params=task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    annotation_artifacts = minecraft_bbox_set_annotation(
        rendered=rendered_scene,
        entity_ids=attempt.annotation_entity_ids,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_minecraft_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        slots=attempt.prompt_slots,
        instance_seed=int(instance_seed),
    )
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=build_minecraft_common_trace_params(
            axes=axes,
            branch_probabilities=branch_probabilities,
            extra_params=dict(attempt.extra_query_params),
        ),
    )
    trace_payload = build_minecraft_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=attempt.annotation_entity_ids,
        axes=axes,
        sample=attempt.sample,
        rendered=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_spec,
        post_noise_meta=post_noise_meta,
        background_meta=background_meta,
        panel_style_meta=panel_style_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=int(attempt.answer_gt.value),
        execution_extra=attempt.execution_extra,
    )
    return MinecraftLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def run_minecraft_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Minecraft lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = minecraft_task_defaults(str(task_obj.task_id))
    return run_minecraft_lifecycle(
        public_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        supported_branches=tuple(str(value) for value in task_obj.supported_query_ids),
        default_branch=str(task_obj._default_branch),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )


def _render_scene(
    *,
    sample: MinecraftSceneSample,
    axes: MinecraftAxes,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[RenderedMinecraftScene, Mapping[str, Any], Mapping[str, Any]]:
    """Resolve panel styling and render one sample into image geometry."""

    max_stack_height = max((int(block.z) + 1 for block in sample.blocks), default=1)
    render_params = resolve_minecraft_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        grid_width=int(sample.grid_width),
        grid_depth=int(sample.grid_depth),
        max_stack_height=max(1, int(max_stack_height)),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene",
        treatment_weights=render_defaults.get("panel_scene_treatment_weights"),
        palette_weights=render_defaults.get("panel_scene_palette_weights"),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_minecraft_block_world_scene(
        grid_width=int(sample.grid_width),
        grid_depth=int(sample.grid_depth),
        terrain_cells=sample.terrain_cells,
        blocks=sample.blocks,
        player_cell=sample.player_cell,
        target_cell=sample.target_cell,
        style_variant=str(sample.style_variant),
        background=background,
        params=render_params,
        ladder_columns=tuple(sample.ladder_columns),
        route_overlays=sample.route_overlays,
    )
    return rendered_scene, dict(background_meta), dict(panel_style_meta)

__all__ = [
    "MinecraftAttemptResult",
    "MinecraftObjectivePlan",
    "minecraft_integer_attempt",
    "run_minecraft_lifecycle",
    "run_minecraft_registered_task",
]
