"""Private lifecycle plumbing for Mini-golf games public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style

from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_minigolf_common_trace_params, build_minigolf_trace_payload
from .shared.prompts import MinigolfPromptSlots, build_minigolf_prompt_artifacts
from .shared.rendering import RenderedMinigolfScene, render_minigolf_scene, resolve_minigolf_render_params
from .shared.sampling import MinigolfAxes
from .shared.state import MinigolfSample


AnnotationBinder = Callable[[RenderedMinigolfScene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, MinigolfAxes], "MinigolfAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Mapping[str, Any]],
    "MinigolfObjectivePlan",
]


@dataclass(frozen=True)
class MinigolfAttemptResult:
    """Task-owned sample plus answer, prompt, and annotation hooks."""

    answer_gt: TypedValue
    sample: MinigolfSample
    prompt_slots: MinigolfPromptSlots
    bind_annotation: AnnotationBinder
    annotation_entity_ids: tuple[str, ...]
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MinigolfObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    axes: MinigolfAxes
    attempt_namespace: str
    construct_attempt: AttemptBuilder


def minigolf_string_answer_attempt(
    *,
    sample: MinigolfSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: str,
    bind_annotation: AnnotationBinder,
    annotation_entity_ids: Sequence[str],
    extra_query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> MinigolfAttemptResult:
    """Package a string-label answer after a public task binds semantics."""

    return MinigolfAttemptResult(
        answer_gt=TypedValue(type="string", value=str(sample.answer)),
        sample=sample,
        prompt_slots=MinigolfPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=example_annotation,
            example_answer=str(example_answer),
        ),
        bind_annotation=bind_annotation,
        annotation_entity_ids=tuple(str(entity_id) for entity_id in annotation_entity_ids),
        extra_query_params=dict(extra_query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


@dataclass(frozen=True)
class MinigolfLifecycleResult:
    """Rendered prompt/image/annotation payload returned by neutral plumbing."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


@lru_cache(maxsize=None)
def minigolf_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache scene defaults for one public Mini-golf task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def run_minigolf_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Mini-golf lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = minigolf_task_defaults(str(task_obj.task_id))
    return run_minigolf_lifecycle(
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


def run_minigolf_lifecycle(
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
    """Run branch selection, retry, render, prompt, annotation, and output assembly."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_id),
        namespace=f"{namespace}.branch",
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
        lifecycle = render_minigolf_lifecycle(
            domain=str(domain),
            selected_branch=str(selected_branch),
            branch_probabilities=dict(branch_probabilities),
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


def render_minigolf_lifecycle(
    *,
    domain: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: MinigolfAxes,
    attempt: MinigolfAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> MinigolfLifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene, background_meta, panel_style_meta, render_params = _render_scene(
        sample=attempt.sample,
        axes=axes,
        params=task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    annotation_artifacts = attempt.bind_annotation(rendered_scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_minigolf_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        slots=attempt.prompt_slots,
        instance_seed=int(instance_seed),
    )
    query_params = build_minigolf_common_trace_params(
        axes=axes,
        branch_probabilities=branch_probabilities,
        extra_params=dict(attempt.extra_query_params),
    )
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=query_params,
    )
    trace_payload = build_minigolf_trace_payload(
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
        answer_value=attempt.answer_gt.value,
        execution_extra=attempt.execution_extra,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_branch)
    trace_payload["execution_trace"]["query_id"] = str(selected_branch)
    trace_payload["render_spec"]["font_asset"] = get_font_family_record(str(render_params.font_family)).to_trace()
    return MinigolfLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def _render_scene(
    *,
    sample: MinigolfSample,
    axes: MinigolfAxes,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[RenderedMinigolfScene, Mapping[str, Any], Mapping[str, Any], Any]:
    """Resolve panel styling and render one Mini-golf sample."""

    render_params = resolve_minigolf_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    raw_treatments = params.get("panel_scene_treatments", group_default(render_defaults, "panel_scene_treatments", None))
    if isinstance(raw_treatments, str):
        allowed_treatments = (str(raw_treatments),)
    elif raw_treatments is None:
        allowed_treatments = None
    else:
        allowed_treatments = tuple(str(item) for item in raw_treatments)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene",
        treatments=allowed_treatments,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(render_defaults, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(render_defaults, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_minigolf_scene(
        obstacles=sample.obstacles,
        shot_options=sample.shot_options,
        mode=str(sample.mode),
        ball_xy_norm=(float(sample.ball_x_norm), float(sample.ball_y_norm)),
        hole_xy_norm=(float(sample.hole_x_norm), float(sample.hole_y_norm)),
        cue_visible_fraction=float(sample.cue_visible_fraction),
        hidden_paths_norm=sample.hidden_paths_norm,
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    return rendered_scene, dict(background_meta), dict(panel_style_meta), render_params


__all__ = [
    "MinigolfAttemptResult",
    "MinigolfObjectivePlan",
    "minigolf_string_answer_attempt",
    "run_minigolf_lifecycle",
    "run_minigolf_registered_task",
]
