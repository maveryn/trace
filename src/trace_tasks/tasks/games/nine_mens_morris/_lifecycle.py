"""Private lifecycle plumbing for Nine Men's Morris public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .shared.annotations import morris_node_point_set_annotation, morris_piece_bbox_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_morris_common_trace_params, build_morris_trace_payload
from .shared.prompts import NineMensMorrisPromptSlots, build_morris_prompt_artifacts
from .shared.rendering import (
    RenderedNineMensMorrisScene,
    render_nine_mens_morris_scene,
    resolve_nine_mens_morris_render_params,
)
from .shared.sampling import NineMensMorrisVisualAxes
from .shared.state import NineMensMorrisBoardState


AnnotationBinder = Callable[[RenderedNineMensMorrisScene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, NineMensMorrisVisualAxes], "NineMensMorrisAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Mapping[str, Any]],
    "NineMensMorrisObjectivePlan",
]


@dataclass(frozen=True)
class NineMensMorrisCountTarget:
    """Resolved integer-answer axis for one Morris count objective."""

    target_answer: int
    target_answer_support: tuple[int, ...]
    target_answer_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class NineMensMorrisAttemptResult:
    """Task-owned sample plus answer, prompt, and annotation hooks."""

    answer_gt: TypedValue
    board_state: NineMensMorrisBoardState
    prompt_slots: NineMensMorrisPromptSlots
    bind_annotation: AnnotationBinder
    annotation_entity_ids: tuple[str, ...]
    construction_mode: str
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NineMensMorrisObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    axes: NineMensMorrisVisualAxes
    attempt_namespace: str
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class NineMensMorrisLifecycleResult:
    """Rendered prompt/image/annotation payload from neutral plumbing."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


def resolve_morris_count_target(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> NineMensMorrisCountTarget:
    """Resolve a task-owned Morris integer target from config or params."""

    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_answer_support = resolve_integer_support(
        task_params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return NineMensMorrisCountTarget(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_answer_support),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def morris_piece_count_attempt(
    *,
    board_state: NineMensMorrisBoardState,
    prompt_key: str,
    annotation_entity_ids: Sequence[str],
    target: NineMensMorrisCountTarget,
    extra_query_params: Mapping[str, Any] | None = None,
) -> NineMensMorrisAttemptResult:
    """Package an all-pieces-in-mills count after task construction."""

    piece_ids = tuple(str(entity_id) for entity_id in annotation_entity_ids)
    return NineMensMorrisAttemptResult(
        answer_gt=TypedValue(type="integer", value=len(piece_ids)),
        board_state=board_state,
        prompt_slots=NineMensMorrisPromptSlots(
            prompt_query_key=str(prompt_key),
            answer_hint_key=f"answer_hint_{prompt_key}",
            annotation_hint_key=f"annotation_hint_{prompt_key}",
            example_annotation=[[180, 220, 224, 264], [340, 220, 384, 264], [500, 220, 544, 264]],
            example_answer=3,
        ),
        bind_annotation=lambda rendered: morris_piece_bbox_set_annotation(
            rendered=rendered,
            piece_ids=piece_ids,
        ),
        annotation_entity_ids=piece_ids,
        construction_mode="exact_piece_mill_count",
        extra_query_params={
            "prompt_query_key": str(prompt_key),
            "target_answer_support": [int(value) for value in target.target_answer_support],
            "target_answer_probabilities": dict(target.target_answer_probabilities),
            **dict(extra_query_params or {}),
        },
        execution_extra={
            "annotation_map_key": "piece_bboxes_px",
        },
    )


def morris_node_count_attempt(
    *,
    board_state: NineMensMorrisBoardState,
    prompt_key: str,
    annotation_entity_ids: Sequence[str],
    color: str,
    target: NineMensMorrisCountTarget,
    extra_query_params: Mapping[str, Any] | None = None,
) -> NineMensMorrisAttemptResult:
    """Package a completion-point count after task construction."""

    node_labels = tuple(str(entity_id) for entity_id in annotation_entity_ids)
    return NineMensMorrisAttemptResult(
        answer_gt=TypedValue(type="integer", value=len(node_labels)),
        board_state=board_state,
        prompt_slots=NineMensMorrisPromptSlots(
            prompt_query_key=str(prompt_key),
            answer_hint_key=f"answer_hint_{prompt_key}",
            annotation_hint_key=f"annotation_hint_{prompt_key}",
            example_annotation=[[242, 202], [362, 202]],
            example_answer=2,
        ),
        bind_annotation=lambda rendered: morris_node_point_set_annotation(
            rendered=rendered,
            node_labels=node_labels,
        ),
        annotation_entity_ids=node_labels,
        construction_mode="exact_mill_completion_count",
        extra_query_params={
            "prompt_query_key": str(prompt_key),
            "player_color": str(color),
            "target_answer_support": [int(value) for value in target.target_answer_support],
            "target_answer_probabilities": dict(target.target_answer_probabilities),
            **dict(extra_query_params or {}),
        },
        execution_extra={
            "annotation_map_key": "node_centers_px",
            "player_color": str(color),
        },
    )


@lru_cache(maxsize=None)
def morris_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache scene defaults for one public Morris task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def run_morris_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Morris lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = morris_task_defaults(str(task_obj.task_id))
    return run_morris_lifecycle(
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


def run_morris_lifecycle(
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
        lifecycle = render_morris_lifecycle(
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


def render_morris_lifecycle(
    *,
    domain: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: NineMensMorrisVisualAxes,
    attempt: NineMensMorrisAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> NineMensMorrisLifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene, background_meta, panel_style_meta, render_params = _render_scene(
        board_state=attempt.board_state,
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
    prompt_defaults_used, prompt_artifacts = build_morris_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        slots=attempt.prompt_slots,
        instance_seed=int(instance_seed),
    )
    query_params = build_morris_common_trace_params(
        axes=axes,
        branch_probabilities=branch_probabilities,
        extra_params=dict(attempt.extra_query_params),
    )
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=query_params,
    )
    trace_payload = build_morris_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=attempt.annotation_entity_ids,
        axes=axes,
        board_state=attempt.board_state,
        rendered=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_spec,
        post_noise_meta=post_noise_meta,
        background_meta=background_meta,
        panel_style_meta=panel_style_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=int(attempt.answer_gt.value),
        construction_mode=str(attempt.construction_mode),
        execution_extra=attempt.execution_extra,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_branch)
    trace_payload["execution_trace"]["query_id"] = str(selected_branch)
    trace_payload["render_spec"]["font_asset"] = get_font_family_record(str(render_params.font_family)).to_trace()
    return NineMensMorrisLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def _render_scene(
    *,
    board_state: NineMensMorrisBoardState,
    axes: NineMensMorrisVisualAxes,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[RenderedNineMensMorrisScene, Mapping[str, Any], Mapping[str, Any], Any]:
    """Resolve panel styling and render one Morris board."""

    render_params = resolve_nine_mens_morris_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene",
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
    rendered_scene = render_nine_mens_morris_scene(
        board_state=board_state,
        background=background,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    return rendered_scene, dict(background_meta), dict(panel_style_meta), render_params


__all__ = [
    "NineMensMorrisAttemptResult",
    "NineMensMorrisObjectivePlan",
    "morris_node_count_attempt",
    "morris_piece_count_attempt",
    "resolve_morris_count_target",
    "run_morris_lifecycle",
    "run_morris_registered_task",
]
