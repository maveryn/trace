"""Private neutral lifecycle helpers for Ludo board public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import LudoAnnotationBundle, keyed_ludo_render_map_point_annotation
from .shared.output import base_ludo_execution_trace, build_ludo_common_trace_params, build_ludo_trace_payload
from .shared.prompts import LudoPromptContext, LudoPromptSlots, build_ludo_prompt_artifacts, make_ludo_prompt_slots_from_keys
from .shared.rendering import RenderedLudoScene, make_ludo_render_state, render_ludo_scene
from .shared.sampling import ValueOptionAxes, ValueOptionAxisConfig, resolve_ludo_scene_axes, resolve_ludo_value_option_axes
from .shared.state import LudoRenderState, LudoSceneAxes, SCENE_ID


AnnotationBuilder = Callable[[RenderedLudoScene], LudoAnnotationBundle]
AttemptBuilder = Callable[[Any, LudoSceneAxes], "LudoAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], LudoSceneAxes, Mapping[str, Any]],
    "LudoObjectivePlan",
]
ValueOptionAttemptBuilder = Callable[[Any, LudoSceneAxes, str, ValueOptionAxes], "LudoAttemptResult"]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class LudoAttemptResult:
    """Task-owned Ludo sample plus answer, prompt, render, and annotation hooks."""

    answer_gt: TypedValue
    render_state: LudoRenderState
    build_annotation: AnnotationBuilder
    scene_kind: str
    execution_trace: Mapping[str, Any]
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    relations_extra: Mapping[str, Any] = field(default_factory=dict)
    render_spec_extra: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def integer_answer(value: int) -> TypedValue:
        """Return the standard typed integer answer for Ludo count/value tasks."""

        return TypedValue(type="integer", value=int(value))

    @staticmethod
    def option_answer(value: str) -> TypedValue:
        """Return the standard typed option-letter answer for Ludo option tasks."""

        return TypedValue(type="option_letter", value=str(value))


@dataclass(frozen=True)
class LudoObjectivePlan:
    """Prepared task-owned Ludo objective for one generated instance."""

    prompt_slots: LudoPromptSlots
    attempt_namespace: str
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class LudoLifecycleResult:
    """Rendered prompt/image/annotation payload returned to public task files."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_bundle: LudoAnnotationBundle
    image: Image.Image
    trace_payload: Mapping[str, Any]


@lru_cache(maxsize=None)
def ludo_task_defaults(task_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache generation/rendering/prompt defaults for one Ludo task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(task_id),
    )


def select_ludo_single_query(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    namespace: str,
) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    """Select and validate the public single-query sentinel for a Ludo task."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=("single",),
        default_query_id="single",
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    return str(selected), dict(probabilities), dict(task_params)


def render_ludo_lifecycle(
    *,
    domain: str,
    selected_query_id: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: LudoSceneAxes,
    render_state: LudoRenderState,
    prompt_context: LudoPromptContext,
    build_annotation: AnnotationBuilder,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    scene_kind: str,
    execution_trace: Mapping[str, Any],
    extra_query_params: Mapping[str, Any] | None = None,
    relations_extra: Mapping[str, Any] | None = None,
    render_spec_extra: Mapping[str, Any] | None = None,
) -> LudoLifecycleResult:
    """Run neutral render, prompt, noise, and trace assembly around task-owned bindings."""

    rendered_scene = render_ludo_scene(
        render_state=render_state,
        axes=axes,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        namespace=str(namespace),
    )
    annotation_bundle = build_annotation(rendered_scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_ludo_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        context=prompt_context,
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=build_ludo_common_trace_params(
            axes=axes,
            branch_probabilities=branch_probabilities,
            extra_params={
                "prompt_query_key": str(prompt_context.prompt_query_key),
                **dict(extra_query_params or {}),
            },
        ),
    )
    trace_payload = build_ludo_trace_payload(
        annotation_bundle=annotation_bundle,
        axes=axes,
        rendered_scene=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=query_spec,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        scene_kind=str(scene_kind),
        selected_branch=str(selected_query_id),
        branch_field_name="query_id",
        execution_trace=execution_trace,
        relations_extra=relations_extra,
        render_spec_extra=render_spec_extra,
    )
    return LudoLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_bundle=annotation_bundle,
        image=image,
        trace_payload=dict(trace_payload),
    )


def build_ludo_attempt_result(
    *,
    answer_type: str,
    answer_value: Any,
    render_state: LudoRenderState,
    build_annotation: AnnotationBuilder,
    selected_query_id: str,
    axes: LudoSceneAxes,
    construction_mode: str,
    token_coords: Mapping[str, Any],
    query_color: str,
    target_color: str | None,
    extra_execution_trace: Mapping[str, Any],
    extra_query_params: Mapping[str, Any] | None = None,
    relations_extra: Mapping[str, Any] | None = None,
    scene_kind: str = "games_ludo_board",
) -> LudoAttemptResult:
    """Assemble neutral answer/annotation/trace plumbing around task-owned fields."""

    execution_trace = base_ludo_execution_trace(
        branch_field_name="query_id",
        selected_query_id=str(selected_query_id),
        axes=axes,
        construction_mode=str(construction_mode),
        token_coords=token_coords,
        query_color=str(query_color),
        target_color=target_color,
    )
    execution_trace.update(dict(extra_execution_trace))
    return LudoAttemptResult(
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        render_state=render_state,
        build_annotation=build_annotation,
        scene_kind=str(scene_kind),
        execution_trace=execution_trace,
        extra_query_params=dict(extra_query_params or {}),
        relations_extra=dict(relations_extra or {}),
    )


def build_ludo_bound_attempt(
    *,
    answer_type: str,
    answer_value: Any,
    selected_query_id: str,
    axes: LudoSceneAxes,
    construction_mode: str,
    token_coords: Mapping[str, Any],
    query_color: str,
    target_color: str | None,
    role_sources: Mapping[str, Sequence[str]],
    role_entity_ids: Mapping[str, str],
    extra_execution_trace: Mapping[str, Any],
    roll_sequence: Sequence[int] = (),
    roll_options: Sequence[Any] = (),
    destination_options: Sequence[Any] = (),
    extra_query_params: Mapping[str, Any] | None = None,
    relations_extra: Mapping[str, Any] | None = None,
) -> LudoAttemptResult:
    """Build a Ludo attempt from task-owned visual roles and symbolic trace fields."""

    return build_ludo_attempt_result(
        answer_type=str(answer_type),
        answer_value=answer_value,
        render_state=make_ludo_render_state(
            style_variant=str(axes.style_variant),
            token_coords=token_coords,
            query_color=str(query_color),
            target_color=target_color,
            roll_sequence=roll_sequence,
            roll_options=roll_options,
            destination_options=destination_options,
        ),
        build_annotation=lambda rendered: keyed_ludo_render_map_point_annotation(
            rendered=rendered,
            role_sources=role_sources,
            role_entity_ids=role_entity_ids,
        ),
        selected_query_id=str(selected_query_id),
        axes=axes,
        construction_mode=str(construction_mode),
        token_coords=token_coords,
        query_color=str(query_color),
        target_color=target_color,
        extra_execution_trace=extra_execution_trace,
        extra_query_params=extra_query_params,
        relations_extra=relations_extra,
    )


def make_ludo_value_option_preparer(
    *,
    prompt_slots: LudoPromptSlots | None = None,
    prompt_keys: tuple[str, str, str, str] | None = None,
    example_annotation: Mapping[str, Any] | None = None,
    example_answer: Any = None,
    axis_config: ValueOptionAxisConfig,
    attempt_namespace: str,
    build_attempt: ValueOptionAttemptBuilder,
) -> ObjectivePreparer:
    """Create a reusable preparer for Ludo tasks with one integer target and visual options."""

    resolved_prompt_slots = prompt_slots
    if resolved_prompt_slots is None:
        if prompt_keys is None or example_annotation is None:
            raise ValueError("value-option Ludo preparer needs prompt_slots or prompt keys with example annotation")
        resolved_prompt_slots = make_ludo_prompt_slots_from_keys(
            keys=prompt_keys,
            example_annotation=example_annotation,
            example_answer=example_answer,
        )

    def prepare(
        instance_seed: int,
        task_params: Mapping[str, Any],
        selected_query_id: str,
        _branch_probabilities: Mapping[str, float],
        _axes: LudoSceneAxes,
        gen_defaults: Mapping[str, Any],
    ) -> LudoObjectivePlan:
        axis_bundle = resolve_ludo_value_option_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=gen_defaults,
            config=axis_config,
        )

        def construct_attempt(rng: Any, axes: LudoSceneAxes) -> LudoAttemptResult:
            return build_attempt(rng, axes, str(selected_query_id), axis_bundle)

        return LudoObjectivePlan(
            prompt_slots=resolved_prompt_slots,
            attempt_namespace=str(attempt_namespace),
            construct_attempt=construct_attempt,
        )

    return prepare


def run_ludo_lifecycle(
    *,
    task_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run shared Ludo plumbing around task-owned objective hooks."""

    selected_query_id, branch_probabilities, task_params = select_ludo_single_query(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        params=dict(params),
        namespace=str(namespace),
    )
    axes = resolve_ludo_scene_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        dict(branch_probabilities),
        axes,
        gen_defaults,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        lifecycle = render_ludo_lifecycle(
            domain=str(domain),
            selected_query_id=str(selected_query_id),
            branch_probabilities=branch_probabilities,
            task_params=task_params,
            axes=axes,
            render_state=attempt.render_state,
            prompt_context=LudoPromptContext(
                prompt_query_key=str(objective.prompt_slots.prompt_query_key),
                rule_slot_name=str(objective.prompt_slots.rule_slot_name),
                answer_hint_key=str(objective.prompt_slots.answer_hint_key),
                annotation_hint_key=str(objective.prompt_slots.annotation_hint_key),
                query_color=str(attempt.render_state.query_color),
                target_color=attempt.render_state.target_color,
                json_example=str(objective.prompt_slots.json_example),
                json_example_answer_only=str(objective.prompt_slots.json_example_answer_only),
            ),
            build_annotation=attempt.build_annotation,
            prompt_defaults=prompt_defaults,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            scene_kind=str(attempt.scene_kind),
            execution_trace=attempt.execution_trace,
            extra_query_params=attempt.extra_query_params,
            relations_extra=attempt.relations_extra,
            render_spec_extra=attempt.render_spec_extra,
        )
        return TaskOutput(
            prompt=str(lifecycle.prompt),
            prompt_variants=dict(lifecycle.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=lifecycle.annotation_bundle.annotation_gt,
            image=lifecycle.image,
            image_id="img0",
            trace_payload=dict(lifecycle.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")


__all__ = [
    "LudoAttemptResult",
    "LudoLifecycleResult",
    "LudoObjectivePlan",
    "LudoSingleQueryTaskBase",
    "build_ludo_bound_attempt",
    "build_ludo_attempt_result",
    "render_ludo_lifecycle",
    "make_ludo_value_option_preparer",
    "run_ludo_registered_task",
    "run_ludo_lifecycle",
    "select_ludo_single_query",
]


class LudoSingleQueryTaskBase:
    """Private base for Ludo public tasks that all expose the single-query sentinel."""

    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = ("single",)
    _namespace: str
    _prepare_objective: ObjectivePreparer


def run_ludo_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Ludo lifecycle for a registered public task instance."""

    gen_defaults, render_defaults, prompt_defaults = ludo_task_defaults(str(task_obj.task_id))
    return run_ludo_lifecycle(
        task_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )
