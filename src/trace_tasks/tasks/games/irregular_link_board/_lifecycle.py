"""Private neutral lifecycle for irregular-link-board public tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import irregular_link_point_set_annotation
from .shared.output import (
    annotation_point_ids,
    build_irregular_link_common_trace_params,
    build_irregular_link_trace_payload,
)
from .shared.prompts import build_irregular_link_prompt_artifacts
from .shared.rendering import render_irregular_link_board_scene
from .shared.sampling import resolve_axes
from .shared.state import (
    SCENE_ID,
    IrregularLinkBoardAxes,
    IrregularLinkBoardSample,
)


SampleBuilder = Callable[[Any, IrregularLinkBoardAxes, Mapping[str, Any]], IrregularLinkBoardSample]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def run_irregular_link_board_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prompt_query_key: str,
    rule_slot_name: str,
    annotation_trace_key: str,
    board_size_support_key: str,
    fallback_board_size_support: Sequence[int],
    sample_builder: SampleBuilder,
    namespace: str,
) -> TaskOutput:
    """Run scene plumbing around task-owned sampling and prompt bindings."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(query_id) for query_id in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    axes = resolve_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        board_size_support_key=str(board_size_support_key),
        fallback_board_size_support=tuple(int(value) for value in fallback_board_size_support),
    )

    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{int(attempt_index)}")
        try:
            sample = sample_builder(rng, axes, gen_defaults)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")

    rendered = render_irregular_link_board_scene(
        sample=sample,
        axes=axes,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        namespace=str(namespace),
    )
    annotation_ids = annotation_point_ids(sample)
    annotation_artifacts = irregular_link_point_set_annotation(rendered, annotation_ids)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_irregular_link_prompt_artifacts(
        domain=str(domain),
        scene_variant=str(axes.scene_variant),
        prompt_query_key=str(prompt_query_key),
        rule_slot_name=str(rule_slot_name),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
    )
    common_query_params = build_irregular_link_common_trace_params(
        axes=axes,
        branch_probabilities=query_probabilities,
        extra_params={"prompt_query_key": str(prompt_query_key)},
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=common_query_params,
    )
    trace_payload = build_irregular_link_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=annotation_ids,
        sample=sample,
        axes=axes,
        rendered_scene=rendered,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=query_spec,
        background_meta=rendered.background_meta,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        selected_branch=str(selected_query_id),
        branch_field_name="query_id",
        annotation_trace_key=str(annotation_trace_key),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=TypedValue(type=str(annotation_artifacts.annotation_type), value=annotation_artifacts.value),
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


__all__ = ["run_irregular_link_board_lifecycle"]
