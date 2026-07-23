"""Private neutral lifecycle plumbing for Sokoban public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import cell_bbox_set, option_cell_bbox, option_pair_bbox_set, option_panel_bbox
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.output import build_sokoban_trace_payload
from .shared.prompts import build_sokoban_prompt_artifacts
from .shared.rendering import apply_panel_style, render_sokoban_scene, resolve_sokoban_render_params
from .shared.sampling import select_scene_axes
from .shared.state import DOMAIN, SCENE_ID, RenderedSokobanScene
from ..shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style


AnnotationBuilder = Callable[[RenderedSokobanScene], AnnotationArtifacts]
ObjectiveBuilder = Callable[[int, Mapping[str, Any], str], "SokobanObjective"]


@dataclass(frozen=True)
class SokobanObjective:
    """Task-owned sample, answer, annotation, and prompt binding."""

    dataset: Mapping[str, Any]
    answer_gt: TypedValue
    prompt_query_key: str
    object_description_key: str
    annotation_hint_key: str
    json_example_key: str
    annotation_source: str
    option_count_support: list[int]
    option_count_probabilities: Mapping[str, float]
    build_annotation: AnnotationBuilder
    answer_hint_key: str = "answer_hint_option_letter"
    json_example_answer_only_key: str = "json_example_answer_only_option_label"
    prompt_dynamic_values: Mapping[str, Any] = field(default_factory=dict)
    trace_extra_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


class SokobanLifecycleTask:
    """Default public metadata shared by Sokoban task classes."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)


def build_path_option_objective(
    *,
    dataset: Mapping[str, Any],
    prompt_query_key: str,
    option_count_support: list[int],
    option_count_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> SokobanObjective:
    """Bind a path-option dataset to scalar option-panel annotation."""

    answer_label = str(dataset["answer_option_label"])
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="option_letter", value=answer_label),
        prompt_query_key=str(prompt_query_key),
        object_description_key="object_description_path_sequence_label",
        annotation_hint_key="annotation_hint_option_panel_bbox",
        json_example_key="json_example_option_label_bbox",
        annotation_source="option_panel_bboxes_px",
        option_count_support=[int(value) for value in option_count_support],
        option_count_probabilities=dict(option_count_probabilities),
        build_annotation=lambda rendered: option_panel_bbox(rendered, answer_label=answer_label),
        trace_extra_params=dict(trace_extra_params or {}),
    )


def build_relation_cell_objective(
    *,
    dataset: Mapping[str, Any],
    prompt_query_key: str,
    option_count_support: list[int],
    option_count_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> SokobanObjective:
    """Bind a single-cell relation dataset to scalar cell annotation."""

    answer_label = str(dataset["answer_option_label"])
    option_specs = tuple(dict(option) for option in dataset.get("option_specs", []))
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="option_letter", value=answer_label),
        prompt_query_key=str(prompt_query_key),
        object_description_key="object_description_box_target_relation_label",
        annotation_hint_key="annotation_hint_relation_cell_bbox",
        json_example_key="json_example_relation_cell_label_bbox",
        annotation_source="cell_bboxes_px",
        option_count_support=[int(value) for value in option_count_support],
        option_count_probabilities=dict(option_count_probabilities),
        build_annotation=lambda rendered: option_cell_bbox(
            rendered,
            option_specs=option_specs,
            answer_label=answer_label,
        ),
        trace_extra_params=dict(trace_extra_params or {}),
    )


def build_relation_pair_objective(
    *,
    dataset: Mapping[str, Any],
    prompt_query_key: str,
    option_count_support: list[int],
    option_count_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> SokobanObjective:
    """Bind a pair-relation dataset to a two-cell bbox-set annotation."""

    answer_label = str(dataset["answer_option_label"])
    option_specs = tuple(dict(option) for option in dataset.get("option_specs", []))
    support = dataset.get("relation_support", {})
    rank_word = str(support.get("rank_word", "requested")) if isinstance(support, Mapping) else "requested"
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="option_letter", value=answer_label),
        prompt_query_key=str(prompt_query_key),
        object_description_key="object_description_box_target_relation_label",
        annotation_hint_key="annotation_hint_relation_pair_bbox_set",
        json_example_key="json_example_relation_pair_label_bbox_set",
        annotation_source="cell_bboxes_px",
        option_count_support=[int(value) for value in option_count_support],
        option_count_probabilities=dict(option_count_probabilities),
        build_annotation=lambda rendered: option_pair_bbox_set(
            rendered,
            option_specs=option_specs,
            answer_label=answer_label,
        ),
        prompt_dynamic_values={"rank_word": rank_word},
        trace_extra_params=dict(trace_extra_params or {}),
    )


def build_box_goal_status_count_objective(
    *,
    dataset: Mapping[str, Any],
    prompt_query_key: str,
    answer_count_support: list[int],
    answer_count_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> SokobanObjective:
    """Bind a box-goal status board to an integer count and counted box bboxes."""

    annotation_cells = tuple(list(cell) for cell in dataset.get("annotation_cells", []))
    answer_value = int(dataset["goal_status_count"])
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="integer", value=answer_value),
        prompt_query_key=str(prompt_query_key),
        object_description_key="object_description_box_goal_status_count",
        annotation_hint_key="annotation_hint_counted_box_bbox_set",
        json_example_key="json_example_counted_box_bbox_set",
        annotation_source="cell_bboxes_px",
        option_count_support=[int(value) for value in answer_count_support],
        option_count_probabilities=dict(answer_count_probabilities),
        build_annotation=lambda rendered: cell_bbox_set(rendered, cells=annotation_cells),
        answer_hint_key="answer_hint_box_goal_count",
        json_example_answer_only_key="json_example_answer_only_integer",
        trace_extra_params=dict(trace_extra_params or {}),
        execution_extra={
            "goal_status_count": int(answer_value),
            "counted_box_labels": list(dataset.get("counted_box_labels", [])),
        },
    )


def run_sokoban_lifecycle(
    *,
    namespace: str,
    supported_queries: tuple[str, ...],
    default_query: str,
    task_params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: ObjectiveBuilder,
) -> TaskOutput:
    """Run shared axes, rendering, prompt, trace, and output assembly."""

    selected_public_query, public_query_probabilities, resolved_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=task_params,
        supported_query_ids=supported_queries,
        default_query_id=str(default_query),
        task_id=str(namespace),
        namespace=f"{namespace}.public_query",
    )
    axes = select_scene_axes(resolved_params, instance_seed=int(instance_seed), namespace=str(namespace))

    last_error: Exception | None = None
    objective: SokobanObjective | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (1009 * int(attempt_index))
        try:
            objective = build_objective(int(attempt_seed), resolved_params, str(selected_public_query))
        except ValueError as exc:
            last_error = exc
            continue
        break
    if objective is None:
        raise RuntimeError(f"{namespace} failed to construct a valid Sokoban instance: {last_error}") from last_error

    scene_style, scene_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene_style",
    )
    render_params = apply_panel_style(
        resolve_sokoban_render_params(resolved_params, instance_seed=int(instance_seed)),
        scene_style,
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_sokoban_scene(
        background,
        dataset=objective.dataset,
        scene_variant=str(axes.scene_variant),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=resolved_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = objective.build_annotation(rendered_scene)
    prompt_defaults, prompt_artifacts = build_sokoban_prompt_artifacts(
        prompt_query_key=str(objective.prompt_query_key),
        object_description_key=str(objective.object_description_key),
        annotation_hint_key=str(objective.annotation_hint_key),
        json_example_key=str(objective.json_example_key),
        answer_hint_key=str(objective.answer_hint_key),
        json_example_answer_only_key=str(objective.json_example_answer_only_key),
        dynamic_values=dict(objective.prompt_dynamic_values),
        instance_seed=int(instance_seed),
    )
    trace_payload = build_sokoban_trace_payload(
        axes=axes,
        dataset=objective.dataset,
        rendered_scene=rendered_scene,
        render_params=render_params,
        prompt_query_key=str(objective.prompt_query_key),
        answer_value=str(objective.answer_gt.value),
        annotation_artifacts=annotation_artifacts,
        annotation_source=str(objective.annotation_source),
        option_count_support=list(objective.option_count_support),
        option_count_probabilities=dict(objective.option_count_probabilities),
        public_query_probabilities=dict(public_query_probabilities),
        background_meta=background_meta,
        scene_style_meta=scene_style_meta,
        post_noise_meta=post_noise_meta,
        trace_extra_params=objective.trace_extra_params,
        execution_extra=objective.execution_extra,
    )
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_public_query),
        params=dict(trace_payload.pop("params_for_prompt")),
    )
    trace_payload["answer_gt"] = objective.answer_gt.to_dict()
    trace_payload["annotation_gt"] = annotation_artifacts.annotation_gt.to_dict()
    trace_payload["render_spec"]["prompt_defaults_bundle_id"] = str(prompt_defaults["bundle_id"])
    trace_payload["execution_trace"]["query_id"] = str(selected_public_query)

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=objective.answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_public_query),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "SokobanLifecycleTask",
    "SokobanObjective",
    "build_box_goal_status_count_objective",
    "build_path_option_objective",
    "build_relation_cell_objective",
    "build_relation_pair_objective",
    "run_sokoban_lifecycle",
]
