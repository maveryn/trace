"""Private neutral lifecycle plumbing for Snake public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import cell_bbox_set
from .shared.defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS
from .shared.output import build_snake_trace_payload, snake_trace_params
from .shared.prompts import build_snake_prompt_artifacts
from .shared.rendering import SnakeRenderParams, render_snake_grid_scene
from .shared.sampling import resolve_scene_axes, select_integer_target
from .shared.state import DOMAIN, SCENE_ID, SnakeSample, SnakeSceneAxes
from ..shared.layout import attach_games_unit_size_jitter, resolve_games_layout_jitter, resolve_games_unit_size_scale, scale_games_px
from ..shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style


SampleBuilder = Callable[[int, Mapping[str, Any], SnakeSceneAxes], "SnakeObjective"]
ImageDecorator = Callable[[Image.Image, Mapping[str, Any], SnakeSample, str], tuple[Image.Image, dict[str, list[float]]]]


@dataclass(frozen=True)
class SnakeObjective:
    """Task-owned sample and answer result consumed by neutral plumbing."""

    sample: SnakeSample
    answer_gt: TypedValue
    prompt_key: str
    prompt_json_example: str
    prompt_json_example_answer_only: str
    answer_support: list[int] | list[str] | None
    annotation_cell_ids: tuple[str, ...]
    trace_extra_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    decorate_image: ImageDecorator | None = None


class SnakeLifecycleTask:
    """Default public class metadata shared by Snake tasks."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)


def snake_prompt_json_examples(*, answer: int | str, annotation_box_count: int) -> tuple[str, str]:
    """Build task-owned prompt examples with matching bbox-set cardinality."""

    annotation = [
        [338 + (72 * index), 332, 410 + (72 * index), 404]
        for index in range(max(0, int(annotation_box_count)))
    ]
    return (
        json.dumps({"annotation": annotation, "answer": answer}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=False),
    )


def select_snake_integer_target(
    task_params: Mapping[str, Any],
    *,
    objective_key: str,
    fallback_support: tuple[int, ...],
    instance_seed: int,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select a balanced integer target using objective-owned key names."""

    return select_integer_target(
        task_params,
        support_key=f"{str(objective_key)}_support",
        explicit_key=f"target_{str(objective_key)}",
        fallback_support=fallback_support,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.{str(objective_key)}",
        balance_flag_key=f"balanced_{str(objective_key)}_sampling",
    )


def build_integer_snake_objective(
    *,
    sample: SnakeSample,
    prompt_key: str,
    prompt_json_example_answer: int,
    prompt_json_example_annotation_count: int,
    answer_support: list[int],
    annotation_cell_ids: tuple[str, ...],
    target_trace_key: str,
    target_value: int,
    target_probabilities: Mapping[str, Any],
) -> SnakeObjective:
    """Create the common integer answer/annotation wrapper for Snake objectives."""

    json_example, json_example_answer_only = snake_prompt_json_examples(
        answer=int(prompt_json_example_answer),
        annotation_box_count=int(prompt_json_example_annotation_count),
    )
    return SnakeObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_key=str(prompt_key),
        prompt_json_example=str(json_example),
        prompt_json_example_answer_only=str(json_example_answer_only),
        answer_support=[int(value) for value in answer_support],
        annotation_cell_ids=tuple(annotation_cell_ids),
        trace_extra_params={
            str(target_trace_key): int(target_value),
            f"{str(target_trace_key)}_probabilities": dict(target_probabilities),
        },
        execution_extra={str(target_trace_key): int(target_value)},
    )


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> SnakeRenderParams:
    """Resolve Snake rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.snake.font_family",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.snake.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.snake.layout",
        ),
        unit_scale_meta,
    )
    return SnakeRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(RENDER_DEFAULTS, "panel_margin_px", DEFAULTS.panel_margin_px))),
        max_board_size_px=scale_games_px(params.get("max_board_size_px", group_default(RENDER_DEFAULTS, "max_board_size_px", DEFAULTS.max_board_size_px)), unit_scale, min_px=360),
        board_border_width_px=scale_games_px(params.get("board_border_width_px", group_default(RENDER_DEFAULTS, "board_border_width_px", DEFAULTS.board_border_width_px)), unit_scale, min_px=2),
        grid_line_width_px=scale_games_px(params.get("grid_line_width_px", group_default(RENDER_DEFAULTS, "grid_line_width_px", DEFAULTS.grid_line_width_px)), unit_scale, min_px=1),
        cell_padding_px=scale_games_px(params.get("cell_padding_px", group_default(RENDER_DEFAULTS, "cell_padding_px", DEFAULTS.cell_padding_px)), unit_scale, min_px=3),
        food_radius_px=scale_games_px(params.get("food_radius_px", group_default(RENDER_DEFAULTS, "food_radius_px", DEFAULTS.food_radius_px)), unit_scale, min_px=8),
        eye_radius_px=scale_games_px(params.get("eye_radius_px", group_default(RENDER_DEFAULTS, "eye_radius_px", DEFAULTS.eye_radius_px)), unit_scale, min_px=2),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def run_snake_lifecycle(
    *,
    namespace: str,
    supported_queries: tuple[str, ...],
    default_query: str,
    task_params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: SampleBuilder,
) -> TaskOutput:
    """Run shared axes, rendering, prompt, trace, and output assembly."""

    selected_query, query_probabilities, resolved_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=task_params,
        supported_query_ids=supported_queries,
        default_query_id=str(default_query),
        task_id=str(namespace),
        namespace=f"{namespace}.public_query",
    )
    axes = resolve_scene_axes(resolved_params, instance_seed=int(instance_seed), namespace=str(namespace))

    last_error: Exception | None = None
    objective: SnakeObjective | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (1009 * int(attempt_index))
        try:
            objective = build_objective(int(attempt_seed), resolved_params, axes)
        except ValueError as exc:
            last_error = exc
            continue
        break
    if objective is None:
        raise RuntimeError(f"{namespace} failed to construct a valid Snake instance: {last_error}") from last_error

    render_params = _resolve_render_params(resolved_params, instance_seed=int(instance_seed))
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.snake.panel_scene_style",
        treatment_weights=resolved_params.get(
            "panel_scene_treatment_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=resolved_params.get(
            "panel_scene_palette_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered = render_snake_grid_scene(
        state=objective.sample.state,
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    render_map = dict(rendered.render_map)
    base_image = rendered.image
    if objective.decorate_image is not None:
        base_image, option_bboxes = objective.decorate_image(
            base_image,
            render_map,
            objective.sample,
            str(render_params.font_family),
        )
        render_map["result_option_bboxes_px"] = dict(option_bboxes)
    image, post_noise_meta = apply_post_image_noise(
        base_image,
        instance_seed=int(instance_seed),
        params=resolved_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts: AnnotationArtifacts = cell_bbox_set(
        render_map,
        cell_ids=tuple(objective.annotation_cell_ids),
    )
    prompt_defaults, prompt_artifacts = build_snake_prompt_artifacts(
        prompt_key=str(objective.prompt_key),
        sample=objective.sample,
        scene_variant=str(axes.scene_variant),
        instance_seed=int(instance_seed),
        json_example=str(objective.prompt_json_example),
        json_example_answer_only=str(objective.prompt_json_example_answer_only),
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    prompt_params = snake_trace_params(
        axes=axes,
        sample=objective.sample,
        answer_value=objective.answer_gt.value,
        answer_support=objective.answer_support,
        extra={
            **dict(objective.trace_extra_params),
            "public_query_probabilities": dict(query_probabilities),
        },
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=prompt_params,
    )
    trace_payload = build_snake_trace_payload(
        axes=axes,
        sample=objective.sample,
        rendered_entities=[dict(entity) for entity in rendered.scene_entities],
        render_map=render_map,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=objective.answer_gt.value,
        prompt_key=str(objective.prompt_key),
        background_meta=background_meta,
        panel_style_meta=panel_style_meta,
        text_style_meta=text_style_meta,
        post_noise_meta=post_noise_meta,
        annotation_artifacts=annotation_artifacts,
        answer_support=objective.answer_support,
        params_extra=objective.trace_extra_params,
        execution_extra=objective.execution_extra,
    )
    trace_payload["query_spec"] = query_spec
    trace_payload["answer_gt"] = objective.answer_gt.to_dict()
    trace_payload["annotation_gt"] = annotation_artifacts.annotation_gt.to_dict()
    trace_payload["execution_trace"]["query_id"] = str(selected_query)
    trace_payload["render_spec"]["prompt_defaults_bundle_id"] = str(prompt_defaults["bundle_id"])

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=objective.answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
    )


def run_snake_task(
    task: SnakeLifecycleTask,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_objective: SampleBuilder,
) -> TaskOutput:
    """Run the lifecycle using metadata owned by the public task class."""

    return run_snake_lifecycle(
        namespace=str(getattr(task, "task_id")),
        supported_queries=tuple(task.supported_query_ids),
        default_query=str(task.supported_query_ids[0]),
        task_params=params,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        build_objective=build_objective,
    )


__all__ = [
    "SnakeLifecycleTask",
    "SnakeObjective",
    "build_integer_snake_objective",
    "run_snake_lifecycle",
    "run_snake_task",
    "select_snake_integer_target",
    "snake_prompt_json_examples",
]
