"""Private neutral lifecycle plumbing for Snakes and Ladders public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .shared.defaults import DEFAULTS, GEN_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS
from .shared.output import build_snakes_ladders_trace_payload
from .shared.prompts import build_snakes_ladders_prompt_artifacts
from .shared.rendering import RenderedSnakesLaddersScene, SnakesLaddersRenderParams, render_snakes_ladders_board_scene
from .shared.rules import board_last_square
from .shared.state import (
    DOMAIN,
    SCENE_ID,
    SUPPORTED_SNAKES_LADDERS_SCENE_VARIANTS,
    SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS,
    SnakesLaddersAxes,
    SnakesLaddersSample,
)
from ..shared.layout import attach_games_unit_size_jitter, resolve_games_layout_jitter, resolve_games_unit_size_scale, scale_games_px
from ..shared.sampling import resolve_games_named_axis
from ..shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style


AnnotationBuilder = Callable[[RenderedSnakesLaddersScene], tuple[AnnotationArtifacts, Mapping[str, Any]]]
ObjectiveBuilder = Callable[[int, Mapping[str, Any], SnakesLaddersAxes, str, Mapping[str, float]], "SnakesLaddersObjective"]


@dataclass(frozen=True)
class SnakesLaddersObjective:
    """Task-owned sample, answer, annotation, and prompt binding."""

    sample: SnakesLaddersSample
    answer_gt: TypedValue
    prompt_query_key: str
    json_example: str
    json_example_answer_only: str
    answer_support: list[int] | None
    build_annotation: AnnotationBuilder
    trace_extra_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    die_value: int | None = None
    horizon_roll_count: int | None = None
    show_roll_panel: bool = True
    highlight_token_square: bool = True


class SnakesLaddersLifecycleTask:
    """Default public class metadata shared by Snakes and Ladders tasks."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)


def resolve_snakes_ladders_scene_axes(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> SnakesLaddersAxes:
    """Resolve scene-only axes before a public task builds its objective."""

    scene_variant, scene_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SNAKES_LADDERS_SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS,
    )
    board_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="board_side_support",
        fallback=DEFAULTS.board_side_support,
    )
    explicit_target = params.get("target_answer")
    if explicit_target is not None:
        target = int(explicit_target)
        board_support = tuple(side for side in board_support if target <= board_last_square(int(side)))
        if not board_support:
            raise ValueError(f"target_answer {target} is incompatible with board_side_support")
    board_params = dict(params)
    board_params["board_side_support"] = [int(side) for side in board_support]
    board_side, board_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=board_params,
        gen_defaults=GEN_DEFAULTS,
        support_key="board_side_support",
        explicit_key="board_side",
        fallback_support=board_support,
        namespace=f"{str(namespace)}.board_side",
        balanced_flag_key="balanced_board_side_sampling",
    )
    return SnakesLaddersAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        board_side=int(board_side),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        board_side_probabilities=dict(board_probs),
    )


def resolve_snakes_ladders_render_params(params: Mapping[str, Any], *, instance_seed: int, board_side: int) -> SnakesLaddersRenderParams:
    """Resolve render dimensions and style controls from scene defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.snakes_ladders.font_family",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.snakes_ladders.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.snakes_ladders.layout",
        ),
        unit_scale_meta,
    )
    return SnakesLaddersRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height))),
        board_side=int(board_side),
        board_left_px=scale_games_px(params.get("board_left_px", group_default(RENDER_DEFAULTS, "board_left_px", DEFAULTS.board_left_px)), unit_scale, min_px=40),
        board_top_px=scale_games_px(params.get("board_top_px", group_default(RENDER_DEFAULTS, "board_top_px", DEFAULTS.board_top_px)), unit_scale, min_px=48),
        board_size_px=scale_games_px(params.get("board_size_px", group_default(RENDER_DEFAULTS, "board_size_px", DEFAULTS.board_size_px)), unit_scale, min_px=max(560, int(board_side) * 80)),
        side_panel_width_px=scale_games_px(params.get("side_panel_width_px", group_default(RENDER_DEFAULTS, "side_panel_width_px", DEFAULTS.side_panel_width_px)), unit_scale, min_px=140),
        cell_gap_px=scale_games_px(params.get("cell_gap_px", group_default(RENDER_DEFAULTS, "cell_gap_px", DEFAULTS.cell_gap_px)), unit_scale, min_px=2),
        cell_radius_px=scale_games_px(params.get("cell_radius_px", group_default(RENDER_DEFAULTS, "cell_radius_px", DEFAULTS.cell_radius_px)), unit_scale, min_px=3),
        number_font_size_px=scale_games_px(params.get("number_font_size_px", group_default(RENDER_DEFAULTS, "number_font_size_px", DEFAULTS.number_font_size_px)), unit_scale, min_px=24),
        token_radius_px=scale_games_px(params.get("token_radius_px", group_default(RENDER_DEFAULTS, "token_radius_px", DEFAULTS.token_radius_px)), unit_scale, min_px=10),
        die_size_px=scale_games_px(params.get("die_size_px", group_default(RENDER_DEFAULTS, "die_size_px", DEFAULTS.die_size_px)), unit_scale, min_px=52),
        jump_width_px=scale_games_px(params.get("jump_width_px", group_default(RENDER_DEFAULTS, "jump_width_px", DEFAULTS.jump_width_px)), unit_scale, min_px=3),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def run_snakes_ladders_task(
    task: SnakesLaddersLifecycleTask,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_objective: ObjectiveBuilder,
    *,
    supported_query_ids: Sequence[str],
    default_query_id: str,
) -> TaskOutput:
    """Run neutral selection, rendering, prompt, trace, and output assembly."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
        namespace=f"{str(task.task_id)}.public_query",
    )
    axes = resolve_snakes_ladders_scene_axes(task_params, instance_seed=int(instance_seed), namespace=str(task.task_id))

    objective: SnakesLaddersObjective | None = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (1009 * int(attempt_index))
        try:
            objective = build_objective(int(attempt_seed), task_params, axes, str(selected_query), dict(query_probabilities))
        except ValueError as exc:
            last_error = exc
            continue
        break
    if objective is None:
        raise RuntimeError(f"{task.task_id} failed to construct a valid Snakes and Ladders instance: {last_error}") from last_error

    render_params = resolve_snakes_ladders_render_params(task_params, instance_seed=int(instance_seed), board_side=int(axes.board_side))
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.snakes_ladders.panel_scene_style",
        treatment_weights=task_params.get(
            "panel_scene_treatment_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=task_params.get(
            "panel_scene_palette_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered = render_snakes_ladders_board_scene(
        jumps=objective.sample.jumps,
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        start_square=int(objective.sample.start_square),
        die_value=None if objective.die_value is None else int(objective.die_value),
        horizon_roll_count=None if objective.horizon_roll_count is None else int(objective.horizon_roll_count),
        show_roll_panel=bool(objective.show_roll_panel),
        highlight_token_square=bool(objective.highlight_token_square),
        panel_style=panel_style,
    )
    annotation_artifacts, witness_symbolic = objective.build_annotation(rendered)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_artifacts = build_snakes_ladders_prompt_artifacts(
        axes=axes,
        prompt_query_key=str(objective.prompt_query_key),
        die_value=objective.die_value,
        horizon_roll_count=objective.horizon_roll_count,
        json_example=str(objective.json_example),
        json_example_answer_only=str(objective.json_example_answer_only),
        instance_seed=int(instance_seed),
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    trace_payload = build_snakes_ladders_trace_payload(
        axes=axes,
        sample=objective.sample,
        rendered_entities=[dict(entity) for entity in rendered.scene_entities],
        render_map=rendered.render_map,
        image_size=(int(image.size[0]), int(image.size[1])),
        prompt_query_key=str(objective.prompt_query_key),
        query_id_probabilities=dict(query_probabilities),
        background_meta=background_meta,
        panel_style_meta=panel_style_meta,
        text_style_meta=text_style_meta,
        post_noise_meta=post_noise_meta,
        annotation_artifacts=annotation_artifacts,
        witness_symbolic=witness_symbolic,
        answer_support=objective.answer_support,
        params_extra=objective.trace_extra_params,
        execution_extra=objective.execution_extra,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query)
    trace_payload["render_spec"]["query_id"] = str(selected_query)
    trace_payload["execution_trace"]["query_id"] = str(selected_query)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=trace_payload.pop("params_for_prompt"),
    )
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


__all__ = [
    "DEFAULT_QUERY_ID",
    "SnakesLaddersLifecycleTask",
    "SnakesLaddersObjective",
    "run_snakes_ladders_task",
]
