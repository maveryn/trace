"""Private neutral lifecycle for Reversi public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import (
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    reversi_cell_bbox_set_annotation,
    reversi_disc_point_set_annotation,
)
from .shared.prompts import build_reversi_prompt_artifacts
from .shared.rendering import ReversiRenderParams, render_reversi_board_scene
from .shared.rules import coord_to_cell_id, player_name
from .shared.sampling import resolve_reversi_visual_axes
from .shared.state import (
    SCENE_ID,
    SCENE_NAMESPACE,
    ReversiTargetAxis,
    ReversiVisualAxes,
    SampledReversiScene,
)
from .shared.defaults import DEFAULTS
from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default

AttemptBuilder = Callable[[Any, ReversiVisualAxes], SampledReversiScene]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], Mapping[str, float], str], "ObjectiveReversiPlan"
]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(
    scene_id=SCENE_ID, apply_prob=0.0
)


@dataclass(frozen=True)
class ObjectiveReversiPlan:
    """Task-owned objective hooks for one Reversi instance."""

    attempt_namespace: str
    prompt_query_key: str
    target_axis: ReversiTargetAxis
    annotation_kind: str
    construct_attempt: AttemptBuilder
    query_params: Mapping[str, Any] = field(default_factory=dict)


def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> ReversiRenderParams:
    """Resolve Reversi rendering controls from scene config and params."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font_family",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout",
        ),
        unit_scale_meta,
    )
    return ReversiRenderParams(
        canvas_width=int(
            params.get(
                "canvas_width",
                group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width),
            )
        ),
        canvas_height=int(
            params.get(
                "canvas_height",
                group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height),
            )
        ),
        panel_margin_px=int(
            params.get(
                "panel_margin_px",
                group_default(
                    render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px
                ),
            )
        ),
        player_badge_height_px=int(
            params.get(
                "player_badge_height_px",
                group_default(
                    render_defaults,
                    "player_badge_height_px",
                    DEFAULTS.player_badge_height_px,
                ),
            )
        ),
        player_badge_width_px=int(
            params.get(
                "player_badge_width_px",
                group_default(
                    render_defaults,
                    "player_badge_width_px",
                    DEFAULTS.player_badge_width_px,
                ),
            )
        ),
        header_gap_px=int(
            params.get(
                "header_gap_px",
                group_default(render_defaults, "header_gap_px", DEFAULTS.header_gap_px),
            )
        ),
        max_board_size_px=scale_games_px(
            params.get(
                "max_board_size_px",
                group_default(
                    render_defaults, "max_board_size_px", DEFAULTS.max_board_size_px
                ),
            ),
            unit_scale,
            min_px=360,
        ),
        board_corner_radius_px=scale_games_px(
            params.get(
                "board_corner_radius_px",
                group_default(
                    render_defaults,
                    "board_corner_radius_px",
                    DEFAULTS.board_corner_radius_px,
                ),
            ),
            unit_scale,
            min_px=10,
        ),
        board_frame_width_px=scale_games_px(
            params.get(
                "board_frame_width_px",
                group_default(
                    render_defaults,
                    "board_frame_width_px",
                    DEFAULTS.board_frame_width_px,
                ),
            ),
            unit_scale,
            min_px=7,
        ),
        cell_line_width_px=scale_games_px(
            params.get(
                "cell_line_width_px",
                group_default(
                    render_defaults, "cell_line_width_px", DEFAULTS.cell_line_width_px
                ),
            ),
            unit_scale,
            min_px=1,
        ),
        marked_square_outline_width_px=scale_games_px(
            params.get(
                "marked_square_outline_width_px",
                group_default(
                    render_defaults,
                    "marked_square_outline_width_px",
                    DEFAULTS.marked_square_outline_width_px,
                ),
            ),
            unit_scale,
            min_px=3,
        ),
        disc_inset_fraction=float(
            params.get(
                "disc_inset_fraction",
                group_default(
                    render_defaults, "disc_inset_fraction", DEFAULTS.disc_inset_fraction
                ),
            )
        ),
        player_badge_font_size_px=int(
            params.get(
                "player_badge_font_size_px",
                group_default(
                    render_defaults,
                    "player_badge_font_size_px",
                    DEFAULTS.player_badge_font_size_px,
                ),
            )
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


def _annotation_for_objective(
    *,
    rendered_scene: Any,
    entity_ids: Sequence[str],
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Project task-owned witness ids through the selected annotation family."""

    if str(annotation_kind) == "cell_bbox_set":
        return reversi_cell_bbox_set_annotation(rendered_scene, entity_ids)
    if str(annotation_kind) == "disc_point_set":
        return reversi_disc_point_set_annotation(rendered_scene, entity_ids)
    raise ValueError(f"unsupported Reversi annotation kind: {annotation_kind}")


def _common_query_params(
    *,
    axes: ReversiVisualAxes,
    target_axis: ReversiTargetAxis,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
) -> dict[str, Any]:
    """Return prompt-query params common to all Reversi lifecycle outputs."""

    return {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size": int(axes.board_size),
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "target_answer": int(target_axis.target_answer),
        "target_answer_support": [
            int(value) for value in target_axis.target_answer_support
        ],
        "target_answer_probabilities": dict(target_axis.target_answer_probabilities),
    }


def _build_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    sample: SampledReversiScene,
    axes: ReversiVisualAxes,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    rendered_scene: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    query_id: str,
    prompt_query_key: str,
    query_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the trace payload once the public task has bound witnesses."""

    current_player_name = player_name(int(sample.current_player))
    legal_move_specs = [
        {
            "coord": [int(coord[0]), int(coord[1])],
            "cell_id": str(coord_to_cell_id(coord)),
            "flip_count": int(len(flips)),
        }
        for coord, flips in sorted(sample.legal_moves.items())
    ]
    annotation_entity_ids = tuple(
        str(entity_id) for entity_id in sample.annotation_entity_ids
    )
    return {
        "scene_ir": {
            "scene_kind": f"games_reversi_board_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "query_id": str(query_id),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(axes.style_variant),
                "board_size": int(axes.board_size),
                "current_player": str(current_player_name),
                "target_answer": int(sample.answer),
                "annotation_entity_ids": list(annotation_entity_ids),
                "marked_move_cell_id": (
                    None
                    if sample.marked_move is None
                    else str(coord_to_cell_id(sample.marked_move))
                ),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(
                rendered_scene.render_map.get("panel_scene_style", {})
            ),
            "text_style": dict(rendered_scene.render_map.get("text_style", {})),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "query_id": str(query_id),
            "prompt_query_key": str(prompt_query_key),
            "style_variant": str(axes.style_variant),
            "board_size": int(axes.board_size),
            "current_player": str(current_player_name),
            "target_answer": int(sample.answer),
            "target_answer_support": [
                int(value) for value in query_params.get("target_answer_support", [])
            ],
            "board_rows": [[int(cell) for cell in row] for row in sample.board],
            "construction_mode": str(sample.construction_mode),
            "legal_destination_total": int(len(sample.legal_moves)),
            "legal_move_specs": legal_move_specs,
            "marked_move": (
                None
                if sample.marked_move is None
                else [int(sample.marked_move[0]), int(sample.marked_move[1])]
            ),
            "marked_move_cell_id": (
                None
                if sample.marked_move is None
                else str(coord_to_cell_id(sample.marked_move))
            ),
            "marked_move_flip_coords": [
                [int(coord[0]), int(coord[1])] for coord in sample.marked_move_flips
            ],
            "annotation_coords": [
                [int(coord[0]), int(coord[1])] for coord in sample.annotation_coords
            ],
            "annotation_entity_ids": list(annotation_entity_ids),
            **dict(query_params),
        },
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "ids": list(annotation_entity_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_defaults": {
            "bundle_id": str(prompt_defaults.get("bundle_id", "")),
            "scene_key": str(prompt_defaults.get("scene_key", "")),
            "task_key": str(prompt_defaults.get("task_key", "")),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def run_reversi_lifecycle(
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
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common Reversi query, render, prompt, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    visual_axes = resolve_reversi_visual_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_root=f"{SCENE_NAMESPACE}.visual",
    )
    objective = prepare_objective(
        int(instance_seed), task_params, query_probabilities, str(query_id)
    )
    render_params = _resolve_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )

    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            sample = objective.construct_attempt(rng, visual_axes)
        except ValueError:
            continue
        if int(sample.answer) == int(objective.target_axis.target_answer):
            break
        sample = None
    if sample is None:
        raise RuntimeError(
            f"{task_id} failed to generate a valid scene after {max_attempts} attempts"
        )

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_scene_style",
        treatment_weights=task_params.get(
            "panel_scene_treatment_weights",
            group_default(render_defaults, "panel_scene_treatment_weights", None),
        ),
        palette_weights=task_params.get(
            "panel_scene_palette_weights",
            group_default(render_defaults, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_reversi_board_scene(
        board=sample.board,
        background=background,
        scene_variant=str(visual_axes.scene_variant),
        style_variant=str(visual_axes.style_variant),
        current_player=int(sample.current_player),
        params=render_params,
        marked_move=sample.marked_move,
        panel_style=panel_style,
    )
    annotation_artifacts = _annotation_for_objective(
        rendered_scene=rendered_scene,
        entity_ids=sample.annotation_entity_ids,
        annotation_kind=str(objective.annotation_kind),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_reversi_prompt_artifacts(
        domain=str(domain),
        scene_variant=str(visual_axes.scene_variant),
        prompt_query_key=str(objective.prompt_query_key),
        current_player_name=player_name(int(sample.current_player)),
        query_player_name=str(objective.query_params.get("query_player", "")),
        annotation_type=str(annotation_artifacts.annotation_type),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
    )
    common_query_params = _common_query_params(
        axes=visual_axes,
        target_axis=objective.target_axis,
        branch_probabilities=query_probabilities,
        prompt_query_key=str(objective.prompt_query_key),
    )
    query_params = {
        **dict(common_query_params),
        **dict(objective.query_params),
        "current_player": player_name(int(sample.current_player)),
    }
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params=query_params,
    )
    trace_payload = _build_trace_payload(
        annotation_artifacts=annotation_artifacts,
        sample=sample,
        axes=visual_axes,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_query_spec,
        rendered_scene=rendered_scene,
        background_meta=background_meta,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        query_id=str(query_id),
        prompt_query_key=str(objective.prompt_query_key),
        query_params=query_params,
    )
    trace_payload["render_spec"]["panel_scene_style"] = dict(panel_style_meta)
    trace_payload["render_spec"]["text_style"] = dict(text_style_meta)

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
    )


__all__ = ["ObjectiveReversiPlan", "run_reversi_lifecycle"]
