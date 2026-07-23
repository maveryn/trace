"""Scene-private lifecycle plumbing for Hex public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import hex_cell_point_annotation, hex_cell_point_set_annotation
from .shared.output import build_hex_common_trace_params, build_hex_trace_payload
from .shared.prompts import HexPromptContext, build_hex_prompt_artifacts
from .shared.rendering import render_hex_board_scene
from .shared.rules import Coord, HexSample, color_name
from .shared.sampling import resolve_hex_render_params, resolve_hex_scene_axes
from .shared.state import HEX_NAMESPACE, SCENE_ID, HexIntegerAxis, HexSceneAxes, HexStringAxis


POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)
AttemptBuilder = Callable[[Any, HexSceneAxes], "HexAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], HexSceneAxes],
    "HexObjectivePlan",
]


@dataclass(frozen=True)
class HexAttemptResult:
    """Task-owned board result plus selected cell witnesses."""

    sample: HexSample
    annotation_coords: Sequence[Coord]
    annotation_contract: str = "point_set"
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HexObjectivePlan:
    """Prepared task-owned objective hooks for one Hex instance."""

    prompt_query_key: str
    answer_gt: TypedValue
    target_axis: HexIntegerAxis | HexStringAxis | None
    candidate_count_axis: HexIntegerAxis | None
    extra_query_params: Mapping[str, Any]
    attempt_namespace: str
    construct_attempt: AttemptBuilder


def _allowed_panel_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...] | None:
    raw = params.get("panel_scene_treatments", group_default(render_defaults, "panel_scene_treatments", None))
    if isinstance(raw, str):
        return (str(raw),)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def run_hex_lifecycle(
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
    """Run neutral Hex scene plumbing around task-owned objective hooks."""

    selected_query_id, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(query_id) for query_id in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_axes = resolve_hex_scene_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(task_id),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        dict(branch_probabilities),
        scene_axes,
    )
    render_params = resolve_hex_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, scene_axes)
        except ValueError:
            continue

        panel_style, panel_style_meta = resolve_game_panel_scene_style(
            instance_seed=int(instance_seed),
            namespace=f"{HEX_NAMESPACE}.panel_scene_style",
            treatments=_allowed_panel_treatments(task_params, render_defaults),
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
        rendered_scene = render_hex_board_scene(
            board=attempt.sample.board,
            background=background,
            scene_variant=str(scene_axes.scene_variant),
            style_variant=str(scene_axes.style_variant),
            player_color=str(scene_axes.player_color),
            candidate_labels_by_coord={
                tuple(spec.coord): str(spec.label)
                for spec in attempt.sample.candidate_specs
            },
            params=render_params,
            reference_coords=(tuple(attempt.sample.reference_coord),)
            if attempt.sample.reference_coord is not None
            else tuple(),
            panel_style=panel_style,
        )
        if str(attempt.annotation_contract) == "point":
            if len(tuple(attempt.annotation_coords)) != 1:
                raise ValueError("scalar Hex point annotation requires exactly one coordinate")
            annotation_entity_id, annotation_artifacts = hex_cell_point_annotation(
                rendered_scene,
                tuple(attempt.annotation_coords)[0],
            )
            annotation_entity_ids = (str(annotation_entity_id),)
        elif str(attempt.annotation_contract) == "point_set":
            annotation_entity_ids, annotation_artifacts = hex_cell_point_set_annotation(
                rendered_scene,
                tuple(attempt.annotation_coords),
            )
        else:
            raise ValueError(f"unsupported Hex annotation contract: {attempt.annotation_contract!r}")
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        query_player = color_name(attempt.sample.player_value)
        neighbor_state = str(attempt.sample.neighbor_target_state or "")
        prompt_defaults_used, prompt_artifacts = build_hex_prompt_artifacts(
            domain=str(domain),
            prompt_defaults=prompt_defaults,
            context=HexPromptContext(
                scene_variant=str(scene_axes.scene_variant),
                prompt_query_key=str(objective.prompt_query_key),
                query_player=str(query_player),
                neighbor_state=str(neighbor_state),
            ),
            answer_type=str(objective.answer_gt.type),
            instance_seed=int(instance_seed),
        )
        common_query_params = build_hex_common_trace_params(
            scene_axes=scene_axes,
            target_axis=objective.target_axis,
            candidate_count_axis=objective.candidate_count_axis,
            branch_probabilities=branch_probabilities,
            sample=attempt.sample,
            neighbor_state=neighbor_state,
            extra_params=objective.extra_query_params,
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=common_query_params,
        )
        text_style_meta = {
            "font_family": str(render_params.font_family),
            "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
        }
        trace_payload = build_hex_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in annotation_entity_ids),
            scene_axes=scene_axes,
            sample=attempt.sample,
            annotation_coords=tuple(attempt.annotation_coords),
            rendered_scene=rendered_scene,
            prompt_defaults=prompt_defaults_used,
            prompt_query_spec=query_spec,
            background_meta=background_meta,
            post_noise_meta=post_noise_meta,
            image_size=(int(image.size[0]), int(image.size[1])),
            execution_extra=dict(attempt.execution_extra),
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
        trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
        trace_payload["render_spec"]["panel_scene_style"] = dict(panel_style_meta)
        trace_payload["render_spec"]["text_style"] = dict(text_style_meta)
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=objective.answer_gt,
            annotation_gt=TypedValue(type=str(annotation_artifacts.annotation_type), value=annotation_artifacts.value),
            image=image,
            image_id="img0",
            trace_payload=dict(trace_payload),
            task_versions=default_task_versions(),
            query_id=str(selected_query_id),
            scene_id=SCENE_ID,
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Hex scene after {max_attempts} attempts")


__all__ = [
    "HexAttemptResult",
    "HexObjectivePlan",
    "run_hex_lifecycle",
]
