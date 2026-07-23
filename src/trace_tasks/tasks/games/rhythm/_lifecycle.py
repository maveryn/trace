"""Private neutral lifecycle for Rhythm public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.layout import resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import rhythm_note_bbox_annotation, rhythm_note_bbox_set_annotation
from .shared.defaults import DEFAULTS
from .shared.prompts import build_rhythm_prompt_artifacts
from .shared.rendering import RhythmRenderParams, render_rhythm_lanes_scene
from .shared.rules import lane_label
from .shared.sampling import resolve_rhythm_visual_axes
from .shared.state import SCENE_ID, SCENE_NAMESPACE, RhythmVisualAxes, SampledRhythmScene


AttemptBuilder = Callable[[Any, RhythmVisualAxes], SampledRhythmScene]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, float], str], "ObjectiveRhythmPlan"]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class ObjectiveRhythmPlan:
    """Task-owned objective hooks for one Rhythm instance."""

    attempt_namespace: str
    prompt_query_key: str
    annotation_kind: str
    construct_attempt: AttemptBuilder
    prompt_rule_keys: Sequence[str] = ("note_object_rule_text",)
    json_example_annotation: Any | None = None
    json_example_answer: int | None = None
    query_params: Mapping[str, Any] = field(default_factory=dict)


def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> RhythmRenderParams:
    """Resolve Rhythm rendering controls from scene config and params."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font_family",
        params=params,
    )
    return RhythmRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        grid_width_px=int(params.get("grid_width_px", group_default(render_defaults, "grid_width_px", DEFAULTS.grid_width_px))),
        grid_height_px=int(params.get("grid_height_px", group_default(render_defaults, "grid_height_px", DEFAULTS.grid_height_px))),
        grid_border_width_px=int(params.get("grid_border_width_px", group_default(render_defaults, "grid_border_width_px", DEFAULTS.grid_border_width_px))),
        row_gap_px=int(params.get("row_gap_px", group_default(render_defaults, "row_gap_px", DEFAULTS.row_gap_px))),
        lane_gap_px=int(params.get("lane_gap_px", group_default(render_defaults, "lane_gap_px", DEFAULTS.lane_gap_px))),
        note_radius_px=int(params.get("note_radius_px", group_default(render_defaults, "note_radius_px", DEFAULTS.note_radius_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout",
        ),
    )


def _annotation_for_objective(
    *,
    rendered_scene: Any,
    entity_ids: Sequence[str],
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Project task-owned witness ids through the selected annotation family."""

    if str(annotation_kind) == "note_bbox_set":
        return rhythm_note_bbox_set_annotation(rendered_scene, entity_ids)
    if str(annotation_kind) == "note_bbox":
        return rhythm_note_bbox_annotation(rendered_scene, entity_ids)
    raise ValueError(f"unsupported Rhythm annotation kind: {annotation_kind}")


def _note_trace(sample: SampledRhythmScene) -> list[dict[str, Any]]:
    """Return trace-friendly note rows with timing and lane metadata."""

    score_values = dict(sample.score_values_by_color or {})
    return [
        {
            "note_id": str(note.note_id),
            "lane_index": int(note.lane_index),
            "lane_label": lane_label(int(note.lane_index)),
            "bottom_row_from_hit_line": int(note.bottom_row),
            "length_rows": int(note.length),
            "color_key": str(note.color_key),
            "score_value": None if not score_values else int(score_values[str(note.color_key)]),
            "kind": str(note.kind),
        }
        for note in sample.notes
    ]


def _common_query_params(
    *,
    axes: RhythmVisualAxes,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
) -> dict[str, Any]:
    """Return prompt-query params common to all Rhythm lifecycle outputs."""

    return {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "lane_count": int(axes.lane_count),
        "lane_count_probabilities": dict(axes.lane_count_probabilities),
        "row_count": int(axes.row_count),
        "row_count_probabilities": dict(axes.row_count_probabilities),
        "beat_window": int(axes.beat_window),
        "beat_window_probabilities": dict(axes.beat_window_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
    }


def _build_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    sample: SampledRhythmScene,
    axes: RhythmVisualAxes,
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
    """Assemble trace payload once the public task has bound witnesses."""

    annotation_entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    note_trace = _note_trace(sample)
    return {
        "scene_ir": {
            "scene_kind": f"games_rhythm_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "query_id": str(query_id),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(axes.style_variant),
                "lane_count": int(sample.lane_count),
                "row_count": int(sample.row_count),
                "beat_window": int(sample.beat_window),
                "annotation_entity_ids": list(annotation_entity_ids),
                "score_values_by_color": None
                if sample.score_values_by_color is None
                else {str(color): int(value) for color, value in sample.score_values_by_color.items()},
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style", {})),
            "text_style": dict(rendered_scene.render_map.get("text_style", {})),
            "score_palette": dict(rendered_scene.render_map.get("score_palette", {})),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "query_id": str(query_id),
            "prompt_query_key": str(prompt_query_key),
            "style_variant": str(axes.style_variant),
            "lane_count": int(sample.lane_count),
            "row_count": int(sample.row_count),
            "beat_window": int(sample.beat_window),
            "selected_lane_index": sample.selected_lane_index,
            "selected_lane_label": sample.selected_lane_label,
            "target_color_key": sample.target_color_key,
            "score_values_by_color": None
            if sample.score_values_by_color is None
            else {str(color): int(value) for color, value in sample.score_values_by_color.items()},
            "answer": int(sample.answer),
            "notes": note_trace,
            "annotation_entity_ids": list(annotation_entity_ids),
            "construction_mode": str(sample.construction_mode),
            **dict(query_params),
        },
        "witness_symbolic": {
            "type": "object" if str(annotation_artifacts.annotation_type) == "bbox" else "object_set",
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


def run_rhythm_lifecycle(
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
    """Run common Rhythm query, render, prompt, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    visual_axes = resolve_rhythm_visual_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_root=SCENE_NAMESPACE,
    )
    objective = prepare_objective(int(instance_seed), task_params, query_probabilities, str(query_id))
    render_params = _resolve_render_params(task_params, render_defaults=render_defaults, instance_seed=int(instance_seed))

    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, visual_axes)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate a valid scene after {max_attempts} attempts")

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_scene_style",
        treatment_weights=task_params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=task_params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_rhythm_lanes_scene(
        lane_count=int(sample.lane_count),
        row_count=int(sample.row_count),
        beat_window=int(sample.beat_window),
        notes=sample.notes,
        score_values_by_color=sample.score_values_by_color,
        background=background,
        style_variant=str(visual_axes.style_variant),
        params=render_params,
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
    prompt_defaults_used, prompt_artifacts = build_rhythm_prompt_artifacts(
        domain=str(domain),
        scene_variant=str(visual_axes.scene_variant),
        prompt_query_key=str(objective.prompt_query_key),
        annotation_type=str(annotation_artifacts.annotation_type),
        prompt_defaults=prompt_defaults,
        selected_lane_label=str(sample.selected_lane_label or ""),
        target_color=str(sample.target_color_key or ""),
        score_values_by_color=sample.score_values_by_color,
        prompt_rule_keys=tuple(str(key) for key in objective.prompt_rule_keys),
        json_example_annotation=objective.json_example_annotation,
        json_example_answer=objective.json_example_answer,
        beat_window=int(sample.beat_window),
        instance_seed=int(instance_seed),
    )
    common_query_params = _common_query_params(
        axes=visual_axes,
        branch_probabilities=query_probabilities,
        prompt_query_key=str(objective.prompt_query_key),
    )
    query_params = {
        **dict(common_query_params),
        **dict(objective.query_params),
        "selected_lane_index": sample.selected_lane_index,
        "selected_lane_label": sample.selected_lane_label,
        "target_color_key": sample.target_color_key,
        "score_values_by_color": None
        if sample.score_values_by_color is None
        else {str(color): int(value) for color, value in sample.score_values_by_color.items()},
        "answer": int(sample.answer),
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


__all__ = ["ObjectiveRhythmPlan", "run_rhythm_lifecycle"]
