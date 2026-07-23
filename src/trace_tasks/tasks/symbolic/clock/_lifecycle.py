"""Private lifecycle shell for symbolic single-clock tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_query_spec, build_prompt_trace_artifacts, render_scene_prompt_variants
from ...shared.text_rendering import temporary_default_font_family
from ...shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
    build_time_artifact_clock_theme,
)
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .shared.defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .shared.rendering import draw_text_option_cards, option_cards_y_below_bbox, render_clock_scene
from .shared.state import (
    SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
    ClockStyleResolution,
    ClockTextOptionSpec,
    RenderedClockScene,
)
from .shared.styles import resolve_clock_render_params


@dataclass(frozen=True)
class SingleClockPlan:
    """Task-owned answer and sampling record before rendering."""

    shown_total_minutes: int
    answer_gt: TypedValue
    query_id: str
    question_format: str
    query_params: Mapping[str, Any]
    execution_fields: Mapping[str, Any]
    json_example: str
    json_example_answer_only: str
    shown_total_seconds: int | None = None
    show_second_hand: bool = False
    answer_options: ClockTextOptionSpec | None = None


@dataclass(frozen=True)
class SingleClockObjective:
    """Task-owned annotation and trace extras after rendering."""

    annotation_gt: TypedValue
    witness_symbolic: Mapping[str, Any]
    projected_annotation: Mapping[str, Any]
    execution_fields: Mapping[str, Any]
    render_map_extra: Mapping[str, Any]


@dataclass(frozen=True)
class SingleClockBinding:
    """Task-specific hooks and prompt keys for one single-clock objective."""

    domain: str
    task_identifier: str
    supported_query_ids: Sequence[str]
    task_prompt_key: str
    prompt_query_key: str
    object_description_prefix: str
    annotation_hint_key: str
    answer_hint_key: str
    build_plan: Callable[..., SingleClockPlan]
    build_objective: Callable[[SingleClockPlan, RenderedClockScene], SingleClockObjective]


def _resolve_axis(
    *,
    task_identifier: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(item) for item in supported_variants),
        task_id=str(task_identifier),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        axis_namespace=str(axis_namespace),
    )


def _resolve_clock_style(*, binding: SingleClockBinding, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], instance_seed: int) -> ClockStyleResolution:
    """Resolve only visual clock axes; task semantics must stay in the public task file."""

    scene_variant, scene_probabilities = _resolve_axis(
        task_identifier=binding.task_identifier,
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )
    style_variant, style_probabilities = _resolve_axis(
        task_identifier=binding.task_identifier,
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        axis_namespace="style_variant",
    )
    accent_name, accent_probabilities = _resolve_axis(
        task_identifier=binding.task_identifier,
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        axis_namespace="accent_color_name",
    )
    return ClockStyleResolution(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_name),
        scene_variant_probabilities=dict(scene_probabilities),
        style_variant_probabilities=dict(style_probabilities),
        accent_color_name_probabilities=dict(accent_probabilities),
    )


def _build_prompt(
    *,
    binding: SingleClockBinding,
    prompt_defaults: Mapping[str, Any],
    style: ClockStyleResolution,
    plan: SingleClockPlan,
    instance_seed: int,
) -> tuple[str, dict[str, str], Any]:
    """Compose the shared single-clock prompt from task-owned prompt keys and slots."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            f"{binding.object_description_prefix}_{style.scene_variant}",
            "json_output_contract",
            "json_output_contract_answer_only",
            binding.annotation_hint_key,
            binding.answer_hint_key,
        ),
        context=f"prompt defaults for {binding.task_identifier}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=binding.domain,
        scene_id="clock",
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        query_key=str(binding.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[f"{binding.object_description_prefix}_{style.scene_variant}"]),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_values["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_values[binding.annotation_hint_key]),
            "answer_hint": str(prompt_values[binding.answer_hint_key]),
            "json_example": str(plan.json_example),
            "json_example_answer_only": str(plan.json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(artifacts.prompt), dict(artifacts.prompt_variants), artifacts


def run_single_clock_task(binding: SingleClockBinding, *, instance_seed: int, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Run neutral single-clock rendering around a task-owned objective."""

    gen_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        binding.domain,
        "clock",
        task_id=binding.task_identifier,
    )
    task_params = dict(params)
    last_error: Exception | None = None
    plan: SingleClockPlan | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            plan = binding.build_plan(
                params=task_params,
                gen_defaults=gen_defaults,
                instance_seed=int(instance_seed) + int(attempt_index),
            )
            break
        except Exception as exc:  # pragma: no cover - exercised by review generation.
            last_error = exc
            plan = None
    if plan is None:
        raise RuntimeError(f"could not generate {binding.task_identifier}: {last_error}") from last_error

    style = _resolve_clock_style(binding=binding, params=task_params, gen_defaults=gen_defaults, instance_seed=int(instance_seed))
    render_params = resolve_clock_render_params(
        task_params,
        render_defaults=render_defaults,
        fallback_values=DEFAULTS.__dict__,
        instance_seed=int(instance_seed),
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{binding.task_identifier}.font",
        params={**dict(render_defaults), **dict(task_params)},
    )
    clock_theme = build_time_artifact_clock_theme(
        accent_color_name=str(style.accent_color_name),
        style_variant=str(style.style_variant),
    )
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{binding.task_identifier}.background",
    )
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_clock_scene(
            background,
            scene_variant=str(style.scene_variant),
            shown_total_minutes=int(plan.shown_total_minutes),
            shown_total_seconds=(
                int(plan.shown_total_seconds)
                if plan.shown_total_seconds is not None
                else None
            ),
            show_second_hand=bool(plan.show_second_hand),
            render_params=render_params,
            visual_theme=clock_theme,
            center_px=(
                (0.5 * float(render_params.canvas_width), 300.0)
                if plan.answer_options is not None
                else None
            ),
            force_show_minor_ticks=bool(plan.show_second_hand and plan.answer_options is not None),
        )
        option_bboxes_px: dict[str, list[float]] = {}
        selected_option_bbox_px: list[float] | None = None
        option_entities: list[dict[str, Any]] = []
        if plan.answer_options is not None:
            raw_option_bboxes, option_entities = draw_text_option_cards(
                rendered_scene.image,
                text_by_label=dict(plan.answer_options.text_by_label),
                correct_label=str(plan.answer_options.correct_label),
                y0_px=option_cards_y_below_bbox(
                    rendered_scene.scene_bbox_px,
                    canvas_height=int(render_params.canvas_height),
                ),
            )
            option_bboxes_px = {
                str(label): [round(float(value), 3) for value in bbox]
                for label, bbox in raw_option_bboxes.items()
            }
            selected_option_bbox_px = list(option_bboxes_px[str(plan.answer_options.correct_label)])
            rendered_scene = RenderedClockScene(
                image=rendered_scene.image,
                scene_bbox_px=(
                    min(float(rendered_scene.scene_bbox_px[0]), min(float(b[0]) for b in raw_option_bboxes.values())),
                    min(float(rendered_scene.scene_bbox_px[1]), min(float(b[1]) for b in raw_option_bboxes.values())),
                    max(float(rendered_scene.scene_bbox_px[2]), max(float(b[2]) for b in raw_option_bboxes.values())),
                    max(float(rendered_scene.scene_bbox_px[3]), max(float(b[3]) for b in raw_option_bboxes.values())),
                ),
                face_bbox_px=rendered_scene.face_bbox_px,
                center_px=rendered_scene.center_px,
                hour_hand_bbox_px=rendered_scene.hour_hand_bbox_px,
                minute_hand_bbox_px=rendered_scene.minute_hand_bbox_px,
                second_hand_bbox_px=rendered_scene.second_hand_bbox_px,
                alarm_hand_bbox_px=rendered_scene.alarm_hand_bbox_px,
                hour_hand_tip_px=rendered_scene.hour_hand_tip_px,
                minute_hand_tip_px=rendered_scene.minute_hand_tip_px,
                second_hand_tip_px=rendered_scene.second_hand_tip_px,
                alarm_hand_tip_px=rendered_scene.alarm_hand_tip_px,
                entities=[*rendered_scene.entities, *option_entities],
            )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_artifacts = _build_prompt(
        binding=binding,
        prompt_defaults=prompt_defaults,
        style=style,
        plan=plan,
        instance_seed=int(instance_seed),
    )
    objective = binding.build_objective(plan, rendered_scene)
    if plan.answer_options is not None:
        if selected_option_bbox_px is None:
            raise ValueError("selected option bbox missing for option task")
        option_annotation = bbox_annotation_artifacts(selected_option_bbox_px)
        objective = SingleClockObjective(
            annotation_gt=option_annotation.annotation_gt,
            witness_symbolic={
                "type": str(option_annotation.annotation_type),
                "value": list(option_annotation.value),
            },
            projected_annotation=dict(option_annotation.projected_annotation),
            execution_fields={
                **dict(objective.execution_fields),
                "supporting_parts": ["selected_answer_option"],
                "selected_option_bbox_px": list(selected_option_bbox_px),
            },
            render_map_extra={
                **dict(objective.render_map_extra),
                "option_bboxes_px": dict(option_bboxes_px),
                "selected_option_label": str(plan.answer_options.correct_label),
                "selected_option_bbox_px": list(selected_option_bbox_px),
            },
        )
    query_params = {
        "query_id": str(plan.query_id),
        "query_id_probabilities": {str(plan.query_id): 1.0},
        "question_format": str(plan.question_format),
        "scene_id": "clock",
        "scene_variant": str(style.scene_variant),
        "style_variant": str(style.style_variant),
        "accent_color_name": str(style.accent_color_name),
        "scene_variant_probabilities": dict(style.scene_variant_probabilities),
        "style_variant_probabilities": dict(style.style_variant_probabilities),
        "accent_color_name_probabilities": dict(style.accent_color_name_probabilities),
            **dict(plan.query_params),
        }
    if plan.answer_options is not None:
        query_params.update(
            {
                "option_labels": [str(label) for label in plan.answer_options.labels],
                "correct_label": str(plan.answer_options.correct_label),
            }
        )
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(plan.query_id),
        params=query_params,
    )
    hand_bboxes_px = {
        "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_bbox_px],
        "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_bbox_px],
    }
    hand_tips_px = {
        "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_tip_px],
        "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_tip_px],
    }
    if rendered_scene.second_hand_bbox_px is not None:
        hand_bboxes_px["second"] = [
            round(float(value), 3) for value in rendered_scene.second_hand_bbox_px
        ]
    if rendered_scene.second_hand_tip_px is not None:
        hand_tips_px["second"] = [
            round(float(value), 3) for value in rendered_scene.second_hand_tip_px
        ]
    trace_payload = {
        "scene_ir": {
            "scene_kind": "symbolic_clock_single",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(query_params),
        },
        "query_spec": {
            **dict(prompt_query_spec),
            "template_id": str(prompt_defaults.get("bundle_id", "")),
        },
        "render_spec": {
            "scene_id": "clock",
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(style.scene_variant),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
            "clock_style": {
                "accent_color_name": str(style.accent_color_name),
                "style_variant": str(style.style_variant),
                "face_radius_px": int(render_params.face_radius_px),
                "bezel_width_px": int(render_params.bezel_width_px),
                "numeral_font_size_px": int(render_params.numeral_font_size_px),
                "hour_hand_width_px": int(render_params.hour_hand_width_px),
                "minute_hand_width_px": int(render_params.minute_hand_width_px),
                "second_hand_width_px": int(render_params.second_hand_width_px),
                "show_second_hand": bool(plan.show_second_hand),
                "font": {
                    "source": "global_font_pool",
                    "font_family": str(font_family),
                    "font_asset_version": font_asset_version(),
                    "scope": "single_clock_face",
                },
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
            "face_bbox_px": [round(float(value), 3) for value in rendered_scene.face_bbox_px],
            "center_px": [round(float(value), 3) for value in rendered_scene.center_px],
            "hand_bboxes_px": dict(hand_bboxes_px),
            "hand_tips_px": dict(hand_tips_px),
            "annotation_source": (
                "selected_answer_option_bbox_px"
                if plan.answer_options is not None
                else "center_px_and_hand_tip_segments_px"
            ),
            **dict(objective.render_map_extra),
        },
        "execution_trace": {
            **dict(query_params),
            "task_id": str(binding.task_identifier),
            "answer_value": plan.answer_gt.value,
            **dict(plan.execution_fields),
            **dict(objective.execution_fields),
        },
        "witness_symbolic": dict(objective.witness_symbolic),
        "projected_annotation": dict(objective.projected_annotation),
        "answer_gt": plan.answer_gt.to_dict(),
        "annotation_gt": objective.annotation_gt.to_dict(),
    }
    return TaskOutput(
        prompt=prompt,
        answer_gt=plan.answer_gt,
        annotation_gt=objective.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id="clock",
        query_id=str(plan.query_id),
        prompt_variants=dict(prompt_variants),
    )
