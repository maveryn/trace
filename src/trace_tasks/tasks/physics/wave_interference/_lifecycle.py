"""Private lifecycle helpers for wave-interference public objectives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import (
    prepare_physics_diagram_style_and_background,
)
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_wave_interference_prompt_artifacts,
    condition_phrase,
)
from .shared.rendering import (
    render_wave_interference_scene,
    resolve_wave_layout_placement,
    resolve_wave_render_defaults,
)
from .shared.sampling import (
    make_choice_scene_spec,
    make_path_scene_spec,
    path_difference_support,
    resolve_choice_axes,
    resolve_common_axes,
    resolve_path_difference_axes,
    sample_choice_scenario,
    sample_path_scenario,
)
from .shared.state import SCENE_ID, SCENE_NAMESPACE, WaveInterferenceDefaults


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def run_point_choice_lifecycle(
    *,
    domain: str,
    task_id: str,
    task_prompt_key: str,
    internal_query_id: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run the point-choice objective from exact source-distance parity."""

    fallback_defaults = WaveInterferenceDefaults()
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index) * 7919
        try:
            common_axes = resolve_common_axes(
                instance_seed=int(attempt_seed),
                params=task_params,
                generation_defaults=generation_defaults,
                namespace=str(lifecycle_namespace),
            )
            choice_axes = resolve_choice_axes(
                instance_seed=int(attempt_seed),
                params=task_params,
                generation_defaults=generation_defaults,
                namespace=str(lifecycle_namespace),
            )
            scenario = sample_choice_scenario(
                spawn_rng(int(attempt_seed), f"{lifecycle_namespace}.scenario"),
                common_axes=common_axes,
                choice_axes=choice_axes,
            )
        except Exception as exc:  # pragma: no cover - surfaced if all attempts fail.
            last_error = exc
            continue

        scene_spec = make_choice_scene_spec(
            common_axes=common_axes,
            scenario=scenario,
        )
        render_defaults = resolve_wave_render_defaults(
            task_params,
            rendering_defaults,
            fallback_defaults=fallback_defaults,
            instance_seed=int(attempt_seed),
            namespace=str(lifecycle_namespace),
        )
        render_defaults, layout_placement_meta = resolve_wave_layout_placement(
            render_defaults=render_defaults,
            rendering_defaults=rendering_defaults,
            params=task_params,
            instance_seed=int(attempt_seed),
            namespace=SCENE_NAMESPACE,
        )
        background, background_meta, diagram_style, diagram_style_meta = (
            prepare_physics_diagram_style_and_background(
                scene_id=SCENE_ID,
                canvas_width=int(render_defaults["canvas_width"]),
                canvas_height=int(render_defaults["canvas_height"]),
                instance_seed=int(attempt_seed),
                params=task_params,
            )
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(attempt_seed),
            namespace=f"{lifecycle_namespace}.render.font",
            params=task_params,
        )
        rendered = render_wave_interference_scene(
            background=background,
            render_defaults=render_defaults,
            accent_color_name=str(common_axes.accent_color_name),
            scene_spec=scene_spec,
            diagram_style=diagram_style,
            font_family=str(font_family),
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(attempt_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        if rendered.annotation_point is None:
            raise ValueError("point-choice scene did not produce a scalar point annotation")

        prompt_defaults_required = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {task_id}",
        )
        prompt_artifacts = build_wave_interference_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_defaults_required.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults_required.get("task_key", task_prompt_key)),
            query_key=str(selected_branch),
            dynamic_slots={
                "target_condition_phrase": condition_phrase(str(choice_axes.target_condition)),
            },
            instance_seed=int(attempt_seed),
        )
        answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_option_letter))
        annotation_gt = TypedValue(type="point", value=list(rendered.annotation_point))
        choice_payload = {
            "phase_relation": str(scenario.phase_relation),
            "target_condition": str(scenario.target_condition),
            "correct_option_letter": str(scenario.correct_option_letter),
            "candidates": [
                {
                    "option_letter": str(candidate.letter),
                    "x_steps": float(candidate.x_steps),
                    "y_steps": float(candidate.y_steps),
                    "s1_distance_steps": int(candidate.r1_steps),
                    "s2_distance_steps": int(candidate.r2_steps),
                    "condition": str(candidate.condition),
                    "is_correct": bool(candidate.is_correct),
                }
                for candidate in scenario.candidates
            ],
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params={
                "query_id": str(selected_branch),
                "internal_query_id": str(internal_query_id),
                "scene_variant": str(common_axes.scene_variant),
                "phase_relation": str(common_axes.phase_relation),
                "target_condition": str(choice_axes.target_condition),
                "correct_option_letter": str(choice_axes.correct_option_letter),
                "accent_color_name": str(common_axes.accent_color_name),
                "target_answer": str(answer_gt.value),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(common_axes.scene_variant_probabilities),
                "phase_relation_probabilities": dict(common_axes.phase_relation_probabilities),
                "target_condition_probabilities": dict(choice_axes.target_condition_probabilities),
                "correct_option_letter_probabilities": dict(choice_axes.correct_option_letter_probabilities),
                "accent_color_name_probabilities": dict(common_axes.accent_color_name_probabilities),
                "target_answer_probabilities": dict(choice_axes.correct_option_letter_probabilities),
            },
        )
        font_record = get_font_family_record(str(font_family))
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_wave_interference_{str(common_axes.scene_variant)}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_branch),
                    "internal_query_id": str(internal_query_id),
                    "scene_variant": str(common_axes.scene_variant),
                    "phase_relation": str(common_axes.phase_relation),
                    "target_condition": str(choice_axes.target_condition),
                    "target_answer": str(answer_gt.value),
                    "answer_type": "option_letter",
                    "choice_scenario": dict(choice_payload),
                    "annotation_entity_ids": list(rendered.annotation_entity_ids),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "scene_variant": str(common_axes.scene_variant),
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "accent_color_name": str(common_axes.accent_color_name),
                "half_wavelength_px": int(render_defaults["half_wavelength_px"]),
                "font": _font_trace(str(font_family), font_record),
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": dict(background_meta),
                "layout_placement": dict(layout_placement_meta),
                "post_image_noise": dict(post_noise_meta),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_branch),
                "internal_query_id": str(internal_query_id),
                "scene_variant": str(common_axes.scene_variant),
                "phase_relation": str(common_axes.phase_relation),
                "target_condition": str(choice_axes.target_condition),
                "correct_option_letter": str(choice_axes.correct_option_letter),
                "accent_color_name": str(common_axes.accent_color_name),
                "target_answer": str(answer_gt.value),
                "answer_type": "option_letter",
                "option_letters": ["A", "B", "C", "D", "E"],
                "choice_scenario": dict(choice_payload),
                "annotation_entity_ids": list(rendered.annotation_entity_ids),
                "sampling_probabilities": {
                    "query_id": dict(branch_probabilities),
                    "scene_variant": dict(common_axes.scene_variant_probabilities),
                    "phase_relation": dict(common_axes.phase_relation_probabilities),
                    "target_condition": dict(choice_axes.target_condition_probabilities),
                    "correct_option_letter": dict(choice_axes.correct_option_letter_probabilities),
                    "accent_color_name": dict(common_axes.accent_color_name_probabilities),
                    "target_answer": dict(choice_axes.correct_option_letter_probabilities),
                },
            },
            "sampling": {
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(common_axes.scene_variant_probabilities),
                "phase_relation_probabilities": dict(common_axes.phase_relation_probabilities),
                "target_condition_probabilities": dict(choice_axes.target_condition_probabilities),
                "correct_option_letter_probabilities": dict(choice_axes.correct_option_letter_probabilities),
                "accent_color_name_probabilities": dict(common_axes.accent_color_name_probabilities),
                "target_answer_probabilities": dict(choice_axes.correct_option_letter_probabilities),
            },
            "witness_symbolic": {
                "type": "object_point",
                "ids": [str(item) for item in rendered.annotation_entity_ids],
            },
            "projected_annotation": {
                "type": "point",
                "point": list(annotation_gt.value),
                "pixel_point": list(annotation_gt.value),
            },
            "background": dict(background_meta),
            "technical_diagram_style": dict(diagram_style_meta),
            "post_image_noise": dict(post_noise_meta),
        }
        return _task_output(
            prompt_artifacts=prompt_artifacts,
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            trace_payload=trace_payload,
            selected_branch=str(selected_branch),
        )

    raise RuntimeError(
        f"{task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}"
    )


def run_path_difference_lifecycle(
    *,
    domain: str,
    task_id: str,
    task_prompt_key: str,
    internal_query_id: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run path-difference generation and bind the two path segments."""

    fallback_defaults = WaveInterferenceDefaults()
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index) * 7919
        try:
            common_axes = resolve_common_axes(
                instance_seed=int(attempt_seed),
                params=task_params,
                generation_defaults=generation_defaults,
                namespace=str(lifecycle_namespace),
            )
            path_axes = resolve_path_difference_axes(
                instance_seed=int(attempt_seed),
                params=task_params,
                generation_defaults=generation_defaults,
                fallback_defaults=fallback_defaults,
                namespace=str(lifecycle_namespace),
            )
            scenario = sample_path_scenario(
                spawn_rng(int(attempt_seed), f"{lifecycle_namespace}.scenario"),
                common_axes=common_axes,
                path_axes=path_axes,
            )
        except Exception as exc:  # pragma: no cover - surfaced if all attempts fail.
            last_error = exc
            continue

        scene_spec = make_path_scene_spec(common_axes=common_axes, scenario=scenario)
        render_defaults = resolve_wave_render_defaults(
            task_params,
            rendering_defaults,
            fallback_defaults=fallback_defaults,
            instance_seed=int(attempt_seed),
            namespace=str(lifecycle_namespace),
        )
        render_defaults, layout_placement_meta = resolve_wave_layout_placement(
            render_defaults=render_defaults,
            rendering_defaults=rendering_defaults,
            params=task_params,
            instance_seed=int(attempt_seed),
            namespace=SCENE_NAMESPACE,
        )
        background, background_meta, diagram_style, diagram_style_meta = (
            prepare_physics_diagram_style_and_background(
                scene_id=SCENE_ID,
                canvas_width=int(render_defaults["canvas_width"]),
                canvas_height=int(render_defaults["canvas_height"]),
                instance_seed=int(attempt_seed),
                params=task_params,
            )
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(attempt_seed),
            namespace=f"{lifecycle_namespace}.render.font",
            params=task_params,
        )
        rendered = render_wave_interference_scene(
            background=background,
            render_defaults=render_defaults,
            accent_color_name=str(common_axes.accent_color_name),
            scene_spec=scene_spec,
            diagram_style=diagram_style,
            font_family=str(font_family),
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(attempt_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        prompt_defaults_required = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {task_id}",
        )
        prompt_artifacts = build_wave_interference_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_defaults_required.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults_required.get("task_key", task_prompt_key)),
            query_key=str(selected_branch),
            dynamic_slots={},
            instance_seed=int(attempt_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.path_difference_steps))
        annotation_gt = TypedValue(
            type="segment_set",
            value=[list(segment) for segment in rendered.annotation_segments],
        )
        path_payload = {
            "phase_relation": str(scenario.phase_relation),
            "point_x_steps": float(scenario.point_x_steps),
            "point_y_steps": float(scenario.point_y_steps),
            "s1_distance_steps": int(scenario.r1_steps),
            "s2_distance_steps": int(scenario.r2_steps),
            "path_difference_steps": int(scenario.path_difference_steps),
            "unit": "lambda/2",
        }
        support = path_difference_support(
            task_params,
            generation_defaults=generation_defaults,
            fallback_defaults=fallback_defaults,
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params={
                "query_id": str(selected_branch),
                "internal_query_id": str(internal_query_id),
                "scene_variant": str(common_axes.scene_variant),
                "phase_relation": str(common_axes.phase_relation),
                "path_difference_steps": int(path_axes.path_difference_steps),
                "path_difference_step_support": list(support),
                "accent_color_name": str(common_axes.accent_color_name),
                "target_answer": int(answer_gt.value),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(common_axes.scene_variant_probabilities),
                "phase_relation_probabilities": dict(common_axes.phase_relation_probabilities),
                "path_difference_steps_probabilities": dict(path_axes.path_difference_steps_probabilities),
                "accent_color_name_probabilities": dict(common_axes.accent_color_name_probabilities),
                "target_answer_probabilities": dict(path_axes.path_difference_steps_probabilities),
            },
        )
        font_record = get_font_family_record(str(font_family))
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_wave_interference_{str(common_axes.scene_variant)}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_branch),
                    "internal_query_id": str(internal_query_id),
                    "scene_variant": str(common_axes.scene_variant),
                    "phase_relation": str(common_axes.phase_relation),
                    "target_answer": int(answer_gt.value),
                    "answer_type": "integer",
                    "path_difference_scenario": dict(path_payload),
                    "annotation_entity_ids": list(rendered.annotation_entity_ids),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "scene_variant": str(common_axes.scene_variant),
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "accent_color_name": str(common_axes.accent_color_name),
                "half_wavelength_px": int(render_defaults["half_wavelength_px"]),
                "font": _font_trace(str(font_family), font_record),
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": dict(background_meta),
                "layout_placement": dict(layout_placement_meta),
                "post_image_noise": dict(post_noise_meta),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_branch),
                "internal_query_id": str(internal_query_id),
                "scene_variant": str(common_axes.scene_variant),
                "phase_relation": str(common_axes.phase_relation),
                "path_difference_steps": int(path_axes.path_difference_steps),
                "path_difference_step_support": list(support),
                "accent_color_name": str(common_axes.accent_color_name),
                "target_answer": int(answer_gt.value),
                "answer_type": "integer",
                "path_difference_scenario": dict(path_payload),
                "annotation_entity_ids": list(rendered.annotation_entity_ids),
                "sampling_probabilities": {
                    "query_id": dict(branch_probabilities),
                    "scene_variant": dict(common_axes.scene_variant_probabilities),
                    "phase_relation": dict(common_axes.phase_relation_probabilities),
                    "path_difference_steps": dict(path_axes.path_difference_steps_probabilities),
                    "accent_color_name": dict(common_axes.accent_color_name_probabilities),
                    "target_answer": dict(path_axes.path_difference_steps_probabilities),
                },
            },
            "sampling": {
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(common_axes.scene_variant_probabilities),
                "phase_relation_probabilities": dict(common_axes.phase_relation_probabilities),
                "path_difference_steps_probabilities": dict(path_axes.path_difference_steps_probabilities),
                "accent_color_name_probabilities": dict(common_axes.accent_color_name_probabilities),
                "target_answer_probabilities": dict(path_axes.path_difference_steps_probabilities),
            },
            "witness_symbolic": {
                "type": "segment_set",
                "ids": [str(item) for item in rendered.annotation_entity_ids],
            },
            "projected_annotation": {
                "type": "segment_set",
                "segment_set": list(annotation_gt.value),
                "pixel_segment_set": list(annotation_gt.value),
            },
            "background": dict(background_meta),
            "technical_diagram_style": dict(diagram_style_meta),
            "post_image_noise": dict(post_noise_meta),
        }
        return _task_output(
            prompt_artifacts=prompt_artifacts,
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            trace_payload=trace_payload,
            selected_branch=str(selected_branch),
        )

    raise RuntimeError(
        f"{task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}"
    )


def _font_trace(font_family: str, font_record: Any) -> dict[str, Any]:
    """Return the common prompt-font metadata block for trace payloads."""

    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "wave_interference_tank",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def _task_output(
    *,
    prompt_artifacts: Any,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    image: Any,
    trace_payload: Mapping[str, Any],
    selected_branch: str,
) -> TaskOutput:
    """Build the final TaskOutput shared by both public objectives."""

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
    )


__all__ = ["run_path_difference_lifecycle", "run_point_choice_lifecycle"]
