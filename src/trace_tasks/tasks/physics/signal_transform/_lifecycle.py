"""Scene lifecycle for signal-transform spectrum-option tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import normalize_signal_transform_annotation_bbox_map
from .shared.prompts import build_signal_transform_prompt_artifacts
from .shared.rendering import render_signal_transform_scene
from .shared.sampling import (
    build_signal_scenario,
    resolve_canvas_size,
    resolve_signal_render_defaults,
    resolve_signal_transform_axes,
    spectrum_payload,
)
from .shared.state import OPTION_LABELS, SCENE_ID, SignalTransformTaskDefaults


@dataclass(frozen=True)
class SignalTransformObjective:
    """Task-owned objective metadata passed into the neutral scene lifecycle."""

    public_task_id: str
    lifecycle_namespace: str
    internal_query_id: str
    task_prompt_key: str
    supported_waveform_families: Sequence[str]


def run_signal_transform_lifecycle(
    *,
    domain: str,
    objective: SignalTransformObjective,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run common signal-rendering lifecycle for one task-bound objective.

    The public task resolves the objective and query branch before calling this
    function. This lifecycle only samples the task's waveform support, renders
    the shared visual grammar, and binds answer/annotation from the same trace.
    """

    defaults = SignalTransformTaskDefaults()
    post_image_noise_defaults = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (attempt_index * 7919)
        try:
            axes = resolve_signal_transform_axes(
                instance_seed=int(attempt_seed),
                params=task_params,
                generation_defaults=generation_defaults,
                supported_waveform_families=objective.supported_waveform_families,
                namespace=str(objective.lifecycle_namespace),
            )
            scenario = build_signal_scenario(
                axes,
                int(attempt_seed),
                namespace=str(objective.lifecycle_namespace),
            )
            render_defaults = resolve_signal_render_defaults(
                task_params,
                rendering_defaults,
                fallback_defaults=defaults,
                instance_seed=int(attempt_seed),
                namespace=str(objective.lifecycle_namespace),
            )
        except Exception as exc:  # pragma: no cover - surfaced if all attempts fail.
            last_error = exc
            continue

        canvas_width, canvas_height = resolve_canvas_size(
            task_params,
            rendering_defaults,
            fallback_defaults=defaults,
        )
        background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
            instance_seed=int(attempt_seed),
            params=task_params,
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(attempt_seed),
            namespace=f"{objective.lifecycle_namespace}.font",
            params=task_params,
        )
        font_record = get_font_family_record(str(font_family))
        rendered = render_signal_transform_scene(
            image=background,
            axes=axes,
            scenario=scenario,
            render_defaults=render_defaults,
            font_family=str(font_family),
            style=diagram_style,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(attempt_seed),
            params=task_params,
            default_config=post_image_noise_defaults,
        )

        prompt_required = required_group_defaults(
            prompt_defaults,
            (
                "bundle_id",
            ),
            context=f"prompt defaults for {objective.public_task_id}",
        )
        prompt_artifacts = build_signal_transform_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_required["bundle_id"]),
            task_key=str(objective.task_prompt_key),
            query_key=str(selected_branch),
            dynamic_slots={},
            instance_seed=int(attempt_seed),
        )

        answer_gt = TypedValue(type="option_letter", value=str(axes.correct_option_letter))
        annotation_bbox_map = normalize_signal_transform_annotation_bbox_map(rendered.annotation_bbox_map)
        annotation_gt = TypedValue(
            type="bbox_map",
            value={str(key): list(value) for key, value in annotation_bbox_map.items()},
        )
        trace_payload = build_signal_transform_trace_payload(
            objective=objective,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            prompt_bundle_id=str(prompt_required["bundle_id"]),
            prompt_artifacts=prompt_artifacts,
            axes=axes,
            scenario=scenario,
            rendered=rendered,
            image=image,
            font_family=str(font_family),
            font_record=font_record,
            diagram_style_meta=diagram_style_meta,
            background_meta=background_meta,
            render_defaults=render_defaults,
            post_noise_meta=post_noise_meta,
            annotation_gt=annotation_gt,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )
    raise RuntimeError(
        "failed to generate signal-transform instance after "
        f"{max_attempts} attempts: {last_error}"
    )


def build_signal_transform_trace_payload(
    *,
    objective: SignalTransformObjective,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_bundle_id: str,
    prompt_artifacts: Any,
    axes: Any,
    scenario: Any,
    rendered: Any,
    image: Any,
    font_family: str,
    font_record: Any,
    diagram_style_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    render_defaults: Mapping[str, int],
    post_noise_meta: Mapping[str, Any],
    annotation_gt: TypedValue,
) -> dict[str, Any]:
    """Assemble trace metadata for one rendered spectrum-match instance.

    The payload binds the selected option, input waveform, and annotation roles
    from the rendered scenario. Objective-specific values arrive as semantic
    arguments rather than public task-id routing inside shared rendering code.
    """

    return {
        "scene_ir": {
            "scene_kind": f"physics_signal_transform_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_branch),
                "internal_query_id": str(objective.internal_query_id),
                "scene_variant": str(axes.scene_variant),
                "waveform_family": str(scenario.waveform_family),
                "correct_option_letter": str(axes.correct_option_letter),
                "target_answer": str(axes.correct_option_letter),
                "correct_spectrum_signature": str(scenario.correct_spectrum.signature),
            },
        },
        "query_spec": {
            "query_id": str(selected_branch),
            "template_id": str(prompt_bundle_id),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "query_id": str(selected_branch),
                "internal_query_id": str(objective.internal_query_id),
                "scene_variant": str(axes.scene_variant),
                "waveform_family": str(scenario.waveform_family),
                "correct_option_letter": str(axes.correct_option_letter),
                "target_answer": str(axes.correct_option_letter),
                "answer_support": list(OPTION_LABELS),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "waveform_family_probabilities": dict(axes.waveform_family_probabilities),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
            },
        },
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "canvas_width": int(image.size[0]),
            "canvas_height": int(image.size[1]),
            "font": {
                "font_family": str(font_family),
                "font_asset_version": font_asset_version(),
                "font_asset": font_record.to_trace(),
                "scope": "signal_transform",
                "selection_policy": {
                    "pool": "global_approved_font_pool",
                    "include_tags": [],
                    "exclude_tags": [],
                    "exclusion_reason": "",
                },
            },
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": background_meta,
            "render_defaults": dict(render_defaults),
            "post_image_noise": post_noise_meta,
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "query_id": str(selected_branch),
            "internal_query_id": str(objective.internal_query_id),
            "scene_variant": str(axes.scene_variant),
            "waveform_family": str(scenario.waveform_family),
            "time_cycles": int(scenario.time_cycles),
            "tone_bins": list(scenario.tone_bins),
            "pulse_width": round(float(scenario.pulse_width), 4),
            "correct_option_letter": str(axes.correct_option_letter),
            "target_answer": str(axes.correct_option_letter),
            "answer_type": "option_letter",
            "answer_option_labels": list(OPTION_LABELS),
            "correct_spectrum": spectrum_payload(scenario.correct_spectrum),
            "option_map": {
                str(label): spectrum_payload(spec)
                for label, spec in scenario.option_specs.items()
            },
            "annotation_entity_ids": [
                "input_waveform",
                f"spectrum_{str(axes.correct_option_letter)}",
            ],
        },
        "sampling": {
            "query_id_probabilities": dict(branch_probabilities),
            "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
            "waveform_family_probabilities": dict(axes.waveform_family_probabilities),
            "target_answer_probabilities": dict(axes.target_answer_probabilities),
        },
        "witness_symbolic": {
            "type": "bbox_map",
            "keys": sorted(annotation_gt.value.keys()),
        },
        "projected_annotation": {
            "type": "bbox_map",
            "bbox_map": {str(key): list(value) for key, value in annotation_gt.value.items()},
            "pixel_bbox_map": {str(key): list(value) for key, value in annotation_gt.value.items()},
        },
        "background": background_meta,
        "technical_diagram_style": dict(diagram_style_meta),
        "post_image_noise": post_noise_meta,
    }


__all__ = ["SignalTransformObjective", "run_signal_transform_lifecycle"]
