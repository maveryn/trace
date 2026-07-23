"""Choose the waveform panel with the requested wave-property extremum."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_waveform_panel_prompt_artifacts,
)
from .shared.rendering import (
    render_waveform_panel_scene,
    resolve_waveform_render_defaults,
)
from .shared.sampling import build_panel_semantic_specs, resolve_waveform_axes
from .shared.state import (
    PANEL_LABELS,
    SCENE_ID,
    SCENE_NAMESPACE,
    WaveformPanelDefaults,
)


TASK_ID = "task_physics__waveform_panel__wave_property_extremum_label"
TASK_NAMESPACE = "physics_waveform_panel_wave_property_extremum_label"
TASK_PROMPT_KEY = "wave_property_extremum_label_query"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "highest_amplitude_label",
    "lowest_amplitude_label",
    "highest_frequency_label",
    "lowest_frequency_label",
    "longest_wavelength_label",
    "shortest_wavelength_label",
)
QUERY_PROPERTY: Dict[str, str] = {
    "highest_amplitude_label": "amplitude",
    "lowest_amplitude_label": "amplitude",
    "highest_frequency_label": "frequency",
    "lowest_frequency_label": "frequency",
    "longest_wavelength_label": "wavelength",
    "shortest_wavelength_label": "wavelength",
}
QUERY_EXTREMUM: Dict[str, str] = {
    "highest_amplitude_label": "highest",
    "lowest_amplitude_label": "lowest",
    "highest_frequency_label": "highest",
    "lowest_frequency_label": "lowest",
    "longest_wavelength_label": "longest",
    "shortest_wavelength_label": "shortest",
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _query_target_modes(query_id: str) -> Tuple[str | None, str | None]:
    """Translate a public query id into semantic amplitude/cycle target modes."""

    if query_id == "highest_amplitude_label":
        return "high", None
    if query_id == "lowest_amplitude_label":
        return "low", None
    if query_id in {"highest_frequency_label", "shortest_wavelength_label"}:
        return None, "high"
    if query_id in {"lowest_frequency_label", "longest_wavelength_label"}:
        return None, "low"
    raise ValueError(f"unsupported waveform-panel query_id: {query_id}")


def _font_trace(font_family: str, font_record: Any) -> dict[str, Any]:
    """Return common font metadata for waveform-panel traces."""

    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "waveform_panel",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


@register_task
class PhysicsWaveformPanelWavePropertyExtremumLabelTask:
    """Choose the labeled panel with the requested wave-property extremum."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one waveform-panel extremum selection instance."""

        _ = int(max_attempts)
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        axes = resolve_waveform_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=TASK_NAMESPACE,
        )
        amplitude_mode, cycle_mode = _query_target_modes(str(selected_query))
        semantic_panels = build_panel_semantic_specs(
            axes=axes,
            amplitude_mode=amplitude_mode,
            cycle_mode=cycle_mode,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}.{selected_query}",
        )

        fallback_defaults = WaveformPanelDefaults()
        canvas_width = int(
            task_params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", fallback_defaults.canvas_width))
        )
        canvas_height = int(
            task_params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", fallback_defaults.canvas_height))
        )
        background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=task_params,
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}.font",
            params=task_params,
        )
        render_defaults = resolve_waveform_render_defaults(
            task_params,
            _RENDER_DEFAULTS,
            fallback_defaults=fallback_defaults,
            instance_seed=int(instance_seed),
            namespace=TASK_NAMESPACE,
        )
        rendered = render_waveform_panel_scene(
            image=background,
            axes=axes,
            semantic_panels=semantic_panels,
            query_property=str(QUERY_PROPERTY[str(selected_query)]),
            query_extremum=str(QUERY_EXTREMUM[str(selected_query)]),
            render_defaults=render_defaults,
            font_family=str(font_family),
            style=diagram_style,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        prompt_defaults_required = required_group_defaults(
            _PROMPT_DEFAULTS,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_waveform_panel_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults_required.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults_required.get("task_key", TASK_PROMPT_KEY)),
            query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="option_letter", value=str(axes.correct_option_letter))
        annotation_artifacts = bbox_annotation_artifacts(rendered.selected_panel_bbox)
        annotation_gt = annotation_artifacts.annotation_gt
        render_map = dict(rendered.render_map)
        render_map["query_id"] = str(selected_query)
        panels_payload = [dict(panel) for panel in render_map["panels"]]
        answer_support = list(PANEL_LABELS[: int(axes.panel_count)])
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "query_id": str(selected_query),
                "query_property": str(QUERY_PROPERTY[str(selected_query)]),
                "query_extremum": str(QUERY_EXTREMUM[str(selected_query)]),
                "scene_variant": str(axes.scene_variant),
                "panel_count": int(axes.panel_count),
                "correct_option_letter": str(axes.correct_option_letter),
                "target_answer": str(axes.correct_option_letter),
                "answer_support": answer_support,
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "panel_count_probabilities": dict(axes.panel_count_probabilities),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
            },
        )
        font_record = get_font_family_record(str(font_family))
        trace_payload: Mapping[str, Any] = {
            "scene_ir": {
                "scene_kind": f"physics_waveform_panel_{str(axes.scene_variant)}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_query),
                    "query_property": str(QUERY_PROPERTY[str(selected_query)]),
                    "query_extremum": str(QUERY_EXTREMUM[str(selected_query)]),
                    "panel_count": int(axes.panel_count),
                    "correct_option_letter": str(axes.correct_option_letter),
                    "target_answer": str(axes.correct_option_letter),
                    "answer_type": "option_letter",
                    "annotation_entity_ids": list(rendered.annotation_entity_ids),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "scene_variant": str(axes.scene_variant),
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "font": _font_trace(str(font_family), font_record),
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
            },
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(selected_query),
                "query_property": str(QUERY_PROPERTY[str(selected_query)]),
                "query_extremum": str(QUERY_EXTREMUM[str(selected_query)]),
                "scene_variant": str(axes.scene_variant),
                "panel_count": int(axes.panel_count),
                "correct_option_letter": str(axes.correct_option_letter),
                "target_answer": str(axes.correct_option_letter),
                "answer_type": "option_letter",
                "answer_option_labels": answer_support,
                "panels": list(panels_payload),
                "annotation_entity_ids": list(rendered.annotation_entity_ids),
                "sampling_probabilities": {
                    "query_id": dict(query_probabilities),
                    "scene_variant": dict(axes.scene_variant_probabilities),
                    "panel_count": dict(axes.panel_count_probabilities),
                    "target_answer": dict(axes.target_answer_probabilities),
                },
            },
            "sampling": {
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "panel_count_probabilities": dict(axes.panel_count_probabilities),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
            },
            "witness_symbolic": {
                "type": "object_bbox",
                "ids": [str(item) for item in rendered.annotation_entity_ids],
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
            "background": dict(background_meta),
            "technical_diagram_style": dict(diagram_style_meta),
            "post_image_noise": dict(post_noise_meta),
        }
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
            query_id=str(selected_query),
        )


__all__ = [
    "PhysicsWaveformPanelWavePropertyExtremumLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
