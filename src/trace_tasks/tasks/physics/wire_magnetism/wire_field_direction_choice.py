"""Choose the field direction near a current-carrying wire."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_map_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_wire_magnetism_prompt_artifacts,
)
from .shared.rendering import render_wire_magnetism_scene, resolve_wire_render_defaults
from .shared.sampling import build_wire_scenario
from .shared.state import (
    OPTION_LABELS,
    SCENE_ID,
    WireMagnetismDefaults,
)


TASK_ID = "task_physics__wire_magnetism__wire_field_direction_choice"
TASK_NAMESPACE = "physics_wire_magnetism_wire_field_direction_choice"
TASK_PROMPT_KEY = "wire_field_direction_choice_query"
INTERNAL_QUERY_ID = "perpendicular_wire_field_direction_at_point"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _font_trace(font_family: str, font_record: Any) -> dict[str, Any]:
    """Return common font metadata for wire-magnetism traces."""

    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "wire_magnetism_diagram",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


@register_task
class PhysicsWireMagnetismFieldDirectionChoiceTask:
    """Choose the magnetic-field direction around a wire perpendicular to the page."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one wire-magnetism field-direction choice instance."""

        _ = int(max_attempts)
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        fallback_defaults = WireMagnetismDefaults()
        render_defaults = resolve_wire_render_defaults(
            task_params,
            _RENDER_DEFAULTS,
            fallback_defaults=fallback_defaults,
            instance_seed=int(instance_seed),
            namespace=TASK_NAMESPACE,
        )
        canvas_width = int(render_defaults["canvas_width"])
        canvas_height = int(render_defaults["canvas_height"])
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
        font_record = get_font_family_record(str(font_family))
        scenario = build_wire_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=TASK_NAMESPACE,
        )
        rendered = render_wire_magnetism_scene(
            image=background,
            scenario=scenario,
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
        prompt_artifacts = build_wire_magnetism_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults_required.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults_required.get("task_key", TASK_PROMPT_KEY)),
            query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_label))
        annotation_artifacts = bbox_map_annotation_artifacts(rendered.annotation_bboxes)
        annotation_gt = annotation_artifacts.annotation_gt
        answer_support = list(OPTION_LABELS)
        sampling_probabilities = {
            "query_id": dict(query_probabilities),
            "current_direction": dict(scenario.current_direction_probabilities),
            "point_position": dict(scenario.point_position_probabilities),
            "target_answer": dict(scenario.target_answer_probabilities),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "query_id": str(selected_query),
                "internal_query_id": INTERNAL_QUERY_ID,
                "current_direction": str(scenario.current_direction),
                "current_z_sign": int(scenario.current_z_sign),
                "point_position": str(scenario.point_position),
                "point_offset_phys": list(scenario.point_offset_phys),
                "field_direction": str(scenario.field_direction),
                "correct_option_letter": str(scenario.correct_label),
                "target_answer": str(scenario.correct_label),
                "answer_support": answer_support,
                "query_id_probabilities": dict(query_probabilities),
                "current_direction_probabilities": dict(scenario.current_direction_probabilities),
                "point_position_probabilities": dict(scenario.point_position_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        trace_payload: Mapping[str, Any] = {
            "scene_ir": {
                "scene_kind": "physics_wire_magnetism_perpendicular_wire",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_query),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "current_direction": str(scenario.current_direction),
                    "current_z_sign": int(scenario.current_z_sign),
                    "point_position": str(scenario.point_position),
                    "point_offset_phys": list(scenario.point_offset_phys),
                    "field_direction": str(scenario.field_direction),
                    "correct_option_letter": str(scenario.correct_label),
                    "target_answer": str(scenario.correct_label),
                    "answer_type": "option_letter",
                    "annotation_entity_ids": list(rendered.annotation_entity_ids),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "font": _font_trace(str(font_family), font_record),
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_query),
                "internal_query_id": INTERNAL_QUERY_ID,
                "current_direction": str(scenario.current_direction),
                "current_z_sign": int(scenario.current_z_sign),
                "point_position": str(scenario.point_position),
                "point_offset_phys": list(scenario.point_offset_phys),
                "field_direction": str(scenario.field_direction),
                "option_map": dict(scenario.option_map),
                "correct_option_letter": str(scenario.correct_label),
                "target_answer": str(scenario.correct_label),
                "answer_type": "option_letter",
                "answer_option_labels": answer_support,
                "annotation_entity_ids": list(rendered.annotation_entity_ids),
                "sampling_probabilities": dict(sampling_probabilities),
            },
            "sampling": dict(sampling_probabilities),
            "witness_symbolic": {
                "type": "object_map",
                "ids": list(rendered.annotation_entity_ids),
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
    "PhysicsWireMagnetismFieldDirectionChoiceTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
