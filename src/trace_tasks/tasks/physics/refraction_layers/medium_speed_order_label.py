"""Medium-speed order task for the refraction-layers scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import normalize_refraction_annotation_bbox_set
from .shared.prompts import build_refraction_prompt_artifacts
from .shared.rendering import render_refraction_layers_scene
from .shared.sampling import make_refraction_scenario
from .shared.state import MEDIUM_LABELS, OPTION_LABELS, SCENE_ID


TASK_ID = "task_physics__refraction_layers__medium_speed_order_label"
TASK_NAMESPACE = "physics_refraction_layers_medium_speed_order_label"
INTERNAL_QUERY_ID = "three_medium_speed_order"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    split_generation_rendering_prompt_defaults(
        _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
        task_id=TASK_ID,
    )
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@register_task
class PhysicsRefractionLayersMediumSpeedOrderLabelTask:
    """Choose the fastest-to-slowest light-speed order from ray bending."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Own scenario sampling, answer binding, annotation binding, and output assembly."""

        del max_attempts
        params = dict(params or {})
        selected_query, query_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        scenario = make_refraction_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
        )
        canvas_width = int(
            task_params.get(
                "canvas_width",
                group_default(_RENDER_DEFAULTS, "canvas_width", 1080),
            )
        )
        canvas_height = int(
            task_params.get(
                "canvas_height",
                group_default(_RENDER_DEFAULTS, "canvas_height", 700),
            )
        )
        background, background_meta, diagram_style, diagram_style_meta = (
            prepare_physics_diagram_style_and_background(
                instance_seed=int(instance_seed),
                params=task_params,
                scene_id=SCENE_ID,
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                require_grid=True,
            )
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}.font",
            params=task_params,
        )
        font_record = get_font_family_record(str(font_family))
        rendered = render_refraction_layers_scene(
            image=background,
            scenario=scenario,
            font_family=str(font_family),
            style=diagram_style,
            instance_seed=int(instance_seed),
            render_defaults=_RENDER_DEFAULTS,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = build_refraction_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults["bundle_id"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )

        answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_label))
        annotation_bbox_set = normalize_refraction_annotation_bbox_set(rendered.annotation_bbox_map)
        annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_bbox_set])
        annotation_entity_ids = ("interface_1_bend", "interface_2_bend")

        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_refraction_layers_three_media",
                "entities": [
                    {
                        "entity_id": str(label),
                        "entity_type": "medium_region",
                        "speed_rank_fastest_to_slowest": int(scenario.speed_order.index(str(label)) + 1),
                        "relative_speed": float(scenario.medium_speeds[str(label)]),
                    }
                    for label in MEDIUM_LABELS
                ],
                "relations": {
                    "query_id": str(selected_query),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "orientation": str(scenario.orientation),
                    "entry_side": str(scenario.entry_side),
                    "speed_order": list(scenario.speed_order),
                    "correct_label": str(scenario.correct_label),
                },
            },
            "query_spec": {
                "query_id": str(selected_query),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(selected_query),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "query_id_probabilities": dict(query_probs),
                    "answer_support": list(OPTION_LABELS),
                    "target_answer": str(scenario.correct_label),
                    "speed_order": list(scenario.speed_order),
                    "orientation": str(scenario.orientation),
                    "entry_side": str(scenario.entry_side),
                },
            },
            "render_spec": {
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "font": {
                    "font_family": str(font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": "refraction_layers_diagram",
                    "selection_policy": {
                        "pool": "global_approved_font_pool",
                        "include_tags": [],
                        "exclude_tags": [],
                        "exclusion_reason": "",
                    },
                },
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": background_meta,
                "post_image_noise": post_noise_meta,
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_query),
                "internal_query_id": INTERNAL_QUERY_ID,
                "orientation": str(scenario.orientation),
                "entry_side": str(scenario.entry_side),
                "speed_order": list(scenario.speed_order),
                "medium_speeds": {str(k): float(v) for k, v in scenario.medium_speeds.items()},
                "angle_by_medium_deg": {str(k): float(v) for k, v in scenario.angle_by_medium_deg.items()},
                "option_map": dict(scenario.option_map),
                "target_answer": str(scenario.correct_label),
                "annotation_entity_ids": list(annotation_entity_ids),
            },
            "witness_symbolic": {
                "type": "bbox_set",
                "entity_ids": list(annotation_entity_ids),
            },
            "projected_annotation": {
                "type": "bbox_set",
                "bbox_set": [list(bbox) for bbox in annotation_gt.value],
                "pixel_bbox_set": [list(bbox) for bbox in annotation_gt.value],
            },
            "background": background_meta,
            "post_image_noise": post_noise_meta,
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(selected_query),
            scene_id=SCENE_ID,
        )


__all__ = ["PhysicsRefractionLayersMediumSpeedOrderLabelTask"]
