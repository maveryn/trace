"""Compute object density from visible submerged fraction and liquid density."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import SCENE_PROMPT_KEY, build_buoyancy_prompt_artifacts
from .shared.rendering import render_buoyancy_density
from .shared.sampling import resolve_buoyancy_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__buoyancy_density__object_density_value"
TASK_NAMESPACE = "physics_buoyancy_density_object_density"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "object_density_value_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsBuoyancyDensityObjectDensityValueTask:
    """Compute object density from floating fraction and liquid density."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one buoyancy-density instance and bind answer/annotation."""

        _ = int(max_attempts)
        raw_params = dict(params or {})
        selected_query, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=raw_params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        scenario = resolve_buoyancy_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_buoyancy_density(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
            render_defaults=_RENDER_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_buoyancy_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="number", value=float(scenario.target_answer))
        annotation_gt = TypedValue(
            type="bbox",
            value=list(rendered.annotation_bbox_map["floating_object"]),
        )
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "scene_variant": str(scenario.scene_variant),
                "object_shape": str(scenario.object_shape),
                "submerged_fraction": [int(scenario.fraction_num), int(scenario.fraction_den)],
                "liquid_density_tenths": int(scenario.liquid_density_tenths),
                "target_answer": float(scenario.target_answer),
                "answer_support": list(scenario.answer_support),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                "object_shape_probabilities": dict(scenario.object_shape_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_buoyancy_density_{scenario.scene_variant}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "scene_variant": str(scenario.scene_variant),
                    "object_shape": str(scenario.object_shape),
                    "submerged_fraction": [int(scenario.fraction_num), int(scenario.fraction_den)],
                    "liquid_density_tenths": int(scenario.liquid_density_tenths),
                    "object_density_tenths": int(scenario.object_density_tenths),
                    "target_answer": float(scenario.target_answer),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": {
                "scene_variant": str(scenario.scene_variant),
                "canvas_width": int(rendered.image.size[0]),
                "canvas_height": int(rendered.image.size[1]),
                "font": {
                    "font_family": str(rendered.font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": SCENE_PROMPT_KEY,
                    "selection_policy": {
                        "pool": "global_approved_font_pool",
                        "include_tags": [],
                        "exclude_tags": [],
                        "exclusion_reason": "",
                    },
                },
                "technical_diagram_style": dict(rendered.render_spec["technical_diagram_style"]),
                "background_style": dict(rendered.render_spec["background_style"]),
                "layout_placement": dict(rendered.render_spec["layout_placement"]),
                "post_image_noise": dict(rendered.render_spec["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_query),
                "scene_variant": str(scenario.scene_variant),
                "object_shape": str(scenario.object_shape),
                "submerged_fraction_num": int(scenario.fraction_num),
                "submerged_fraction_den": int(scenario.fraction_den),
                "liquid_density_g_cm3": float(scenario.liquid_density_tenths) / 10.0,
                "object_density_g_cm3": float(scenario.target_answer),
                "target_answer": float(scenario.target_answer),
                "answer_rounding": "one_decimal",
                "answer_support": list(scenario.answer_support),
                "annotation_entity_ids": ["floating_object"],
                "sampling_probabilities": {
                    "query_id": dict(branch_probabilities),
                    "scene_variant": dict(scenario.scene_variant_probabilities),
                    "object_shape": dict(scenario.object_shape_probabilities),
                    "target_answer": dict(scenario.target_answer_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object",
                "ids": ["floating_object"],
                "entity_id": "floating_object",
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_gt.value),
                "pixel_bbox": list(annotation_gt.value),
            },
            "background": dict(rendered.render_spec["background_style"]),
            "post_image_noise": dict(rendered.render_spec["post_image_noise"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
        )


__all__ = ["PhysicsBuoyancyDensityObjectDensityValueTask"]
