"""Choose the brightest or dimmest bulb in a visible circuit."""

from __future__ import annotations

from typing import Any, Dict, Tuple

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
from .shared.prompts import SCENE_PROMPT_KEY, build_bulb_prompt_artifacts
from .shared.rendering import render_bulb_circuit
from .shared.sampling import resolve_bulb_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__bulb_circuit__brightness_extremum_label"
TASK_NAMESPACE = "physics_bulb_circuit_brightness_extremum"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "brightest_bulb_label",
    "dimmest_bulb_label",
)
TASK_PROMPT_KEY = "brightness_extremum_query"
_TARGET_DIRECTION_BY_QUERY = {
    "brightest_bulb_label": "brightest",
    "dimmest_bulb_label": "dimmest",
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsBulbCircuitBrightnessExtremumLabelTask:
    """Choose the brightest or dimmest labeled bulb in a visible circuit."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate, bind, and trace one bulb-brightness objective instance."""

        _ = int(max_attempts)
        raw_params = dict(params or {})
        selected_query, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=raw_params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        target_direction = _TARGET_DIRECTION_BY_QUERY[str(selected_query)]
        scenario = resolve_bulb_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            target_direction=str(target_direction),
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_bulb_circuit(
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
        prompt_artifacts = build_bulb_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="string", value=str(scenario.correct_label))
        annotation_gt = TypedValue(
            type="bbox",
            value=list(rendered.annotation_bbox_map[str(scenario.correct_label)]),
        )
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "scene_variant": str(scenario.scene_variant),
                "target_direction": str(scenario.target_direction),
                "accent_color_name": str(scenario.accent_color_name),
                "target_answer": str(scenario.correct_label),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                "accent_color_name_probabilities": dict(
                    scenario.accent_color_name_probabilities
                ),
                "target_label_probabilities": dict(scenario.target_label_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_bulb_circuit_{scenario.scene_variant}",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "scene_variant": str(scenario.scene_variant),
                    "target_direction": str(scenario.target_direction),
                    "brightest_label": str(scenario.brightest_label),
                    "dimmest_label": str(scenario.dimmest_label),
                    "correct_label": str(scenario.correct_label),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": {
                "scene_variant": str(scenario.scene_variant),
                "canvas_width": int(rendered.image.size[0]),
                "canvas_height": int(rendered.image.size[1]),
                "accent_color_name": str(scenario.accent_color_name),
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
                "technical_diagram_style": dict(rendered.render_map["technical_diagram_style"]),
                "background_style": rendered.render_map["background_style"],
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "scene_variant": str(scenario.scene_variant),
                "query_id": str(selected_query),
                "target_direction": str(scenario.target_direction),
                "accent_color_name": str(scenario.accent_color_name),
                "target_answer": str(scenario.correct_label),
                "brightest_label": str(scenario.brightest_label),
                "dimmest_label": str(scenario.dimmest_label),
                "bulb_specs": [
                    {
                        "slot_id": str(spec.slot_id),
                        "label": str(spec.label),
                        "resistance_ohm": int(spec.resistance_ohm),
                        "relative_power": float(spec.relative_power),
                    }
                    for spec in scenario.bulbs
                ],
                "annotation_entity_ids": [str(scenario.correct_label)],
                "sampling_probabilities": {
                    "query_id": dict(branch_probabilities),
                    "scene_variant": dict(scenario.scene_variant_probabilities),
                    "accent_color_name": dict(scenario.accent_color_name_probabilities),
                    "target_label": dict(scenario.target_label_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object",
                "ids": [str(scenario.correct_label)],
                "entity_id": str(scenario.correct_label),
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_gt.value),
                "pixel_bbox": list(annotation_gt.value),
            },
            "background": rendered.render_map["background_style"],
            "post_image_noise": rendered.render_map["post_image_noise"],
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
            query_id=str(selected_query),
            scene_id=SCENE_ID,
        )


__all__ = ["PhysicsBulbCircuitBrightnessExtremumLabelTask"]
