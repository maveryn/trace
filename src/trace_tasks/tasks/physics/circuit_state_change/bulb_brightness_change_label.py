"""Select the bulb whose brightness changes after a switch action."""

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
from .shared.prompts import SCENE_PROMPT_KEY, build_circuit_state_change_prompt_artifacts
from .shared.rendering import render_circuit_state_change
from .shared.sampling import resolve_state_change_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__circuit_state_change__bulb_brightness_change_label"
TASK_NAMESPACE = "physics_circuit_state_change_bulb_brightness_change"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "brightens_after_switch_change",
    "dims_after_switch_change",
    "turns_on_after_switch_change",
    "turns_off_after_switch_change",
)
TASK_PROMPT_KEY = "bulb_brightness_change_query"
_TARGET_CHANGE_CLASS_BY_QUERY = {
    "brightens_after_switch_change": "brightens",
    "dims_after_switch_change": "dims",
    "turns_on_after_switch_change": "turns_on",
    "turns_off_after_switch_change": "turns_off",
}
_ALLOWED_SWITCH_ACTIONS_BY_QUERY = {
    "brightens_after_switch_change": ("closes", "opens"),
    "dims_after_switch_change": ("closes", "opens"),
    "turns_on_after_switch_change": ("closes",),
    "turns_off_after_switch_change": ("opens",),
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsCircuitStateChangeBulbBrightnessLabelTask:
    """Select the bulb whose brightness changes in the requested way."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate, bind, and trace one circuit state-change instance."""

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
        target_change_class = _TARGET_CHANGE_CLASS_BY_QUERY[str(selected_query)]
        allowed_switch_actions = _ALLOWED_SWITCH_ACTIONS_BY_QUERY[str(selected_query)]
        scenario = resolve_state_change_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            target_change_class=str(target_change_class),
            allowed_switch_actions=allowed_switch_actions,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_circuit_state_change(
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
        prompt_artifacts = build_circuit_state_change_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="string", value=str(scenario.correct_label))
        annotation_gt = TypedValue(
            type="bbox_map",
            value={
                str(key): list(value)
                for key, value in rendered.annotation_bbox_map.items()
            },
        )
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "switch_action": str(scenario.switch_action),
                "target_change_class": str(scenario.target_change_class),
                "accent_color_name": str(scenario.accent_color_name),
                "target_label": str(scenario.target_label),
                "target_answer": str(scenario.correct_label),
                "query_id_probabilities": dict(branch_probabilities),
                "switch_action_probabilities": dict(scenario.switch_action_probabilities),
                "target_label_probabilities": dict(scenario.target_label_probabilities),
                "accent_color_name_probabilities": dict(scenario.accent_color_name_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_circuit_state_change_switch_branch",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(selected_query),
                    "switch_action": str(scenario.switch_action),
                    "correct_label": str(scenario.correct_label),
                    "target_change_class": str(scenario.target_change_class),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": {
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
                "background_style": dict(rendered.render_map["background_style"]),
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_query),
                "switch_action": str(scenario.switch_action),
                "accent_color_name": str(scenario.accent_color_name),
                "target_label": str(scenario.target_label),
                "target_answer": str(scenario.correct_label),
                "target_change_class": str(scenario.target_change_class),
                "bulb_specs": [
                    {
                        "role": str(spec.role),
                        "label": str(spec.label),
                        "resistance_ohm": int(spec.resistance_ohm),
                        "power_before": float(spec.power_before),
                        "power_after": float(spec.power_after),
                        "change_class": str(spec.change_class),
                    }
                    for spec in scenario.bulbs
                ],
                "annotation_entity_ids": sorted(annotation_gt.value.keys()),
                "sampling_probabilities": {
                    "query_id": dict(branch_probabilities),
                    "switch_action": dict(scenario.switch_action_probabilities),
                    "target_label": dict(scenario.target_label_probabilities),
                    "accent_color_name": dict(scenario.accent_color_name_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object_map",
                "ids": sorted(annotation_gt.value.keys()),
                "key_to_entity_id": {
                    str(key): str(key)
                    for key in annotation_gt.value.keys()
                },
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_gt.value),
                "pixel_bbox_map": dict(annotation_gt.value),
            },
            "background": dict(rendered.render_map["background_style"]),
            "post_image_noise": dict(rendered.render_map["post_image_noise"]),
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


__all__ = ["PhysicsCircuitStateChangeBulbBrightnessLabelTask"]
