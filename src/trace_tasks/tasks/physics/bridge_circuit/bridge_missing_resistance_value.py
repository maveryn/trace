"""Solve a missing resistance in a balanced bridge circuit."""

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
from .shared.prompts import SCENE_PROMPT_KEY, build_bridge_prompt_artifacts
from .shared.rendering import render_bridge_circuit
from .shared.sampling import resolve_bridge_scenario
from .shared.state import BRIDGE_EQUATION, SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__bridge_circuit__bridge_missing_resistance_value"
TASK_NAMESPACE = "physics_bridge_circuit_bridge_missing_resistance"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "bridge_missing_resistance_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsBridgeCircuitMissingResistanceValueTask:
    """Solve a missing resistance from a balanced bridge circuit."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('topology', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate, bind, and trace one bridge-circuit objective instance."""

        _ = int(max_attempts)
        raw_params = dict(params or {})
        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=raw_params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        scenario = resolve_bridge_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_bridge_circuit(
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
        prompt_artifacts = build_bridge_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.target_answer))
        annotation_gt = TypedValue(
            type="bbox",
            value=list(rendered.annotation_bbox_map["target_resistor"]),
        )
        resistor_values = {
            str(spec.label): int(spec.value_ohm)
            for spec in scenario.resistors
        }
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "scene_variant": str(scenario.scene_variant),
                "accent_color_name": str(scenario.accent_color_name),
                "missing_resistor": str(scenario.missing_resistor),
                "target_answer": int(scenario.target_answer),
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                "accent_color_name_probabilities": dict(scenario.accent_color_name_probabilities),
                "missing_resistor_probabilities": dict(scenario.missing_resistor_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_bridge_circuit_{scenario.scene_variant}",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "bridge_equation": BRIDGE_EQUATION,
                    "zero_meter_reading": 0,
                    "missing_resistor": str(scenario.missing_resistor),
                    "target_answer": int(scenario.target_answer),
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
                "background_style": dict(rendered.render_map["background_style"]),
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "scene_variant": str(scenario.scene_variant),
                "query_id": str(query_id),
                "accent_color_name": str(scenario.accent_color_name),
                "resistor_values": dict(resistor_values),
                "missing_resistor": str(scenario.missing_resistor),
                "target_answer": int(scenario.target_answer),
                "bridge_balance_product_left": int(resistor_values["R1"]) * int(resistor_values["R4"]),
                "bridge_balance_product_right": int(resistor_values["R2"]) * int(resistor_values["R3"]),
                "zero_meter_reading": 0,
                "annotation_entity_ids": ["target_resistor"],
                "sampling_probabilities": {
                    "query_id": dict(query_probabilities),
                    "scene_variant": dict(scenario.scene_variant_probabilities),
                    "accent_color_name": dict(scenario.accent_color_name_probabilities),
                    "missing_resistor": dict(scenario.missing_resistor_probabilities),
                    "target_answer": dict(scenario.target_answer_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object",
                "id": "target_resistor",
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_gt.value),
                "pixel_bbox": list(annotation_gt.value),
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
            query_id=str(query_id),
            scene_id=SCENE_ID,
        )


__all__ = ["PhysicsBridgeCircuitMissingResistanceValueTask"]
