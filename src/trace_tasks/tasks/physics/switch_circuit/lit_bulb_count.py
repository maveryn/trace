"""Public task for switch-circuit lit-bulb counting."""

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

from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_switch_circuit_prompt_artifacts,
)
from .shared.rendering import render_switch_circuit
from .shared.sampling import (
    make_switch_circuit_scenario,
    resolve_switch_circuit_render_defaults,
)
from .shared.state import (
    BULB_LABELS,
    NEG_NODE,
    POS_NODE,
    SCENE_ID,
    SCENE_NAMESPACE,
    SWITCH_LABELS,
    TARGET_SUPPORT,
    SwitchCircuitDefaults,
)


TASK_ID = "task_physics__switch_circuit__lit_bulb_count"
TASK_NAMESPACE = "physics_switch_circuit_lit_bulb_count"
TASK_PROMPT_KEY = "lit_bulb_count_query"
INTERNAL_QUERY_ID = "lit_bulb_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)

_DEFAULTS = SwitchCircuitDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


@register_task
class PhysicsSwitchCircuitLitBulbCountTask:
    """Count bulbs that are on in a mixed branch circuit."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
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
        """Generate, bind, and trace one switch-circuit instance."""

        raw_params = dict(params or {})
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + (attempt_index * 7919)
            try:
                selected_query, query_probs, task_params = select_task_query_id(
                    instance_seed=int(attempt_seed),
                    params=raw_params,
                    supported_query_ids=SUPPORTED_QUERY_IDS,
                    default_query_id=SUPPORTED_QUERY_IDS[0],
                    task_id=TASK_ID,
                    namespace=f"{TASK_NAMESPACE}.query",
                )
                scenario = make_switch_circuit_scenario(
                    instance_seed=int(attempt_seed),
                    params=task_params,
                    public_query_id=str(selected_query),
                    public_query_probabilities=query_probs,
                    generation_defaults=_GEN_DEFAULTS,
                )
                render_defaults = resolve_switch_circuit_render_defaults(
                    task_params,
                    _RENDER_DEFAULTS,
                    instance_seed=int(attempt_seed),
                    defaults=_DEFAULTS,
                )
            except Exception as exc:  # pragma: no cover - surfaced below if all attempts fail.
                last_error = exc
                continue

            rendered = render_switch_circuit(
                instance_seed=int(attempt_seed),
                params=task_params,
                scenario=scenario,
                rendering_defaults=_RENDER_DEFAULTS,
                render_defaults=render_defaults,
                defaults=_DEFAULTS,
            )
            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                ("bundle_id", "task_key"),
                context=f"prompt defaults for {TASK_ID}",
            )
            prompt_artifacts = build_switch_circuit_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
                query_key=str(selected_query),
                dynamic_slots={},
                instance_seed=int(attempt_seed),
            )

            answer_gt = TypedValue(type="integer", value=int(scenario.target_answer))
            annotation_gt = TypedValue(
                type="bbox_set",
                value=[list(item) for item in rendered.annotation_bboxes],
            )
            font_record = get_font_family_record(str(rendered.font_family))
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query),
                params={
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "target_answer": int(scenario.target_answer),
                    "answer_support": list(TARGET_SUPPORT),
                    "scene_variant": str(scenario.scene_variant),
                    "query_id_probabilities": dict(scenario.query_id_probabilities),
                    "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                    "accent_color_name_probabilities": dict(scenario.accent_color_name_probabilities),
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_switch_circuit_mixed_branch",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(selected_query),
                        "internal_query_id": INTERNAL_QUERY_ID,
                        "positive_node": POS_NODE,
                        "negative_node": NEG_NODE,
                        "lit_bulb_count": int(scenario.target_answer),
                        "lit_bulbs": list(scenario.lit_bulbs),
                    },
                },
                "query_spec": query_spec,
                "render_spec": {
                    "canvas_width": int(rendered.image.size[0]),
                    "canvas_height": int(rendered.image.size[1]),
                    "font": {
                        "font_family": str(rendered.font_family),
                        "font_asset_version": font_asset_version(),
                        "font_asset": font_record.to_trace(),
                        "scope": "switch_circuit_diagram",
                        "selection_policy": {
                            "pool": "global_approved_font_pool",
                            "include_tags": [],
                            "exclude_tags": [],
                            "exclusion_reason": "",
                        },
                    },
                    "technical_diagram_style": dict(rendered.render_map["technical_diagram_style"]),
                    "background_style": dict(rendered.render_map["background_style"]),
                    "render_defaults": dict(render_defaults),
                    "post_image_noise": dict(rendered.render_map["post_image_noise"]),
                },
                "render_map": dict(rendered.render_map),
                "execution_trace": {
                    "query_id": str(selected_query),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "target_answer": int(scenario.target_answer),
                    "lit_bulbs": list(scenario.lit_bulbs),
                    "switch_states": {
                        label: ("closed" if bool(scenario.switch_states[label]) else "open")
                        for label in SWITCH_LABELS
                    },
                    "edges": [
                        {
                            "edge_id": str(edge.edge_id),
                            "kind": str(edge.kind),
                            "node_a": str(edge.node_a),
                            "node_b": str(edge.node_b),
                            "label": str(edge.label),
                            "conductive": bool(edge.conductive),
                        }
                        for edge in scenario.edges
                    ],
                    "annotation_entity_ids": list(scenario.lit_bulbs),
                    "sampling_probabilities": {
                        "query_id": dict(scenario.query_id_probabilities),
                        "scene_variant": dict(scenario.scene_variant_probabilities),
                        "target_answer": dict(scenario.target_answer_probabilities),
                        "accent_color_name": dict(scenario.accent_color_name_probabilities),
                    },
                },
                "witness_symbolic": {
                    "type": "bbox_set",
                    "entity_ids": list(scenario.lit_bulbs),
                },
                "projected_annotation": {
                    "type": "bbox_set",
                    "bbox_set": [list(item) for item in annotation_gt.value],
                    "pixel_bbox_set": [list(item) for item in annotation_gt.value],
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
                scene_id=SCENE_ID,
                query_id=str(selected_query),
            )
        raise RuntimeError(
            f"failed to generate switch-circuit instance after {max_attempts} attempts: {last_error}"
        )


__all__ = [
    "BULB_LABELS",
    "PhysicsSwitchCircuitLitBulbCountTask",
    "SUPPORTED_QUERY_IDS",
    "SWITCH_LABELS",
    "TASK_ID",
]
