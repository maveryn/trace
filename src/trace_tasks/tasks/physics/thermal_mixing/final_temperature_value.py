"""Public task for equal-amount thermal-mixing final temperature."""

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

from .shared.formulas import integer_average
from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_thermal_mixing_prompt_artifacts,
)
from .shared.rendering import render_thermal_mixing
from .shared.sampling import (
    make_thermal_mixing_scenario,
    resolve_thermal_mixing_render_defaults,
)
from .shared.state import (
    FINAL_TEMPERATURE_SUPPORT,
    SCENE_ID,
    ThermalMixingDefaults,
)


TASK_ID = "task_physics__thermal_mixing__final_temperature_value"
TASK_NAMESPACE = "physics_thermal_mixing_final_temperature_value"
TASK_PROMPT_KEY = "final_temperature_value_query"
INTERNAL_QUERY_ID = "equal_amount_final_temperature"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)

_DEFAULTS = ThermalMixingDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


@register_task
class PhysicsThermalMixingFinalTemperatureValueTask:
    """Compute final equilibrium temperature for equal amounts of the same liquid."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
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
        """Generate, bind, and trace one thermal-mixing instance."""

        raw_params = dict(params or {})
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + attempt_index * 7919
            try:
                selected_query, query_probs, task_params = select_task_query_id(
                    instance_seed=int(attempt_seed),
                    params=raw_params,
                    supported_query_ids=SUPPORTED_QUERY_IDS,
                    default_query_id=SUPPORTED_QUERY_IDS[0],
                    task_id=TASK_ID,
                    namespace=f"{TASK_NAMESPACE}.query",
                )
                scenario = make_thermal_mixing_scenario(
                    instance_seed=int(attempt_seed),
                    params=task_params,
                    generation_defaults=_GEN_DEFAULTS,
                )
                render_defaults = resolve_thermal_mixing_render_defaults(
                    task_params,
                    _RENDER_DEFAULTS,
                    instance_seed=int(attempt_seed),
                    defaults=_DEFAULTS,
                )
            except Exception as exc:  # pragma: no cover - surfaced below if all attempts fail.
                last_error = exc
                continue

            rendered = render_thermal_mixing(
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
            prompt_artifacts = build_thermal_mixing_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
                query_key=str(selected_query),
                dynamic_slots={},
                instance_seed=int(attempt_seed),
            )

            answer_gt = TypedValue(type="integer", value=int(scenario.final_temperature_c))
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
                    "target_answer": int(scenario.final_temperature_c),
                    "answer_support": list(FINAL_TEMPERATURE_SUPPORT),
                    "cup_count": int(scenario.cup_count),
                    "query_id_probabilities": dict(query_probs),
                    "cup_count_probabilities": dict(scenario.cup_count_probabilities),
                    "final_temperature_probabilities": dict(scenario.final_temperature_probabilities),
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_thermal_mixing_equal_amounts",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(selected_query),
                        "internal_query_id": INTERNAL_QUERY_ID,
                        "cup_count": int(scenario.cup_count),
                        "same_liquid": True,
                        "equal_amounts": True,
                        "closed_insulated_system": True,
                        "final_temperature_c": int(scenario.final_temperature_c),
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
                        "scope": "thermal_mixing_setup",
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
                    "initial_temperatures_c": [
                        int(value) for value in scenario.initial_temperatures_c
                    ],
                    "final_temperature_c": int(scenario.final_temperature_c),
                    "calculation": {
                        "operation": "average",
                        "sum_temperatures_c": int(sum(scenario.initial_temperatures_c)),
                        "cup_count": int(scenario.cup_count),
                        "computed_average_c": int(
                            integer_average([int(value) for value in scenario.initial_temperatures_c])
                        ),
                    },
                    "annotation_entity_ids": [
                        f"initial_cup_{chr(ord('a') + idx)}"
                        for idx in range(int(scenario.cup_count))
                    ],
                    "sampling_probabilities": {
                        "query_id": dict(query_probs),
                        "cup_count": dict(scenario.cup_count_probabilities),
                        "final_temperature": dict(scenario.final_temperature_probabilities),
                    },
                },
                "sampling": {
                    "query_id_probabilities": dict(query_probs),
                    "cup_count_probabilities": dict(scenario.cup_count_probabilities),
                    "final_temperature_probabilities": dict(scenario.final_temperature_probabilities),
                },
                "witness_symbolic": {
                    "type": "bbox_set",
                    "count": len(annotation_gt.value),
                },
                "projected_annotation": {
                    "type": "bbox_set",
                    "bboxes": [list(item) for item in annotation_gt.value],
                    "pixel_bboxes": [list(item) for item in annotation_gt.value],
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
            f"failed to generate thermal-mixing instance after {max_attempts} attempts: {last_error}"
        )


__all__ = [
    "INTERNAL_QUERY_ID",
    "PhysicsThermalMixingFinalTemperatureValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
