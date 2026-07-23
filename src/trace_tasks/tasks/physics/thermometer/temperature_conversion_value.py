"""Public task for thermometer temperature conversion."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

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
    build_thermometer_prompt_artifacts,
)
from .shared.rendering import render_thermometer
from .shared.sampling import make_thermometer_scenario
from .shared.state import SCENE_ID, ThermometerDefaults


TASK_ID = "task_physics__thermometer__temperature_conversion_value"
TASK_NAMESPACE = "physics_thermometer_temperature_conversion_value"
TASK_PROMPT_KEY = "temperature_conversion_query"
CELSIUS_TO_FAHRENHEIT_QUERY_ID = "celsius_to_fahrenheit_value"
FAHRENHEIT_TO_CELSIUS_QUERY_ID = "fahrenheit_to_celsius_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (CELSIUS_TO_FAHRENHEIT_QUERY_ID, FAHRENHEIT_TO_CELSIUS_QUERY_ID)
_QUERY_TO_UNITS: Mapping[str, Tuple[str, str]] = {
    CELSIUS_TO_FAHRENHEIT_QUERY_ID: ("C", "F"),
    FAHRENHEIT_TO_CELSIUS_QUERY_ID: ("F", "C"),
}

_DEFAULTS = ThermometerDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


def units_for_query(query_id: str) -> Tuple[str, str]:
    """Map the public query branch onto semantic source and target units."""

    try:
        return _QUERY_TO_UNITS[str(query_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {query_id}") from exc


@register_task
class PhysicsThermometerTemperatureConversionValueTask:
    """Read a thermometer and convert the source temperature to the other unit."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate, bind, and trace one thermometer conversion instance."""

        raw_params = dict(params or {})
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + attempt_index * 7919
            try:
                query_id, query_probabilities, task_params = select_task_query_id(
                    instance_seed=int(attempt_seed),
                    params=raw_params,
                    supported_query_ids=SUPPORTED_QUERY_IDS,
                    default_query_id=CELSIUS_TO_FAHRENHEIT_QUERY_ID,
                    task_id=TASK_ID,
                    namespace=f"{TASK_NAMESPACE}.query",
                )
                source_unit, target_unit = units_for_query(str(query_id))
                scenario = make_thermometer_scenario(
                    instance_seed=int(attempt_seed),
                    params=task_params,
                    generation_defaults=_GEN_DEFAULTS,
                    source_unit=str(source_unit),
                    target_unit=str(target_unit),
                )
            except Exception as exc:  # pragma: no cover - surfaced below if all attempts fail.
                last_error = exc
                continue

            rendered = render_thermometer(
                instance_seed=int(attempt_seed),
                params=task_params,
                scenario=scenario,
                render_defaults=_RENDER_DEFAULTS,
                defaults=_DEFAULTS,
            )
            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                ("bundle_id", "task_key"),
                context=f"prompt defaults for {TASK_ID}",
            )
            prompt_artifacts = build_thermometer_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
                query_key=str(query_id),
                dynamic_slots={},
                instance_seed=int(attempt_seed),
            )

            answer_gt = TypedValue(type="integer", value=int(scenario.target_temperature))
            annotation_gt = TypedValue(
                type="segment",
                value=[list(point) for point in rendered.annotation_segment],
            )
            font_record = get_font_family_record(str(rendered.font_family))
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(query_id),
                params={
                    "scale_profile": str(scenario.profile.profile_id),
                    "source_temperature": int(scenario.source_temperature),
                    "source_unit": str(scenario.profile.source_unit),
                    "target_answer": int(scenario.target_temperature),
                    "target_unit": str(scenario.profile.target_unit),
                    "query_id_probabilities": dict(query_probabilities),
                    "scale_profile_probabilities": dict(scenario.scale_profile_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"physics_thermometer_{scenario.profile.profile_id}",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(query_id),
                        "scale_profile": str(scenario.profile.profile_id),
                        "source_temperature": int(scenario.source_temperature),
                        "source_unit": str(scenario.profile.source_unit),
                        "target_temperature": int(scenario.target_temperature),
                        "target_unit": str(scenario.profile.target_unit),
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
                        "scope": "thermometer_diagram",
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
                    "query_id": str(query_id),
                    "scale_profile": str(scenario.profile.profile_id),
                    "source_temperature": int(scenario.source_temperature),
                    "source_unit": str(scenario.profile.source_unit),
                    "target_temperature": int(scenario.target_temperature),
                    "target_unit": str(scenario.profile.target_unit),
                    "target_answer": int(scenario.target_temperature),
                    "annotation_entity_ids": ["liquid_level"],
                    "sampling_probabilities": {
                        "query_id": dict(query_probabilities),
                        "scale_profile": dict(scenario.scale_profile_probabilities),
                        "target_answer": dict(scenario.target_answer_probabilities),
                    },
                },
                "sampling": {
                    "query_id_probabilities": dict(query_probabilities),
                    "scale_profile_probabilities": dict(scenario.scale_profile_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                },
                "witness_symbolic": {
                    "type": "segment",
                    "id": "liquid_level",
                },
                "projected_annotation": {
                    "type": "segment",
                    "segment": [list(point) for point in annotation_gt.value],
                    "pixel_segment": [list(point) for point in annotation_gt.value],
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
                query_id=str(query_id),
            )
        raise RuntimeError(
            f"failed to generate thermometer conversion instance after {max_attempts} attempts: {last_error}"
        )


__all__ = ["PhysicsThermometerTemperatureConversionValueTask"]
