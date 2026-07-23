"""Read an analog ammeter or voltmeter needle value."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, get_font_family_record
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import SCENE_PROMPT_KEY, build_analog_meter_prompt_artifacts
from .shared.rendering import render_analog_meter
from .shared.sampling import probability_map, resolve_meter_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__analog_meter__meter_readout_value"
TASK_NAMESPACE = "physics_analog_meter_meter_readout"
AMMETER_QUERY_ID = "ammeter_readout"
VOLTMETER_QUERY_ID = "voltmeter_readout"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (AMMETER_QUERY_ID, VOLTMETER_QUERY_ID)
TASK_PROMPT_KEY = "analog_meter_readout_query"
_QUERY_TO_METER_KIND = {
    AMMETER_QUERY_ID: "ammeter",
    VOLTMETER_QUERY_ID: "voltmeter",
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


def _select_query_id(instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate the task-owned semantic meter readout branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=AMMETER_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_NAMESPACE}.query",
    )


def _meter_kind_for_query(selected: str) -> str:
    """Map the public query branch onto a semantic meter apparatus kind."""

    try:
        return _QUERY_TO_METER_KIND[str(selected)]
    except KeyError as exc:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected}") from exc


@register_task
class PhysicsAnalogMeterReadoutValueTask:
    """Read an analog ammeter or voltmeter needle value."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one meter readout instance and bind the answer/annotation trace."""

        _ = int(max_attempts)
        raw_params = dict(params or {})
        query_id, query_probabilities, task_params = _select_query_id(int(instance_seed), raw_params)
        meter_kind = _meter_kind_for_query(str(query_id))
        scenario = resolve_meter_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            meter_kind=str(meter_kind),
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_analog_meter(
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
        prompt_artifacts = build_analog_meter_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.readout_value))
        needle_segment = [
            [float(value) for value in point]
            for point in rendered.render_map["needle_segment_px"]
        ]
        annotation_gt = TypedValue(type="segment", value=needle_segment)
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "meter_kind": str(meter_kind),
                "meter_profile": str(scenario.profile.profile_id),
                "unit": str(scenario.profile.unit),
                "target_answer": int(scenario.readout_value),
                "query_id_probabilities": dict(query_probabilities),
                "meter_profile_probabilities": dict(scenario.meter_profile_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_analog_meter_{scenario.profile.profile_id}",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(query_id),
                    "meter_kind": str(meter_kind),
                    "meter_profile": str(scenario.profile.profile_id),
                    "unit": str(scenario.profile.unit),
                    "readout_value": int(scenario.readout_value),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": {
                "canvas_width": int(rendered.image.size[0]),
                "canvas_height": int(rendered.image.size[1]),
                "font": {
                    "font_family": str(rendered.font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": SCENE_PROMPT_KEY,
                },
                "technical_diagram_style": dict(rendered.render_map["technical_diagram_style"]),
                "background_style": dict(rendered.render_map["background_style"]),
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(query_id),
                "meter_kind": str(meter_kind),
                "meter_profile": str(scenario.profile.profile_id),
                "unit": str(scenario.profile.unit),
                "scale_max": int(scenario.profile.scale_max),
                "readout_value": int(scenario.readout_value),
                "target_answer": int(scenario.readout_value),
                "annotation_entity_ids": ["needle"],
                "sampling_probabilities": {
                    "query_id": dict(query_probabilities),
                    "meter_profile": dict(scenario.meter_profile_probabilities),
                    "target_answer": dict(scenario.target_answer_probabilities),
                    "meter_kind": probability_map(tuple(sorted(set(_QUERY_TO_METER_KIND.values()))), selected=str(meter_kind)),
                },
            },
            "witness_symbolic": {
                "type": "segment",
                "ids": ["needle"],
            },
            "projected_annotation": {
                "type": "segment",
                "segment": list(annotation_gt.value),
                "pixel_segment": list(annotation_gt.value),
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


__all__ = ["PhysicsAnalogMeterReadoutValueTask"]
