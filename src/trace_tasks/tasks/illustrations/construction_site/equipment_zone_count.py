"""Count construction vehicles in a requested site zone."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from .shared.annotations import (
    construction_equipment_bbox_map,
    construction_worker_bbox_map,
    sort_construction_bbox_centers,
    sort_construction_bboxes,
)
from .shared.labels import construction_zone_display_name
from .shared.output import construction_count_trace_sections, serialize_construction_scene
from .shared.prompts import render_construction_prompt, required_construction_prompt_defaults
from .shared.rendering import render_construction_site_scene
from .shared.state import (
    ConstructionEquipmentSpec,
    ConstructionMaterialSpec,
    ConstructionWorkerSpec,
)
from .shared.sampling import (
    bounds,
    color_support,
    equipment_support,
    material_support,
    query_support,
    render_params,
    sample_count,
    setting_weights,
    spawned_task_rng,
    style_weights,
    tool_support,
    uniform_string_probability_map,
    zone_support,
)


TASK_ID = "task_illustrations__construction_site__equipment_zone_count"
SCENE_ID = "construction_site"
QUERY_IDS: Tuple[str, ...] = (
    "vehicle_in_zone_count",
)


@dataclass(frozen=True)
class _Defaults:
    equipment_count_min: int = 5
    equipment_count_max: int = 9
    target_count_min: int = 0
    target_count_max: int = 4
    worker_count_min: int = 3
    worker_count_max: int = 6
    material_count_min: int = 5
    material_count_max: int = 8
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    target_zone_id: str
    target_count: int
    equipment_count: int
    worker_count: int
    material_count: int
    equipment_specs: Tuple[ConstructionEquipmentSpec, ...]
    worker_specs: Tuple[ConstructionWorkerSpec, ...]
    material_specs: Tuple[ConstructionMaterialSpec, ...]
    zone_name: str
    query_probabilities: Dict[str, float]
    zone_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    equipment_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "illustrations",
    SCENE_ID,
    task_id=TASK_ID,
)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Build equipment specs with a unique target-zone count answer."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    query_values = query_support(params, _GEN_DEFAULTS, QUERY_IDS)
    zones = zone_support(params, _GEN_DEFAULTS)
    equipment_values = equipment_support(params, _GEN_DEFAULTS)
    materials = material_support(params, _GEN_DEFAULTS)
    colors = color_support(params, _GEN_DEFAULTS)
    tools = tool_support(params, _GEN_DEFAULTS)

    explicit_query = params.get("query_id")
    if explicit_query is not None:
        query_id = str(explicit_query)
        if query_id not in set(query_values):
            raise ValueError("query_id is outside configured support")
        query_probabilities = uniform_string_probability_map(query_values, selected=query_id)
    else:
        query_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:query")
        query_id, query_probabilities = uniform_choice_with_probabilities(
            query_rng,
            query_values,
            sort_keys=False,
        )
        query_id = str(query_id)
    explicit_zone = params.get("target_zone_id")
    if explicit_zone is not None:
        target_zone = str(explicit_zone)
        if target_zone not in set(zones):
            raise ValueError("target_zone_id is outside configured zone_support")
        zone_probabilities = uniform_string_probability_map(zones, selected=target_zone)
    else:
        zone_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:target_zone")
        target_zone, zone_probabilities = uniform_choice_with_probabilities(
            zone_rng,
            zones,
            sort_keys=False,
        )
        target_zone = str(target_zone)
    if target_zone not in set(zones):
        raise ValueError("target_zone_id is outside configured zone_support")

    target_min, target_max = bounds(
        params,
        _GEN_DEFAULTS,
        "target_count_min",
        "target_count_max",
        _DEFAULTS.target_count_min,
        _DEFAULTS.target_count_max,
    )
    target_count, target_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    equipment_min, equipment_max = bounds(
        params,
        _GEN_DEFAULTS,
        "equipment_count_min",
        "equipment_count_max",
        _DEFAULTS.equipment_count_min,
        _DEFAULTS.equipment_count_max,
    )
    equipment_low = max(int(equipment_min), int(target_count) + 1)
    equipment_count, equipment_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:equipment_count",
        low=int(equipment_low),
        high=int(equipment_max),
        explicit_key="equipment_count",
    )
    other_zones = [str(zone) for zone in zones if str(zone) != str(target_zone)]
    if not other_zones:
        raise ValueError("equipment-in-zone task needs at least one non-target zone")
    equipment_specs = [
        ConstructionEquipmentSpec(equipment_type=str(rng.choice(equipment_values)), zone_id=target_zone, role="target")
        for _ in range(int(target_count))
    ]
    equipment_specs.extend(
        ConstructionEquipmentSpec(
            equipment_type=str(rng.choice(equipment_values)),
            zone_id=str(rng.choice(tuple(other_zones))),
            role="distractor",
        )
        for _ in range(int(equipment_count) - int(target_count))
    )
    rng.shuffle(equipment_specs)

    worker_min, worker_max = bounds(params, _GEN_DEFAULTS, "worker_count_min", "worker_count_max", _DEFAULTS.worker_count_min, _DEFAULTS.worker_count_max)
    worker_count = int(rng.randint(int(worker_min), int(worker_max)))
    worker_specs = tuple(
        ConstructionWorkerSpec(
            hard_hat_color=str(rng.choice(colors)),
            vest_color=str(rng.choice(colors)),
            tool_type=str(rng.choice(tools)) if rng.random() < 0.40 else None,
            role="decor",
        )
        for _ in range(worker_count)
    )
    material_min, material_max = bounds(
        params,
        _GEN_DEFAULTS,
        "material_count_min",
        "material_count_max",
        _DEFAULTS.material_count_min,
        _DEFAULTS.material_count_max,
    )
    material_count = int(rng.randint(int(material_min), int(material_max)))
    material_specs = tuple(
        ConstructionMaterialSpec(material_type=str(rng.choice(materials)), role="decor")
        for _ in range(material_count)
    )

    return _SampleSpec(
        query_id=str(query_id),
        target_zone_id=str(target_zone),
        target_count=int(target_count),
        equipment_count=int(equipment_count),
        worker_count=int(worker_count),
        material_count=int(material_count),
        equipment_specs=tuple(equipment_specs),
        worker_specs=tuple(worker_specs),
        material_specs=tuple(material_specs),
        zone_name=construction_zone_display_name(str(target_zone)),
        query_probabilities=dict(query_probabilities),
        zone_probabilities=dict(zone_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        equipment_count_probabilities=dict(equipment_count_probabilities),
    )


def _equipment_ids_in_target_zone(scene: Any, sample: _SampleSpec) -> Tuple[str, ...]:
    """Select construction equipment whose rendered zone matches the query zone."""

    return tuple(
        str(equipment.equipment_id)
        for equipment in scene.equipment
        if str(equipment.zone_id) == str(sample.target_zone_id)
    )


def _equipment_dynamic_slots(prompt_defaults: Mapping[str, Any], sample: _SampleSpec) -> Dict[str, str]:
    """Format the prompt slots for the selected construction zone."""

    return {
        "object_description": str(prompt_defaults["object_description"]).format(worker_count=int(sample.worker_count)),
        "question_text": str(prompt_defaults["question_text_vehicle_in_zone_count"]).format(zone_name=str(sample.zone_name)),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_equipment_in_zone"]).format(zone_name=str(sample.zone_name)),
        "annotation_hint": str(prompt_defaults["annotation_hint_equipment_in_zone"]).format(zone_name=str(sample.zone_name)),
        "json_example": str(prompt_defaults["json_example_equipment_in_zone"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_equipment_in_zone"]),
    }




@register_task
class IllustrationsCountingEquipmentInZoneCountTask:
    """Count construction vehicles placed in one zone."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "illustrations"
    supported_queries = QUERY_IDS
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one scene and bind target-zone equipment to annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                rp = render_params(
                    params,
                    _RENDER_DEFAULTS,
                    fallback_width=_DEFAULTS.canvas_width,
                    fallback_height=_DEFAULTS.canvas_height,
                    fallback_scale=_DEFAULTS.render_scale,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:canvas_profile",
                )
                scene = render_construction_site_scene(
                    rng=scene_rng,
                    worker_specs=sample.worker_specs,
                    material_specs=sample.material_specs,
                    equipment_specs=sample.equipment_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(params, _RENDER_DEFAULTS),
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                    instance_seed=int(instance_seed),
                    font_params={**dict(_RENDER_DEFAULTS), **dict(params)},
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                scene = None
        if scene is None or sample is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        counted_equipment_ids = _equipment_ids_in_target_zone(scene, sample)
        if len(counted_equipment_ids) != int(sample.target_count):
            raise RuntimeError("rendered equipment count did not match sample target")
        equipment_bbox_map = construction_equipment_bbox_map(scene)
        counted_equipment_bboxes = sort_construction_bboxes(equipment_bbox_map, counted_equipment_ids)
        counted_equipment_points = sort_construction_bbox_centers(equipment_bbox_map, counted_equipment_ids)
        annotation_artifacts = bbox_set_annotation_artifacts(counted_equipment_bboxes)
        serialized_scene, bbox_map = serialize_construction_scene(scene)

        prompt_defaults = required_construction_prompt_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                "question_text_vehicle_in_zone_count",
                "answer_hint_equipment_in_zone",
                "annotation_hint_equipment_in_zone",
                "json_example_equipment_in_zone",
                "json_example_answer_only_equipment_in_zone",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        dynamic_slots = _equipment_dynamic_slots(prompt_defaults, sample)
        prompt_artifacts = render_construction_prompt(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            dynamic_slots=dynamic_slots,
            instance_seed=int(instance_seed),
        )
        zone_counts = dict(Counter(str(equipment.zone_id) for equipment in scene.equipment))
        query_params = {
            "query_id": str(sample.query_id),
            "target_zone_id": str(sample.target_zone_id),
            "target_count": int(sample.target_count),
            "equipment_count": int(sample.equipment_count),
            "query_id_probabilities": dict(sample.query_probabilities),
            "query_probabilities": dict(sample.query_probabilities),
            "zone_probabilities": dict(sample.zone_probabilities),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "equipment_count_probabilities": dict(sample.equipment_count_probabilities),
        }
        trace_payload = {
            **construction_count_trace_sections(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "query_id": str(sample.query_id),
                    "target_zone_id": str(sample.target_zone_id),
                },
                render_map={
                    "bboxes_px": bbox_map,
                    "equipment_bboxes_px": equipment_bbox_map,
                    "worker_bboxes_px": construction_worker_bbox_map(scene),
                    "counted_equipment_ids": list(counted_equipment_ids),
                    "counted_equipment_bboxes_px": counted_equipment_bboxes,
                    "counted_equipment_points_px": counted_equipment_points,
                },
                execution_trace={
                    "query_id": str(sample.query_id),
                    "scene_id": SCENE_ID,
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "target_count": int(sample.target_count),
                    "equipment_count": int(sample.equipment_count),
                    "target_zone_id": str(sample.target_zone_id),
                    "equipment_zone_counts": zone_counts,
                    "counted_equipment_ids": list(counted_equipment_ids),
                    "scene": serialized_scene[0],
                },
                witness_symbolic={"counted_equipment_ids": list(counted_equipment_ids), "answer": int(sample.target_count)},
                projected_annotation=annotation_artifacts.projected_annotation,
            ),
            "query_spec": {
                "task_id": self.task_id,
                "query_id": str(sample.query_id),
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": query_params,
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(sample.target_count)),
            annotation_gt=annotation_artifacts.annotation_gt,
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
        )


__all__ = ["IllustrationsCountingEquipmentInZoneCountTask"]
