"""Count indoor objects of a named type on a named surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.task_support import sample_count as _shared_sample_count
from ..shared.task_support import bounds as _shared_bounds
from .shared.annotations import serialize_indoor_scene, sort_bbox_centers_by_ids, sort_bboxes_by_ids
from .shared.output import (
    indoor_base_render_map,
    indoor_render_spec,
    object_type_map,
    render_fallback_from_defaults,
)
from .shared.prompts import build_indoor_prompt_artifacts, indoor_setting_name
from .shared.rendering import indoor_scene_entities, render_indoor_scene_from_specs
from .shared.sampling import display_name, support_choice, theme_support, typed_support
from .shared.state import (
    INDOOR_CONTAINER_TYPES,
    INDOOR_OBJECT_TYPES,
    INDOOR_SURFACE_TYPES,
    IndoorObjectSpec,
)


TASK_ID = "task_illustrations__indoor_room__surface_object_count"
SCENE_ID = "indoor_room"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "object_type_on_surface_count"


@dataclass(frozen=True)
class _Defaults:
    object_count_min: int = 10
    object_count_max: int = 16
    target_count_min: int = 1
    target_count_max: int = 5
    canvas_width: int = 1280
    canvas_height: int = 840
    object_size_min_px: int = 52
    object_size_max_px: int = 86
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    theme_id: str
    surface_type: str
    object_type: str
    object_name: str
    target_count: int
    object_count: int
    specs: Tuple[IndoorObjectSpec, ...]
    theme_probabilities: Dict[str, float]
    surface_type_probabilities: Dict[str, float]
    object_type_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)






def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample a surface-count room with exact target witnesses and distractors."""

    object_support = typed_support(
        params,
        _GEN_DEFAULTS,
        param_key="object_type_support",
        default_key="indoor_object_type_support",
        fallback=INDOOR_OBJECT_TYPES,
        error_name="object_type_support",
    )
    surface_support = typed_support(
        params,
        _GEN_DEFAULTS,
        param_key="surface_type_support",
        default_key="surface_type_support",
        fallback=INDOOR_SURFACE_TYPES,
        error_name="surface_type_support",
    )
    theme_id, theme_probabilities = support_choice(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:theme",
        support=theme_support(params, _GEN_DEFAULTS),
        explicit_key="theme_id",
    )
    surface_type, surface_probabilities = support_choice(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:surface",
        support=surface_support,
        explicit_key="surface_type",
    )
    object_type, object_type_probabilities = support_choice(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:object_type",
        support=object_support,
        explicit_key="object_type",
    )
    target_min, target_max = _shared_bounds(params, _GEN_DEFAULTS, "target_count_min", "target_count_max", _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    target_count, target_probabilities = _shared_sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    object_min, object_max = _shared_bounds(params, _GEN_DEFAULTS, "object_count_min", "object_count_max", _DEFAULTS.object_count_min, _DEFAULTS.object_count_max)
    object_low = max(int(object_min), int(target_count) + 4)
    object_count, object_count_probabilities = _shared_sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:object_count",
        low=int(object_low),
        high=int(object_max),
        explicit_key="object_count",
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:spec", int(attempt_index))
    specs = [
        IndoorObjectSpec(str(object_type), "surface", str(surface_type), "target")
        for _ in range(int(target_count))
    ]
    non_target_types = tuple(value for value in object_support if str(value) != str(object_type)) or tuple(
        value for value in INDOOR_OBJECT_TYPES if str(value) != str(object_type)
    )
    other_surfaces = tuple(value for value in INDOOR_SURFACE_TYPES if str(value) != str(surface_type))
    distractor_kinds = [
        ("same_type_other_surface", max(1, min(2, int(object_count) - int(target_count)))),
        ("other_type_target_surface", max(1, min(3, int(object_count) - int(target_count)))),
    ]
    for kind, count in distractor_kinds:
        for _ in range(int(count)):
            if len(specs) >= int(object_count):
                break
            if kind == "same_type_other_surface":
                specs.append(IndoorObjectSpec(str(object_type), "surface", str(rng.choice(other_surfaces)), "distractor"))
            else:
                specs.append(IndoorObjectSpec(str(rng.choice(non_target_types)), "surface", str(surface_type), "distractor"))
    while len(specs) < int(object_count):
        placement_kind = str(rng.choice(("surface", "container")))
        if placement_kind == "surface":
            target = str(rng.choice(other_surfaces))
        else:
            target = str(rng.choice(INDOOR_CONTAINER_TYPES))
        specs.append(IndoorObjectSpec(str(rng.choice(non_target_types)), placement_kind, target, "distractor"))
    rng.shuffle(specs)
    return _SampleSpec(
        theme_id=str(theme_id),
        surface_type=str(surface_type),
        object_type=str(object_type),
        object_name=display_name(str(object_type)),
        target_count=int(target_count),
        object_count=int(object_count),
        specs=tuple(specs),
        theme_probabilities=dict(theme_probabilities),
        surface_type_probabilities=dict(surface_probabilities),
        object_type_probabilities=dict(object_type_probabilities),
        target_count_probabilities=dict(target_probabilities),
        object_count_probabilities=dict(object_count_probabilities),
    )




@register_task
class IllustrationsIndoorRoomSurfaceObjectCountTask:
    """Count objects of one type on a named indoor surface."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one surface-object count instance and bind answer/annotation locally."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        fallback = render_fallback_from_defaults(_DEFAULTS)
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene = render_indoor_scene_from_specs(
                    render_namespace="surface_object_count",
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    specs=sample.specs,
                    theme_id=str(sample.theme_id),
                    params=params,
                    render_defaults=_RENDER_DEFAULTS,
                    fallback=fallback,
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                scene = None
        if scene is None or sample is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_objects, object_bboxes, part_bboxes = serialize_indoor_scene(scene)
        counted_ids = tuple(
            str(placement.object_id)
            for placement in scene.placements
            if str(placement.surface_type) == str(sample.surface_type)
            and str(placement.object_type) == str(sample.object_type)
        )
        if len(counted_ids) != int(sample.target_count):
            raise RuntimeError("rendered type-on-surface count did not match sample target")
        counted_bboxes = sort_bboxes_by_ids(object_bboxes, counted_ids)
        counted_points = sort_bbox_centers_by_ids(object_bboxes, counted_ids)
        annotation_artifacts = bbox_set_annotation_artifacts(counted_bboxes)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_object_type_on_surface",
                "annotation_hint_object_type_on_surface",
                "json_example_object_type_on_surface",
                "json_example_answer_only_object_type_on_surface",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        slots = {
            "object_count": int(sample.object_count),
            "room_setting": indoor_setting_name(str(scene.theme_id)),
            "object_name": str(sample.object_name),
            "surface_name": str(sample.surface_type),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults["answer_hint_object_type_on_surface"]).format(
                object_name=str(sample.object_name),
                surface_name=str(sample.surface_type),
            ),
            "annotation_hint": str(prompt_defaults["annotation_hint_object_type_on_surface"]).format(
                object_name=str(sample.object_name),
                surface_name=str(sample.surface_type),
            ),
            "json_example": str(prompt_defaults["json_example_object_type_on_surface"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only_object_type_on_surface"]),
        }
        prompt_artifacts = build_indoor_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=slots,
            instance_seed=int(instance_seed),
        )
        render_map = indoor_base_render_map(scene, object_bboxes=object_bboxes, part_bboxes=part_bboxes)
        render_map["counted_object_ids"] = list(counted_ids)
        render_map["counted_object_bboxes_px"] = list(counted_bboxes)
        render_map["counted_object_points_px"] = list(counted_points)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "entities": indoor_scene_entities(scene),
                "relations": {
                    "query_id": QUERY_ID,
                    "surface_type": str(sample.surface_type),
                    "object_type": str(sample.object_type),
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "query_id": QUERY_ID,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "theme": str(sample.theme_id),
                    "theme_id": str(sample.theme_id),
                    "surface_type": str(sample.surface_type),
                    "surface_name": str(sample.surface_type),
                    "object_type": str(sample.object_type),
                    "object_name": str(sample.object_name),
                    "target_count": int(sample.target_count),
                    "object_count": int(sample.object_count),
                    "theme_probabilities": dict(sample.theme_probabilities),
                    "surface_type_probabilities": dict(sample.surface_type_probabilities),
                    "object_type_probabilities": dict(sample.object_type_probabilities),
                    "target_count_probabilities": dict(sample.target_count_probabilities),
                    "object_count_probabilities": dict(sample.object_count_probabilities),
                },
            },
            "render_spec": indoor_render_spec(scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": QUERY_ID,
                "scene_id": SCENE_ID,
                "theme_id": str(scene.theme_id),
                "theme": str(scene.theme_id),
                "surface_type": str(sample.surface_type),
                "object_type": str(sample.object_type),
                "object_name": str(sample.object_name),
                "target_count": int(sample.target_count),
                "object_count": int(sample.object_count),
                "counted_object_ids": list(counted_ids),
                "object_types": object_type_map(serialized_objects),
            },
            "witness_symbolic": {
                "counted_object_ids": list(counted_ids),
                "surface_type": str(sample.surface_type),
                "object_type": str(sample.object_type),
                "answer": int(sample.target_count),
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
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
            query_id=QUERY_ID,
        )


__all__ = ["IllustrationsIndoorRoomSurfaceObjectCountTask"]
