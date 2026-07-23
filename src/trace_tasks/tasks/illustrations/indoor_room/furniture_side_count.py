"""Count indoor objects on one side of named furniture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
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
    INDOOR_FURNITURE_TYPES,
    INDOOR_OBJECT_TYPES,
    IndoorObjectSpec,
)


TASK_ID = "task_illustrations__indoor_room__furniture_side_count"
SCENE_ID = "indoor_room"
RELATION_SUPPORT: Tuple[str, ...] = ("left", "right")
QUERY_ID_TO_RELATION: Dict[str, str] = {
    "left_side": "left",
    "right_side": "right",
}
RELATION_TO_QUERY_ID: Dict[str, str] = {relation: query_id for query_id, relation in QUERY_ID_TO_RELATION.items()}
QUERY_IDS = tuple(QUERY_ID_TO_RELATION.keys())
PROMPT_QUERY_KEY = "furniture_side_count"
OPPOSITE_RELATION: Dict[str, str] = {"left": "right", "right": "left"}
VALID_FURNITURE_RELATION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("table", "left"),
    ("table", "right"),
)


@dataclass(frozen=True)
class _Defaults:
    object_count_min: int = 8
    object_count_max: int = 14
    target_count_min: int = 1
    target_count_max: int = 6
    canvas_width: int = 1280
    canvas_height: int = 840
    object_size_min_px: int = 52
    object_size_max_px: int = 86
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    theme_id: str
    furniture_type: str
    query_id: str
    relation: str
    object_type: str
    object_name: str
    target_count: int
    object_count: int
    specs: Tuple[IndoorObjectSpec, ...]
    theme_probabilities: Dict[str, float]
    furniture_type_probabilities: Dict[str, float]
    relation_probabilities: Dict[str, float]
    object_type_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _decoupled_sampling_params(
    params: Mapping[str, Any],
    *,
    multiplier: int,
    offset: int,
    block_divisor: int = 1,
) -> Mapping[str, Any]:
    """No-op hook for local cycling call sites."""

    _ = int(multiplier), int(offset), int(block_divisor)
    return params






def _choose_furniture_relation(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    furniture_support: Sequence[str],
    relation_support: Sequence[str],
) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
    """Resolve the public side query to one feasible furniture/relation pair."""

    explicit_furniture = params.get("furniture_type")
    explicit_query_relation = params.get("query_id")
    explicit_relation = params.get("relation")
    furniture_values = tuple(str(value) for value in furniture_support)
    relation_values = tuple(str(value) for value in relation_support)
    if explicit_query_relation is not None:
        explicit_query_id = str(explicit_query_relation)
        if explicit_query_id not in set(QUERY_IDS):
            raise ValueError(f"query_id must be one of {QUERY_IDS}")
        explicit_query_relation = QUERY_ID_TO_RELATION[str(explicit_query_id)]
        if explicit_relation is not None and str(explicit_relation) != str(explicit_query_relation):
            raise ValueError("query_id and relation must refer to the same furniture-side relation")
        explicit_relation = str(explicit_query_relation)
    pairs = tuple(
        (furniture, relation)
        for furniture, relation in VALID_FURNITURE_RELATION_PAIRS
        if furniture in set(furniture_values) and relation in set(relation_values)
    )
    if explicit_furniture is not None:
        explicit_furniture = str(explicit_furniture)
        if explicit_furniture not in set(furniture_values):
            raise ValueError(f"furniture_type must be one of {furniture_values}")
        pairs = tuple(pair for pair in pairs if pair[0] == explicit_furniture)
    if explicit_relation is not None:
        explicit_relation = str(explicit_relation)
        if explicit_relation not in set(relation_values):
            raise ValueError(f"relation must be one of {relation_values}")
        pairs = tuple(pair for pair in pairs if pair[1] == explicit_relation)
    if not pairs:
        raise ValueError("no feasible furniture/relation pair for indoor relation task")
    if explicit_furniture is not None and explicit_relation is not None:
        furniture_type, relation = str(explicit_furniture), str(explicit_relation)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}:furniture_relation")
        pair, _probabilities = uniform_choice_with_probabilities(rng, pairs, sort_keys=True)
        furniture_type, relation = str(pair[0]), str(pair[1])

    furniture_probabilities: Dict[str, float] = {}
    relation_probabilities: Dict[str, float] = {}
    if explicit_furniture is not None:
        furniture_probabilities[str(furniture_type)] = 1.0
    else:
        for furniture, _relation in pairs:
            furniture_probabilities[str(furniture)] = furniture_probabilities.get(str(furniture), 0.0) + 1.0 / float(len(pairs))
    if explicit_relation is not None:
        relation_probabilities[str(relation)] = 1.0
    else:
        for _furniture, pair_relation in pairs:
            relation_probabilities[str(pair_relation)] = relation_probabilities.get(str(pair_relation), 0.0) + 1.0 / float(len(pairs))
    return str(furniture_type), str(relation), dict(sorted(furniture_probabilities.items())), dict(sorted(relation_probabilities.items()))


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample one room state while keeping the queried side unique by construction."""

    furniture_support = typed_support(
        params,
        _GEN_DEFAULTS,
        param_key="furniture_type_support",
        default_key="furniture_type_support",
        fallback=INDOOR_FURNITURE_TYPES,
        error_name="furniture_type_support",
    )
    relation_support = typed_support(
        params,
        _GEN_DEFAULTS,
        param_key="relation_support",
        default_key="relation_support",
        fallback=RELATION_SUPPORT,
        error_name="relation_support",
    )
    object_support = typed_support(
        params,
        _GEN_DEFAULTS,
        param_key="object_type_support",
        default_key="indoor_object_type_support",
        fallback=INDOOR_OBJECT_TYPES,
        error_name="object_type_support",
    )
    theme_id, theme_probabilities = support_choice(
        params=_decoupled_sampling_params(params, multiplier=5, offset=2, block_divisor=4),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:theme",
        support=theme_support(params, _GEN_DEFAULTS),
        explicit_key="theme_id",
    )
    furniture_type, relation, furniture_probabilities, relation_probabilities = _choose_furniture_relation(
        params=_decoupled_sampling_params(params, multiplier=3, offset=1, block_divisor=6),
        instance_seed=int(instance_seed),
        furniture_support=furniture_support,
        relation_support=relation_support,
    )
    object_type, object_type_probabilities = support_choice(
        params=_decoupled_sampling_params(params, multiplier=7, offset=3, block_divisor=20),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:object_type",
        support=object_support,
        explicit_key="object_type",
    )
    target_min, target_max = _shared_bounds(params, _GEN_DEFAULTS, "target_count_min", "target_count_max", _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    target_count, target_probabilities = _shared_sample_count(
        params=_decoupled_sampling_params(params, multiplier=1, offset=0),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    object_min, object_max = _shared_bounds(params, _GEN_DEFAULTS, "object_count_min", "object_count_max", _DEFAULTS.object_count_min, _DEFAULTS.object_count_max)
    object_low = max(int(object_min), int(target_count) + 4)
    object_count, object_probabilities = _shared_sample_count(
        params=_decoupled_sampling_params(params, multiplier=5, offset=5, block_divisor=6),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:object_count",
        low=int(object_low),
        high=int(object_max),
        explicit_key="object_count",
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:spec", int(attempt_index))
    target_region = f"{furniture_type}:{relation}"
    distractor_region = f"{furniture_type}:{OPPOSITE_RELATION[str(relation)]}"
    specs = [
        IndoorObjectSpec(str(object_type), "region", str(target_region), "target")
        for _ in range(int(target_count))
    ]
    non_target_types = tuple(value for value in object_support if str(value) != str(object_type)) or tuple(
        value for value in INDOOR_OBJECT_TYPES if str(value) != str(object_type)
    )
    distractor_kinds = (
        ("same_type_opposite_region", max(1, min(2, int(object_count) - int(target_count)))),
        ("other_type_target_region", max(1, min(3, int(object_count) - int(target_count)))),
    )
    for kind, count in distractor_kinds:
        for _ in range(int(count)):
            if len(specs) >= int(object_count):
                break
            if str(kind) == "same_type_opposite_region":
                specs.append(IndoorObjectSpec(str(object_type), "region", str(distractor_region), "distractor"))
            else:
                specs.append(IndoorObjectSpec(str(rng.choice(non_target_types)), "region", str(target_region), "distractor"))
    while len(specs) < int(object_count):
        specs.append(IndoorObjectSpec(str(rng.choice(non_target_types)), "region", str(distractor_region), "distractor"))
    rng.shuffle(specs)
    return _SampleSpec(
        theme_id=str(theme_id),
        furniture_type=str(furniture_type),
        query_id=str(RELATION_TO_QUERY_ID[str(relation)]),
        relation=str(relation),
        object_type=str(object_type),
        object_name=display_name(str(object_type)),
        target_count=int(target_count),
        object_count=int(object_count),
        specs=tuple(specs),
        theme_probabilities=dict(theme_probabilities),
        furniture_type_probabilities=dict(furniture_probabilities),
        relation_probabilities=dict(relation_probabilities),
        object_type_probabilities=dict(object_type_probabilities),
        target_count_probabilities=dict(target_probabilities),
        object_count_probabilities=dict(object_probabilities),
    )




@register_task
class IllustrationsIndoorRoomFurnitureSideCountTask:
    """Count objects left or right of named furniture."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one furniture-side count instance and bind answer/annotation locally."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        fallback = render_fallback_from_defaults(_DEFAULTS)
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene = render_indoor_scene_from_specs(
                    render_namespace="furniture_side_count",
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
        furniture_id = f"furniture_{sample.furniture_type}"
        counted_ids = tuple(
            str(placement.object_id)
            for placement in scene.placements
            if bool(placement.relations[str(furniture_id)][str(sample.relation)])
            and str(placement.object_type) == str(sample.object_type)
        )
        if len(counted_ids) != int(sample.target_count):
            raise RuntimeError("rendered furniture-side count did not match sample target")
        query_id = str(sample.query_id)
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
                "answer_hint_furniture_side",
                "annotation_hint_furniture_side",
                "json_example_furniture_side",
                "json_example_answer_only_furniture_side",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        relation_word = {
            "left": "to the left of",
            "right": "to the right of",
        }.get(str(sample.relation), str(sample.relation))
        slots = {
            "object_count": int(sample.object_count),
            "room_setting": indoor_setting_name(str(scene.theme_id)),
            "furniture_name": str(sample.furniture_type),
            "object_name": str(sample.object_name),
            "relation_word": str(relation_word),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults["answer_hint_furniture_side"]).format(
                object_name=str(sample.object_name),
                relation_word=str(relation_word),
                furniture_name=str(sample.furniture_type),
            ),
            "annotation_hint": str(prompt_defaults["annotation_hint_furniture_side"]).format(
                object_name=str(sample.object_name),
                relation_word=str(relation_word),
                furniture_name=str(sample.furniture_type),
            ),
            "json_example": str(prompt_defaults["json_example_furniture_side"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only_furniture_side"]),
        }
        prompt_artifacts = build_indoor_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=slots,
            instance_seed=int(instance_seed),
        )
        query_params = {
            "theme": str(sample.theme_id),
            "theme_id": str(sample.theme_id),
            "furniture_type": str(sample.furniture_type),
            "furniture_name": str(sample.furniture_type),
            "furniture_id": str(furniture_id),
            "query_relation": str(sample.relation),
            "relation": str(sample.relation),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "object_type": str(sample.object_type),
            "object_name": str(sample.object_name),
            "target_count": int(sample.target_count),
            "object_count": int(sample.object_count),
            "theme_probabilities": dict(sample.theme_probabilities),
            "furniture_type_probabilities": dict(sample.furniture_type_probabilities),
            "relation_probabilities": dict(sample.relation_probabilities),
            "object_type_probabilities": dict(sample.object_type_probabilities),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "object_count_probabilities": dict(sample.object_count_probabilities),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=query_id,
            params=query_params,
        )
        query_spec["task_id"] = self.task_id
        query_spec["scene_id"] = SCENE_ID
        render_map = indoor_base_render_map(scene, object_bboxes=object_bboxes, part_bboxes=part_bboxes)
        render_map["target_furniture_id"] = str(furniture_id)
        render_map["counted_object_ids"] = list(counted_ids)
        render_map["counted_object_bboxes_px"] = list(counted_bboxes)
        render_map["counted_object_points_px"] = list(counted_points)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "entities": indoor_scene_entities(scene),
                "relations": {
                    "query_id": query_id,
                    "furniture_type": str(sample.furniture_type),
                    "furniture_id": str(furniture_id),
                    "relation": str(sample.relation),
                    "object_type": str(sample.object_type),
                },
            },
            "query_spec": query_spec,
            "render_spec": indoor_render_spec(scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": query_id,
                "scene_id": SCENE_ID,
                "theme_id": str(scene.theme_id),
                "theme": str(scene.theme_id),
                "furniture_type": str(sample.furniture_type),
                "furniture_id": str(furniture_id),
                "relation": str(sample.relation),
                "object_type": str(sample.object_type),
                "object_name": str(sample.object_name),
                "target_count": int(sample.target_count),
                "object_count": int(sample.object_count),
                "counted_object_ids": list(counted_ids),
                "object_types": object_type_map(serialized_objects),
            },
            "witness_symbolic": {
                "counted_object_ids": list(counted_ids),
                "furniture_id": str(furniture_id),
                "relation": str(sample.relation),
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
            query_id=query_id,
        )


__all__ = ["IllustrationsIndoorRoomFurnitureSideCountTask"]
