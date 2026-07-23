"""Count quarry objects on the highest or lowest terrain level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile

from .shared.output import (
    bbox_set_projection,
    isometric_quarry_object_count_render_map,
    isometric_quarry_render_spec,
    isometric_quarry_scene_ir,
    rounded_bbox,
)
from .shared.prompts import build_isometric_quarry_prompt_artifacts
from .shared.rendering import OBJECT_COUNT_QUARRY_OBJECT_TYPES, SCENE_ID, render_isometric_quarry_scene
from .shared.state import IsoQuarryEntity, IsoQuarryScene


TASK_ID = "task_illustrations__isometric_quarry__terrain_level_object_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("highest_terrain_object_count", "lowest_terrain_object_count")
TARGET_OBJECT_TYPES: Tuple[str, ...] = ("ore_vein", "mine_cart")
OBJECT_LABELS: Mapping[str, str] = {
    "ore_vein": "ore veins",
    "mine_cart": "mine carts",
}


@dataclass(frozen=True)
class _SampleSpec:
    selected_query: str
    prompt_query_key: str
    query_probabilities: dict[str, float]
    target_object_type: str
    object_label: str
    target_object_type_probabilities: dict[str, float]
    target_count: int
    target_count_probabilities: dict[str, float]
    answer_count_support: tuple[int, ...]
    answer_count_probabilities: dict[str, float]
    canvas_width: int
    canvas_height: int
    canvas_profile: str
    canvas_profile_probabilities: dict[str, float]


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _support_strings(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    support_key: str,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    raw = params.get(str(support_key), group_default(defaults, str(support_key), tuple(fallback)))
    values = (raw,) if isinstance(raw, str) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(str(value) for value in values))
    if not support:
        raise ValueError(f"{support_key} must include at least one value")
    return support


def _support_ints(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    support_key: str,
    fallback: Sequence[int],
) -> tuple[int, ...]:
    raw = params.get(str(support_key), group_default(defaults, str(support_key), tuple(fallback)))
    values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in values))
    if not support:
        raise ValueError(f"{support_key} must include at least one value")
    return support


def _uniform_string_probability_map(values: Sequence[str]) -> dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _select_target_object_type(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    support = _support_strings(params, defaults, support_key="target_object_type_support", fallback=TARGET_OBJECT_TYPES)
    explicit = params.get("target_object_type")
    if explicit is not None:
        value = str(explicit)
        if value not in set(support):
            raise ValueError(f"target_object_type must be one of {support}")
        return value, {value: 1.0}
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:target_object_type")
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(value), dict(probabilities)


def _select_target_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    support = _support_ints(params, defaults, support_key="answer_count_support", fallback=(0, 1, 2, 3, 4, 5))
    explicit = params.get("target_count", params.get("answer_count"))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"target_count must be one of {support}")
        return value, support_probability_map(support, selected=value, sort_keys=True), support
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:target_count")
    value, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(value), dict(probabilities), support


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Resolve task operands while keeping quarry subtype and count as parameters."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="highest_terrain_object_count",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    target_object_type, target_object_type_probabilities = _select_target_object_type(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
    )
    if target_object_type not in OBJECT_LABELS:
        raise ValueError(f"target_object_type has no prompt label: {target_object_type}")
    target_count, target_count_probabilities, answer_count_support = _select_target_count(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
    )
    profile = resolve_canvas_profile(
        params=task_params,
        defaults=_RENDER_DEFAULTS,
        fallback_width=1200,
        fallback_height=800,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:canvas_profile",
    )
    return _SampleSpec(
        selected_query=str(selected_query),
        prompt_query_key=str(selected_query),
        query_probabilities=dict(query_probabilities),
        target_object_type=str(target_object_type),
        object_label=str(OBJECT_LABELS[str(target_object_type)]),
        target_object_type_probabilities=dict(target_object_type_probabilities),
        target_count=int(target_count),
        target_count_probabilities=dict(target_count_probabilities),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        answer_count_probabilities=dict(support_probability_map(answer_count_support, sort_keys=True)),
        canvas_width=int(profile.width),
        canvas_height=int(profile.height),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
    )


def _target_level(scene: IsoQuarryScene, *, selected_query: str) -> tuple[int, str]:
    active_levels = tuple(int(level) for level in scene.trace.get("levels", ()))
    levels = active_levels or tuple(sorted({int(tile.level) for tile in scene.tiles}))
    if not levels:
        raise ValueError("quarry scene has no terrain levels")
    if str(selected_query) == "highest_terrain_object_count":
        return max(levels), "highest"
    if str(selected_query) == "lowest_terrain_object_count":
        return min(levels), "lowest"
    raise ValueError(f"unsupported terrain-level object count query: {selected_query}")


def _matching_entities(
    scene: IsoQuarryScene,
    *,
    target_object_type: str,
    target_level: int,
) -> tuple[IsoQuarryEntity, ...]:
    return tuple(
        sorted(
            (
                entity
                for entity in scene.entities
                if str(entity.object_type) == "quarry_object"
                and str(entity.metadata.get("quarry_object_type", "")) == str(target_object_type)
                and int(entity.level) == int(target_level)
            ),
            key=lambda entity: (float(entity.bbox_xyxy[1]), float(entity.bbox_xyxy[0]), str(entity.entity_id)),
        )
    )


def _prompt_slots(prompt_defaults: Mapping[str, Any], sample: _SampleSpec) -> dict[str, str]:
    return {
        "object_label": str(sample.object_label),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_terrain_level_object_count"]).format(object_label=str(sample.object_label)),
        "annotation_hint": str(prompt_defaults["annotation_hint_terrain_level_object_count"]).format(object_label=str(sample.object_label)),
        "json_example": str(prompt_defaults["json_example_terrain_level_object_count"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_terrain_level_object_count"]),
    }


@register_task
class IllustrationsIsometricQuarryTerrainLevelObjectCountTask:
    """Count quarry ore veins or mine carts on the requested terrain level."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'logical_composition')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one count instance with answer and annotation from one trace."""

        sample = _sample_spec(instance_seed=int(instance_seed), params=params)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_terrain_level_object_count",
                "annotation_hint_terrain_level_object_count",
                "json_example_terrain_level_object_count",
                "json_example_answer_only_terrain_level_object_count",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )

        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                scene_seed = int(instance_seed) + int(attempt) * 1009
                base_scene = render_isometric_quarry_scene(
                    scene_seed,
                    width=sample.canvas_width,
                    height=sample.canvas_height,
                    canvas_profile=sample.canvas_profile,
                    canvas_profile_probabilities=sample.canvas_profile_probabilities,
                    context_object_types=OBJECT_COUNT_QUARRY_OBJECT_TYPES,
                    quarry_patch_mode="none",
                    render_style_params=params,
                    render_style_defaults=_RENDER_DEFAULTS,
                )
                target_level, extremum_mode = _target_level(base_scene, selected_query=sample.selected_query)
                scene = render_isometric_quarry_scene(
                    scene_seed,
                    width=sample.canvas_width,
                    height=sample.canvas_height,
                    canvas_profile=sample.canvas_profile,
                    canvas_profile_probabilities=sample.canvas_profile_probabilities,
                    required_entity_counts_by_level_type={
                        str(sample.target_object_type): {int(target_level): int(sample.target_count)}
                    },
                    context_object_types=OBJECT_COUNT_QUARRY_OBJECT_TYPES,
                    quarry_patch_mode="none",
                    render_style_params=params,
                    render_style_defaults=_RENDER_DEFAULTS,
                )
                target_level, extremum_mode = _target_level(scene, selected_query=sample.selected_query)
                matching_entities = _matching_entities(
                    scene,
                    target_object_type=sample.target_object_type,
                    target_level=int(target_level),
                )
                if len(matching_entities) != int(sample.target_count):
                    raise ValueError(
                        f"count {len(matching_entities)} did not match target_count {sample.target_count}"
                    )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        annotation_value = [rounded_bbox(entity.bbox_xyxy) for entity in matching_entities]
        counted_entity_ids = tuple(str(entity.entity_id) for entity in matching_entities)
        prompt_artifacts = build_isometric_quarry_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=sample.prompt_query_key,
            slots=_prompt_slots(prompt_defaults, sample),
            instance_seed=int(instance_seed),
        )
        render_map = isometric_quarry_object_count_render_map(
            scene=scene,
            target_object_type=sample.target_object_type,
            target_level=int(target_level),
            counted_entity_ids=counted_entity_ids,
        )
        query_params = {
            "query_id": str(sample.selected_query),
            "prompt_query_key": str(sample.prompt_query_key),
            "query_id_probabilities": dict(sample.query_probabilities),
            "target_object_type": str(sample.target_object_type),
            "target_object_type_probabilities": dict(sample.target_object_type_probabilities),
            "object_label": str(sample.object_label),
            "target_level": int(target_level),
            "extremum_mode": str(extremum_mode),
            "target_count": int(sample.target_count),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "answer_count_support": list(sample.answer_count_support),
            "answer_count_probabilities": dict(sample.answer_count_probabilities),
            "answer_count": int(len(matching_entities)),
            "counted_entity_ids": list(counted_entity_ids),
            "canvas_profile": str(sample.canvas_profile),
            "canvas_profile_probabilities": dict(sample.canvas_profile_probabilities),
        }
        trace_payload = {
            "scene_ir": isometric_quarry_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "operation": "count_quarry_objects_on_elevation_extremum",
                    "extremum": str(extremum_mode),
                    "target_object_type": str(sample.target_object_type),
                    "target_object_label": str(sample.object_label),
                    "target_level": int(target_level),
                    "answer_count": int(len(matching_entities)),
                    "counted_entity_ids": list(counted_entity_ids),
                    "counted_entity_bboxes_px": [list(bbox) for bbox in annotation_value],
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": str(sample.selected_query),
                "prompt_query_key": str(sample.prompt_query_key),
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": query_params,
            },
            "render_spec": isometric_quarry_render_spec(scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(sample.selected_query),
                "prompt_query_key": str(sample.prompt_query_key),
                "scene_id": SCENE_ID,
                "answer": int(len(matching_entities)),
                "target_object_type": str(sample.target_object_type),
                "target_object_label": str(sample.object_label),
                "target_level": int(target_level),
                "extremum_mode": str(extremum_mode),
                "counted_entity_ids": list(counted_entity_ids),
                "renderer": dict(scene.trace),
            },
            "witness_symbolic": {
                "answer_count": int(len(matching_entities)),
                "target_object_type": str(sample.target_object_type),
                "target_object_label": str(sample.object_label),
                "target_level": int(target_level),
                "extremum_mode": str(extremum_mode),
                "counted_entity_ids": list(counted_entity_ids),
                "counted_entity_bboxes": [list(bbox) for bbox in annotation_value],
            },
            "projected_annotation": bbox_set_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(len(matching_entities))),
            annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_value]),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.selected_query),
        )


__all__ = [
    "IllustrationsIsometricQuarryTerrainLevelObjectCountTask",
    "OBJECT_LABELS",
    "SUPPORTED_QUERY_IDS",
    "TARGET_OBJECT_TYPES",
    "TASK_ID",
]
