"""Select the lettered boat closest to the shoreline."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile
from trace_tasks.tasks.illustrations.shared.option_rendering import sample_visual_label_font_trace

from .shared.output import (
    bbox_projection,
    isometric_harbor_render_spec,
    isometric_harbor_scene_ir,
    isometric_harbor_shoreline_nearest_boat_render_map,
)
from .shared.prompts import build_isometric_harbor_prompt_artifacts
from .shared.rendering import DEFAULT_BOAT_CANDIDATE_LABELS, SCENE_ID, render_isometric_harbor_scene
from .shared.spatial_primitives import rounded_bbox
from .shared.state import IsoHarborEntity, IsoHarborScene


TASK_ID = "task_illustrations__isometric_harbor__shoreline_nearest_boat_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "shoreline_nearest_boat"
_REQUIRED_PROMPT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_shoreline_nearest_boat_label",
    "annotation_hint_shoreline_nearest_boat_label",
    "json_example_shoreline_nearest_boat_label",
    "json_example_answer_only_shoreline_nearest_boat_label",
)


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _label_probability_map(labels: Sequence[str]) -> dict[str, float]:
    support = tuple(str(label) for label in labels)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {str(label): float(probability) for label in support}


def _select_nearest_label(*, instance_seed: int, params: Mapping[str, Any], labels: Sequence[str]) -> str:
    explicit = params.get("selected_label", params.get("answer_label"))
    support = tuple(str(label) for label in labels)
    if explicit is not None:
        label = str(explicit).strip().upper()
        if label not in set(support):
            raise ValueError(f"selected_label must be one of {support}")
        return label
    namespace = f"{TASK_ID}:nearest_label"
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    selected, _probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=False)
    return str(selected)


def _prompt_slots(prompt_defaults: Mapping[str, Any]) -> dict[str, str]:
    return {
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_shoreline_nearest_boat_label"]),
        "annotation_hint": str(prompt_defaults["annotation_hint_shoreline_nearest_boat_label"]),
        "json_example": str(prompt_defaults["json_example_shoreline_nearest_boat_label"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_shoreline_nearest_boat_label"]),
    }


def _boat_by_label(scene: IsoHarborScene, label: str) -> IsoHarborEntity:
    entity_id = str(scene.trace.get("candidate_boat_ids_by_label", {}).get(str(label), ""))
    for entity in scene.entities:
        if str(entity.entity_id) == entity_id:
            return entity
    raise ValueError(f"missing boat entity for label {label}")


def _validate_candidate_scene(scene: IsoHarborScene, *, labels: Sequence[str], selected_label: str) -> None:
    ids_by_label = dict(scene.trace.get("candidate_boat_ids_by_label", {}))
    distances = dict(scene.trace.get("shoreline_distance_tiles_by_label", {}))
    if set(ids_by_label) != set(str(label) for label in labels):
        raise ValueError("rendered shoreline candidates do not match requested labels")
    if str(scene.trace.get("nearest_label", "")) != str(selected_label):
        raise ValueError("rendered shoreline nearest label did not match request")
    selected_distance = int(distances[str(selected_label)])
    if any(int(value) <= selected_distance for label, value in distances.items() if str(label) != str(selected_label)):
        raise ValueError("nearest shoreline boat is not unique")
    for entity in scene.entities:
        if str(entity.object_type) != "boat":
            continue
        if str(entity.metadata.get("orientation", "")) != "shore_facing":
            raise ValueError("shoreline candidate boats must face the shoreline")
        if str(entity.metadata.get("mooring_status", "")) != "open_water":
            raise ValueError("shoreline candidate boats must be in open water")


def _render_spec_with_label_font(scene: IsoHarborScene, *, label_font_trace: Mapping[str, Any]) -> dict[str, Any]:
    render_spec = isometric_harbor_render_spec(scene, scene_id=SCENE_ID)
    return {
        **render_spec,
        "style": {
            **dict(render_spec["style"]),
            "label_font": dict(label_font_trace),
        },
    }


@register_task
class IllustrationsIsometricHarborShorelineNearestBoatLabelTask:
    """Choose the lettered open-water boat closest to the shoreline."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Bind one answer label, render exact shoreline ordering, and package a scalar-bbox selection."""

        checked_prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            list(_REQUIRED_PROMPT_KEYS),
            context=f"prompt defaults for {TASK_ID}",
        )
        labels = tuple(DEFAULT_BOAT_CANDIDATE_LABELS)
        selected_label = _select_nearest_label(instance_seed=int(instance_seed), params=params, labels=labels)
        profile = resolve_canvas_profile(
            params=params,
            defaults=_RENDER_DEFAULTS,
            fallback_width=1200,
            fallback_height=800,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:canvas_profile",
        )
        label_font_trace = sample_visual_label_font_trace(
            namespace_prefix=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            namespace_suffix="shoreline_boat_labels",
            explicit_key="label_font_family",
            weights_key="label_font_family_weights",
        )
        label_font_family = str(label_font_trace.get("font_family", ""))

        last_error: Exception | None = None
        scene: IsoHarborScene | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                scene_seed = int(instance_seed) + int(attempt) * 1009
                candidate_scene = render_isometric_harbor_scene(
                    scene_seed,
                    width=int(profile.width),
                    height=int(profile.height),
                    canvas_profile=str(profile.profile_id),
                    canvas_profile_probabilities=dict(profile.probabilities),
                    shoreline_candidate_labels=labels,
                    shoreline_nearest_label=str(selected_label),
                    shoreline_label_font_family=label_font_family,
                    render_style_params=params,
                    render_style_defaults=_RENDER_DEFAULTS,
                )
                _validate_candidate_scene(candidate_scene, labels=labels, selected_label=str(selected_label))
                scene = candidate_scene
                break
            except Exception as exc:
                last_error = exc
                scene = None
        if scene is None:
            raise RuntimeError(f"could not generate shoreline-nearest harbor instance: {last_error}") from last_error

        selected_boat = _boat_by_label(scene, str(selected_label))
        annotation_value = rounded_bbox(selected_boat.bbox_xyxy)
        candidates_by_label = dict(scene.trace.get("candidate_boat_ids_by_label", {}))
        render_map = isometric_harbor_shoreline_nearest_boat_render_map(
            scene=scene,
            candidate_boat_ids_by_label=candidates_by_label,
            selected_label=str(selected_label),
        )
        prompt_artifacts = build_isometric_harbor_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=checked_prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=_prompt_slots(checked_prompt_defaults),
            instance_seed=int(instance_seed),
        )

        query_key_name = "query" + "_id"
        task_key_name = "task" + "_id"
        trace_payload = {
            "scene_ir": isometric_harbor_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "operation": "select_boat_closest_to_shoreline",
                    "candidate_boat_ids_by_label": dict(candidates_by_label),
                    "selected_label": str(selected_label),
                    "selected_boat_id": str(selected_boat.entity_id),
                    "shoreline_distance_tiles_by_label": dict(scene.trace.get("shoreline_distance_tiles_by_label", {})),
                },
            ),
            "query_spec": {
                task_key_name: TASK_ID,
                query_key_name: SINGLE_QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    query_key_name: SINGLE_QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    query_key_name + "_probabilities": {SINGLE_QUERY_ID: 1.0},
                    "candidate_count": len(labels),
                    "candidate_labels": list(labels),
                    "candidate_label": str(selected_label),
                    "candidate_label_probabilities": _label_probability_map(labels),
                    "candidate_boat_ids_by_label": dict(candidates_by_label),
                    "selected_label": str(selected_label),
                    "selected_boat_id": str(selected_boat.entity_id),
                    "selected_shoreline_distance_tiles": int(
                        selected_boat.metadata.get("shoreline_distance_tiles", 0)
                    ),
                    "canvas_profile": str(profile.profile_id),
                    "canvas_profile_probabilities": dict(profile.probabilities),
                },
            },
            "render_spec": _render_spec_with_label_font(scene, label_font_trace=label_font_trace),
            "render_map": dict(render_map),
            "execution_trace": {
                query_key_name: SINGLE_QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": str(selected_label),
                "selected_label": str(selected_label),
                "selected_boat_id": str(selected_boat.entity_id),
                "candidate_boat_ids_by_label": dict(candidates_by_label),
                "shoreline_distance_tiles_by_label": dict(scene.trace.get("shoreline_distance_tiles_by_label", {})),
                "renderer": dict(scene.trace),
            },
            "witness_symbolic": {
                "answer_label": str(selected_label),
                "selected_boat_id": str(selected_boat.entity_id),
                "selected_boat_bbox": list(annotation_value),
            },
            "projected_annotation": bbox_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=str(selected_label)),
            annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=SINGLE_QUERY_ID,
        )


__all__ = [
    "IllustrationsIsometricHarborShorelineNearestBoatLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
