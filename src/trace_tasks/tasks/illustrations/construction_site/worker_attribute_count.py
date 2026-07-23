"""Count construction workers matching visible safety-gear conditions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from .shared.annotations import construction_worker_bbox_map, sort_construction_bbox_centers, sort_construction_bboxes
from .shared.labels import construction_color_display_name, construction_color_hex
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
    query_support as _shared_query_support,
    render_params,
    sample_count,
    setting_weights,
    spawned_task_rng,
    style_weights,
    tool_support,
    uniform_string_probability_map,
)


TASK_ID = "task_illustrations__construction_site__worker_attribute_count"
SCENE_ID = "construction_site"
QUERY_IDS: Tuple[str, ...] = (
    "hard_hat_color_worker_count",
    "vest_color_worker_count",
)


@dataclass(frozen=True)
class _Defaults:
    worker_count_min: int = 9
    worker_count_max: int = 15
    target_count_min: int = 0
    target_count_max: int = 5
    material_count_min: int = 5
    material_count_max: int = 8
    equipment_count_min: int = 3
    equipment_count_max: int = 5
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    target_color: str | None
    target_count: int
    worker_count: int
    worker_specs: Tuple[ConstructionWorkerSpec, ...]
    material_specs: Tuple[ConstructionMaterialSpec, ...]
    equipment_specs: Tuple[ConstructionEquipmentSpec, ...]
    match_phrase: str
    query_probabilities: Dict[str, float]
    color_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    worker_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "illustrations",
    SCENE_ID,
    task_id=TASK_ID,
)




def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Build worker specs with exactly one visible attribute count answer."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    query_values = _shared_query_support(params, _GEN_DEFAULTS, QUERY_IDS)
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

    if "color" in query_id:
        explicit_color = params.get("target_color")
        if explicit_color is not None:
            target_color = str(explicit_color)
            if target_color not in set(colors):
                raise ValueError("target_color is outside configured support")
            color_probabilities = uniform_string_probability_map(colors, selected=target_color)
        else:
            color_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:target_color")
            target_color, color_probabilities = uniform_choice_with_probabilities(
                color_rng,
                colors,
                sort_keys=False,
            )
            target_color = str(target_color)
    else:
        target_color = None
        color_probabilities = uniform_string_probability_map(colors)

    target_min, target_max = bounds(params, _GEN_DEFAULTS, "target_count_min", "target_count_max", _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    target_count, target_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    worker_min, worker_max = bounds(params, _GEN_DEFAULTS, "worker_count_min", "worker_count_max", _DEFAULTS.worker_count_min, _DEFAULTS.worker_count_max)
    worker_low = max(int(worker_min), int(target_count) + 2)
    worker_count, worker_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:worker_count",
        low=int(worker_low),
        high=int(worker_max),
        explicit_key="worker_count",
    )

    non_target_colors = [str(color) for color in colors if str(color) != str(target_color)]
    if not non_target_colors:
        raise ValueError("worker safety task needs at least one non-target color")
    worker_specs = []
    if query_id == "hard_hat_color_worker_count":
        match_phrase = f"workers wearing {construction_color_display_name(target_color)} hard hats"
        for _ in range(int(target_count)):
            worker_specs.append(
                ConstructionWorkerSpec(
                    hard_hat_color=target_color,
                    vest_color=str(rng.choice(colors)),
                    tool_type=str(rng.choice(tools)) if rng.random() < 0.45 else None,
                    role="target",
                )
            )
        for _ in range(int(worker_count) - int(target_count)):
            worker_specs.append(
                ConstructionWorkerSpec(
                    hard_hat_color=str(rng.choice(tuple(non_target_colors))),
                    vest_color=str(rng.choice(colors)),
                    tool_type=str(rng.choice(tools)) if rng.random() < 0.40 else None,
                    role="distractor",
                )
            )
    elif query_id == "vest_color_worker_count":
        match_phrase = f"workers wearing {construction_color_display_name(target_color)} safety vests"
        for _ in range(int(target_count)):
            worker_specs.append(
                ConstructionWorkerSpec(
                    hard_hat_color=str(rng.choice(colors)),
                    vest_color=target_color,
                    tool_type=str(rng.choice(tools)) if rng.random() < 0.45 else None,
                    role="target",
                )
            )
        for _ in range(int(worker_count) - int(target_count)):
            worker_specs.append(
                ConstructionWorkerSpec(
                    hard_hat_color=str(rng.choice(colors)),
                    vest_color=str(rng.choice(tuple(non_target_colors))),
                    tool_type=str(rng.choice(tools)) if rng.random() < 0.40 else None,
                    role="distractor",
                )
            )
    else:
        raise ValueError(f"unsupported worker attribute query_id: {query_id}")
    rng.shuffle(worker_specs)

    material_min, material_max = bounds(params, _GEN_DEFAULTS, "material_count_min", "material_count_max", _DEFAULTS.material_count_min, _DEFAULTS.material_count_max)
    material_count = int(rng.randint(int(material_min), int(material_max)))
    materials = material_support(params, _GEN_DEFAULTS)
    material_specs = tuple(ConstructionMaterialSpec(material_type=str(rng.choice(materials)), role="decor") for _ in range(material_count))
    equipment_min, equipment_max = bounds(params, _GEN_DEFAULTS, "equipment_count_min", "equipment_count_max", _DEFAULTS.equipment_count_min, _DEFAULTS.equipment_count_max)
    equipment_count = int(rng.randint(int(equipment_min), int(equipment_max)))
    equipment_values = equipment_support(params, _GEN_DEFAULTS)
    equipment_specs = tuple(ConstructionEquipmentSpec(equipment_type=str(rng.choice(equipment_values)), role="decor") for _ in range(equipment_count))

    return _SampleSpec(
        query_id=str(query_id),
        target_color=target_color if "color" in query_id else None,
        target_count=int(target_count),
        worker_count=int(worker_count),
        worker_specs=tuple(worker_specs),
        material_specs=tuple(material_specs),
        equipment_specs=tuple(equipment_specs),
        match_phrase=str(match_phrase),
        query_probabilities=dict(query_probabilities),
        color_probabilities=dict(color_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        worker_count_probabilities=dict(worker_count_probabilities),
    )


def _counted_worker_ids(scene: Any, sample: _SampleSpec) -> Tuple[str, ...]:
    """Select workers that satisfy the active worker-attribute predicate."""

    if sample.query_id == "hard_hat_color_worker_count":
        return tuple(str(worker.worker_id) for worker in scene.workers if str(worker.hard_hat_color) == str(sample.target_color))
    if sample.query_id == "vest_color_worker_count":
        return tuple(str(worker.worker_id) for worker in scene.workers if str(worker.vest_color) == str(sample.target_color))
    raise ValueError(f"unsupported worker attribute query_id: {sample.query_id}")


def _worker_dynamic_slots(prompt_defaults: Mapping[str, Any], sample: _SampleSpec) -> Dict[str, str]:
    """Format the prompt slots for the selected worker predicate."""

    return {
        "object_description": str(prompt_defaults["object_description"]).format(worker_count=int(sample.worker_count)),
        "question_text": str(prompt_defaults[f"question_text_{sample.query_id}"]).format(
            color_label=construction_color_display_name(str(sample.target_color or "")),
            match_phrase=str(sample.match_phrase),
        ),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_worker_safety_gear"]).format(match_phrase=str(sample.match_phrase)),
        "annotation_hint": str(prompt_defaults["annotation_hint_worker_safety_gear"]).format(match_phrase=str(sample.match_phrase)),
        "json_example": str(prompt_defaults["json_example_worker_safety_gear"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_worker_safety_gear"]),
    }


def _worker_query_params(sample: _SampleSpec, *, target_color_label: str | None, target_color_hex: str | None) -> Dict[str, Any]:
    """Return verifier-facing query parameters for the active worker predicate."""

    return {
        "query_id": str(sample.query_id),
        "target_color": str(sample.target_color) if sample.target_color else None,
        "target_color_label": target_color_label,
        "target_color_hex": target_color_hex,
        "match_phrase": str(sample.match_phrase),
        "target_count": int(sample.target_count),
        "worker_count": int(sample.worker_count),
        "query_id_probabilities": dict(sample.query_probabilities),
        "query_probabilities": dict(sample.query_probabilities),
        "color_probabilities": dict(sample.color_probabilities),
        "target_count_probabilities": dict(sample.target_count_probabilities),
        "worker_count_probabilities": dict(sample.worker_count_probabilities),
    }




@register_task
class IllustrationsCountingWorkerSafetyGearCountTask:
    """Count workers matching a visible hard-hat or vest color condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "illustrations"
    supported_queries = QUERY_IDS
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one scene and bind counted workers to prompt and annotation."""

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

        counted_worker_ids = _counted_worker_ids(scene, sample)
        if len(counted_worker_ids) != int(sample.target_count):
            raise RuntimeError("rendered worker count did not match sample target")
        worker_bbox_map = construction_worker_bbox_map(scene)
        counted_worker_bboxes = sort_construction_bboxes(worker_bbox_map, counted_worker_ids)
        counted_worker_points = sort_construction_bbox_centers(worker_bbox_map, counted_worker_ids)
        annotation_artifacts = bbox_set_annotation_artifacts(counted_worker_bboxes)
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
                "question_text_hard_hat_color_worker_count",
                "question_text_vest_color_worker_count",
                "answer_hint_worker_safety_gear",
                "annotation_hint_worker_safety_gear",
                "json_example_worker_safety_gear",
                "json_example_answer_only_worker_safety_gear",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        dynamic_slots = _worker_dynamic_slots(prompt_defaults, sample)
        prompt_artifacts = render_construction_prompt(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            dynamic_slots=dynamic_slots,
            instance_seed=int(instance_seed),
        )
        target_color_label = construction_color_display_name(str(sample.target_color)) if sample.target_color else None
        target_color_hex = construction_color_hex(str(sample.target_color)) if sample.target_color else None
        trace_payload = {
            **construction_count_trace_sections(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={"query_id": str(sample.query_id), "match_phrase": str(sample.match_phrase)},
                render_map={
                    "bboxes_px": bbox_map,
                    "worker_bboxes_px": worker_bbox_map,
                    "counted_worker_ids": list(counted_worker_ids),
                    "counted_worker_bboxes_px": counted_worker_bboxes,
                    "counted_worker_points_px": counted_worker_points,
                },
                execution_trace={
                    "query_id": str(sample.query_id),
                    "scene_id": SCENE_ID,
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "target_count": int(sample.target_count),
                    "worker_count": int(sample.worker_count),
                    "target_color_label": target_color_label,
                    "target_color_hex": target_color_hex,
                    "worker_color_counts": {
                        "hard_hat": dict(Counter(str(worker.hard_hat_color) for worker in scene.workers)),
                        "vest": dict(Counter(str(worker.vest_color) for worker in scene.workers)),
                    },
                    "counted_worker_ids": list(counted_worker_ids),
                    "scene": serialized_scene[0],
                },
                witness_symbolic={"counted_worker_ids": list(counted_worker_ids), "answer": int(sample.target_count)},
                projected_annotation=annotation_artifacts.projected_annotation,
            ),
            "query_spec": {
                "task_id": self.task_id,
                "query_id": str(sample.query_id),
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": _worker_query_params(
                    sample,
                    target_color_label=target_color_label,
                    target_color_hex=target_color_hex,
                ),
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


__all__ = ["IllustrationsCountingWorkerSafetyGearCountTask"]
