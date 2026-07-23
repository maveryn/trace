"""Private lifecycle plumbing for isometric harbor count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile

from .shared.output import (
    bbox_set_projection,
    isometric_harbor_heading_status_count_render_map,
    isometric_harbor_render_spec,
    isometric_harbor_scene_ir,
)
from .shared.prompts import build_isometric_harbor_prompt_artifacts
from .shared.rendering import BOAT_HEADING_STATUS_VALUES, SCENE_ID, render_isometric_harbor_scene
from .shared.sampling import CountTaskSampleSpec, select_count
from .shared.spatial_primitives import rounded_bbox
from .shared.state import IsoHarborEntity, IsoHarborScene


@dataclass(frozen=True)
class HarborCountPlan:
    """Public-owned hooks for one harbor count objective."""

    public_id: str
    operation: str
    required_prompt_keys: tuple[str, ...]
    sample_spec: Callable[[int, Mapping[str, Any]], CountTaskSampleSpec]
    prompt_slots: Callable[[Mapping[str, Any], CountTaskSampleSpec], Mapping[str, Any]]
    scene_builder: Callable[[int, CountTaskSampleSpec, Mapping[str, Any]], IsoHarborScene]
    entity_selector: Callable[[IsoHarborScene, CountTaskSampleSpec], Sequence[IsoHarborEntity]]
    render_map: Callable[[IsoHarborScene, CountTaskSampleSpec, tuple[str, ...]], Mapping[str, Any]]
    identity_fields: Callable[[CountTaskSampleSpec], Mapping[str, Any]]
    extra_query_params: Callable[[CountTaskSampleSpec], Mapping[str, Any]]
    scene_validator: Callable[[IsoHarborScene, CountTaskSampleSpec], None] | None = None


@dataclass(frozen=True)
class HarborHeadingCountConfig:
    """Task-owned parameters for the generic heading-status count lifecycle."""

    public_id: str
    supported_query_ids: tuple[str, ...]
    query_to_heading_status: Mapping[str, str]
    heading_status_labels: Mapping[str, str]
    required_prompt_keys: tuple[str, ...]
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]
    total_boats: int = 6


@dataclass(frozen=True)
class HarborHeadingCountSampleSpec(CountTaskSampleSpec):
    """Resolved heading-status count sample owned by one public task."""

    target_heading_status: str
    target_heading_label: str
    heading_status_counts: dict[str, int]


def sorted_harbor_boats(
    scene: IsoHarborScene,
    *,
    predicate: Callable[[IsoHarborEntity], bool],
) -> tuple[IsoHarborEntity, ...]:
    """Return countable boat witnesses in stable annotation order."""

    return tuple(
        sorted(
            (
                entity
                for entity in scene.entities
                if str(entity.object_type) == "boat" and bool(predicate(entity))
            ),
            key=lambda entity: (float(entity.bbox_xyxy[1]), float(entity.bbox_xyxy[0]), str(entity.entity_id)),
        )
    )


def _heading_status_counts(
    *,
    config: HarborHeadingCountConfig,
    instance_seed: int,
    params: Mapping[str, Any],
    target_status: str,
    target_count: int,
) -> dict[str, int]:
    counts = {str(status): 0 for status in BOAT_HEADING_STATUS_VALUES}
    counts[str(target_status)] = int(target_count)
    remaining = int(config.total_boats) - int(target_count)
    if int(remaining) < 0:
        raise ValueError("target_count cannot exceed total heading-status boats")
    other_statuses = [str(status) for status in BOAT_HEADING_STATUS_VALUES if str(status) != str(target_status)]
    if len(other_statuses) == 1:
        counts[other_statuses[0]] = int(remaining)
    elif len(other_statuses) > 1:
        rng = spawn_rng(
            int(instance_seed),
            f"{config.public_id}:other_heading_split:{target_status}",
        )
        first_count = int(rng.randint(0, int(remaining)))
        counts[other_statuses[0]] = int(first_count)
        counts[other_statuses[1]] = int(remaining) - int(first_count)
    return counts


def _heading_sample_spec(
    *,
    config: HarborHeadingCountConfig,
    instance_seed: int,
    params: Mapping[str, Any],
) -> HarborHeadingCountSampleSpec:
    """Resolve the heading query, target count, distractor split, and canvas profile."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=config.supported_query_ids,
        default_query_id=str(config.supported_query_ids[0]),
        task_id=config.public_id,
        namespace=f"{config.public_id}:query",
    )
    target_count, target_count_probabilities, answer_count_support = select_count(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=config.generation_defaults,
        support_key="answer_count_support",
        explicit_key="target_count",
        fallback=(1, 2, 3, 4, 5),
        namespace=f"{config.public_id}:target_count",
    )
    profile = resolve_canvas_profile(
        params=task_params,
        defaults=config.rendering_defaults,
        fallback_width=1200,
        fallback_height=800,
        instance_seed=int(instance_seed),
        namespace=f"{config.public_id}:canvas_profile",
    )
    target_status = str(config.query_to_heading_status[str(selected_query)])
    counts = _heading_status_counts(
        config=config,
        instance_seed=int(instance_seed),
        params=task_params,
        target_status=target_status,
        target_count=int(target_count),
    )
    return HarborHeadingCountSampleSpec(
        selected_key=str(selected_query),
        prompt_query_key=str(selected_query),
        query_probabilities=dict(query_probabilities),
        target_count=int(target_count),
        target_count_probabilities=dict(target_count_probabilities),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        answer_count_probabilities=dict(support_probability_map(answer_count_support, sort_keys=True)),
        canvas_width=int(profile.width),
        canvas_height=int(profile.height),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
        target_heading_status=target_status,
        target_heading_label=str(config.heading_status_labels[target_status]),
        heading_status_counts=counts,
    )


def _heading_prompt_slots(
    prompt_defaults: Mapping[str, Any],
    sample: CountTaskSampleSpec,
) -> dict[str, str]:
    heading_sample = sample if isinstance(sample, HarborHeadingCountSampleSpec) else None
    target_heading_label = "" if heading_sample is None else str(heading_sample.target_heading_label)
    return {
        "target_heading_label": target_heading_label,
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_boat_heading_status_count"]).format(
            target_heading_label=target_heading_label
        ),
        "annotation_hint": str(prompt_defaults["annotation_hint_boat_heading_status_count"]).format(
            target_heading_label=target_heading_label
        ),
        "json_example": str(prompt_defaults["json_example_boat_heading_status_count"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_boat_heading_status_count"]),
    }


def _heading_identity_fields(sample: CountTaskSampleSpec) -> dict[str, Any]:
    heading_sample = sample if isinstance(sample, HarborHeadingCountSampleSpec) else None
    return {
        "target_heading_status": "" if heading_sample is None else str(heading_sample.target_heading_status),
        "target_heading_label": "" if heading_sample is None else str(heading_sample.target_heading_label),
    }


def _heading_extra_query_params(sample: CountTaskSampleSpec) -> dict[str, Any]:
    if not isinstance(sample, HarborHeadingCountSampleSpec):
        return {"allowed_heading_statuses": list(BOAT_HEADING_STATUS_VALUES)}
    return {
        "allowed_heading_statuses": list(BOAT_HEADING_STATUS_VALUES),
        "heading_status_counts": dict(sample.heading_status_counts),
        "total_boats": int(sum(int(value) for value in sample.heading_status_counts.values())),
    }


def _render_heading_scene_from_sample(
    scene_seed: int,
    sample: CountTaskSampleSpec,
    params: Mapping[str, Any],
    *,
    rendering_defaults: Mapping[str, Any],
) -> IsoHarborScene:
    """Render open-water boats with exact shoreline-relative heading counts."""

    if not isinstance(sample, HarborHeadingCountSampleSpec):
        raise TypeError("heading-status sample expected")
    return render_isometric_harbor_scene(
        scene_seed,
        width=sample.canvas_width,
        height=sample.canvas_height,
        canvas_profile=sample.canvas_profile,
        canvas_profile_probabilities=sample.canvas_profile_probabilities,
        required_heading_status_counts=sample.heading_status_counts,
        render_style_params=params,
        render_style_defaults=rendering_defaults,
    )


def _validate_heading_scene_from_sample(scene: IsoHarborScene, sample: CountTaskSampleSpec) -> None:
    """Validate renderer heading counts against the sampled exact count map."""

    if not isinstance(sample, HarborHeadingCountSampleSpec):
        raise TypeError("heading-status sample expected")
    rendered_counts = dict(scene.trace.get("boat_counts_by_heading_status", {}))
    for status, count in sample.heading_status_counts.items():
        if int(rendered_counts.get(str(status), -1)) != int(count):
            raise ValueError(f"rendered heading count for {status} did not match request")


def build_harbor_heading_count_plan(config: HarborHeadingCountConfig) -> HarborCountPlan:
    """Build a public-owned count plan for shoreline-relative boat headings."""

    return HarborCountPlan(
        public_id=str(config.public_id),
        operation="count_boats_by_shoreline_relative_heading",
        required_prompt_keys=tuple(str(key) for key in config.required_prompt_keys),
        sample_spec=lambda instance_seed, params: _heading_sample_spec(
            config=config,
            instance_seed=int(instance_seed),
            params=params,
        ),
        prompt_slots=_heading_prompt_slots,
        scene_builder=lambda scene_seed, sample, params: _render_heading_scene_from_sample(
            scene_seed,
            sample,
            params,
            rendering_defaults=config.rendering_defaults,
        ),
        entity_selector=lambda scene, sample: sorted_harbor_boats(
            scene,
            predicate=lambda entity: (
                isinstance(sample, HarborHeadingCountSampleSpec)
                and str(entity.metadata.get("heading_status", "")) == str(sample.target_heading_status)
            ),
        ),
        render_map=lambda scene, sample, counted_ids: isometric_harbor_heading_status_count_render_map(
            scene=scene,
            target_status=str(sample.target_heading_status) if isinstance(sample, HarborHeadingCountSampleSpec) else "",
            counted_entity_ids=counted_ids,
        ),
        identity_fields=_heading_identity_fields,
        extra_query_params=_heading_extra_query_params,
        scene_validator=_validate_heading_scene_from_sample,
    )


def run_harbor_count_lifecycle(
    *,
    plan: HarborCountPlan,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Render one harbor scene, bind task-owned witnesses, and package `TaskOutput`."""

    sample = plan.sample_spec(int(instance_seed), dict(params))
    checked_prompt_defaults = required_group_defaults(
        prompt_defaults,
        list(plan.required_prompt_keys),
        context=f"prompt defaults for {plan.public_id}",
    )
    last_error: Exception | None = None
    scene: IsoHarborScene | None = None
    counted_entities: tuple[IsoHarborEntity, ...] = ()
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_seed = int(instance_seed) + int(attempt) * 1009
            scene = plan.scene_builder(scene_seed, sample, params)
            counted_entities = tuple(plan.entity_selector(scene, sample))
            if len(counted_entities) != int(sample.target_count):
                raise ValueError(f"count {len(counted_entities)} did not match target_count {sample.target_count}")
            if plan.scene_validator is not None:
                plan.scene_validator(scene, sample)
            break
        except Exception as exc:
            last_error = exc
            scene = None
            counted_entities = ()
    if scene is None:
        raise RuntimeError(f"could not generate harbor task instance: {last_error}") from last_error

    annotation_value = [rounded_bbox(entity.bbox_xyxy) for entity in counted_entities]
    counted_entity_ids = tuple(str(entity.entity_id) for entity in counted_entities)
    prompt_artifacts = build_isometric_harbor_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=checked_prompt_defaults,
        prompt_query_key=str(sample.prompt_query_key),
        slots=dict(plan.prompt_slots(checked_prompt_defaults, sample)),
        instance_seed=int(instance_seed),
    )
    identity_fields = dict(plan.identity_fields(sample))
    extra_query_params = dict(plan.extra_query_params(sample))
    query_params = {
        "query_id": str(sample.selected_key),
        "prompt_query_key": str(sample.prompt_query_key),
        "query_id_probabilities": dict(sample.query_probabilities),
        "target_count": int(sample.target_count),
        "target_count_probabilities": dict(sample.target_count_probabilities),
        "answer_count_support": list(sample.answer_count_support),
        "answer_count_probabilities": dict(sample.answer_count_probabilities),
        "answer_count": int(len(counted_entities)),
        "counted_entity_ids": list(counted_entity_ids),
        "canvas_profile": str(sample.canvas_profile),
        "canvas_profile_probabilities": dict(sample.canvas_profile_probabilities),
        **identity_fields,
        **extra_query_params,
    }
    trace_payload = {
        "scene_ir": isometric_harbor_scene_ir(
            domain=str(domain),
            scene_id=SCENE_ID,
            scene=scene,
            relations={
                "operation": str(plan.operation),
                **identity_fields,
                "answer_count": int(len(counted_entities)),
                "counted_entity_ids": list(counted_entity_ids),
                "counted_entity_bboxes_px": [list(bbox) for bbox in annotation_value],
            },
        ),
        "query_spec": {
            "task_id": str(plan.public_id),
            "query_id": str(sample.selected_key),
            "prompt_query_key": str(sample.prompt_query_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": isometric_harbor_render_spec(scene, scene_id=SCENE_ID),
        "render_map": dict(plan.render_map(scene, sample, counted_entity_ids)),
        "execution_trace": {
            "query_id": str(sample.selected_key),
            "prompt_query_key": str(sample.prompt_query_key),
            "scene_id": SCENE_ID,
            "answer": int(len(counted_entities)),
            **identity_fields,
            "counted_entity_ids": list(counted_entity_ids),
            "renderer": dict(scene.trace),
        },
        "witness_symbolic": {
            "answer_count": int(len(counted_entities)),
            **identity_fields,
            "counted_entity_ids": list(counted_entity_ids),
            "counted_entity_bboxes": [list(bbox) for bbox in annotation_value],
        },
        "projected_annotation": bbox_set_projection(annotation_value),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        answer_gt=TypedValue(type="integer", value=int(len(counted_entities))),
        annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_value]),
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(sample.selected_key),
    )


__all__ = [
    "HarborCountPlan",
    "HarborHeadingCountConfig",
    "build_harbor_heading_count_plan",
    "run_harbor_count_lifecycle",
    "sorted_harbor_boats",
]
