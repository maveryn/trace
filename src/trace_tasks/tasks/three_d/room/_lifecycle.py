"""Neutral lifecycle for 3D room public task files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.scene_config import get_domain_defaults, get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from trace_tasks.tasks.three_d.room.shared.relations import (
    REFERENCE_WALL_OBJECT_TYPES,
    build_room_wall_same_wall_reference_dataset,
)
from trace_tasks.tasks.three_d.room.shared.rendering import render_room_scene_3d
from trace_tasks.tasks.three_d.room.shared.state import SCENE_ID
from trace_tasks.tasks.three_d.shared.canvas import render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import (
    ObjectSceneRenderParams,
    POINT_LABELS,
    resolve_object_scene_render_params,
)
from trace_tasks.tasks.three_d.shared.task_support import (
    resolve_axis_variant_for_namespace,
    resolve_count_for_namespace,
    resolve_support_choice_for_namespace,
)
from trace_tasks.tasks.three_d.room.shared.state import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.three_d.shared.option_panel import build_text_option_choices


PrepareRoomObjective = Callable[
    [int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], ObjectSceneRenderParams],
    "RoomObjectivePlan",
]


@dataclass(frozen=True)
class RoomObjectivePlan:
    """Task-local objective data returned by a public room task hook."""

    dataset: Mapping[str, Any]
    public_query_id: str
    query_probabilities: Mapping[str, float]
    prompt_query_key: str | None
    prompt_dynamic_slots: Mapping[str, Any]
    answer_gt: TypedValue
    annotation_schema: str
    query_params: Mapping[str, Any]
    use_option_panel: bool = False


@dataclass(frozen=True)
class RoomOptionCounts:
    """Shared count operands for room option-panel tasks."""

    candidate_count: int
    candidate_count_probabilities: Mapping[str, float]
    context_wall_count: int
    context_wall_count_probabilities: Mapping[str, float]
    floor_context_count: int
    floor_context_count_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class RoomOptionContext:
    """Resolved shared scene/count operands for one room option-panel task."""

    scene_variant: str
    scene_probabilities: Mapping[str, float]
    counts: RoomOptionCounts


_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _task_defaults(task_identifier: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve room scene defaults for one public task without branching on task identity."""

    scene_defaults = get_scene_defaults("three_d", SCENE_ID)
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(task_identifier),
    )
    return dict(gen_defaults), dict(render_defaults), dict(prompt_defaults)


def resolve_room_choice(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    key: str,
    support: tuple[str, ...],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one task-owned room operand from an explicit value or uniform support."""

    values = tuple(str(value) for value in support)
    explicit = params.get(str(key))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(values):
            raise ValueError(f"unsupported {key}: {selected}")
        return selected, {value: (1.0 if value == selected else 0.0) for value in values}
    selected, probabilities = resolve_support_choice_for_namespace(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{key}",
        support_values=values,
        explicit_key=str(key),
    )
    return str(selected), {str(value): float(probability) for value, probability in probabilities.items()}


def require_single_query(params: Mapping[str, Any], *, expected: str, task_identifier: str) -> None:
    """Validate the repo-wide single-query sentinel for one public room task."""

    requested = str(params.get("query_id", str(expected)))
    if requested != str(expected):
        raise ValueError(f"unsupported query_id for {task_identifier}: {requested}")


def resolve_room_scene_variant(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve a room scene variant from explicit params or scene defaults."""

    scene_variant, scene_probabilities = resolve_axis_variant_for_namespace(
        params,
        namespace=f"{namespace}.scene_variant",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )
    return str(scene_variant), dict(scene_probabilities)


def resolve_room_option_counts(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    candidate_min: int,
    candidate_max: int,
    candidate_lower: int,
    candidate_upper: int,
) -> RoomOptionCounts:
    """Resolve shared candidate/context/floor counts for option-panel room tasks."""

    candidate_count, candidate_count_probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.candidate_count",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="candidate_count",
        default_min=int(candidate_min),
        default_max=int(candidate_max),
        lower=int(candidate_lower),
        upper=int(candidate_upper),
    )
    context_wall_count, context_wall_count_probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.context_wall_count",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="context_wall_count",
        default_min=0,
        default_max=0,
        lower=0,
        upper=0,
    )
    floor_context_count, floor_context_count_probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.floor_context_count",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="floor_context_count",
        default_min=6,
        default_max=6,
        lower=3,
        upper=8,
    )
    return RoomOptionCounts(
        candidate_count=int(candidate_count),
        candidate_count_probabilities=dict(candidate_count_probabilities),
        context_wall_count=int(context_wall_count),
        context_wall_count_probabilities=dict(context_wall_count_probabilities),
        floor_context_count=int(floor_context_count),
        floor_context_count_probabilities=dict(floor_context_count_probabilities),
    )


def resolve_room_option_context(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    candidate_min: int,
    candidate_max: int,
    candidate_lower: int,
    candidate_upper: int,
) -> RoomOptionContext:
    """Resolve the scene variant and shared option counts for room tasks."""

    scene_variant, scene_probabilities = resolve_room_scene_variant(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    counts = resolve_room_option_counts(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        candidate_min=int(candidate_min),
        candidate_max=int(candidate_max),
        candidate_lower=int(candidate_lower),
        candidate_upper=int(candidate_upper),
    )
    return RoomOptionContext(
        scene_variant=str(scene_variant),
        scene_probabilities=dict(scene_probabilities),
        counts=counts,
    )


def room_option_answer_label_probabilities(candidate_count: int) -> dict[str, float]:
    """Return the uniform label prior for an option-panel room task."""

    return {
        str(label): round(1.0 / float(candidate_count), 8)
        for label in POINT_LABELS[: int(candidate_count)]
    }


def build_room_option_objective_plan(
    *,
    dataset: Mapping[str, Any],
    public_query_id: str,
    query_probabilities: Mapping[str, float],
    prompt_query_key: str,
    prompt_dynamic_slots: Mapping[str, Any],
    context: RoomOptionContext,
    extra_query_params: Mapping[str, Any] | None = None,
) -> RoomObjectivePlan:
    """Assemble the common option-letter RoomObjectivePlan for public tasks."""

    return RoomObjectivePlan(
        dataset=dict(dataset),
        public_query_id=str(public_query_id),
        query_probabilities=dict(query_probabilities),
        prompt_query_key=str(prompt_query_key),
        prompt_dynamic_slots=dict(prompt_dynamic_slots),
        answer_gt=TypedValue(type="option_letter", value=str(dataset["answer_label"])),
        annotation_schema="bbox",
        use_option_panel=True,
        query_params={
            **room_option_query_params(
                scene_variant=str(context.scene_variant),
                scene_probabilities=context.scene_probabilities,
                counts=context.counts,
                dataset=dataset,
            ),
            **dict(extra_query_params or {}),
        },
    )


def prepare_same_wall_reference_objective_from_semantics(
    *,
    task_identifier: str,
    single_branch: str,
    objective_seed: int,
    objective_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> RoomObjectivePlan:
    """Build the same-wall reference objective from public task bindings."""

    require_single_query(
        objective_params,
        expected=str(single_branch),
        task_identifier=str(task_identifier),
    )
    context = resolve_room_option_context(
        objective_params,
        gen_defaults,
        instance_seed=int(objective_seed),
        namespace=str(task_identifier),
        candidate_min=6,
        candidate_max=6,
        candidate_lower=4,
        candidate_upper=6,
    )
    reference_wall, reference_wall_probabilities = resolve_room_choice(
        objective_params,
        instance_seed=int(objective_seed),
        key="reference_wall",
        support=("back", "left", "right"),
        namespace=str(task_identifier),
    )
    reference_object_type, reference_object_type_probabilities = resolve_room_choice(
        objective_params,
        instance_seed=int(objective_seed),
        key="reference_object_type",
        support=REFERENCE_WALL_OBJECT_TYPES,
        namespace=str(task_identifier),
    )
    dataset = build_room_wall_same_wall_reference_dataset(
        scene_variant=str(context.scene_variant),
        candidate_count=int(context.counts.candidate_count),
        context_wall_count=int(context.counts.context_wall_count),
        floor_context_count=int(context.counts.floor_context_count),
        reference_wall=str(reference_wall),
        reference_object_type=str(reference_object_type),
        render_params=render_params,
        namespace=str(task_identifier),
        instance_seed=int(objective_seed),
    )
    reference_name = str(dataset["reference_object"]["prompt_name"])
    return build_room_option_objective_plan(
        dataset=dict(dataset),
        public_query_id=str(single_branch),
        query_probabilities={str(single_branch): 1.0},
        prompt_query_key="same_wall_reference",
        prompt_dynamic_slots={"reference_name": reference_name},
        context=context,
        extra_query_params={
            "reference_wall": str(reference_wall),
            "reference_wall_probabilities": dict(reference_wall_probabilities),
            "reference_object_type": str(reference_object_type),
            "reference_object_type_probabilities": dict(reference_object_type_probabilities),
            "reference_name": reference_name,
        },
    )


def room_option_query_params(
    *,
    scene_variant: str,
    scene_probabilities: Mapping[str, float],
    counts: RoomOptionCounts,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Return shared trace query params for option-panel room tasks."""

    return {
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "candidate_count": int(counts.candidate_count),
        "candidate_count_probabilities": dict(counts.candidate_count_probabilities),
        "context_wall_count": int(counts.context_wall_count),
        "context_wall_count_probabilities": dict(counts.context_wall_count_probabilities),
        "floor_context_count": int(counts.floor_context_count),
        "floor_context_count_probabilities": dict(counts.floor_context_count_probabilities),
        "object_count": int(dataset["object_count"]),
        "answer_label_probabilities": room_option_answer_label_probabilities(int(counts.candidate_count)),
    }


def run_room_lifecycle(
    instance_seed: int,
    *,
    params: Dict[str, Any],
    max_attempts: int,
    task_identifier: str,
    prepare_objective: PrepareRoomObjective,
) -> TaskOutput:
    """Retry a task-local objective hook, render the room, and bind verifier output."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = (
            int(instance_seed)
            if attempt_index == 0
            else int(spawn_rng(int(instance_seed), f"{task_identifier}.attempt_seed.{attempt_index}").randrange(1, 2**62))
        )
        try:
            return _run_once(
                int(attempt_seed),
                params=params,
                task_identifier=str(task_identifier),
                prepare_objective=prepare_objective,
            )
        except Exception as exc:  # pragma: no cover - stochastic retry guard.
            last_error = exc
    raise RuntimeError(f"{task_identifier} failed to generate a valid scene after {max_attempts} attempts: {last_error}")


def _run_once(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    task_identifier: str,
    prepare_objective: PrepareRoomObjective,
) -> TaskOutput:
    """Resolve defaults, call the objective hook, render, project annotation, and return output."""

    gen_defaults, render_defaults, prompt_defaults_config = _task_defaults(str(task_identifier))
    render_params = resolve_object_scene_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{task_identifier}.canvas",
    )
    plan = prepare_objective(
        int(instance_seed),
        params,
        gen_defaults,
        prompt_defaults_config,
        render_params,
    )
    dataset = dict(plan.dataset)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=_BACKGROUND_DEFAULTS,
    )
    option_choices = build_text_option_choices(dataset["candidate_object_specs"]) if bool(plan.use_option_panel) else []
    rendered_scene = render_room_scene_3d(
        background,
        dataset=dataset,
        render_params=render_params,
        option_choices=option_choices,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    annotation_bboxes = [[round(float(value), 3) for value in bbox] for bbox in rendered_scene.annotation_bboxes]
    if str(plan.annotation_schema) == "bbox":
        if len(annotation_bboxes) != 1:
            raise RuntimeError(f"{task_identifier} expected exactly one annotation bbox")
        annotation_payload = bbox_annotation_artifacts(annotation_bboxes[0])
        annotation_gt = annotation_payload.annotation_gt
        projected_annotation = dict(annotation_payload.projected_annotation)
    elif str(plan.annotation_schema) == "bbox_set":
        annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_bboxes])
        projected_annotation = {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in annotation_bboxes],
            "pixel_bbox_set": [list(bbox) for bbox in annotation_bboxes],
        }
    else:
        raise ValueError(f"unsupported room annotation schema: {plan.annotation_schema}")

    prompt_defaults = required_group_defaults(
        prompt_defaults_config,
        ("bundle_id", "scene_key", "task_key"),
        context=f"prompt defaults for {task_identifier}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain="three_d",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=(str(plan.prompt_query_key) if plan.prompt_query_key is not None else None),
        dynamic_slots=dict(plan.prompt_dynamic_slots),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    option_descriptor_by_label = {
        str(choice["label"]): str(choice["descriptor"])
        for choice in option_choices
    }
    execution_trace = {
        "query_id": str(plan.public_query_id),
        "scene_id": SCENE_ID,
        **dict(dataset),
        "option_choices": [dict(choice) for choice in option_choices],
        "option_descriptor_by_label": dict(option_descriptor_by_label),
        "view_family": "synthetic_perspective_3d_room",
    }
    relations = {
        key: value
        for key, value in dict(dataset).items()
        if key not in {"wall_object_specs", "floor_object_specs", "object_specs", "candidate_object_specs"}
    }
    relations["view_family"] = "synthetic_perspective_3d_room"
    query_params = {
        "query_id": str(plan.public_query_id),
        "query_id_probabilities": dict(plan.query_probabilities),
        **dict(plan.query_params),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "three_d_room_scene",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(relations),
        },
        "query_spec": {
            "query_id": str(plan.public_query_id),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_params),
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(image.height),
            "scene_canvas_preset": str(render_params.canvas_preset),
            "scene_canvas_width": int(render_params.canvas_width),
            "scene_canvas_height": int(render_params.canvas_height),
            "scene_canvas_policy": str(render_params.canvas_policy),
            **render_params_canvas_metadata(render_params),
            "final_canvas_width": int(image.width),
            "final_canvas_height": int(image.height),
            "final_canvas_pixels": int(image.width) * int(image.height),
            "option_panel_height_px": int(rendered_scene.option_panel_height_px),
            "coord_space": "pixel",
            "scene_variant": str(dataset["scene_variant"]),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "label_font_size_px": int(render_params.label_font_size_px),
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "room_bbox_px": list(rendered_scene.room_bbox_px),
            "object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.object_bboxes_px.items()},
            "object_centers_px": {str(key): list(value) for key, value in rendered_scene.object_centers_px.items()},
            "option_panel_bbox_px": list(rendered_scene.option_panel_bbox_px),
            "option_panel_height_px": int(rendered_scene.option_panel_height_px),
            "option_choice_bboxes_px": {str(key): list(value) for key, value in rendered_scene.option_choice_bboxes_px.items()},
            "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
            "wall_object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.wall_object_bboxes_px.items()},
            "wall_object_centers_px": {str(key): list(value) for key, value in rendered_scene.wall_object_centers_px.items()},
            "floor_object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.floor_object_bboxes_px.items()},
            "floor_object_centers_px": {str(key): list(value) for key, value in rendered_scene.floor_object_centers_px.items()},
            "target_object_bboxes_px": {
                str(key): list(rendered_scene.object_bboxes_px[str(key)])
                for key in dataset["target_object_ids"]
            },
        },
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "object_set" if str(plan.annotation_schema) == "bbox_set" else "object",
            "ids": [str(value) for value in dataset["target_object_ids"]],
            "answer": plan.answer_gt.value,
        },
        "projected_annotation": dict(projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    reference_object = dataset.get("reference_object")
    if isinstance(reference_object, Mapping):
        trace_payload["render_map"]["reference_object_bbox_px"] = list(rendered_scene.object_bboxes_px[str(reference_object["object_id"])])

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.public_query_id),
    )


__all__ = [
    "RoomObjectivePlan",
    "RoomOptionCounts",
    "RoomOptionContext",
    "build_room_option_objective_plan",
    "prepare_same_wall_reference_objective_from_semantics",
    "require_single_query",
    "resolve_room_option_context",
    "resolve_room_choice",
    "resolve_room_option_counts",
    "resolve_room_scene_variant",
    "room_option_query_params",
    "run_room_lifecycle",
]
