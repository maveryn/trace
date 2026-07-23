"""Neutral render lifecycle for organic-structure public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.common import get_int_range as _get_range
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ..shared.unit_size_jitter import with_symbolic_unit_size_jitter
from ..shared.visual_defaults import load_symbolic_noise_defaults

from .shared.annotations import atom_point_set, bond_segment_set, ring_bbox_set
from .shared.output import (
    atom_trace_records,
    bond_trace_records,
    ring_trace_records,
    text_label_trace_records,
)
from .shared.prompts import render_organic_prompt
from .shared.rendering import render_organic_scene
from .shared.rules import (
    build_constrained_organic_ring_size_structure,
    build_constrained_organic_structure,
    organic_ring_item_ids,
    validate_organic_structure,
)
from .shared.state import (
    ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT,
    ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT,
    SUPPORTED_BOND_ORDERS,
    SUPPORTED_ORGANIC_RING_SIZES,
    OrganicRenderParams,
    OrganicStructureSpec,
    RenderedOrganicScene,
    SCENE_ID,
)
from .shared.sampling import build_with_retries, resolve_scene_variant
from .shared.styles import resolve_render_params


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class OrganicRenderBundle:
    image: Image.Image
    rendered: RenderedOrganicScene
    render_params: OrganicRenderParams
    scene_style_metadata: dict[str, Any]
    background_metadata: dict[str, Any]
    post_noise_metadata: dict[str, Any]
    task_versions: dict[str, str]


@dataclass(frozen=True)
class OrganicCountPlan:
    """Task-owned organic count dataset bound before rendering."""

    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    selected_query_id: str
    selected_query_probabilities: dict[str, float]
    answer_value: int
    target_answer_support: tuple[int, ...]
    structure: OrganicStructureSpec
    annotation_item_ids: tuple[str, ...]
    metadata: dict[str, Any]
    prompt_dynamic_slots: dict[str, Any]
    query_extra: dict[str, Any]
    relations_extra: dict[str, Any]
    annotation_family: str
    annotation_source: str
    execution_extra: dict[str, Any]


@dataclass(frozen=True)
class OrganicLifecycleSpec:
    """Static lifecycle inputs supplied by one public organic task file."""

    registry_key: str
    domain: str
    internal_branch: str
    supported_query_ids: tuple[str, ...]
    gen_defaults: Mapping[str, Any]
    render_defaults: Mapping[str, Any]
    prompt_defaults: Mapping[str, Any]
    task_prompt_key: str
    prompt_key_stem: str


def organic_answer_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    lower: int,
    upper: int,
    owner_key: str,
) -> tuple[int, ...]:
    """Resolve a task-owned integer answer support within a fixed count range."""

    answer_min, answer_max = _get_range(
        params,
        gen_defaults,
        min_key="target_answer_min",
        max_key="target_answer_max",
        fallback_min=int(lower),
        fallback_max=int(upper),
    )
    if int(answer_min) < int(lower) or int(answer_max) > int(upper):
        raise ValueError(f"{owner_key} supports target_answer_min/max only within {int(lower)}..{int(upper)}")
    return tuple(range(int(answer_min), int(answer_max) + 1))


def sample_organic_answer_count(rng: Any, params: Mapping[str, Any], support: tuple[int, ...]) -> int:
    """Sample or validate the integer count requested by a public task."""

    if "answer_value" in params:
        answer_count = int(params["answer_value"])
        if answer_count not in support:
            raise ValueError(f"answer_value={answer_count} is outside configured support {support}")
        return int(answer_count)
    return int(rng.choice(support))


def prepare_bond_order_count_plan(
    *,
    owner_key: str,
    gen_defaults: Mapping[str, Any],
    target_orders: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    selected_query_id: str,
    selected_query_probabilities: Mapping[str, float],
) -> OrganicCountPlan:
    """Build the task-bound plan for counting double or triple bonds."""

    target_order, target_probabilities = resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=target_orders,
        task_id=str(owner_key),
        explicit_key="target_bond_order",
        weights_key="target_bond_order_weights",
        balance_flag_key="balanced_target_bond_order_sampling",
        axis_namespace="target_bond_order",
    )
    rng = spawn_rng(int(instance_seed), f"{owner_key}.dataset")
    support = organic_answer_support(
        params,
        gen_defaults,
        lower=1,
        upper=ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT,
        owner_key=str(owner_key),
    )
    answer_count = sample_organic_answer_count(rng, params, support)
    structure = build_constrained_organic_structure(rng, target_bond_order=str(target_order), answer_count=int(answer_count))
    constraint_report = validate_organic_structure(structure)
    annotation_ids = tuple(bond.item_id for bond in structure.bonds if bond.order == str(target_order))
    if len(annotation_ids) != int(answer_count):
        raise RuntimeError("constrained organic structure did not preserve requested answer value")
    return make_organic_count_plan(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_variant_probabilities,
        selected_query_id=str(selected_query_id),
        selected_query_probabilities=selected_query_probabilities,
        answer_value=int(len(annotation_ids)),
        answer_support=tuple(support),
        structure=structure,
        annotation_item_ids=tuple(annotation_ids),
        metadata={
            "target_bond_order": str(target_order),
            "target_bond_order_support": list(target_orders),
            "bond_order_support": list(SUPPORTED_BOND_ORDERS),
            "scaffold_id": str(structure.scaffold_id),
            "scaffold_family": str(structure.scaffold_family),
            "constraint_policy": str(structure.constraint_policy),
            "constraint_report": constraint_report.to_metadata(),
            "chemical_validity_policy": "basic atom valence and line-angle geometry constraints are enforced; molecule identity is not required",
            "text_label_policy": "hetero atom and substituent labels may be rendered as scene context; they are not answer targets",
        },
        prompt_dynamic_slots={"target_bond_order": str(target_order)},
        query_extra={
            "target_bond_order": str(target_order),
            "target_bond_order_probabilities": {str(key): float(value) for key, value in target_probabilities.items()},
        },
        relations_extra={"target_bond_order": str(target_order)},
        annotation_family="bond_segments",
        annotation_source="bond_segments_px",
    )


def prepare_ring_size_count_plan(
    *,
    owner_key: str,
    gen_defaults: Mapping[str, Any],
    target_sizes: tuple[int, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    selected_query_id: str,
    selected_query_probabilities: Mapping[str, float],
) -> OrganicCountPlan:
    """Build the task-bound plan for counting pentagonal or hexagonal rings."""

    target_size_text, target_probabilities = resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=[str(item) for item in target_sizes],
        task_id=str(owner_key),
        explicit_key="target_ring_size",
        weights_key="target_ring_size_weights",
        balance_flag_key="balanced_target_ring_size_sampling",
        axis_namespace="target_ring_size",
    )
    target_size = int(target_size_text)
    rng = spawn_rng(int(instance_seed), f"{owner_key}.dataset")
    support = organic_answer_support(
        params,
        gen_defaults,
        lower=1,
        upper=ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT,
        owner_key=str(owner_key),
    )
    answer_count = sample_organic_answer_count(rng, params, support)
    structure = build_constrained_organic_ring_size_structure(rng, target_ring_size=target_size, answer_count=answer_count)
    constraint_report = validate_organic_structure(structure)
    annotation_ids = organic_ring_item_ids(structure, target_size)
    if len(annotation_ids) != int(answer_count):
        raise RuntimeError("constrained organic ring-size structure did not preserve requested answer value")
    ring_name = "pentagonal" if target_size == 5 else "hexagonal"
    return make_organic_count_plan(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_variant_probabilities,
        selected_query_id=str(selected_query_id),
        selected_query_probabilities=selected_query_probabilities,
        answer_value=int(len(annotation_ids)),
        answer_support=tuple(support),
        structure=structure,
        annotation_item_ids=tuple(annotation_ids),
        metadata={
            "target_property": "ring_size",
            "target_ring_size": int(target_size),
            "target_ring_name": ring_name,
            "target_ring_size_support": list(target_sizes),
            "ring_size_support": list(SUPPORTED_ORGANIC_RING_SIZES),
            "ring_count_support": list(range(1, ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT + 1)),
            "scaffold_id": str(structure.scaffold_id),
            "scaffold_family": str(structure.scaffold_family),
            "constraint_policy": str(structure.constraint_policy),
            "constraint_report": constraint_report.to_metadata(),
            "chemical_validity_policy": "basic atom valence and line-angle geometry constraints are enforced; molecule identity is not required",
            "ring_layout_policy": "rings are explicit five- or six-member records arranged in bent or branched linked-ring clusters; boundaries must remain unambiguous",
            "text_label_policy": "hetero atom and substituent labels may be rendered as scene context; they are not answer targets",
        },
        prompt_dynamic_slots={"target_ring_name": ring_name},
        query_extra={
            "target_ring_size": int(target_size),
            "target_ring_name": ring_name,
            "target_ring_size_probabilities": {str(key): float(value) for key, value in target_probabilities.items()},
        },
        relations_extra={"target_ring_size": int(target_size)},
        annotation_family="ring_bboxes",
        annotation_source="item_bboxes_px",
        execution_extra={
            "matching_ring_item_ids": [str(item) for item in annotation_ids],
            "rings": ring_trace_records(structure, target_ring_ids=set(annotation_ids)),
        },
    )


def make_organic_count_plan(
    *,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    selected_query_id: str,
    selected_query_probabilities: Mapping[str, float],
    answer_value: int,
    answer_support: tuple[int, ...],
    structure: OrganicStructureSpec,
    annotation_item_ids: tuple[str, ...],
    metadata: Mapping[str, Any],
    prompt_dynamic_slots: Mapping[str, Any] | None = None,
    query_extra: Mapping[str, Any] | None = None,
    relations_extra: Mapping[str, Any] | None = None,
    annotation_family: str,
    annotation_source: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> OrganicCountPlan:
    """Package a task-owned dataset into the shared organic lifecycle state."""

    return OrganicCountPlan(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        selected_query_id=str(selected_query_id),
        selected_query_probabilities={str(key): float(value) for key, value in selected_query_probabilities.items()},
        answer_value=int(answer_value),
        target_answer_support=tuple(int(value) for value in answer_support),
        structure=structure,
        annotation_item_ids=tuple(str(item) for item in annotation_item_ids),
        metadata={str(key): value for key, value in metadata.items()},
        prompt_dynamic_slots={str(key): value for key, value in (prompt_dynamic_slots or {}).items()},
        query_extra={str(key): value for key, value in (query_extra or {}).items()},
        relations_extra={str(key): value for key, value in (relations_extra or {}).items()},
        annotation_family=str(annotation_family),
        annotation_source=str(annotation_source),
        execution_extra={str(key): value for key, value in (execution_extra or {}).items()},
    )


def render_organic_task_scene(
    *,
    structure: OrganicStructureSpec,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    instance_seed: int,
    sampling_scope: str,
) -> OrganicRenderBundle:
    """Resolve render settings, render the scene, and apply post-image noise."""

    render_params = resolve_render_params(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )
    scene_style, scene_style_metadata = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{sampling_scope}.organic_structure_background",
    )
    background, background_metadata = make_symbolic_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_organic_scene(
        background,
        structure=structure,
        render_params=render_params,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )
    image, post_noise_metadata = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return OrganicRenderBundle(
        image=image,
        rendered=rendered_scene,
        render_params=render_params,
        scene_style_metadata=dict(scene_style_metadata),
        background_metadata=dict(background_metadata),
        post_noise_metadata=dict(post_noise_metadata),
        task_versions=default_task_versions(),
    )


def organic_render_spec(
    *,
    scene_variant: str,
    bundle: OrganicRenderBundle,
) -> dict[str, Any]:
    """Serialize shared render metadata for one organic scene."""

    return {
        "scene_id": SCENE_ID,
        "canvas_width": int(bundle.render_params.canvas_width),
        "canvas_height": int(bundle.render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "scene_style": dict(bundle.scene_style_metadata),
        "organic_style": dict(bundle.rendered.style_metadata),
        "background_style": dict(bundle.background_metadata),
        "post_image_noise": dict(bundle.post_noise_metadata),
        "scene_bbox_px": list(bundle.rendered.scene_bbox_px),
        "unit_size_jitter": dict(bundle.render_params.unit_size_jitter),
        "layout_jitter": dict(bundle.rendered.layout_jitter),
    }


def organic_render_map(
    *,
    bundle: OrganicRenderBundle,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize shared final geometry maps for one organic scene."""

    payload = {
        "image_id": "img0",
        "scene_bbox_px": list(bundle.rendered.scene_bbox_px),
        "item_bboxes_px": {str(key): list(value) for key, value in bundle.rendered.item_bboxes.items()},
        "bond_segments_px": {
            str(key): [list(point) for point in value]
            for key, value in bundle.rendered.item_segments.items()
        },
        "atom_points_px": {str(key): list(value) for key, value in bundle.rendered.item_points.items()},
        "layout_jitter": dict(bundle.rendered.layout_jitter),
    }
    if extra:
        payload.update({str(key): value for key, value in extra.items()})
    return with_symbolic_unit_size_jitter(payload, bundle.render_params.unit_size_jitter)


def organic_query_params(
    *,
    public_branch: str,
    public_probabilities: Mapping[str, float],
    internal_branch: str,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    answer_support: list[int],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build shared query metadata from task-owned branch and operand fields."""

    payload: dict[str, Any] = {
        "query_id": str(public_branch),
        "query_id_probabilities": {str(key): float(value) for key, value in public_probabilities.items()},
        "internal_query_id": str(internal_branch),
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": {str(key): float(value) for key, value in scene_variant_probabilities.items()},
        "answer_support": [int(value) for value in answer_support],
        "target_answer_support": [int(value) for value in answer_support],
        "question_format": str(internal_branch),
    }
    if extra:
        payload.update({str(key): value for key, value in extra.items()})
    return payload


def organic_prompt_query_spec(
    *,
    prompt_artifacts: Any,
    public_branch: str,
    internal_branch: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build prompt query metadata while preserving the internal task branch."""

    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_branch),
        params=params,
    )
    query_spec["internal_query_id"] = str(internal_branch)
    return query_spec


def compose_organic_trace_payload(
    *,
    rendered: RenderedOrganicScene,
    prompt_query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    render_map: Mapping[str, Any],
    relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    annotation_artifacts: Any,
    answer_gt: Any,
) -> dict[str, Any]:
    """Serialize shared trace sections from task-bound answer and annotation values."""

    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relations),
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": dict(render_spec),
        "render_map": dict(render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {"type": str(annotation_artifacts.annotation_type), "value": annotation_artifacts.value},
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
    }


def _organic_annotation_artifacts(rendered: RenderedOrganicScene, plan: OrganicCountPlan) -> Any:
    family = str(plan.annotation_family)
    if family == "bond_segments":
        return bond_segment_set(rendered.item_segments, plan.annotation_item_ids)
    if family == "atom_points":
        return atom_point_set(rendered.item_points, plan.annotation_item_ids)
    if family == "ring_bboxes":
        return ring_bbox_set(rendered.item_bboxes, plan.annotation_item_ids)
    raise ValueError(f"unsupported organic annotation family: {family!r}")


def _organic_render_map_extra(plan: OrganicCountPlan, bundle: OrganicRenderBundle) -> dict[str, Any]:
    extra: dict[str, Any] = {"annotation_source": str(plan.annotation_source)}
    if str(plan.annotation_family) == "ring_bboxes":
        extra["ring_bboxes_px"] = {
            f"ring_{ring_index + 1:02d}": list(bundle.rendered.item_bboxes[f"ring_{ring_index + 1:02d}"])
            for ring_index in range(len(plan.structure.ring_atom_sets))
        }
    return extra


def run_organic_count_lifecycle(
    spec: OrganicLifecycleSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Callable[..., OrganicCountPlan],
) -> TaskOutput:
    """Run the neutral organic count scene lifecycle for one public task.

    The public task supplies the constrained dataset, annotation binding, and
    objective-specific trace extras; this helper performs only repeated query
    selection, rendering, prompt rendering, trace serialization, and final
    output packaging.
    """

    selected_query_id, selected_query_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=spec.supported_query_ids,
        default_query_id=spec.supported_query_ids[0],
        task_id=str(spec.registry_key),
        namespace=f"{spec.registry_key}.query",
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        clean_params,
        gen_defaults=spec.gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(spec.registry_key),
    )
    plan = build_with_retries(
        lambda retry_seed: build_plan(
            instance_seed=int(retry_seed),
            params=clean_params,
            scene_variant=str(scene_variant),
            scene_variant_probabilities=scene_variant_probabilities,
            selected_query_id=str(selected_query_id),
            selected_query_probabilities=selected_query_probabilities,
        ),
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        failure_message=f"failed to generate organic-structure {spec.internal_branch} instance",
    )
    render_bundle = render_organic_task_scene(
        structure=plan.structure,
        params=clean_params,
        render_defaults=spec.render_defaults,
        scene_variant=str(plan.scene_variant),
        instance_seed=int(instance_seed),
        sampling_scope=str(spec.registry_key),
    )
    prompt_runtime = render_organic_prompt(
        spec.prompt_defaults,
        domain=str(spec.domain),
        scene_id=SCENE_ID,
        scene_variant=str(plan.scene_variant),
        task_key=str(spec.task_prompt_key),
        annotation_hint_key=f"annotation_hint_{spec.prompt_key_stem}",
        answer_hint_key=f"answer_hint_{spec.prompt_key_stem}",
        json_example_key=f"json_example_{spec.prompt_key_stem}",
        json_example_answer_only_key=f"json_example_answer_only_{spec.prompt_key_stem}",
        instance_seed=int(instance_seed),
        context=f"prompt defaults for {spec.registry_key}",
        extra_dynamic_slots=dict(plan.prompt_dynamic_slots),
    )
    annotation_artifacts = _organic_annotation_artifacts(render_bundle.rendered, plan)
    answer_gt = TypedValue(type="integer", value=int(plan.answer_value))
    query_params = organic_query_params(
        public_branch=str(plan.selected_query_id),
        public_probabilities=plan.selected_query_probabilities,
        internal_branch=str(spec.internal_branch),
        scene_variant=str(plan.scene_variant),
        scene_variant_probabilities=plan.scene_variant_probabilities,
        answer_support=[int(value) for value in plan.target_answer_support],
        extra=plan.query_extra,
    )
    prompt_query_spec = organic_prompt_query_spec(
        prompt_artifacts=prompt_runtime.artifacts,
        public_branch=str(plan.selected_query_id),
        internal_branch=str(spec.internal_branch),
        params=query_params,
    )
    render_map = organic_render_map(
        bundle=render_bundle,
        extra=_organic_render_map_extra(plan, render_bundle),
    )
    execution_trace = {
        **dict(query_params),
        "task_id": str(spec.registry_key),
        "answer_value": int(plan.answer_value),
        "answer_type": "integer",
        "annotation_item_ids": [str(item) for item in plan.annotation_item_ids],
        "organic_metadata": dict(plan.metadata),
        "atoms": atom_trace_records(plan.structure),
        "bonds": bond_trace_records(plan.structure),
        "text_labels": text_label_trace_records(plan.structure),
        "ring_vertex_sets": [[int(idx) for idx in ring] for ring in plan.structure.ring_atom_sets],
        **dict(plan.execution_extra),
    }
    trace_payload = compose_organic_trace_payload(
        rendered=render_bundle.rendered,
        prompt_query_spec=prompt_query_spec,
        render_spec=organic_render_spec(scene_variant=str(plan.scene_variant), bundle=render_bundle),
        render_map=render_map,
        relations={
            "query_id": str(plan.selected_query_id),
            "internal_query_id": str(spec.internal_branch),
            "scene_id": SCENE_ID,
            "scene_variant": str(plan.scene_variant),
            "answer_value": int(plan.answer_value),
            "scaffold_id": str(plan.structure.scaffold_id),
            "scaffold_family": str(plan.structure.scaffold_family),
            **dict(plan.relations_extra),
        },
        execution_trace=execution_trace,
        annotation_artifacts=annotation_artifacts,
        answer_gt=answer_gt,
    )
    return TaskOutput(
        prompt=str(prompt_runtime.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=render_bundle.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=dict(render_bundle.task_versions),
        scene_id=SCENE_ID,
        query_id=str(plan.selected_query_id),
        prompt_variants=dict(prompt_runtime.prompt_variants),
    )


__all__ = [
    "OrganicCountPlan",
    "OrganicLifecycleSpec",
    "OrganicRenderBundle",
    "compose_organic_trace_payload",
    "make_organic_count_plan",
    "organic_render_map",
    "organic_render_spec",
    "organic_prompt_query_spec",
    "organic_query_params",
    "prepare_bond_order_count_plan",
    "prepare_ring_size_count_plan",
    "organic_answer_support",
    "render_organic_task_scene",
    "run_organic_count_lifecycle",
    "sample_organic_answer_count",
]
