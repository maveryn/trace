"""Private lifecycle helpers for graph binary-tree public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from ....core.sampling import uniform_choice, weighted_support_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_default
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.task_support import resolve_graph_render_params
from ..shared.task_support import format_graph_prompt_label
from .shared.algorithms import traversal_labels
from .shared.annotations import (
    keyed_bboxes_for_roles,
    keyed_points_for_roles,
    project_node_label_bboxes,
    role_to_label_map,
    roles_by_label,
    rounded_points,
)
from .shared.defaults import BinaryTreeDefaults, BinaryTreeVisualAxes, resolve_visual_axes
from .shared.output import common_execution_fields, common_query_params, edge_entities, node_entities, rendered_trace_sections, scene_ir
from .shared.prompts import (
    count_prompt_json_examples,
    heap_violation_prompt_json_examples,
    keyed_node_prompt_json_examples,
    operation_path_prompt_json_examples,
    traversal_prompt_json_examples,
)
from .shared.rendering import render_binary_tree_scene
from .shared.sampling import (
    depth_support_for_target,
    labels_for_count_mode,
    sample_relation_tree,
    sample_search_tree_operation,
    sample_structure_count_tree,
    sample_traversal_tree,
    support_for_count_mode,
)
from .shared.state import BinaryTreeSample, RenderedBinaryTreeScene, SCENE_ID


@dataclass(frozen=True)
class BinaryTreeFrameContext:
    """Resolved style axes, render params, and background canvas for one scene."""

    forced_params: Mapping[str, Any]
    visual_axes: BinaryTreeVisualAxes
    render_params: Any
    image: Image.Image
    background_meta: Mapping[str, Any]


@dataclass(frozen=True)
class BinaryTreeRenderedContext:
    """Rendered scene plus the final image and post-render noise metadata."""

    rendered_scene: RenderedBinaryTreeScene
    image: Image.Image
    post_noise_meta: Mapping[str, Any]


@dataclass(frozen=True)
class BinaryTreeCountPlan:
    """Public-owned semantic settings for a node-count objective."""

    owner_id: str
    supported_branch_names: tuple[str, ...]
    default_branch_name: str
    count_mode_by_branch: Mapping[str, str]
    uses_depth_operand: bool = False


@dataclass(frozen=True)
class BinaryTreeTraversalPlan:
    """Public-owned semantic settings for a traversal-order objective."""

    owner_id: str
    supported_branch_names: tuple[str, ...]
    default_branch_name: str
    traversal_order_by_branch: Mapping[str, str]


@dataclass(frozen=True)
class BinaryTreeRelationPlan:
    """Public-owned semantic settings for node-label relation objectives."""

    owner_id: str
    supported_branch_names: tuple[str, ...]
    default_branch_name: str
    relation_kind_by_branch: Mapping[str, str]
    annotation_roles_by_branch: Mapping[str, tuple[str, ...]]
    relation_answer_scope_weights_by_branch: Mapping[str, Mapping[str, float]] | None = None


@dataclass(frozen=True)
class BinaryTreeOperationPlan:
    """Public-owned semantic settings for numeric tree operation objectives."""

    owner_id: str
    supported_branch_names: tuple[str, ...]
    default_branch_name: str
    operation_kind_by_branch: Mapping[str, str]
    scene_title: str
    object_description_key: str
    prompt_family: str
    annotation_roles: tuple[str, ...] = ()


def _prompt_default(prompt_defaults: Mapping[str, Any], key: str) -> Any:
    """Resolve prompt defaults from scene config or the v1 prompt asset."""

    return required_group_default(prompt_defaults, str(key), context="binary_tree prompt defaults")


def prepare_binary_tree_frame(
    *,
    instance_seed: int,
    forced_params: Mapping[str, Any],
    owner_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
    background_defaults: Mapping[str, Any],
    include_label_variant: bool = True,
) -> BinaryTreeFrameContext:
    """Resolve reusable visual axes and background before task-specific sampling."""

    visual_axes = resolve_visual_axes(
        instance_seed=int(instance_seed),
        params=dict(forced_params),
        gen_defaults=gen_defaults,
        sampling_namespace=str(owner_id),
        include_label_variant=bool(include_label_variant),
    )
    render_params = resolve_graph_render_params(
        dict(forced_params),
        instance_seed=int(instance_seed),
        task_id=str(owner_id),
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        node_color_name=str(visual_axes.node_color_name),
        node_shape_variant=str(visual_axes.node_shape_variant),
        edge_routing_variant="straight",
    )
    image, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=dict(forced_params),
        default_config=background_defaults,
    )
    return BinaryTreeFrameContext(
        forced_params=dict(forced_params),
        visual_axes=visual_axes,
        render_params=render_params,
        image=image,
        background_meta=dict(background_meta),
    )


def select_binary_tree_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    owner_id: str,
    branch_names: tuple[str, ...],
    default_branch_name: str,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Select a public-file-supported branch and return forced params."""

    branch_name, branch_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in branch_names),
        default_query_id=str(default_branch_name),
        task_id=str(owner_id),
        namespace=f"{owner_id}.query",
    )
    forced_params = force_query_id_params(task_params, query_id=str(branch_name))
    return str(branch_name), dict(branch_probs), dict(forced_params)


def render_binary_tree_frame(
    *,
    sample: BinaryTreeSample,
    frame: BinaryTreeFrameContext,
    scene_title: str,
    instance_seed: int,
    noise_defaults: Mapping[str, Any],
) -> BinaryTreeRenderedContext:
    """Render the sampled tree and apply scene-level post-image noise."""

    rendered_scene = render_binary_tree_scene(
        sample=sample,
        render_params=frame.render_params,
        scene_variant=str(frame.visual_axes.scene_variant),
        scene_title=str(scene_title),
        layout_seed=int(instance_seed),
        base_image=frame.image,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(frame.forced_params),
        default_config=noise_defaults,
    )
    return BinaryTreeRenderedContext(
        rendered_scene=rendered_scene,
        image=image,
        post_noise_meta=dict(post_noise_meta),
    )


def render_binary_tree_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    branch_name: str,
    slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render scene prompt variants and return normalized prompt trace artifacts."""

    numeric_slot_keys = {"target_depth", "target_key", "traversal_position"}
    dynamic_slots: dict[str, Any] = {}
    for key, value in dict(slots).items():
        if value == "":
            continue
        if str(key) in numeric_slot_keys:
            dynamic_slots[str(key)] = int(value)
        else:
            dynamic_slots[str(key)] = value
    prompt_selection = render_scene_prompt_variants(
        domain="graph",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(branch_name),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def common_prompt_slots(
    *,
    prompt_defaults: Mapping[str, Any],
    json_example: str,
    json_example_answer_only: str,
    annotation_hint: str,
    answer_hint: str,
    object_description: str | None = None,
    target_depth: str = "",
    target_key: str = "",
    query_label: str = "",
    query_label_a: str = "",
    query_label_b: str = "",
) -> dict[str, str]:
    """Return the common prompt slot map used by binary-tree templates."""

    return {
        "object_description": str(
            object_description
            if object_description is not None
            else _prompt_default(prompt_defaults, "object_description")
        ),
        "target_depth": str(target_depth),
        "target_key": str(target_key),
        "query_label": str(query_label),
        "query_label_a": str(query_label_a),
        "query_label_b": str(query_label_b),
        "json_output_contract": str(_prompt_default(prompt_defaults, "json_output_contract")),
        "json_output_contract_answer_only": str(_prompt_default(prompt_defaults, "json_output_contract_answer_only")),
        "annotation_hint": str(annotation_hint),
        "answer_hint": str(answer_hint),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


def binary_tree_trace_payload(
    *,
    owner_id: str,
    branch_name: str,
    prompt_artifacts: PromptTraceArtifacts,
    scene_payload: Mapping[str, Any],
    query_params: Mapping[str, Any],
    rendered: BinaryTreeRenderedContext,
    frame: BinaryTreeFrameContext,
    sample: BinaryTreeSample,
    execution_fields: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    extra_style: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common trace skeleton from task-owned semantic bindings."""

    scene_ir = dict(scene_payload)
    scene_ir["task_id"] = str(owner_id)
    full_query_params = {
        **dict(query_params),
        **common_query_params(sample=sample, visual_axes=frame.visual_axes),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_name),
        params=full_query_params,
    )
    query_spec["task_id"] = str(owner_id)
    return {
        "scene_ir": scene_ir,
        "query_spec": query_spec,
        **rendered_trace_sections(
            rendered_scene=rendered.rendered_scene,
            render_params=frame.render_params,
            background_meta=frame.background_meta,
            post_noise_meta=rendered.post_noise_meta,
            extra_style=extra_style,
        ),
        "execution_trace": {
            "task_id": str(owner_id),
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            **dict(execution_fields),
            **common_execution_fields(
                sample=sample,
                rendered_scene=rendered.rendered_scene,
                visual_axes=frame.visual_axes,
            ),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


def binary_tree_task_output(
    *,
    prompt_artifacts: PromptTraceArtifacts,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    rendered: BinaryTreeRenderedContext,
    trace_payload: Mapping[str, Any],
    branch_name: str,
) -> TaskOutput:
    """Build the final output after the public task binds answer and annotation."""

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_name),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def _resolve_axis_value(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    owner_id: str,
    axis_name: str,
    support: tuple[int, ...],
) -> tuple[int, dict[str, float]]:
    explicit_value = params.get(str(axis_name))
    if explicit_value is not None:
        value = int(explicit_value)
        if value not in set(support):
            raise ValueError(f"{axis_name} is outside configured binary-tree support")
        return int(value), uniform_probability_map(support, selected=int(value))
    value = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{owner_id}:{axis_name}"),
            support,
        )
    )
    return int(value), uniform_probability_map(support)


def _resolve_weighted_string_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    owner_id: str,
    axis_name: str,
    weights: Mapping[str, float],
) -> tuple[str, dict[str, float]]:
    support = tuple(str(key) for key, value in weights.items() if float(value) > 0.0)
    if not support:
        return "", {}
    explicit_value = params.get(str(axis_name))
    if explicit_value is not None:
        value = str(explicit_value)
        if value not in support:
            raise ValueError(f"{axis_name} is outside configured binary-tree support")
        return value, {str(key): (1.0 if str(key) == value else 0.0) for key in support}
    selected, probabilities = weighted_support_choice(
        spawn_rng(int(instance_seed), f"{owner_id}:{axis_name}"),
        support,
        weights=weights,
    )
    return str(selected), dict(probabilities)


def run_binary_tree_count_plan(
    *,
    plan: BinaryTreeCountPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Run a count objective from public-owned branch semantics."""

    branch_name, branch_probs, forced_params = select_binary_tree_branch(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=str(plan.owner_id),
        branch_names=tuple(plan.supported_branch_names),
        default_branch_name=str(plan.default_branch_name),
    )
    count_mode = str(plan.count_mode_by_branch[str(branch_name)])
    count_support = tuple(
        support_for_count_mode(count_mode, gen_defaults=gen_defaults, defaults=defaults)
    )
    target_count, target_count_probs = _resolve_axis_value(
        instance_seed=int(instance_seed),
        params=forced_params,
        owner_id=str(plan.owner_id),
        axis_name="target_count",
        support=tuple(int(value) for value in count_support),
    )
    target_depth = None
    target_depth_probs: dict[str, float] = {}
    if bool(plan.uses_depth_operand):
        depth_support = depth_support_for_target(int(target_count), gen_defaults=gen_defaults, defaults=defaults)
        target_depth, target_depth_probs = _resolve_axis_value(
            instance_seed=int(instance_seed),
            params=forced_params,
            owner_id=str(plan.owner_id),
            axis_name="target_depth",
            support=tuple(int(value) for value in depth_support),
        )
    frame = prepare_binary_tree_frame(
        instance_seed=int(instance_seed),
        forced_params=forced_params,
        owner_id=str(plan.owner_id),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        defaults=defaults,
        background_defaults=background_defaults,
    )
    sample = sample_structure_count_tree(
        int(instance_seed),
        count_mode=count_mode,
        target_count=int(target_count),
        target_depth=target_depth,
        node_count_min=int(group_default(gen_defaults, "node_count_min", defaults.node_count_min)),
        node_count_max=int(group_default(gen_defaults, "node_count_max", defaults.node_count_max)),
        max_depth=int(group_default(gen_defaults, "max_depth", defaults.max_depth)),
        label_variant=str(frame.visual_axes.label_variant),
        label_max_chars=int(group_default(gen_defaults, "label_max_chars", defaults.label_max_chars)),
        max_attempts=max(1, int(max_attempts)),
    )
    target_labels = labels_for_count_mode(sample, count_mode=count_mode, target_depth=target_depth)
    if len(target_labels) != int(target_count):
        raise ValueError("binary-tree count sample does not match requested target count")
    rendered = render_binary_tree_frame(
        sample=sample,
        frame=frame,
        scene_title="Binary Tree",
        instance_seed=int(instance_seed),
        noise_defaults=noise_defaults,
    )
    annotation_projection = project_node_label_bboxes(rendered.rendered_scene, target_labels)
    annotation_points = rounded_points(annotation_projection["pixel_point_set"])
    prompt_defaults_map = dict(prompt_defaults)
    json_example, json_example_answer_only = count_prompt_json_examples()
    annotation_hint = str(_prompt_default(prompt_defaults_map, f"annotation_hint_{branch_name}"))
    if target_depth is not None:
        annotation_hint = annotation_hint.format(target_depth=str(target_depth))
    prompt_artifacts = render_binary_tree_prompt_artifacts(
        prompt_defaults=prompt_defaults_map,
        branch_name=str(branch_name),
        slots=common_prompt_slots(
            prompt_defaults=prompt_defaults_map,
            target_depth="" if target_depth is None else str(target_depth),
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
            annotation_hint=annotation_hint,
            answer_hint=str(_prompt_default(prompt_defaults_map, "answer_hint_count")),
        ),
        instance_seed=int(instance_seed),
    )
    entities = [
        *node_entities(
            rendered_scene=rendered.rendered_scene,
            sample=sample,
            counted_labels=target_labels,
            annotation_labels=target_labels,
        ),
        *edge_entities(rendered.rendered_scene),
    ]
    scene_payload = scene_ir(
        scene_kind="binary_tree",
        rendered_scene=rendered.rendered_scene,
        sample=sample,
        entities=entities,
        relations={
            "query_id": str(branch_name),
            "count_mode": str(count_mode),
            "target_labels": list(target_labels),
            "target_depth": target_depth,
        },
    )
    trace_payload = binary_tree_trace_payload(
        owner_id=str(plan.owner_id),
        branch_name=str(branch_name),
        prompt_artifacts=prompt_artifacts,
        scene_payload=scene_payload,
        query_params={
            "count_mode": str(count_mode),
            "query_id_probabilities": dict(branch_probs),
            "target_count": int(target_count),
            "target_count_probabilities": dict(target_count_probs),
            "target_depth": target_depth,
            "target_depth_probabilities": dict(target_depth_probs),
        },
        rendered=rendered,
        frame=frame,
        sample=sample,
        execution_fields={
            "count_mode": str(count_mode),
            "answer": int(len(target_labels)),
            "target_labels": list(target_labels),
            "target_depth": target_depth,
        },
        witness_symbolic={
            "type": "node_label_set",
            "labels": list(target_labels),
            "query_id": str(branch_name),
            "count_mode": str(count_mode),
        },
        projected_annotation={
            "type": "point_set",
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
            "pixel_bbox_set": list(annotation_projection["pixel_bbox_set"]),
        },
        extra_style={"node_color_name": str(frame.visual_axes.node_color_name)},
    )
    return binary_tree_task_output(
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="integer", value=int(len(target_labels))),
        annotation_gt=TypedValue(type="point_set", value=list(annotation_points)),
        rendered=rendered,
        trace_payload=trace_payload,
        branch_name=str(branch_name),
    )


def run_binary_tree_traversal_plan(
    *,
    plan: BinaryTreeTraversalPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Run a traversal-position objective from public-owned branch semantics."""

    branch_name, branch_probs, forced_params = select_binary_tree_branch(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=str(plan.owner_id),
        branch_names=tuple(plan.supported_branch_names),
        default_branch_name=str(plan.default_branch_name),
    )
    traversal_order = str(plan.traversal_order_by_branch[str(branch_name)])
    lower = int(group_default(gen_defaults, "traversal_position_min", defaults.traversal_position_min))
    upper = int(group_default(gen_defaults, "traversal_position_max", defaults.traversal_position_max))
    traversal_position, traversal_position_probs = _resolve_axis_value(
        instance_seed=int(instance_seed),
        params=forced_params,
        owner_id=str(plan.owner_id),
        axis_name="traversal_position",
        support=tuple(range(int(lower), int(upper) + 1)),
    )
    frame = prepare_binary_tree_frame(
        instance_seed=int(instance_seed),
        forced_params=forced_params,
        owner_id=str(plan.owner_id),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        defaults=defaults,
        background_defaults=background_defaults,
    )
    node_count_min = max(int(group_default(gen_defaults, "node_count_min", defaults.node_count_min)), int(traversal_position))
    sample = sample_traversal_tree(
        int(instance_seed),
        node_count_min=int(node_count_min),
        node_count_max=int(group_default(gen_defaults, "node_count_max", defaults.node_count_max)),
        max_depth=int(group_default(gen_defaults, "max_depth", defaults.max_depth)),
        label_variant=str(frame.visual_axes.label_variant),
        label_max_chars=int(group_default(gen_defaults, "label_max_chars", defaults.label_max_chars)),
        max_attempts=max(1, int(max_attempts)),
    )
    ordered_labels = traversal_labels(sample, traversal_order)
    if int(traversal_position) > len(ordered_labels):
        raise ValueError("sampled binary tree is smaller than requested traversal position")
    answer_value = str(ordered_labels[int(traversal_position) - 1])
    annotation_labels = tuple(str(label) for label in ordered_labels[: int(traversal_position)])
    rendered = render_binary_tree_frame(
        sample=sample,
        frame=frame,
        scene_title="Binary Tree",
        instance_seed=int(instance_seed),
        noise_defaults=noise_defaults,
    )
    annotation_projection = project_node_label_bboxes(rendered.rendered_scene, annotation_labels)
    annotation_points = rounded_points(annotation_projection["pixel_point_sequence"])
    prompt_defaults_map = dict(prompt_defaults)
    json_example, json_example_answer_only = traversal_prompt_json_examples()
    prompt_artifacts = render_binary_tree_prompt_artifacts(
        prompt_defaults=prompt_defaults_map,
        branch_name=str(branch_name),
        slots={
            **common_prompt_slots(
                prompt_defaults=prompt_defaults_map,
                json_example=str(json_example),
                json_example_answer_only=str(json_example_answer_only),
                annotation_hint=str(_prompt_default(prompt_defaults_map, f"annotation_hint_{branch_name}")),
                answer_hint=str(_prompt_default(prompt_defaults_map, "answer_hint_node_label")),
            ),
            "traversal_position": str(traversal_position),
        },
        instance_seed=int(instance_seed),
    )
    answer_node = next(node for node in sample.nodes if str(node.label) == str(answer_value))
    entities = [
        *node_entities(
            rendered_scene=rendered.rendered_scene,
            sample=sample,
            answer_node_id=str(answer_node.node_id),
            annotation_labels=annotation_labels,
        ),
        *edge_entities(rendered.rendered_scene),
    ]
    scene_payload = scene_ir(
        scene_kind="binary_tree",
        rendered_scene=rendered.rendered_scene,
        sample=sample,
        entities=entities,
        relations={
            "query_id": str(branch_name),
            "traversal_order": str(traversal_order),
            "traversal_position": int(traversal_position),
            "answer_label": str(answer_value),
        },
    )
    trace_payload = binary_tree_trace_payload(
        owner_id=str(plan.owner_id),
        branch_name=str(branch_name),
        prompt_artifacts=prompt_artifacts,
        scene_payload=scene_payload,
        query_params={
            "traversal_order": str(traversal_order),
            "query_id_probabilities": dict(branch_probs),
            "traversal_position": int(traversal_position),
            "traversal_position_probabilities": dict(traversal_position_probs),
        },
        rendered=rendered,
        frame=frame,
        sample=sample,
        execution_fields={
            "traversal_order": str(traversal_order),
            "answer": str(answer_value),
            "annotation_labels": list(annotation_labels),
            "traversal_position": int(traversal_position),
        },
        witness_symbolic={
            "type": "node_label_sequence",
            "labels": list(annotation_labels),
            "query_id": str(branch_name),
            "traversal_order": str(traversal_order),
            "answer_label": str(answer_value),
        },
        projected_annotation={
            "type": "point_sequence",
            "point_sequence": list(annotation_points),
            "pixel_point_sequence": list(annotation_points),
            "pixel_bbox_sequence": list(annotation_projection["pixel_bbox_sequence"]),
        },
        extra_style={"node_color_name": str(frame.visual_axes.node_color_name)},
    )
    return binary_tree_task_output(
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type="point_sequence", value=list(annotation_points)),
        rendered=rendered,
        trace_payload=trace_payload,
        branch_name=str(branch_name),
    )


def run_binary_tree_relation_plan(
    *,
    plan: BinaryTreeRelationPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Run a node-label relation objective from public-owned branch semantics."""

    branch_name, branch_probs, forced_params = select_binary_tree_branch(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=str(plan.owner_id),
        branch_names=tuple(plan.supported_branch_names),
        default_branch_name=str(plan.default_branch_name),
    )
    relation_kind = str(plan.relation_kind_by_branch[str(branch_name)])
    annotation_roles = tuple(str(role) for role in plan.annotation_roles_by_branch[str(branch_name)])
    answer_scope_weights_by_branch = plan.relation_answer_scope_weights_by_branch or {}
    answer_scope_weights = answer_scope_weights_by_branch.get(str(branch_name), {})
    relation_answer_scope, relation_answer_scope_probs = _resolve_weighted_string_axis(
        instance_seed=int(instance_seed),
        params=forced_params,
        owner_id=str(plan.owner_id),
        axis_name="relation_answer_scope",
        weights=answer_scope_weights,
    )
    frame = prepare_binary_tree_frame(
        instance_seed=int(instance_seed),
        forced_params=forced_params,
        owner_id=str(plan.owner_id),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        defaults=defaults,
        background_defaults=background_defaults,
    )
    sample, relation = sample_relation_tree(
        int(instance_seed),
        relation_kind=relation_kind,
        relation_answer_scope=str(relation_answer_scope),
        node_count_min=int(group_default(gen_defaults, "node_count_min", defaults.node_count_min)),
        node_count_max=int(group_default(gen_defaults, "node_count_max", defaults.node_count_max)),
        max_depth=int(group_default(gen_defaults, "max_depth", defaults.max_depth)),
        label_variant=str(frame.visual_axes.label_variant),
        label_max_chars=int(group_default(gen_defaults, "label_max_chars", defaults.label_max_chars)),
        max_attempts=max(1, int(max_attempts)),
    )
    rendered = render_binary_tree_frame(
        sample=sample,
        frame=frame,
        scene_title="Binary Tree",
        instance_seed=int(instance_seed),
        noise_defaults=noise_defaults,
    )
    annotation_projection = project_node_label_bboxes(rendered.rendered_scene, relation.annotation_labels)
    annotation_keyed_bboxes = keyed_bboxes_for_roles(roles=annotation_roles, labels=relation.annotation_labels, projection=annotation_projection)
    annotation_keyed_points = keyed_points_for_roles(roles=annotation_roles, projection=annotation_projection)
    annotation_role_to_label = role_to_label_map(roles=annotation_roles, labels=relation.annotation_labels)
    annotation_roles_by_label = roles_by_label(annotation_role_to_label)
    relation_answer_scope_fields = (
        {"answer_scope": str(relation.answer_scope)}
        if str(relation.answer_scope)
        else {}
    )
    query_answer_scope_fields = (
        {
            "relation_answer_scope": str(relation_answer_scope),
            "relation_answer_scope_probabilities": dict(relation_answer_scope_probs),
        }
        if str(relation_answer_scope)
        else {}
    )
    prompt_defaults_map = dict(prompt_defaults)
    prompt_query_labels = tuple(
        format_graph_prompt_label(str(label), label_variant=str(frame.visual_axes.label_variant))
        for label in relation.query_labels
    )
    json_example, json_example_answer_only = keyed_node_prompt_json_examples(annotation_roles)
    prompt_artifacts = render_binary_tree_prompt_artifacts(
        prompt_defaults=prompt_defaults_map,
        branch_name=str(branch_name),
        slots=common_prompt_slots(
            prompt_defaults=prompt_defaults_map,
            query_label=str(prompt_query_labels[0]) if prompt_query_labels else "",
            query_label_a=str(prompt_query_labels[0]) if prompt_query_labels else "",
            query_label_b=str(prompt_query_labels[1]) if len(prompt_query_labels) > 1 else "",
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
            annotation_hint=str(_prompt_default(prompt_defaults_map, f"annotation_hint_{branch_name}")),
            answer_hint=str(_prompt_default(prompt_defaults_map, "answer_hint_node_label")),
        ),
        instance_seed=int(instance_seed),
    )
    entities = [
        *node_entities(
            rendered_scene=rendered.rendered_scene,
            sample=sample,
            query_node_ids=relation.query_node_ids,
            answer_node_id=relation.answer_node_id,
            annotation_labels=relation.annotation_labels,
            annotation_roles_by_label=annotation_roles_by_label,
        ),
        *edge_entities(rendered.rendered_scene),
    ]
    scene_payload = scene_ir(
        scene_kind="binary_tree",
        rendered_scene=rendered.rendered_scene,
        sample=sample,
        entities=entities,
        relations={
            "query_id": str(branch_name),
            "relation_kind": str(relation_kind),
            "query_labels": list(relation.query_labels),
            "answer_label": str(relation.answer_label),
            "answer_node_id": str(relation.answer_node_id),
            "annotation_role_to_label": dict(annotation_role_to_label),
            **relation_answer_scope_fields,
        },
    )
    trace_payload = binary_tree_trace_payload(
        owner_id=str(plan.owner_id),
        branch_name=str(branch_name),
        prompt_artifacts=prompt_artifacts,
        scene_payload=scene_payload,
        query_params={
            "relation_kind": str(relation_kind),
            "query_id_probabilities": dict(branch_probs),
            **query_answer_scope_fields,
        },
        rendered=rendered,
        frame=frame,
        sample=sample,
        execution_fields={
            "relation_kind": str(relation_kind),
            "query_labels": list(relation.query_labels),
            "answer": str(relation.answer_label),
            "answer_label": str(relation.answer_label),
            "answer_node_id": str(relation.answer_node_id),
            "annotation_roles": list(annotation_roles),
            "annotation_role_to_label": dict(annotation_role_to_label),
            "annotation_labels": list(relation.annotation_labels),
            **relation_answer_scope_fields,
        },
        witness_symbolic={
            "type": "binary_tree_node_label_relation",
            "query_id": str(branch_name),
            "relation_kind": str(relation_kind),
            "query_labels": list(relation.query_labels),
            "answer_label": str(relation.answer_label),
            "annotation_role_to_label": dict(annotation_role_to_label),
            **relation_answer_scope_fields,
        },
        projected_annotation={
            "type": "point_map",
            "point_map": dict(annotation_keyed_points),
            "pixel_point_map": dict(annotation_keyed_points),
            "bbox_map": dict(annotation_keyed_bboxes),
            "pixel_bbox_map": dict(annotation_keyed_bboxes),
            "bbox_sequence": list(annotation_keyed_bboxes.values()),
            "pixel_bbox_sequence": list(annotation_keyed_bboxes.values()),
            "pixel_point_sequence": list(annotation_projection["pixel_point_set"]),
        },
        extra_style={"node_color_name": str(frame.visual_axes.node_color_name)},
    )
    return binary_tree_task_output(
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="string", value=str(relation.answer_label)),
        annotation_gt=TypedValue(type="point_map", value=dict(annotation_keyed_points)),
        rendered=rendered,
        trace_payload=trace_payload,
        branch_name=str(branch_name),
    )


def run_binary_tree_operation_plan(
    *,
    plan: BinaryTreeOperationPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: BinaryTreeDefaults,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Run a numeric tree operation objective from public-owned branch semantics."""

    branch_name, branch_probs, forced_params = select_binary_tree_branch(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=str(plan.owner_id),
        branch_names=tuple(plan.supported_branch_names),
        default_branch_name=str(plan.default_branch_name),
    )
    lower = int(group_default(gen_defaults, "node_count_min", defaults.node_count_min))
    upper = int(group_default(gen_defaults, "node_count_max", defaults.node_count_max))
    node_count, node_count_probs = _resolve_axis_value(
        instance_seed=int(instance_seed),
        params=forced_params,
        owner_id=str(plan.owner_id),
        axis_name="node_count",
        support=tuple(range(int(lower), int(upper) + 1)),
    )
    frame = prepare_binary_tree_frame(
        instance_seed=int(instance_seed),
        forced_params=forced_params,
        owner_id=str(plan.owner_id),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        defaults=defaults,
        background_defaults=background_defaults,
        include_label_variant=False,
    )
    operation_kind = str(plan.operation_kind_by_branch[str(branch_name)])
    sample, operation = sample_search_tree_operation(
        int(instance_seed),
        operation_kind=operation_kind,
        node_count=int(node_count),
        key_min=int(group_default(gen_defaults, "key_min", defaults.key_min)),
        key_max=int(group_default(gen_defaults, "key_max", defaults.key_max)),
        heap_violation_gap_min=int(group_default(gen_defaults, "heap_violation_gap_min", defaults.heap_violation_gap_min)),
        heap_violation_gap_max=int(group_default(gen_defaults, "heap_violation_gap_max", defaults.heap_violation_gap_max)),
        max_depth=int(group_default(gen_defaults, "max_depth", defaults.max_depth)),
        max_attempts=max(1, int(max_attempts)),
    )
    rendered = render_binary_tree_frame(
        sample=sample,
        frame=frame,
        scene_title=str(plan.scene_title),
        instance_seed=int(instance_seed),
        noise_defaults=noise_defaults,
    )
    annotation_projection = project_node_label_bboxes(rendered.rendered_scene, operation.annotation_labels)
    prompt_defaults_map = dict(prompt_defaults)
    object_description = str(_prompt_default(prompt_defaults_map, str(plan.object_description_key)))
    if str(plan.prompt_family) == "heap":
        json_example, json_example_answer_only = heap_violation_prompt_json_examples()
        annotation_roles = tuple(plan.annotation_roles)
        annotation_keyed_bboxes = keyed_bboxes_for_roles(roles=annotation_roles, labels=operation.annotation_labels, projection=annotation_projection)
        annotation_keyed_points = keyed_points_for_roles(roles=annotation_roles, projection=annotation_projection)
        annotation_role_to_label = role_to_label_map(roles=annotation_roles, labels=operation.annotation_labels)
        annotation_roles_by_label = roles_by_label(annotation_role_to_label)
        annotation_gt = TypedValue(type="point_map", value=dict(annotation_keyed_points))
        projected_annotation = {
            "type": "point_map",
            "point_map": dict(annotation_keyed_points),
            "pixel_point_map": dict(annotation_keyed_points),
            "bbox_map": dict(annotation_keyed_bboxes),
            "pixel_bbox_map": dict(annotation_keyed_bboxes),
            "pixel_point_sequence": list(annotation_projection["pixel_point_sequence"]),
        }
        annotation_labels_for_entities = operation.annotation_labels
    else:
        json_example, json_example_answer_only = operation_path_prompt_json_examples()
        annotation_role_to_label = {}
        annotation_roles_by_label = {}
        annotation_points = rounded_points(annotation_projection["pixel_point_sequence"])
        annotation_gt = TypedValue(type="point_sequence", value=list(annotation_points))
        projected_annotation = {
            "type": "point_sequence",
            "point_sequence": list(annotation_points),
            "pixel_point_sequence": list(annotation_points),
            "pixel_bbox_sequence": list(annotation_projection["pixel_bbox_sequence"]),
        }
        annotation_labels_for_entities = operation.annotation_labels
    target_key = "" if operation.target_key is None else str(operation.target_key)
    prompt_artifacts = render_binary_tree_prompt_artifacts(
        prompt_defaults=prompt_defaults_map,
        branch_name=str(branch_name),
        slots=common_prompt_slots(
            prompt_defaults=prompt_defaults_map,
            object_description=object_description,
            target_key=str(target_key),
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
            annotation_hint=str(_prompt_default(prompt_defaults_map, f"annotation_hint_{branch_name}")),
            answer_hint=str(_prompt_default(prompt_defaults_map, "answer_hint_numeric_key_label")),
        ),
        instance_seed=int(instance_seed),
    )
    entities = [
        *node_entities(
            rendered_scene=rendered.rendered_scene,
            sample=sample,
            query_node_ids=operation.query_node_ids,
            answer_node_id=operation.answer_node_id,
            annotation_labels=annotation_labels_for_entities,
            annotation_roles_by_label=annotation_roles_by_label,
            numeric_labels=True,
            path_node_ids=operation.query_node_ids,
        ),
        *edge_entities(rendered.rendered_scene),
    ]
    scene_payload = scene_ir(
        scene_kind="search_tree_operation_diagram",
        rendered_scene=rendered.rendered_scene,
        sample=sample,
        entities=entities,
        relations={
            "query_id": str(branch_name),
            "operation_kind": str(operation.operation_kind),
            "target_key": int(operation.target_key) if operation.target_key is not None else None,
            "answer_label": str(operation.answer_label),
            "answer_node_id": str(operation.answer_node_id),
            "annotation_labels": list(operation.annotation_labels),
            "annotation_role_to_label": dict(annotation_role_to_label),
        },
    )
    trace_payload = binary_tree_trace_payload(
        owner_id=str(plan.owner_id),
        branch_name=str(branch_name),
        prompt_artifacts=prompt_artifacts,
        scene_payload=scene_payload,
        query_params={
            "operation_kind": str(operation.operation_kind),
            "query_id_probabilities": dict(branch_probs),
            "node_count_probabilities": dict(node_count_probs),
        },
        rendered=rendered,
        frame=frame,
        sample=sample,
        execution_fields={
            "operation_kind": str(operation.operation_kind),
            "target_key": int(operation.target_key) if operation.target_key is not None else None,
            "answer": str(operation.answer_label),
            "answer_label": str(operation.answer_label),
            "answer_node_id": str(operation.answer_node_id),
            "annotation_labels": list(operation.annotation_labels),
            "annotation_role_to_label": dict(annotation_role_to_label),
        },
        witness_symbolic={
            "type": "search_tree_operation_path",
            "query_id": str(branch_name),
            "operation_kind": str(operation.operation_kind),
            "target_key": int(operation.target_key) if operation.target_key is not None else None,
            "answer_label": str(operation.answer_label),
            "annotation_labels": list(operation.annotation_labels),
            "annotation_role_to_label": dict(annotation_role_to_label),
        },
        projected_annotation=projected_annotation,
        extra_style={
            "scene_title": str(plan.scene_title),
            "node_color_name": str(frame.visual_axes.node_color_name),
        },
    )
    return binary_tree_task_output(
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="string", value=str(operation.answer_label)),
        annotation_gt=annotation_gt,
        rendered=rendered,
        trace_payload=trace_payload,
        branch_name=str(branch_name),
    )


__all__ = [
    "BinaryTreeFrameContext",
    "BinaryTreeCountPlan",
    "BinaryTreeOperationPlan",
    "BinaryTreeRelationPlan",
    "BinaryTreeRenderedContext",
    "BinaryTreeTraversalPlan",
    "binary_tree_trace_payload",
    "binary_tree_task_output",
    "common_prompt_slots",
    "prepare_binary_tree_frame",
    "render_binary_tree_frame",
    "render_binary_tree_prompt_artifacts",
    "run_binary_tree_count_plan",
    "run_binary_tree_operation_plan",
    "run_binary_tree_relation_plan",
    "run_binary_tree_traversal_plan",
    "select_binary_tree_branch",
]
