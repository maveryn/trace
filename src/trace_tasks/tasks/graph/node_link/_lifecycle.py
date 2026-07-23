"""Scene-private output plumbing for migrated node-link tasks.

The public task files own their objective/query sampling, answer binding, and
annotation binding. This module only carries neutral bundle metadata rewriting
and final ``TaskOutput`` assembly that is identical for every node-link task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.graph.shared.task_support import format_graph_prompt_label, graph_edge_label_entries, resolve_graph_named_variant, resolve_graph_render_params
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .shared.annotations import annotation_value, answer_value
from .shared.defaults import NodeLinkDefaults
from .shared.output import edge_entities, node_entities
from .shared.prompts import build_graph_prompt_json_examples, resolve_prompt_slot
from .shared.rendering import render_node_link_sample
from .shared.sampling import SUPPORTED_TOPOLOGY_PROFILES, resolve_node_link_visual_axes
from .shared.state import SCENE_ID


class _SafePromptSlots(dict):
    """Leave unknown prompt slots intact instead of failing at runtime."""

    def __missing__(self, key: str) -> str:
        return "{" + str(key) + "}"


@dataclass(frozen=True)
class NodeLinkAxes:
    """Resolved query, semantic support, and non-semantic visual axes."""

    query_id: str
    node_count: int
    values: dict[str, Any]
    topology_profile: str
    layout_variant: str
    label_variant: str
    node_shape_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    probabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeLinkTaskBundle:
    """Built node-link instance before public TaskOutput assembly."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Any
    image_id: str
    trace_payload: dict[str, Any]
    prompt_variants: dict[str, Any]
    scene_id: str = SCENE_ID
    query_id: str = ""


@dataclass(frozen=True)
class NodeLinkObjectivePlan:
    """Task-owned semantic plan consumed by scene-private lifecycle code."""

    public_id: str
    class_name: str
    supported_query_ids: tuple[str, ...]
    sample_graph: Callable[[Any, NodeLinkAxes, int], Any]
    answer_type: str
    answer_field: str
    annotation_type: str
    annotation_kind: str
    annotation_field: str
    prompt_query_key: Callable[[NodeLinkAxes], str] | str
    object_description_key: Callable[[NodeLinkAxes], str] | str = "object_description_undirected"
    annotation_hint_key: Callable[[NodeLinkAxes], str] | str = "annotation_hint"
    answer_hint_key: str = "answer_hint"
    prompt_bundle_id: str = ""
    prompt_scene_key: str = ""
    prompt_task_key: str = ""
    graph_directionality: Callable[[NodeLinkAxes], str] | str = "undirected"
    question_format: Callable[[NodeLinkAxes], str] | str = ""
    scene_kind: str = "node_link_graph"
    value_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    fixed_values: dict[str, Any] = field(default_factory=dict)
    semantic_colors: tuple[str, ...] = ("red", "blue", "green", "yellow", "orange", "purple")
    annotation_example: Any = field(default_factory=lambda: [[180, 220], [310, 180]])
    answer_example: Any = 2
    strict_edge_label_placement: bool = False


def _resolve_int_axis(
    *,
    key: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback: tuple[int, int],
    instance_seed: int,
    task_id: str,
) -> tuple[int, dict[str, float]]:
    """Resolve one integer axis with deterministic cycling over support."""

    if key in params:
        value = int(params[key])
        return value, {str(value): 1.0}
    lower = int(params.get(f"{key}_min", group_default(gen_defaults, f"{key}_min", int(fallback[0]))))
    upper = int(params.get(f"{key}_max", group_default(gen_defaults, f"{key}_max", int(fallback[1]))))
    if upper < lower:
        raise ValueError(f"{key} support is empty")
    support = tuple(range(lower, upper + 1))
    value = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{task_id}:{key}"),
            support,
        )
    )
    return value, {str(item): 1.0 / float(len(support)) for item in support}


def _trace_with_public_task_id(trace_payload: Mapping[str, Any], *, public_task_id: str) -> dict[str, Any]:
    """Attach the public task id to task-owned trace sections."""

    trace = dict(trace_payload)
    for section in ("scene_ir", "query_spec", "execution_trace"):
        if isinstance(trace.get(section), Mapping):
            trace[section] = {**dict(trace[section]), "task_id": str(public_task_id)}
    return trace


def _format_prompt_default(value: Any, slots: Mapping[str, Any]) -> str:
    """Format task prompt-default prose with runtime slots when present."""

    text = str(value)
    if "{" not in text:
        return text
    return text.format_map(_SafePromptSlots({str(key): value for key, value in slots.items()}))


def _sample_label_variant(sample: Any, axes: NodeLinkAxes) -> str:
    """Return the label variant used by the rendered sample."""

    return str(getattr(sample, "label_variant", axes.label_variant))


def _prompt_label(sample: Any, axes: NodeLinkAxes, label: Any) -> str:
    """Return one prompt-facing graph label with named labels quoted."""

    text = str(label)
    if not text:
        return ""
    return format_graph_prompt_label(text, label_variant=_sample_label_variant(sample, axes))


def _prompt_color_label(color_name: Any) -> str:
    """Return prompt-facing color text with canonical hex code."""

    text = str(color_name).strip().lower()
    if not text:
        return ""
    try:
        from trace_tasks.tasks.shared.named_colors import named_color

        return format_named_color_with_hex(text, named_color(text))
    except Exception:
        return text


def _resolve_axes(
    *,
    plan: NodeLinkObjectivePlan,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: NodeLinkDefaults,
    instance_seed: int,
) -> NodeLinkAxes:
    """Resolve task-supplied query ids, numeric supports, and visual axes."""

    supported_queries = tuple(str(value) for value in plan.supported_query_ids)
    if not supported_queries:
        raise ValueError(f"{plan.public_id} has no supported query ids")
    requested_query = params.get("query_id")
    if requested_query is not None and str(requested_query) not in {"", "default"}:
        selected_query = str(requested_query)
        if selected_query not in set(supported_queries):
            raise ValueError(f"unsupported query_id for {plan.public_id}: {selected_query}")
        query_probabilities = {str(selected_query): 1.0}
    else:
        selected_query = str(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{plan.public_id}:query_id"),
                supported_queries,
            )
        )
        probability = 1.0 / float(len(supported_queries))
        query_probabilities = {str(query_id): float(probability) for query_id in supported_queries}
    node_count, node_probabilities = _resolve_int_axis(
        key="node_count",
        params=params,
        gen_defaults=gen_defaults,
        fallback=(defaults.node_count_min, defaults.node_count_max),
        instance_seed=int(instance_seed),
        task_id=str(plan.public_id),
    )
    values: dict[str, Any] = dict(plan.fixed_values)
    for semantic_key in ("source_color_name", "target_color_name", "target_edge_label"):
        raw_value = params.get(semantic_key)
        if raw_value is not None and str(raw_value).strip():
            values[str(semantic_key)] = str(raw_value).strip().lower()
    raw_edge_label_support = params.get("edge_label_support", group_default(gen_defaults, "edge_label_support", None))
    if raw_edge_label_support is not None:
        if isinstance(raw_edge_label_support, str):
            values["edge_label_support"] = [
                str(item).strip().lower()
                for item in raw_edge_label_support.split(",")
                if str(item).strip()
            ]
        else:
            values["edge_label_support"] = [
                str(item).strip().lower()
                for item in raw_edge_label_support
                if str(item).strip()
            ]
    for semantic_key in (
        "edge_label_support_size",
        "edge_label_min_chars",
        "edge_label_max_chars",
        "max_labeled_edge_count",
    ):
        raw_value = params.get(semantic_key, group_default(gen_defaults, semantic_key, None))
        if raw_value is not None:
            values[str(semantic_key)] = int(raw_value)
    raw_edge_label_bucket_weights = params.get(
        "edge_label_bucket_weights",
        group_default(gen_defaults, "edge_label_bucket_weights", None),
    )
    if isinstance(raw_edge_label_bucket_weights, Mapping):
        values["edge_label_bucket_weights"] = dict(raw_edge_label_bucket_weights)
    probabilities: dict[str, Any] = {
        "query_id_probabilities": dict(query_probabilities),
        "node_count_probabilities": dict(node_probabilities),
    }
    fallback_ranges = {
        "target_count": (defaults.target_count_min, defaults.target_count_max),
        "query_degree": (defaults.query_degree_min, defaults.query_degree_max),
        "target_component_size": (defaults.target_count_min, defaults.target_count_max),
        "component_count": (defaults.component_count_min, defaults.component_count_max),
        "target_cycle_size": (defaults.cycle_size_min, defaults.cycle_size_max),
        "target_shortest_path_length": (defaults.path_length_min, defaults.path_length_max),
        "target_longest_path_length": (defaults.path_length_min, defaults.path_length_max),
        "target_position": (1, max(1, int(node_count))),
        "extra_edge_count": (defaults.extra_edge_count_min, defaults.extra_edge_count_max),
        "target_degree": (defaults.query_degree_min, defaults.query_degree_max),
    }
    for key, fallback in {**fallback_ranges, **plan.value_ranges}.items():
        if key in values:
            continue
        axis_defaults = {} if key in plan.value_ranges else gen_defaults
        selected_value, selected_probabilities = _resolve_int_axis(
            key=key,
            params=params,
            gen_defaults=axis_defaults,
            fallback=tuple(int(item) for item in fallback),
            instance_seed=int(instance_seed),
            task_id=str(plan.public_id),
        )
        values[key] = int(selected_value)
        probabilities[f"{key}_probabilities"] = dict(selected_probabilities)

    topology_rng = spawn_rng(int(instance_seed), f"{plan.public_id}.topology")
    topology_profile, topology_probabilities = resolve_graph_named_variant(
        topology_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="topology_profile",
        weights_key="topology_profile_weights",
        balance_flag_key="balanced_topology_profile_sampling",
        supported=SUPPORTED_TOPOLOGY_PROFILES,
        instance_seed=int(instance_seed),
        task_id=str(plan.public_id),
        namespace="topology_profile",
    )
    visual_axes = resolve_node_link_visual_axes(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selection_salt=str(plan.public_id),
    )
    return NodeLinkAxes(
        query_id=str(selected_query),
        node_count=int(node_count),
        values=values,
        topology_profile=str(topology_profile),
        layout_variant=str(visual_axes.layout_variant),
        label_variant=str(visual_axes.label_variant),
        node_shape_variant=str(visual_axes.node_shape_variant),
        layout_transform_variant=str(visual_axes.layout_transform_variant),
        edge_routing_variant=str(visual_axes.edge_routing_variant),
        node_color_name=str(visual_axes.node_color_name),
        probabilities={
            **probabilities,
            "topology_profile_probabilities": dict(topology_probabilities),
            "layout_variant_probabilities": dict(visual_axes.layout_variant_probabilities),
            "label_variant_probabilities": dict(visual_axes.label_variant_probabilities),
            "node_shape_variant_probabilities": dict(visual_axes.node_shape_variant_probabilities),
            "layout_transform_variant_probabilities": dict(visual_axes.layout_transform_variant_probabilities),
            "edge_routing_variant_probabilities": dict(visual_axes.edge_routing_variant_probabilities),
            "node_color_name_probabilities": dict(visual_axes.node_color_name_probabilities),
        },
    )


def run_node_link_plan(
    *,
    plan: NodeLinkObjectivePlan,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate one node-link task from public-owned objective hooks."""

    scene_id = SCENE_ID
    scene_id_defaults = {}
    from trace_tasks.core.scene_config import get_scene_defaults

    loaded_defaults = get_scene_defaults("graph", scene_id)
    if isinstance(loaded_defaults, Mapping):
        scene_id_defaults = dict(loaded_defaults)
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_id_defaults,
        task_id=str(plan.public_id),
    )
    if not prompt_defaults and str(plan.prompt_bundle_id):
        prompt_defaults = {
            "bundle_id": str(plan.prompt_bundle_id),
            "scene_key": str(plan.prompt_scene_key or "single_graph_counting"),
            "task_key": str(plan.prompt_task_key),
            "json_output_contract": 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.',
            "json_output_contract_answer_only": 'Use a valid JSON object with key "answer" for the final answer.',
            "object_description": "a labeled graph",
            "annotation_hint": 'set "annotation" to the requested pixel-space witnesses for the answer',
            "answer_hint": 'set "answer" to the requested value',
        }
    defaults = NodeLinkDefaults()
    axes = _resolve_axes(
        plan=plan,
        params=params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        instance_seed=int(instance_seed),
    )
    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(plan.public_id),
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        node_color_name=str(axes.node_color_name),
        node_shape_variant=str(axes.node_shape_variant),
        edge_routing_variant=str(axes.edge_routing_variant),
    )
    raw_edge_text_label_font_size_px = params.get(
        "edge_text_label_font_size_px",
        group_default(render_defaults, "edge_text_label_font_size_px", None),
    )
    edge_text_label_font_size_px = (
        int(raw_edge_text_label_font_size_px)
        if raw_edge_text_label_font_size_px is not None
        else None
    )
    resolved_edge_text_label_font_size_px = (
        int(edge_text_label_font_size_px)
        if edge_text_label_font_size_px is not None
        else max(14, int(render_params.label_font_size_px) - 4)
    )
    search_attempts = int(params.get("graph_search_attempts", group_default(gen_defaults, "graph_search_attempts", defaults.graph_search_attempts)))

    sample = None
    rendered_scene = None
    image = None
    background_meta: dict[str, Any] = {}
    post_noise_meta: dict[str, Any] = {}
    last_error: Exception | None = None
    graph_rng = spawn_rng(int(instance_seed), "graph_structure")
    for attempt in range(max(1, int(max_attempts))):
        try:
            sample = plan.sample_graph(graph_rng, axes, max(80, int(search_attempts) // max(1, int(max_attempts)) + 80))
            directionality = resolve_prompt_slot(plan.graph_directionality, axes)
            rendered = render_node_link_sample(
                sample=sample,
                layout_variant=str(axes.layout_variant),
                layout_transform_variant=str(axes.layout_transform_variant),
                render_params=render_params,
                layout_seed=int(instance_seed + attempt),
                directed=str(directionality) == "directed",
                params=params,
                instance_seed=int(instance_seed),
                scene_id=scene_id,
                strict_edge_label_placement=bool(plan.strict_edge_label_placement),
                edge_text_label_font_size_px=edge_text_label_font_size_px,
            )
            rendered_scene = rendered.rendered_scene
            image = rendered.image
            background_meta = dict(rendered.background_meta)
            post_noise_meta = dict(rendered.post_noise_meta)
            break
        except Exception as exc:  # pragma: no cover - retry loop depends on random graph feasibility
            last_error = exc
            continue
    else:
        raise RuntimeError(f"failed to generate node-link task instance for {plan.class_name}") from last_error

    answer_gt = TypedValue(
        type=str(plan.answer_type),
        value=answer_value(
            sample,
            answer_field=str(plan.answer_field),
            answer_type=str(plan.answer_type),
        ),
    )
    annotation_gt, projected_annotation, witness_symbolic = annotation_value(
        sample,
        rendered_scene,
        annotation_kind=str(plan.annotation_kind),
        annotation_field=str(plan.annotation_field),
    )
    directionality = resolve_prompt_slot(plan.graph_directionality, axes)
    object_description_key = resolve_prompt_slot(plan.object_description_key, axes)
    if (
        str(directionality) == "directed"
        and str(object_description_key) == "object_description_undirected"
        and "object_description_directed" in prompt_defaults
    ):
        object_description_key = "object_description_directed"
    if object_description_key not in prompt_defaults and "object_description" in prompt_defaults:
        object_description_key = "object_description"
    annotation_hint_key = resolve_prompt_slot(plan.annotation_hint_key, axes)
    query_annotation_hint_key = f"annotation_hint_{axes.query_id}"
    if annotation_hint_key not in prompt_defaults and query_annotation_hint_key in prompt_defaults:
        annotation_hint_key = query_annotation_hint_key
    if annotation_hint_key not in prompt_defaults and "annotation_hint" in prompt_defaults:
        annotation_hint_key = "annotation_hint"
    prompt_defaults_required = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(object_description_key),
            str(annotation_hint_key),
            str(plan.answer_hint_key),
        ),
        context=f"prompt defaults for {plan.public_id}",
    )
    prompt_json_example, prompt_json_example_answer_only = build_graph_prompt_json_examples(
        annotation_value=plan.annotation_example,
        answer_value=plan.answer_example,
    )
    query_edge = tuple(str(value) for value in getattr(sample, "query_edge", ("", ""))[:2])
    if len(query_edge) < 2:
        query_edge = ("", "")
    edit_edge = tuple(str(value) for value in getattr(sample, "edit_edge", ("", ""))[:2])
    if len(edit_edge) < 2:
        edit_edge = ("", "")
    target_color_name = str(getattr(sample, "target_color_name", axes.values.get("target_color_name", "")))
    source_color_name = str(getattr(sample, "source_color_name", axes.values.get("source_color_name", "")))
    runtime_slots = {
        "query_node_label": _prompt_label(sample, axes, getattr(sample, "query_label", "")),
        "query_label": _prompt_label(sample, axes, getattr(sample, "query_label", "")),
        "query_node_label_a": _prompt_label(sample, axes, getattr(sample, "query_label_a", "")),
        "query_node_label_b": _prompt_label(sample, axes, getattr(sample, "query_label_b", "")),
        "query_label_a": _prompt_label(sample, axes, getattr(sample, "query_label_a", "")),
        "query_label_b": _prompt_label(sample, axes, getattr(sample, "query_label_b", "")),
        "source_node_label": _prompt_label(sample, axes, getattr(sample, "source_label", "")),
        "target_node_label": _prompt_label(sample, axes, getattr(sample, "goal_label", "")),
        "source_label": _prompt_label(sample, axes, getattr(sample, "source_label", query_edge[0])),
        "target_label": _prompt_label(sample, axes, getattr(sample, "goal_label", query_edge[1])),
        "goal_label": _prompt_label(sample, axes, getattr(sample, "goal_label", "")),
        "edit_label_a": _prompt_label(sample, axes, edit_edge[0]),
        "edit_label_b": _prompt_label(sample, axes, edit_edge[1]),
        "orientation_start_label": _prompt_label(sample, axes, getattr(sample, "orientation_start_label", "")),
        "orientation_next_label": _prompt_label(sample, axes, getattr(sample, "orientation_next_label", "")),
        "query_degree": int(axes.values.get("query_degree", axes.values.get("target_degree", 0))),
        "target_color": _prompt_color_label(target_color_name),
        "target_color_label": _prompt_color_label(target_color_name),
        "target_color_name": target_color_name,
        "source_color": _prompt_color_label(source_color_name),
        "source_color_label": _prompt_color_label(source_color_name),
        "source_color_name": source_color_name,
        "target_edge_label": str(getattr(sample, "target_edge_label", "")),
        "path_edge_term": "arrow" if str(directionality) == "directed" else "edge",
        "path_follow_clause": " following arrow direction" if str(directionality) == "directed" else "",
        "orientation_final_label": _prompt_label(sample, axes, getattr(sample, "orientation_final_label", "")),
    }
    prompt_selection = render_scene_prompt_variants(
        domain="graph",
        scene_id=scene_id,
        bundle_id=str(prompt_defaults_required["bundle_id"]),
        scene_key=str(prompt_defaults_required["scene_key"]),
        task_key=str(prompt_defaults_required["task_key"]),
        query_key=resolve_prompt_slot(plan.prompt_query_key, axes),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": _format_prompt_default(prompt_defaults_required[str(object_description_key)], runtime_slots),
            "json_output_contract": str(prompt_defaults_required["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults_required["json_output_contract_answer_only"]),
            "annotation_hint": _format_prompt_default(prompt_defaults_required[str(annotation_hint_key)], runtime_slots),
            "answer_hint": _format_prompt_default(prompt_defaults_required[str(plan.answer_hint_key)], runtime_slots),
            "json_example": str(prompt_json_example),
            "json_example_answer_only": str(prompt_json_example_answer_only),
            **runtime_slots,
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    realized_query_values: dict[str, Any] = {}
    for key in (
        "target_degree",
        "target_count",
        "query_degree",
        "target_position",
        "target_reachable_count",
        "target_component_size",
        "component_count",
        "target_shortest_path_length",
        "target_cycle_size",
    ):
        if hasattr(sample, key):
            realized_query_values[str(key)] = int(getattr(sample, key))
    for key in ("source_color_name", "target_color_name"):
        if hasattr(sample, key):
            realized_query_values[str(key)] = str(getattr(sample, key))
    realized_execution_values: dict[str, Any] = {}
    for key in ("target_degree", "target_count", "query_degree", "target_position", "target_reachable_count", "target_shortest_path_length", "target_component_size", "component_count"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = int(getattr(sample, key))
    for key in ("target_cycle_size", "attachment_count", "extra_edge_count"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = int(getattr(sample, key))
    for key in (
        "answer_label",
        "degree_mode",
        "extremum_mode",
        "orientation_final_label",
        "orientation_next_label",
        "orientation_start_label",
        "query_label",
        "relation_mode",
        "source_color_name",
        "source_label",
        "target_color_name",
        "target_edge_label",
        "answer_label",
        "goal_label",
        "removed_node_label",
        "graph_directionality",
        "topology_profile",
    ):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = str(getattr(sample, key))
    for key in ("adjacency_by_label", "successors_by_label", "predecessors_by_label"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = {
                str(item_key): [str(value) for value in values]
                for item_key, values in getattr(sample, key).items()
            }
    for key in (
        "pre_removal_adjacency_by_label",
        "post_removal_adjacency_by_label",
        "pre_removal_successors_by_label",
        "post_removal_successors_by_label",
        "pre_removal_predecessors_by_label",
        "post_removal_predecessors_by_label",
    ):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = {
                str(item_key): [str(value) for value in values]
                for item_key, values in getattr(sample, key).items()
            }
    for key in (
        "pre_removal_degrees_by_label",
        "post_removal_degrees_by_label",
        "pre_removal_in_degrees_by_label",
        "post_removal_in_degrees_by_label",
        "pre_removal_out_degrees_by_label",
        "post_removal_out_degrees_by_label",
    ):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = {
                str(item_key): int(value)
                for item_key, value in getattr(sample, key).items()
            }
    for key in ("post_removal_edge_labels",):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = [
                [str(left), str(right)]
                for left, right in getattr(sample, key)
            ]
    for key in ("queried_degrees_by_label", "node_color_names_by_label"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = {
                str(item_key): str(item_value) if key == "node_color_names_by_label" else int(item_value)
                for item_key, item_value in getattr(sample, key).items()
            }
    for key in ("chordless_cycle_sizes", "chordless_cycle_labels"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = [
                list(item) if isinstance(item, tuple) else int(item)
                for item in getattr(sample, key)
            ]
    for key in ("annotation_labels", "target_labels"):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = [
                str(item)
                for item in getattr(sample, key)
            ]
    if hasattr(sample, "target_shortest_path_length") and hasattr(sample, "target_labels"):
        realized_execution_values["shortest_path_labels"] = [
            str(item)
            for item in getattr(sample, "target_labels")
        ]
    if hasattr(sample, "target_position") and hasattr(sample, "target_labels"):
        realized_execution_values["topological_order_labels"] = [
            str(item)
            for item in getattr(sample, "target_labels")
        ]
    if hasattr(sample, "color_counts_by_name"):
        realized_execution_values["color_counts_by_name"] = {
            str(key): int(value)
            for key, value in getattr(sample, "color_counts_by_name").items()
        }
    if hasattr(sample, "edge_label_counts_by_value"):
        realized_execution_values["edge_label_counts_by_value"] = {
            str(key): int(value)
            for key, value in getattr(sample, "edge_label_counts_by_value").items()
        }
    if hasattr(sample, "edge_attribute_labels_by_label"):
        edge_label_entries = list(graph_edge_label_entries(getattr(sample, "edge_attribute_labels_by_label")))
        realized_execution_values["edge_attribute_labels_by_label_pair"] = edge_label_entries
    if hasattr(sample, "edge_label_support"):
        realized_execution_values["edge_label_support"] = [
            str(label)
            for label in getattr(sample, "edge_label_support")
        ]
    for key in (
        "edge_label_source_kind",
        "edge_label_bucket",
        "edge_label_manifest",
    ):
        if hasattr(sample, key):
            realized_execution_values[str(key)] = str(getattr(sample, key))
    if hasattr(sample, "edge_label_filter"):
        realized_execution_values["edge_label_filter"] = dict(getattr(sample, "edge_label_filter"))
    if hasattr(sample, "edge_label_bucket_probabilities"):
        realized_execution_values["edge_label_bucket_probabilities"] = {
            str(key): float(value)
            for key, value in getattr(sample, "edge_label_bucket_probabilities").items()
        }
    realized_node_count = len(tuple(getattr(sample, "node_labels", ()))) or int(axes.node_count)
    query_params = {
        "query_id": str(axes.query_id),
        "graph_directionality": str(directionality),
        "node_count": int(realized_node_count),
        "edge_count": int(getattr(sample, "edge_count", 0)),
        **{str(key): value for key, value in axes.values.items()},
        **realized_query_values,
        **dict(axes.probabilities),
    }
    if int(realized_node_count) != int(axes.node_count):
        query_params["requested_node_count"] = int(axes.node_count)
    trace_payload = {
        "scene_ir": {
            "scene_kind": str(plan.scene_kind),
            "scene_id": scene_id,
            "entities": [*node_entities(rendered_scene, sample), *edge_entities(rendered_scene, sample)],
            "relations": {
                "graph_directionality": str(directionality),
                "adjacency_by_label": {str(key): list(values) for key, values in getattr(sample, "adjacency_by_label", {}).items()},
                "successors_by_label": {str(key): list(values) for key, values in getattr(sample, "successors_by_label", {}).items()},
                "predecessors_by_label": {str(key): list(values) for key, values in getattr(sample, "predecessors_by_label", {}).items()},
                "edge_labels": [list(edge) for edge in getattr(sample, "edge_labels", ())],
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(rendered_scene.panel_geometry),
            },
        },
        "query_spec": {
            "query_id": str(axes.query_id),
            "template_id": str(prompt_defaults_required["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": {
            "canvas_size": list(rendered_scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "panel_geometry": dict(rendered_scene.panel_geometry),
            "style": {
                "node_color_name": str(axes.node_color_name),
                "node_shape_variant": str(render_params.node_shape_variant),
                "edge_routing_variant": str(rendered_scene.edge_routing_variant),
                "theme_tone": str(render_params.theme_tone),
                "panel_style_variant": str(render_params.panel_style_variant),
                "node_fill_rgb": list(render_params.node_fill_rgb),
                "node_border_rgb": list(render_params.node_border_rgb),
                "edge_color_rgb": list(render_params.edge_color_rgb),
                "label_font_size_px": int(render_params.label_font_size_px),
                "edge_text_label_font_size_px": (
                    int(edge_text_label_font_size_px)
                    if edge_text_label_font_size_px is not None
                    else None
                ),
                "resolved_edge_text_label_font_size_px": int(resolved_edge_text_label_font_size_px),
                "background_meta": dict(background_meta),
                "post_image_noise_meta": dict(post_noise_meta),
            },
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "query_id": str(axes.query_id),
            "scene_id": scene_id,
            "question_format": resolve_prompt_slot(plan.question_format, axes) or str(axes.query_id),
            "graph_directionality": str(directionality),
            "node_count": int(realized_node_count),
            "edge_count": int(getattr(sample, "edge_count", 0)),
            "layout_variant_used": str(rendered_scene.layout_variant),
            "layout_variant_requested": str(axes.layout_variant),
            "layout_transform_variant": str(rendered_scene.layout_transform_variant),
            "edge_routing_variant": str(rendered_scene.edge_routing_variant),
            "node_shape_variant": str(axes.node_shape_variant),
            "node_color_name": str(axes.node_color_name),
            "label_variant": str(getattr(sample, "label_variant", axes.label_variant)),
            "matching_labels": list(getattr(sample, "annotation_labels", ()) or getattr(sample, "target_labels", ())),
            "matching_edges": [list(edge) for edge in getattr(sample, "target_edges", ())],
            **realized_execution_values,
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }
    bundle = NodeLinkTaskBundle(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        query_id=str(axes.query_id),
    )
    return task_output_from_bundle(
        bundle,
        public_task_id=str(plan.public_id),
        scene_id=scene_id,
        query_id=str(axes.query_id),
    )


def rewrite_node_link_bundle_query_id(
    bundle: NodeLinkTaskBundle,
    *,
    query_id: str,
    query_id_probabilities: Mapping[str, float] | None = None,
) -> NodeLinkTaskBundle:
    """Return a copy of ``bundle`` with query metadata set consistently."""

    trace_payload = dict(bundle.trace_payload)
    query_spec = dict(trace_payload.get("query_spec") or {})
    query_spec["query_id"] = str(query_id)
    params = dict(query_spec.get("params") or {})
    params["query_id"] = str(query_id)
    if query_id_probabilities is not None:
        params["query_id_probabilities"] = {
            str(key): float(value)
            for key, value in query_id_probabilities.items()
        }
    query_spec["params"] = params
    trace_payload["query_spec"] = query_spec
    execution_trace = dict(trace_payload.get("execution_trace") or {})
    execution_trace["query_id"] = str(query_id)
    trace_payload["execution_trace"] = execution_trace
    return NodeLinkTaskBundle(
        prompt=str(bundle.prompt),
        answer_gt=bundle.answer_gt,
        annotation_gt=bundle.annotation_gt,
        image=bundle.image,
        image_id=str(bundle.image_id),
        trace_payload=trace_payload,
        prompt_variants=dict(bundle.prompt_variants),
        scene_id=str(bundle.scene_id),
        query_id=str(query_id),
    )


def bundle_query_id(bundle: NodeLinkTaskBundle, *, fallback_query_id: str) -> str:
    """Return the query id selected during task-owned sampling."""

    selected_key = str(getattr(bundle, "query_id", ""))
    if selected_key:
        return selected_key
    trace = bundle.trace_payload if isinstance(bundle.trace_payload, Mapping) else {}
    query_spec = trace.get("query_spec") if isinstance(trace, Mapping) else {}
    if isinstance(query_spec, Mapping) and str(query_spec.get("query_id", "")):
        return str(query_spec["query_id"])
    execution_trace = trace.get("execution_trace") if isinstance(trace, Mapping) else {}
    if isinstance(execution_trace, Mapping) and str(execution_trace.get("query_id", "")):
        return str(execution_trace["query_id"])
    return str(fallback_query_id)


def task_output_from_bundle(
    bundle: NodeLinkTaskBundle,
    *,
    public_task_id: str,
    scene_id: str,
    query_id: str,
) -> TaskOutput:
    """Temporary final TaskOutput assembly until public node-link tasks own it."""

    answer_gt = TypedValue(type=str(bundle.answer_gt.type), value=bundle.answer_gt.value)
    annotation_gt = TypedValue(type=str(bundle.annotation_gt.type), value=bundle.annotation_gt.value)
    return TaskOutput(
        prompt=str(bundle.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=bundle.image,
        image_id=str(bundle.image_id),
        trace_payload=_trace_with_public_task_id(bundle.trace_payload, public_task_id=str(public_task_id)),
        task_versions=default_task_versions(),
        scene_id=str(scene_id),
        query_id=str(query_id),
        prompt_variants=dict(bundle.prompt_variants),
    )
