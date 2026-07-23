"""Scene-private lifecycle orchestration for automaton acceptance tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ..shared.task_support import resolve_graph_render_params
from .shared.annotations import projected_state_path_points
from .shared.prompts import build_automaton_prompt_artifacts
from .shared.rendering import render_acceptance_scene
from .shared.sampling import AcceptanceDefaults, resolve_acceptance_axes, sample_acceptance_automaton
from .shared.state import AcceptanceAxes, AcceptanceRender, AcceptanceSample, SCENE_ID


ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, float], str], "AcceptanceObjectivePlan"]


@dataclass(frozen=True)
class AcceptanceObjectivePlan:
    """Task-owned semantic settings for one acceptance objective."""

    automaton_kind: str
    task_prompt_key: str
    object_description: str
    acceptance_rule: str


def _transition_entries(sample: AcceptanceSample) -> list[dict[str, Any]]:
    """Return trace-friendly transition records for one automaton sample."""

    return [
        {
            "source_state": str(left),
            "target_state": str(right),
            "symbols": str(symbols).split(","),
        }
        for (left, right), symbols in sample.transition_labels_by_edge.items()
    ]


def _acceptance_trace_payload(
    *,
    task_id: str,
    query_id: str,
    query_probabilities: Mapping[str, float],
    axes: AcceptanceAxes,
    sample: AcceptanceSample,
    rendered: AcceptanceRender,
    annotation_path: list[list[int]],
    annotation_projection: Mapping[str, Any],
    prompt_bundle_id: str,
    prompt_artifacts: Any,
    acceptance_rule: str,
) -> dict[str, Any]:
    """Build trace metadata after task-owned answer and annotation binding."""

    render_params = rendered.render_params
    rendered_scene = rendered.rendered_scene
    image = rendered.image
    option_bboxes = dict(rendered.option_bboxes)
    path_label_set = set(str(label) for label in sample.accepting_path_labels)
    node_entities = [
        {
            "entity_id": f"state_{node.label}",
            "entity_kind": "automaton_state",
            "label": str(node.label),
            "is_start_state": bool(str(node.label) == str(sample.start_label)),
            "is_accepting_state": bool(str(node.label) in set(sample.accepting_labels)),
            "is_in_annotation_path": bool(str(node.label) in path_label_set),
            "path_positions": [int(index) for index, label in enumerate(sample.accepting_path_labels) if str(label) == str(node.label)],
            "center_px": list(node.center_xy),
            "bbox_xyxy": list(node.bbox_xyxy),
            "successors": list(sample.graph_sample.successors_by_label[str(node.label)]),
            "predecessors": list(sample.graph_sample.predecessors_by_label[str(node.label)]),
        }
        for node in rendered_scene.nodes
    ]
    edge_entities = [
        {
            "entity_id": str(edge.edge_id),
            "entity_kind": "automaton_transition",
            "source_state_label": str(edge.node_u_label),
            "target_state_label": str(edge.node_v_label),
            "transition_symbols": str(sample.transition_labels_by_edge[(str(edge.node_u_label), str(edge.node_v_label))]).split(","),
            "segment_px": [list(edge.segment_px[0]), list(edge.segment_px[1])],
            "route_variant": str(edge.route_variant),
            "control_px": list(edge.control_px) if edge.control_px is not None else None,
            "edge_label_bbox_xyxy": list(edge.edge_label_bbox_xyxy) if edge.edge_label_bbox_xyxy is not None else None,
        }
        for edge in rendered_scene.edges
    ]
    option_entities = [
        {
            "entity_id": f"candidate_option_{option_label}",
            "entity_kind": "candidate_input_string",
            "option_label": str(option_label),
            "input_string": str(input_string),
            "is_answer": bool(str(option_label) == str(sample.answer_option_label)),
            "is_accepted": bool(str(option_label) in set(sample.accepted_option_labels)),
            "bbox_xyxy": list(option_bboxes[str(option_label)]),
        }
        for option_label, input_string in sample.candidate_strings_by_option.items()
    ]
    transition_entries = _transition_entries(sample)
    panel_geometry = dict(rendered_scene.panel_geometry)
    panel_geometry["canvas_size"] = [int(image.width), int(image.height)]
    panel_geometry.update(dict(rendered.option_panel_meta))
    transition_function_json = {
        str(state): {str(symbol): list(targets) for symbol, targets in per_symbol.items()}
        for state, per_symbol in sample.transition_function.items()
    }
    return {
        "scene_ir": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "scene_kind": "automaton_string_acceptance",
            "entities": [*node_entities, *edge_entities, *option_entities],
            "relations": {
                "query_id": str(query_id),
                "automaton_kind": str(axes.automaton_kind),
                "acceptance_rule": str(acceptance_rule),
                "start_state_label": str(sample.start_label),
                "accepting_state_labels": list(sample.accepting_labels),
                "candidate_strings_by_option": dict(sample.candidate_strings_by_option),
                "answer_option_label": str(sample.answer_option_label),
                "answer_input_string": str(sample.answer_input_string),
                "accepted_option_labels": list(sample.accepted_option_labels),
                "accepting_path_labels": list(sample.accepting_path_labels),
                "transition_function": dict(transition_function_json),
                "transition_labels_by_edge": list(transition_entries),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(panel_geometry),
            },
        },
        "query_spec": {
            "task_id": str(task_id),
            "query_id": str(query_id),
            "template_id": str(prompt_bundle_id),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "query_id": str(query_id),
                "query_id_probabilities": dict(query_probabilities),
                "automaton_kind": str(axes.automaton_kind),
                "state_count": int(axes.state_count),
                "state_count_probabilities": dict(axes.state_count_probabilities),
                "input_length": int(axes.input_length),
                "input_length_min": int(axes.input_length_min),
                "input_length_max": int(axes.input_length_max),
                "input_length_probabilities": dict(axes.input_length_probabilities),
                "candidate_count": int(axes.candidate_count),
                "candidate_count_support": [int(value) for value in axes.candidate_count_support],
                "candidate_count_probabilities": dict(axes.candidate_count_probabilities),
                "answer_option_label": str(sample.answer_option_label),
                "answer_option_probabilities": dict(axes.answer_option_probabilities),
                "answer_input_string": str(sample.answer_input_string),
                "start_state_label": str(sample.start_label),
                "accepting_state_labels": list(sample.accepting_labels),
                "layout_variant": str(axes.layout_variant),
                "layout_variant_probabilities": dict(axes.layout_variant_probabilities),
                "layout_transform_variant": str(axes.layout_transform_variant),
                "layout_transform_variant_probabilities": dict(axes.layout_transform_variant_probabilities),
                "edge_routing_variant": str(axes.edge_routing_variant),
                "edge_routing_variant_probabilities": dict(axes.edge_routing_variant_probabilities),
                "node_color_name": str(axes.node_color_name),
                "node_color_name_probabilities": dict(axes.node_color_name_probabilities),
            },
        },
        "render_spec": {
            "canvas_size": [int(image.width), int(image.height)],
            "coord_space": "pixel",
            "panel_geometry": dict(panel_geometry),
            "style": {
                "node_color_name": str(axes.node_color_name),
                "theme_tone": str(render_params.theme_tone),
                "panel_style_variant": str(render_params.panel_style_variant),
                "background_color_rgb": list(render_params.background_color_rgb),
                "panel_fill_rgb": list(render_params.panel_fill_rgb),
                "panel_border_rgb": list(render_params.panel_border_rgb),
                "title_color_rgb": list(render_params.title_color_rgb),
                "edge_color_rgb": list(render_params.edge_color_rgb),
                "node_fill_rgb": list(render_params.node_fill_rgb),
                "node_border_rgb": list(render_params.node_border_rgb),
                "label_text_rgb": list(render_params.label_text_rgb),
                "label_stroke_rgb": list(render_params.label_stroke_rgb),
                "node_shape_variant": "circle",
                "node_radius_px": int(render_params.node_radius_px),
                "edge_width_px": int(render_params.edge_width_px),
                "edge_routing_variant": str(rendered_scene.edge_routing_variant),
                "arrow_length_px": int(render_params.arrow_length_px),
                "arrow_width_px": int(render_params.arrow_width_px),
                "node_border_width_px": int(render_params.node_border_width_px),
                "label_font_size_px": int(render_params.label_font_size_px),
                "resolved_label_font_size_px": int(rendered_scene.resolved_label_font_size_px),
                "label_stroke_width_px": int(rendered_scene.resolved_label_stroke_width_px),
                "font_family": str(render_params.font_family or ""),
                "font_asset": dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {},
                "font_asset_version": str(render_params.font_asset_version or ""),
                "font_exclusion_reason": str(render_params.font_exclusion_reason),
                "context_text_elements": list(panel_geometry.get("context_text_elements", [])),
                "automaton_accepting_state_glyph": "double_inner_ring",
                "automaton_start_state_glyph": "incoming_start_arrow",
                "transition_labels_by_edge": list(transition_entries),
                "candidate_option_bboxes_xyxy": dict(option_bboxes),
                "background_meta": dict(rendered.background_meta),
                "post_image_noise_meta": dict(rendered.post_noise_meta),
            },
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "query_id_probabilities": dict(query_probabilities),
            "scene_variant": str(rendered_scene.layout_variant),
            "automaton_kind": str(axes.automaton_kind),
            "state_count": int(axes.state_count),
            "edge_count": int(sample.graph_sample.edge_count),
            "input_length": int(axes.input_length),
            "candidate_count": int(axes.candidate_count),
            "candidate_count_support": [int(value) for value in axes.candidate_count_support],
            "candidate_count_probabilities": dict(axes.candidate_count_probabilities),
            "answer": str(sample.answer_option_label),
            "answer_option_label": str(sample.answer_option_label),
            "answer_input_string": str(sample.answer_input_string),
            "candidate_strings_by_option": dict(sample.candidate_strings_by_option),
            "accepted_option_labels": list(sample.accepted_option_labels),
            "accepting_path_labels": list(sample.accepting_path_labels),
            "start_state_label": str(sample.start_label),
            "accepting_state_labels": list(sample.accepting_labels),
            "transition_function": dict(transition_function_json),
            "transition_labels_by_edge": list(transition_entries),
            "option_bboxes": dict(option_bboxes),
            "layout_variant_requested": str(axes.layout_variant),
            "layout_variant_used": str(rendered_scene.layout_variant),
            "layout_transform_variant": str(rendered_scene.layout_transform_variant),
            "edge_routing_variant": str(rendered_scene.edge_routing_variant),
            "node_color_name": str(axes.node_color_name),
            "crossing_count": int(rendered_scene.crossing_count),
        },
        "witness_symbolic": {
            "type": "automaton_accepting_path",
            "answer_option_label": str(sample.answer_option_label),
            "answer_input_string": str(sample.answer_input_string),
            "state_path_labels": list(sample.accepting_path_labels),
        },
        "projected_annotation": {
            "type": "point_sequence",
            "point_sequence": list(annotation_path),
            "pixel_point_sequence": list(annotation_path),
            "pixel_bbox_set": list(annotation_projection["pixel_bbox_set"]),
        },
    }


def run_acceptance_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_bundle_id: str,
    post_image_background_defaults: Mapping[str, Any],
    post_image_noise_defaults: Mapping[str, Any],
    defaults: AcceptanceDefaults,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common acceptance-task query, render, prompt, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    objective = prepare_objective(int(instance_seed), task_params, query_probabilities, query_id)
    axes = resolve_acceptance_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        automaton_kind=str(objective.automaton_kind),
    )
    render_params = resolve_graph_render_params(
        task_params,
        instance_seed=int(instance_seed),
        task_id=str(task_id),
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        node_color_name=str(axes.node_color_name),
        node_shape_variant="circle",
        edge_routing_variant=str(axes.edge_routing_variant),
    )
    graph_rng = spawn_rng(int(instance_seed), f"{task_id}.graph")
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            sample = sample_acceptance_automaton(graph_rng, axes=axes)
            rendered = render_acceptance_scene(
                instance_seed=int(instance_seed),
                params=task_params,
                render_defaults=render_defaults,
                post_image_background_defaults=post_image_background_defaults,
                post_image_noise_defaults=post_image_noise_defaults,
                axes=axes,
                sample=sample,
                render_params=render_params,
                option_panel_height_fallback=int(defaults.option_panel_height_px),
                layout_seed=int(instance_seed + attempt),
            )
            annotation_path, annotation_projection = projected_state_path_points(
                rendered.rendered_scene,
                sample.accepting_path_labels,
            )
            break
        except Exception as exc:  # pragma: no cover - exercised by stochastic retry
            last_error = exc
            continue
    else:
        raise RuntimeError("failed to generate automaton acceptance instance") from last_error

    answer_gt = TypedValue(type="string", value=str(sample.answer_option_label))
    annotation_gt = TypedValue(type="point_sequence", value=list(annotation_path))
    prompt_artifacts = build_automaton_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_bundle_id),
        task_key=str(objective.task_prompt_key),
        prompt_query_key=str(query_id),
        dynamic_slots={"object_description": str(objective.object_description)},
        instance_seed=int(instance_seed),
    )

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=_acceptance_trace_payload(
            task_id=str(task_id),
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            axes=axes,
            sample=sample,
            rendered=rendered,
            annotation_path=annotation_path,
            annotation_projection=annotation_projection,
            prompt_bundle_id=str(prompt_bundle_id),
            prompt_artifacts=prompt_artifacts,
            acceptance_rule=str(objective.acceptance_rule),
        ),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "AcceptanceObjectivePlan",
    "run_acceptance_lifecycle",
]
