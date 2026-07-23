"""Private neutral lifecycle helpers for radial Sankey public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import annotation_payload
from .shared.defaults import font_assets_payload
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_radial_sankey_dataset
from .shared.sampling import link_dict, node_dict
from .shared.state import RadialSankeyDataset, RadialSankeyRenderResult, SCENE_ID


@dataclass(frozen=True)
class RadialSankeyTaskPlan:
    dataset: RadialSankeyDataset
    prompt_key: str
    question_format: str
    witness_type: str
    trace_params: dict[str, Any]


def attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_radial_sankey_plan(
    *,
    dataset: RadialSankeyDataset,
    prompt_key: str,
    question_format: str,
    witness_type: str,
    trace_params: Mapping[str, Any] | None = None,
) -> RadialSankeyTaskPlan:
    return RadialSankeyTaskPlan(
        dataset=dataset,
        prompt_key=str(prompt_key),
        question_format=str(question_format),
        witness_type=str(witness_type),
        trace_params=dict(trace_params or {}),
    )


def _trace_payload(
    *,
    plan: RadialSankeyTaskPlan,
    rendered: RadialSankeyRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble trace metadata after a public task has selected operands and witnesses."""

    dataset = plan.dataset
    frame = dataset.frame
    question = dataset.question
    rendered_scene = rendered.rendered_scene
    render_params = rendered.render_params
    sources = [node_dict(node) for node in frame.sources]
    targets = [node_dict(node) for node in frame.targets]
    links = [link_dict(link) for link in frame.links]
    link_lookup = {str(link["link_id"]): dict(link) for link in links}
    question_params = {
        "query_id": str(question.branch_id),
        "query_id_probabilities": dict(question.branch_probabilities),
        "scene_variant": str(frame.scene_variant),
        "scene_variant_probabilities": dict(frame.scene_probabilities),
        "source_count": int(len(frame.sources)),
        "target_count": int(len(frame.targets)),
        "link_count": int(len(frame.links)),
        "max_links_per_node_side": int(frame.max_links_per_node_side),
        "link_side_counts": dict(frame.link_side_counts),
        "annotation_link_ids": list(annotation["link_refs"]),
        "annotation_node_ids": list(annotation["node_refs"]),
        **dict(question.params),
        **dict(plan.trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": "chart_radial_sankey",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(question.branch_id),
                "scene_variant": str(frame.scene_variant),
                "answer_value": question.answer,
                "annotation_type": str(annotation["type"]),
                "annotation_link_ids": list(annotation["link_refs"]),
                "annotation_node_ids": list(annotation["node_refs"]),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(question.branch_id),
            params=dict(question_params),
        ),
        "render_spec": {
            "scene_variant": str(frame.scene_variant),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "source_count": int(len(frame.sources)),
            "target_count": int(len(frame.targets)),
            "link_count": int(len(frame.links)),
            "max_links_per_node_side": int(frame.max_links_per_node_side),
            "value_min": int(frame.value_min),
            "value_max": int(frame.value_max),
            "min_flow_width_px": int(render_params.min_flow_width_px),
            "max_flow_width_px": int(render_params.max_flow_width_px),
            "flow_alpha": int(render_params.flow_alpha),
            "radial_color_scheme": str(render_params.color_scheme_name),
            "flow_palette_rgb": [list(color) for color in render_params.flow_palette_rgb],
            "source_node_fill_rgb": list(render_params.source_node_fill_rgb),
            "target_node_fill_rgb": list(render_params.target_node_fill_rgb),
            "ring_line_rgb": list(render_params.ring_line_rgb),
            "layout_jitter": dict(render_params.layout_jitter_meta),
            "background_style": dict(rendered.background_meta),
            "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
            "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered_scene.panel_bbox_px),
            "title_bbox_px": list(rendered_scene.title_bbox_px),
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "node_bboxes_px": dict(rendered_scene.node_bbox_map),
            "node_label_bboxes_px": dict(rendered_scene.node_label_bbox_map),
            "link_bboxes_px": dict(rendered_scene.link_bbox_map),
            "link_label_bboxes_px": dict(rendered_scene.link_label_bbox_map),
            "link_centers_px": dict(rendered_scene.link_center_map),
        },
        "execution_trace": {
            "query_id": str(question.branch_id),
            "scene_variant": str(frame.scene_variant),
            "query_id_probabilities": dict(question.branch_probabilities),
            "scene_variant_probabilities": dict(frame.scene_probabilities),
            "question_format": str(plan.question_format),
            "scene_title": str(frame.scene_title),
            "sources": list(sources),
            "targets": list(targets),
            "links": list(links),
            "links_by_id": dict(link_lookup),
            "source_count": int(len(frame.sources)),
            "target_count": int(len(frame.targets)),
            "link_count": int(len(frame.links)),
            "max_links_per_node_side": int(frame.max_links_per_node_side),
            "link_side_counts": dict(frame.link_side_counts),
            "value_min": int(frame.value_min),
            "value_max": int(frame.value_max),
            "answer_value": question.answer,
            "answer_type": str(question.answer_type),
            "annotation_type": str(annotation["type"]),
            "annotation_link_ids": list(annotation["link_refs"]),
            "annotation_node_ids": list(annotation["node_refs"]),
            **dict(question.params),
            **dict(plan.trace_params),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "answer_value": question.answer,
            "annotation_type": str(annotation["type"]),
            "annotation_link_ids": list(annotation["link_refs"]),
            "annotation_node_ids": list(annotation["node_refs"]),
            **dict(question.params),
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_radial_sankey_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: RadialSankeyTaskPlan,
) -> TaskOutput:
    rendered = render_radial_sankey_dataset(dataset=plan.dataset, params=dict(params), instance_seed=int(instance_seed))
    annotation = annotation_payload(dataset=plan.dataset, rendered=rendered)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(plan.prompt_key),
        dynamic_slot_values=dynamic_slots(dataset=plan.dataset),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        plan=plan,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation=annotation,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(plan.dataset.question.answer_type), value=plan.dataset.question.answer),
        annotation_gt=TypedValue(type=str(annotation["type"]), value=annotation["value"]),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.question.branch_id),
    )


def run_radial_sankey_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        current_seed = attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            plan = task._build_plan(
                dict(task_params),
                int(current_seed),
                str(selected),
                dict(probabilities),
            )
            return materialize_radial_sankey_plan(
                params=dict(task_params),
                instance_seed=int(current_seed),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate radial Sankey task: {last_error}") from last_error


__all__ = [
    "RadialSankeyTaskPlan",
    "build_radial_sankey_plan",
    "materialize_radial_sankey_plan",
    "run_radial_sankey_task",
]
