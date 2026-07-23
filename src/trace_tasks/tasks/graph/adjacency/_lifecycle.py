"""Scene-private artifact preparation for adjacency-scene task families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import component_topmost_row_labels, row_label_bbox_set_artifacts
from .shared.output import (
    component_edge_entities,
    component_node_entities,
    execution_trace_body,
    label_query_params,
    pixel_panel_frames,
    query_spec_body,
    render_spec_fragment,
    scene_ir_body,
)
from .shared.prompts import build_adjacency_prompt_artifacts
from .shared.rendering import render_component_adjacency_panel
from .shared.sampling import (
    ComponentCountAxes,
    ComponentCountDefaults,
    resolve_component_count_axes,
    resolve_adjacency_labels,
    sample_component_adjacency,
)
from .shared.state import SCENE_ID, AdjacencyGraphSample, AdjacencyLabelSet, AdjacencyRepresentationRender
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts


@dataclass(frozen=True)
class ComponentCountArtifacts:
    """Prepared scene artifacts for one component-count objective."""

    labels: AdjacencyLabelSet
    sample: AdjacencyGraphSample
    rendered: AdjacencyRepresentationRender
    image: Any
    canvas_width: int
    canvas_height: int
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
    topmost_row_labels: tuple[str, ...]
    annotation_artifacts: AnnotationArtifacts
    prompt_artifacts: PromptTraceArtifacts


@dataclass(frozen=True)
class ComponentCountPlan:
    """Task-owned semantic settings for one component-count objective."""

    directed: bool
    object_description: str


@dataclass(frozen=True)
class SinglePanelRenderArtifacts:
    """Rendered image artifacts for one non-component adjacency task."""

    rendered: AdjacencyRepresentationRender
    image: Any
    canvas_width: int
    canvas_height: int
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


def render_single_panel_artifacts(
    *,
    task_params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    defaults: Any,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    render_panel: Any,
) -> SinglePanelRenderArtifacts:
    """Render one adjacency panel on a styled canvas and apply post-image noise."""

    canvas_width = int(task_params.get("canvas_width", group_default(render_defaults, "canvas_width", defaults.canvas_width)))
    canvas_height = int(task_params.get("canvas_height", group_default(render_defaults, "canvas_height", defaults.canvas_height)))
    base_image, background_meta = make_background_canvas(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=background_defaults,
    )
    rendered = render_panel(base_image)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=noise_defaults,
    )
    return SinglePanelRenderArtifacts(
        rendered=rendered,
        image=image,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def single_panel_render_kwargs(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    defaults: Any,
    instance_seed: int,
) -> dict[str, Any]:
    """Return common typography and context controls for adjacency panels."""

    return {
        "font_size_px": int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", defaults.label_font_size_px))),
        "layout_seed": int(instance_seed),
        "font_family": params.get("font_family"),
        "context_text_probability": float(params.get("context_text_probability", group_default(render_defaults, "context_text_probability", 0.35))),
    }


def single_panel_trace_payload(
    *,
    task_id: str,
    scene_id: str,
    query_id: str,
    prompt_bundle_id: str,
    prompt_artifacts: PromptTraceArtifacts,
    render_artifacts: SinglePanelRenderArtifacts,
    entities: list[dict[str, Any]],
    relations: Mapping[str, Any],
    query_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble common single-panel trace payload fields."""

    rendered = render_artifacts.rendered
    return {
        "scene_ir": {
            "task_id": str(task_id),
            "scene_id": str(scene_id),
            **scene_ir_body(rendered=rendered, entities=entities, relations=relations),
        },
        "query_spec": {
            "task_id": str(task_id),
            "query_id": str(query_id),
            **query_spec_body(
                prompt_bundle_id=str(prompt_bundle_id),
                prompt_artifacts=prompt_artifacts,
                params=query_params,
            ),
        },
        "render_spec": render_spec_fragment(
            canvas_width=int(render_artifacts.canvas_width),
            canvas_height=int(render_artifacts.canvas_height),
            rendered=rendered,
            background_meta=render_artifacts.background_meta,
            post_noise_meta=render_artifacts.post_noise_meta,
        ),
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": str(scene_id),
            "query_id": str(query_id),
            **execution_trace_body(rendered=rendered, fields=execution_fields),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


def single_panel_task_output(
    *,
    prompt_artifacts: PromptTraceArtifacts,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    render_artifacts: SinglePanelRenderArtifacts,
    trace_payload: Mapping[str, Any],
    scene_id: str,
    query_id: str,
) -> TaskOutput:
    """Build the final TaskOutput after the public task binds answer and annotation."""

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=render_artifacts.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=str(scene_id),
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def prepare_component_count_artifacts(
    *,
    task_id: str,
    domain: str,
    query_id: str,
    prompt_key: str | None = None,
    task_params: Mapping[str, Any],
    axes: ComponentCountAxes,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    defaults: ComponentCountDefaults,
    prompt_bundle_id: str,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    directed: bool,
    object_description: str,
    instance_seed: int,
) -> ComponentCountArtifacts:
    """Prepare shared visual artifacts without deciding the public objective."""

    labels = resolve_adjacency_labels(
        instance_seed=int(instance_seed),
        rng_namespace=str(task_id),
        label_variant=str(axes.label_variant),
        node_count=int(axes.node_count),
        max_chars=int(group_default(gen_defaults, "label_max_chars", defaults.label_max_chars)),
    )
    sample = sample_component_adjacency(
        instance_seed=int(instance_seed),
        rng_namespace=str(task_id),
        labels=labels.labels,
        component_count=int(axes.component_count),
        directed=bool(directed),
        extra_edge_count=int(axes.extra_edge_count),
    )
    topmost_row_labels = component_topmost_row_labels(sample.labels, sample.components)
    canvas_width = int(task_params.get("canvas_width", group_default(render_defaults, "canvas_width", defaults.canvas_width)))
    canvas_height = int(task_params.get("canvas_height", group_default(render_defaults, "canvas_height", defaults.canvas_height)))
    base_image, background_meta = make_background_canvas(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=background_defaults,
    )
    rendered = render_component_adjacency_panel(
        sample=sample,
        base_image=base_image,
        representation_variant=str(axes.scene_variant),
        directed=bool(directed),
        params=task_params,
        render_defaults=render_defaults,
        defaults=defaults,
        layout_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=noise_defaults,
    )
    prompt_artifacts = build_adjacency_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_bundle_id),
        prompt_key=str(prompt_key or query_id),
        dynamic_slots={"object_description": str(object_description)},
        instance_seed=int(instance_seed),
    )
    return ComponentCountArtifacts(
        labels=labels,
        sample=sample,
        rendered=rendered,
        image=image,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        topmost_row_labels=tuple(str(label) for label in topmost_row_labels),
        annotation_artifacts=row_label_bbox_set_artifacts(rendered, topmost_row_labels),
        prompt_artifacts=prompt_artifacts,
    )


def component_count_trace_payload(
    *,
    task_id: str,
    query_id: str,
    prompt_bundle_id: str,
    query_probabilities: Mapping[str, float],
    axes: ComponentCountAxes,
    artifacts: ComponentCountArtifacts,
    directed: bool,
) -> dict[str, Any]:
    """Build component-count trace fields after task-owned semantic binding."""

    sample = artifacts.sample
    labels = artifacts.labels
    rendered = artifacts.rendered
    topmost_row_labels = artifacts.topmost_row_labels
    prompt_artifacts = artifacts.prompt_artifacts
    return {
        "scene_ir": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "scene_kind": "adjacency",
            "entities": [
                *component_node_entities(sample, rendered, topmost_row_labels),
                *component_edge_entities(sample, directed=bool(directed)),
            ],
            "relations": {
                "representation_variant": str(rendered.representation_variant),
                "query_id": str(query_id),
                "directed": bool(directed),
                "components": [list(component) for component in sample.components],
                "component_topmost_row_labels": list(topmost_row_labels),
                "adjacency": {str(key): list(values) for key, values in sample.adjacency.items()},
            },
            "frames": pixel_panel_frames(rendered),
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
                "scene_variant": str(axes.scene_variant),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "node_count": int(axes.node_count),
                "node_count_probabilities": dict(axes.node_count_probabilities),
                "component_count": int(len(sample.components)),
                "component_count_probabilities": dict(axes.component_count_probabilities),
                "extra_edge_count": int(axes.extra_edge_count),
                "extra_edge_count_probabilities": dict(axes.extra_edge_count_probabilities),
                **label_query_params(labels, label_variant_probabilities=axes.label_variant_probabilities),
            },
        },
        "render_spec": render_spec_fragment(
            canvas_width=int(artifacts.canvas_width),
            canvas_height=int(artifacts.canvas_height),
            rendered=rendered,
            background_meta=artifacts.background_meta,
            post_noise_meta=artifacts.post_noise_meta,
        ),
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "representation_variant": str(rendered.representation_variant),
            "answer": int(len(sample.components)),
            "component_topmost_row_labels": list(topmost_row_labels),
            "components": [list(component) for component in sample.components],
            "node_count": int(axes.node_count),
            "edge_count": int(len(sample.edges)),
            "directed": bool(directed),
            "label_variant": str(labels.label_variant),
        },
        "witness_symbolic": {
            "type": "component_topmost_row_label_set",
            "labels": list(topmost_row_labels),
            "components": [list(component) for component in sample.components],
        },
        "projected_annotation": dict(artifacts.annotation_artifacts.projected_annotation),
    }


def run_component_count_lifecycle(
    *,
    task_id: str,
    domain: str,
    scene_id: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_bundle_id: str,
    prompt_key: str | None = None,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    defaults: ComponentCountDefaults,
    instance_seed: int,
    params: Mapping[str, Any],
    prepare_objective: Any,
) -> TaskOutput:
    """Run neutral component-count plumbing using task-owned objective settings."""

    from trace_tasks.tasks.shared.fixed_query import select_task_query_id

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace="query_id",
    )
    axes = resolve_component_count_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        rng_namespace=str(task_id),
    )
    plan = prepare_objective(axes)
    artifacts = prepare_component_count_artifacts(
        task_id=str(task_id),
        domain=str(domain),
        query_id=str(query_id),
        prompt_key=prompt_key,
        task_params=task_params,
        axes=axes,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        defaults=defaults,
        prompt_bundle_id=str(prompt_bundle_id),
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        directed=bool(plan.directed),
        object_description=str(plan.object_description),
        instance_seed=int(instance_seed),
    )
    answer_value = int(len(artifacts.sample.components))
    return TaskOutput(
        prompt=str(artifacts.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=answer_value),
        annotation_gt=artifacts.annotation_artifacts.annotation_gt,
        image=artifacts.image,
        image_id="img0",
        trace_payload=component_count_trace_payload(
            task_id=str(task_id),
            query_id=str(query_id),
            prompt_bundle_id=str(prompt_bundle_id),
            query_probabilities=query_probabilities,
            axes=axes,
            artifacts=artifacts,
            directed=bool(plan.directed),
        ),
        task_versions=default_task_versions(),
        scene_id=str(scene_id),
        query_id=str(query_id),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "ComponentCountArtifacts",
    "ComponentCountPlan",
    "SinglePanelRenderArtifacts",
    "component_count_trace_payload",
    "prepare_component_count_artifacts",
    "render_single_panel_artifacts",
    "run_component_count_lifecycle",
    "single_panel_render_kwargs",
    "single_panel_task_output",
    "single_panel_trace_payload",
]
