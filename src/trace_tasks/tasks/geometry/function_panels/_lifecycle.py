"""Scene-private component assembly for function-panel public tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    PanelAnnotationArtifacts,
    intersection_panel_annotation,
    selected_panel_bbox_annotation,
)
from .shared.defaults import split_defaults_for
from .shared.prompts import render_panel_prompt_artifacts
from .shared.rendering import render_intersection_scene, render_property_scene
from .shared.sampling import (
    build_intersection_panels,
    build_property_relations,
    panel_trace_payload,
    relation_trace_payload,
    resolve_intersection_selection,
    resolve_property_selection,
    selected_interval,
)
from .shared.state import RULE_SIGN_INTERVAL, SCENE_ID, SIGN_NEGATIVE, SIGN_POSITIVE


@dataclass(frozen=True)
class PropertyObjectivePlan:
    """Task-owned semantic fields for one selected function-property objective."""

    prompt_key: str
    rule_kind: str
    sign_kind: str | None
    prompt_defaults: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class IntersectionObjectivePlan:
    """Task-owned semantic fields for one selected intersection objective."""

    prompt_key: str
    pair_kind: str
    relation_class: str
    prompt_defaults: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class FunctionPanelComponents:
    """Prompt/image/annotation/trace sections before public TaskOutput binding."""

    prompt: str
    prompt_variants: Mapping[str, str]
    image: Any
    annotation: PanelAnnotationArtifacts
    trace_payload: Mapping[str, Any]


def property_defaults(public_name: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return scene defaults for one public property file."""

    return split_defaults_for(str(public_name))


def property_plan(
    *,
    prompt_key: str,
    rule_kind: str,
    sign_kind: str | None,
    defaults: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> PropertyObjectivePlan:
    """Bind semantic property fields selected by a public task."""

    generation_defaults, rendering_defaults, prompt_defaults = defaults
    return PropertyObjectivePlan(
        prompt_key=str(prompt_key),
        rule_kind=str(rule_kind),
        sign_kind=str(sign_kind) if sign_kind is not None else None,
        prompt_defaults=prompt_defaults,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
    )


def intersection_plan(
    *,
    prompt_key: str,
    pair_kind: str,
    relation_class: str,
    defaults: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> IntersectionObjectivePlan:
    """Bind semantic intersection fields selected by a public task."""

    generation_defaults, rendering_defaults, prompt_defaults = defaults
    return IntersectionObjectivePlan(
        prompt_key=str(prompt_key),
        pair_kind=str(pair_kind),
        relation_class=str(relation_class),
        prompt_defaults=prompt_defaults,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
    )


def _property_prompt_slots(plan: PropertyObjectivePlan, rendered) -> dict[str, str]:
    interval = ""
    if plan.rule_kind == RULE_SIGN_INTERVAL and plan.sign_kind in {SIGN_POSITIVE, SIGN_NEGATIVE}:
        interval_tuple = selected_interval(plan.rule_kind, sign_kind=plan.sign_kind)
        interval = rendered.target_interval if interval_tuple is not None else ""
    return {
        "target_range": str(rendered.target_range),
        "target_interval": str(interval),
    }


def build_property_components(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    namespace: str,
    plan: PropertyObjectivePlan,
) -> FunctionPanelComponents:
    """Assemble render, prompt, annotation, and trace components for one property objective."""

    selection = resolve_property_selection(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=plan.generation_defaults,
        namespace=str(namespace),
        max_panel_count=6,
    )
    relations = build_property_relations(
        instance_seed=int(instance_seed),
        params=params,
        selection=selection,
        rule_kind=plan.rule_kind,
        sign_kind=plan.sign_kind,
        namespace=str(namespace),
    )
    selection = replace(selection, target_interval=selected_interval(plan.rule_kind, sign_kind=plan.sign_kind))
    rendered = render_property_scene(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=plan.rendering_defaults,
        selection=selection,
        relations_by_label=relations,
    )
    annotation = selected_panel_bbox_annotation(rendered, label=selection.selected_label)
    prompt_artifacts = render_panel_prompt_artifacts(
        prompt_defaults=plan.prompt_defaults,
        prompt_key=plan.prompt_key,
        instance_seed=int(instance_seed),
        **_property_prompt_slots(plan, rendered),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_name),
        params={
            "scene_id": SCENE_ID,
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_key": str(plan.prompt_key),
            "property_rule": str(plan.rule_kind),
            "answer_label": str(selection.selected_label),
            "answer_label_probabilities": dict(selection.label_probabilities),
            "candidate_label_pool": list(selection.label_pool),
            "panel_count_probabilities": dict(selection.panel_count_probabilities),
            "target_range": str(rendered.target_range),
            "target_interval": str(rendered.target_interval),
        },
    )
    relations_trace = {str(label): relation_trace_payload(relation) for label, relation in relations.items()}
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "scene_kind": "geometry_function_property_panel_grid",
            "entities": [{"label": str(label), "kind": "coordinate_panel"} for label in selection.label_pool],
            "relations": {
                "selected_label": str(selection.selected_label),
                "property_rule": str(plan.rule_kind),
                "relations_by_label": relations_trace,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_size": [int(rendered.image.width), int(rendered.image.height)],
            "coord_space": "pixel",
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "diagram_style": dict(rendered.diagram_style_meta),
            "panel_style": dict(rendered.panel_style_meta),
            "line_color_selection": dict(rendered.line_color_meta),
            "line_colors": [list(color) for color in rendered.line_colors],
            "panel_columns": int(rendered.panel_columns),
            "panel_rows": int(rendered.panel_rows),
        },
        "render_map": {"coord_space": "pixel"},
        "projected_annotation": dict(annotation.projected_annotation),
        "witness_symbolic": dict(annotation.witness_symbolic),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_key": str(plan.prompt_key),
            "property_rule": str(plan.rule_kind),
            "answer_type": "option_letter",
            "answer_label": str(selection.selected_label),
            "relations_by_label": relations_trace,
            "winner_relation": relation_trace_payload(relations[str(selection.selected_label)]),
            "target_range": str(rendered.target_range),
            "target_interval": str(rendered.target_interval),
        },
    }
    return FunctionPanelComponents(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=rendered.image,
        annotation=annotation,
        trace_payload=trace_payload,
    )


def build_intersection_components(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    namespace: str,
    plan: IntersectionObjectivePlan,
) -> FunctionPanelComponents:
    """Assemble render, prompt, annotation, and trace components for one intersection objective."""

    scene_params = {**dict(params), "canvas_height": int(params.get("canvas_height", 1024))}
    selection = resolve_intersection_selection(
        instance_seed=int(instance_seed),
        params=scene_params,
        gen_defaults=plan.generation_defaults,
        namespace=str(namespace),
    )
    panels = build_intersection_panels(
        instance_seed=int(instance_seed),
        params=scene_params,
        selection=selection,
        pair_kind=plan.pair_kind,
        relation_class=plan.relation_class,
        namespace=str(namespace),
    )
    rendered = render_intersection_scene(
        instance_seed=int(instance_seed),
        params=scene_params,
        render_defaults=plan.rendering_defaults,
        selection=selection,
        panels_by_label=panels,
    )
    annotation = intersection_panel_annotation(rendered, label=selection.selected_label)
    prompt_artifacts = render_panel_prompt_artifacts(
        prompt_defaults=plan.prompt_defaults,
        prompt_key=plan.prompt_key,
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_name),
        params={
            "scene_id": SCENE_ID,
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_key": str(plan.prompt_key),
            "pair_kind": str(plan.pair_kind),
            "relation_class": str(plan.relation_class),
            "answer_label": str(selection.selected_label),
            "answer_label_probabilities": dict(selection.label_probabilities),
            "candidate_label_pool": list(selection.label_pool),
            "panel_count_probabilities": dict(selection.panel_count_probabilities),
        },
    )
    panel_trace = {str(label): panel_trace_payload(panel) for label, panel in panels.items()}
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "scene_kind": "geometry_intersection_property_panel_grid",
            "entities": [{"label": str(label), "kind": "coordinate_panel"} for label in selection.label_pool],
            "relations": {
                "selected_label": str(selection.selected_label),
                "pair_kind": str(plan.pair_kind),
                "relation_class": str(plan.relation_class),
                "panels_by_label": panel_trace,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_size": [int(rendered.image.width), int(rendered.image.height)],
            "coord_space": "pixel",
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "panel_style": dict(rendered.panel_style_meta),
            "panel_columns": int(rendered.panel_columns),
            "panel_rows": int(rendered.panel_rows),
            "object_color_selection": dict(rendered.object_color_meta),
            "object_colors": [list(color) for color in rendered.object_colors],
            "intersection_color_selection": dict(rendered.intersection_color_meta),
        },
        "render_map": {"coord_space": "pixel"},
        "projected_annotation": dict(annotation.projected_annotation),
        "witness_symbolic": dict(annotation.witness_symbolic),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_key": str(plan.prompt_key),
            "pair_kind": str(plan.pair_kind),
            "relation_class": str(plan.relation_class),
            "answer_type": "option_letter",
            "answer_label": str(selection.selected_label),
            "target_quadrant": "",
            "panels_by_label": panel_trace,
            "winner_panel": panel_trace[str(selection.selected_label)],
        },
    }
    return FunctionPanelComponents(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=rendered.image,
        annotation=annotation,
        trace_payload=trace_payload,
    )
