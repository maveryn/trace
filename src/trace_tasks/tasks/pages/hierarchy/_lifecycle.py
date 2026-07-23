"""Scene-private rendering and response assembly for hierarchy tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.pages.shared.information_style import (
    PagesInformationStyle,
    make_pages_information_background,
    resolve_pages_information_style,
)
from trace_tasks.tasks.pages.shared.diagram.common import (
    round_diagram_bbox,
    projected_diagram_bbox_annotation,
    projected_diagram_bbox_sequence_annotation,
)
from trace_tasks.tasks.pages.shared.diagram.hierarchy_common import (
    build_hierarchy_tree_count_dataset,
    resolve_hierarchy_render_params,
    resolve_hierarchy_tree_count_scene_variant,
)
from trace_tasks.tasks.pages.shared.diagram.hierarchy_scene import render_hierarchy_scene
from trace_tasks.tasks.pages.shared.diagram.visual_defaults import (
    load_diagrams_background_defaults,
    load_diagrams_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)


DOMAIN = "pages"
SCENE = "hierarchy"
SCENE_GENERATOR_KEY = "pages.hierarchy.org_chart"
PROMPT_BUNDLE = "pages_hierarchy_v1"
PROMPT_SCENE_KEY = "hierarchy_diagram"
PROMPT_TASK_KEY = "org_chart_query"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_diagrams_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_diagrams_noise_defaults(scene_id=SCENE, apply_prob=0.0)


@dataclass(frozen=True)
class HierarchyObjectiveBinding:
    """Task-owned semantic branch, answer schema, and prompt binding."""

    semantic_branch_key: str
    prompt_branch_key: str
    answer_type: str
    annotation_type: str
    question_format: str


def _hierarchy_render_params_from_information_style(render_params: Any, style: PagesInformationStyle) -> Any:
    """Map shared Pages information-style roles into hierarchy diagram colors."""

    return replace(
        render_params,
        root_fill_rgb=tuple(int(value) for value in style.callout_fill_rgb),
        node_fill_rgb=tuple(int(value) for value in style.surface_alt_rgb),
        panel_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
        panel_border_rgb=tuple(int(value) for value in style.panel_border_rgb),
        title_color_rgb=tuple(int(value) for value in style.text_rgb),
        node_border_rgb=tuple(int(value) for value in style.panel_border_rgb),
        label_color_rgb=tuple(int(value) for value in style.text_rgb),
        label_stroke_rgb=tuple(int(value) for value in style.text_stroke_rgb),
        connector_color_rgb=tuple(int(value) for value in style.connector_rgb),
    )


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the public branch using the shared single-query policy."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(selected), dict(probabilities), dict(task_params)


def _annotation_for_rendered_tree(
    *,
    rendered_scene: Any,
    node_bbox_ids: list[str],
    annotation_type: str,
) -> tuple[TypedValue, Dict[str, Any], str]:
    """Project either unordered counted-node boxes or an ordered path sequence."""

    if str(annotation_type) == "bbox":
        if len(node_bbox_ids) != 1:
            raise ValueError("scalar bbox annotation requires exactly one node bbox id")
        bbox = round_diagram_bbox(rendered_scene.node_bbox_map[str(node_bbox_ids[0])])
        projection = {"type": "bbox", "bbox": list(bbox)}
        witness_type = "selected_id"
        return TypedValue(type="bbox", value=list(bbox)), dict(projection), str(witness_type)
    if str(annotation_type) == "bbox_sequence":
        projection = projected_diagram_bbox_sequence_annotation(rendered_scene.node_bbox_map, node_bbox_ids)
        boxes = [
            [round(float(value), 3) for value in bbox]
            for bbox in projection["bbox_sequence"]
        ]
        witness_type = "ordered_id_path"
    else:
        projection = projected_diagram_bbox_annotation(rendered_scene.node_bbox_map, node_bbox_ids)
        boxes = [
            [round(float(value), 3) for value in bbox]
            for bbox in projection["bbox_set"]
        ]
        witness_type = "id_set"
    return TypedValue(type=str(annotation_type), value=list(boxes)), dict(projection), str(witness_type)


def render_bound_hierarchy(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    objective: HierarchyObjectiveBinding,
) -> TaskOutput:
    """Render the hierarchy scene and assemble the task-bound response."""

    scene_variant, scene_variant_probabilities = resolve_hierarchy_tree_count_scene_variant(
        params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        task_id=SCENE_GENERATOR_KEY,
    )
    dataset = build_hierarchy_tree_count_dataset(
        query_id=str(objective.semantic_branch_key),
        scene_variant=str(scene_variant),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        task_id=SCENE_GENERATOR_KEY,
    )
    render_params = resolve_hierarchy_render_params(
        params,
        render_defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    information_style, information_style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE,
    )
    render_params = _hierarchy_render_params_from_information_style(
        render_params,
        information_style,
    )
    background, background_meta = make_pages_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace="pages.hierarchy.information_scene_background",
    )
    background_meta = dict(background_meta)
    background_meta["information_scene_style"] = dict(information_style_meta)
    rendered_scene = render_hierarchy_scene(
        background,
        scene_title=str(dataset["scene_title"]),
        root_node_id=str(dataset["root_node_id"]),
        node_specs=list(dataset["node_specs"]),
        edge_specs=list(dataset["edge_specs"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(objective.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dataset["query_prompt_slots"]),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    annotation_node_bbox_ids = [str(bbox_id) for bbox_id in dataset["annotation_node_bbox_ids"]]
    annotation_gt, annotation_projection, witness_type = _annotation_for_rendered_tree(
        rendered_scene=rendered_scene,
        node_bbox_ids=annotation_node_bbox_ids,
        annotation_type=str(objective.annotation_type),
    )
    if str(objective.answer_type) == "integer":
        answer_value: int | str = int(dataset["answer_value"])
        answer_gt = TypedValue(type="integer", value=int(answer_value))
    elif str(objective.answer_type) == "string":
        answer_value = str(dataset["answer_value"])
        answer_gt = TypedValue(type="string", value=str(answer_value))
    else:
        raise ValueError(f"unsupported hierarchy answer type: {objective.answer_type}")
    probabilities = {str(key): float(value) for key, value in branch_probabilities.items()}
    semantic_key = str(objective.semantic_branch_key)
    query_params = {
        "query_id": str(selected_branch),
        "source_query_id": semantic_key,
        "prompt_query_key": str(objective.prompt_branch_key),
        "scene_variant": str(scene_variant),
        "query_id_probabilities": dict(probabilities),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "tree_node_count": int(dataset["tree_node_count"]),
        "tree_depth": int(dataset["tree_depth"]),
        "leaf_count": int(dataset["leaf_count"]),
        "query_relationship": str(dataset["query_relationship"]),
        "answer_value": answer_value,
        "answer_count": int(dataset["answer_count"]),
        "information_scene_treatment": str(information_style_meta.get("treatment", "")),
        "information_scene_palette_id": str(information_style_meta.get("palette_id", "")),
        "information_scene_style_pack": str(information_style_meta.get("style_pack", "")),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": f"diagram_hierarchy_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(selected_branch),
                "source_query_id": semantic_key,
                "prompt_query_key": str(objective.prompt_branch_key),
                "scene_variant": str(scene_variant),
                "root_node_id": str(dataset["root_node_id"]),
                "query_node_ids": [str(node_id) for node_id in dataset["query_node_ids"]],
                "annotation_node_ids": [str(node_id) for node_id in dataset["annotation_node_ids"]],
                "view_family": str(dataset["view_family"]),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(selected_branch),
            "source_query_id": semantic_key,
            "scene_variant": str(scene_variant),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "node_width_px": int(render_params.node_width_px),
            "node_height_px": int(render_params.node_height_px),
            "connector_width_px": int(render_params.connector_width_px),
            "layout_jitter": dict(rendered_scene.layout_jitter_meta),
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "context_text_policy": {
                "container_bboxes_are_background": True,
            },
            "hierarchy_style": {
                "information_scene_treatment": str(information_style_meta.get("treatment", "")),
                "information_scene_palette_id": str(information_style_meta.get("palette_id", "")),
                "information_scene_style_pack": str(information_style_meta.get("style_pack", "")),
                "resolved_colors_rgb": {
                    "panel_fill": [int(value) for value in render_params.panel_fill_rgb],
                    "panel_border": [int(value) for value in render_params.panel_border_rgb],
                    "node_fill": [int(value) for value in render_params.node_fill_rgb],
                    "root_fill": [int(value) for value in render_params.root_fill_rgb],
                    "node_border": [int(value) for value in render_params.node_border_rgb],
                    "label_color": [int(value) for value in render_params.label_color_rgb],
                    "connector_color": [int(value) for value in render_params.connector_color_rgb],
                },
            },
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered_scene.panel_bbox_px),
            "title_bbox_px": list(rendered_scene.title_bbox_px),
            "node_bboxes_px": dict(rendered_scene.node_bbox_map),
            "node_label_bboxes_px": dict(rendered_scene.node_label_bbox_map),
            "edge_bboxes_px": dict(rendered_scene.edge_bbox_map),
        },
        "execution_trace": {
            "query_id": str(selected_branch),
            "source_query_id": semantic_key,
            "prompt_query_key": str(objective.prompt_branch_key),
            "scene_variant": str(scene_variant),
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "question_format": str(objective.question_format),
            "view_family": str(dataset["view_family"]),
            "scene_title": str(dataset["scene_title"]),
            "query_prompt_slots": dict(dataset["query_prompt_slots"]),
            "template_id": str(dataset["template_id"]),
            "tree_node_count": int(dataset["tree_node_count"]),
            "tree_depth": int(dataset["tree_depth"]),
            "leaf_count": int(dataset["leaf_count"]),
            "root_node_id": str(dataset["root_node_id"]),
            "node_specs": [dict(spec) for spec in dataset["node_specs"]],
            "edge_specs": [dict(spec) for spec in dataset["edge_specs"]],
            "query_node_ids": [str(node_id) for node_id in dataset["query_node_ids"]],
            "query_node_labels": [str(label) for label in dataset["query_node_labels"]],
            "query_depths": [int(depth) for depth in dataset["query_depths"]],
            "query_relationship": str(dataset["query_relationship"]),
            "answer_value": answer_value,
            "answer_count": int(dataset["answer_count"]),
            "answer_node_id": str(dataset["answer_node_id"]),
            "answer_node_label": str(dataset["answer_node_label"]),
            "answer_node_bbox_id": str(dataset["answer_node_bbox_id"]),
            "answer_metric_name": str(dataset["answer_metric_name"]),
            "answer_metric_count": int(dataset["answer_metric_count"]),
            "candidate_manager_counts": [dict(row) for row in dataset.get("candidate_manager_counts", [])],
            "annotation_node_ids": [str(node_id) for node_id in dataset["annotation_node_ids"]],
            "annotation_node_bbox_ids": [str(bbox_id) for bbox_id in annotation_node_bbox_ids],
            "annotation_semantics": str(dataset["annotation_semantics"]),
            "descendant_node_ids": [str(node_id) for node_id in dataset["descendant_node_ids"]],
            "descendant_count": int(dataset["descendant_count"]),
            "leaf_descendant_node_ids": [str(node_id) for node_id in dataset["leaf_descendant_node_ids"]],
            "leaf_descendant_count": int(dataset["leaf_descendant_count"]),
            "supporting_node_bbox_ids": [str(bbox_id) for bbox_id in annotation_node_bbox_ids],
            "information_scene_treatment": str(information_style_meta.get("treatment", "")),
            "information_scene_palette_id": str(information_style_meta.get("palette_id", "")),
            "information_scene_style_pack": str(information_style_meta.get("style_pack", "")),
        },
        "witness_symbolic": {
            "type": str(witness_type),
            "ids": [str(node_id) for node_id in dataset["annotation_node_ids"]],
        },
        "projected_annotation": dict(annotation_projection),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE,
        query_id=str(selected_branch),
    )


__all__ = [
    "DOMAIN",
    "HierarchyObjectiveBinding",
    "SCENE",
    "select_public_branch",
    "render_bound_hierarchy",
]
