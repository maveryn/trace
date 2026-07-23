"""Private lifecycle helpers for pedigree-chart graph tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...shared.config_defaults import required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.mcq import option_label_for_index
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ...shared.variant_sampling import resolve_variant
from ..shared.style import SUPPORTED_NODE_COLOR_NAMES
from ..shared.task_support import resolve_graph_render_params
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from .shared.annotations import projected_pedigree_person_bbox_set_annotation
from .shared.option_rendering import OPTION_LABELS, draw_pedigree_options
from .shared.rendering import pedigree_connector_relations, pedigree_scene_entities, render_pedigree_chart_scene
from .shared.sampling import sample_pedigree_relatedness, sample_pedigree_relationship
from .shared.state import (
    PEDIGREE_RELATEDNESS_LABELS,
    PEDIGREE_RELATEDNESS_OPTION_LABELS,
    PEDIGREE_RELATIONSHIP_LABELS,
    SUPPORTED_PEDIGREE_SCENE_VARIANTS,
    PedigreeRelatednessQuerySample,
    PedigreeRelationshipQuerySample,
)

SCENE_ID = "pedigree_chart"


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable rendering fallbacks for pedigree relation tasks."""

    person_count_min: int = 8
    person_count_max: int = 16
    canvas_width: int = 980
    canvas_height: int = 700
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 18
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    edge_routing_variant: str = "straight"
    node_radius_min_px: int = 18
    node_radius_max_px: int = 22
    edge_width_px: int = 3
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 3
    label_font_size_px: int = 18
    node_color_name: str = "blue"
    background_color_rgb: Tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: Tuple[int, int, int] = (205, 212, 224)
    title_color_rgb: Tuple[int, int, int] = (70, 78, 96)
    edge_color_rgb: Tuple[int, int, int] = (92, 104, 126)
    node_fill_rgb: Tuple[int, int, int] = (42, 72, 140)
    node_border_rgb: Tuple[int, int, int] = (42, 72, 140)
    label_text_rgb: Tuple[int, int, int] = (20, 26, 36)
    label_stroke_rgb: Tuple[int, int, int] = (255, 255, 255)
    option_count: int = 6


@dataclass(frozen=True)
class _ResolvedStyle:
    """Resolved nonsemantic style axes."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    node_color_name: str
    node_color_name_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_SCENE_DEFAULTS = get_scene_defaults("graph", SCENE_ID)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _sections_for_task(task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=str(task_id),
    )


def _resolve_style(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng_namespace: str,
) -> _ResolvedStyle:
    scene_variant, scene_probs = resolve_variant(
        spawn_rng(int(instance_seed), f"{rng_namespace}.scene_variant"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_PEDIGREE_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    node_color_name, color_probs = resolve_variant(
        spawn_rng(int(instance_seed), f"{rng_namespace}.node_color_name"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_NODE_COLOR_NAMES,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
    )
    return _ResolvedStyle(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        node_color_name=str(node_color_name),
        node_color_name_probabilities=dict(color_probs),
    )


def _select_variant(
    instance_seed: int,
    *,
    rng_namespace: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    supported: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
) -> Tuple[str, Dict[str, float]]:
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{rng_namespace}.{explicit_key}"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    return str(selected), dict(probabilities)


def _person_label(sample, person_id: str) -> str:
    for person in sample.people:
        if str(person.person_id) == str(person_id):
            return str(person.label)
    raise ValueError(f"missing pedigree person id: {person_id}")


def _render_sample(
    *,
    render_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    style: _ResolvedStyle,
    sample,
    highlighted_person_ids=(),
    bottom_reserved_px: int = 0,
):
    """Render one neutral pedigree scene while preserving projected person-symbol geometry."""

    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(render_namespace),
        render_defaults=render_defaults,
        fallback_defaults=_DEFAULTS,
        node_color_name=str(style.node_color_name),
        node_shape_variant="circle",
        edge_routing_variant="straight",
    )
    image, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered_scene = render_pedigree_chart_scene(
        sample=sample,
        render_params=render_params,
        scene_variant=str(style.scene_variant),
        scene_title="Pedigree Chart",
        base_image=image,
        highlighted_person_ids=tuple(str(person_id) for person_id in highlighted_person_ids),
        bottom_reserved_px=int(bottom_reserved_px),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return image, rendered_scene, render_params, background_meta, post_noise_meta


def _prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    query_id: str,
    slots: Mapping[str, Any],
    instance_seed: int,
):
    prompt_defaults_required = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context=f"prompt defaults for {scene_id}:{query_id}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults_required["bundle_id"]),
        scene_key=str(prompt_defaults_required["scene_key"]),
        task_key=str(prompt_defaults_required["task_key"]),
        query_key=str(query_id),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _json_examples(answer: str) -> Tuple[str, str]:
    annotation = [[220, 210, 260, 250], [410, 340, 450, 380]]
    return (
        json.dumps({"annotation": annotation, "answer": str(answer)}, separators=(",", ":")),
        json.dumps({"answer": str(answer)}, separators=(",", ":")),
    )


def _common_slots(
    prompt_defaults: Mapping[str, Any],
    *,
    answer_example: str,
) -> Dict[str, str]:
    prompt_defaults_required = required_group_defaults(
        prompt_defaults,
        (
            "object_description",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="prompt defaults for pedigree_chart",
    )
    json_example, json_example_answer_only = _json_examples(str(answer_example))
    return {
        "object_description": str(prompt_defaults_required["object_description"]),
        "json_output_contract": str(prompt_defaults_required["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults_required["json_output_contract_answer_only"]),
        "annotation_hint": str(prompt_defaults_required["annotation_hint"]),
        "answer_hint": str(prompt_defaults_required["answer_hint"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


def _select_relationship_options(answer: str) -> Tuple[Dict[str, str], str]:
    option_values = {
        str(option_label_for_index(index)): str(value)
        for index, value in enumerate(PEDIGREE_RELATIONSHIP_LABELS)
    }
    correct_letter = next(str(label) for label, value in option_values.items() if str(value) == str(answer))
    return dict(option_values), str(correct_letter)


def _select_relatedness_options(
    *,
    instance_seed: int,
    answer: str,
    rng_namespace: str,
) -> Tuple[Dict[str, str], str]:
    rng = spawn_rng(int(instance_seed), f"{rng_namespace}.fraction_options")
    distractors = [str(value) for value in PEDIGREE_RELATEDNESS_OPTION_LABELS if str(value) != str(answer)]
    rng.shuffle(distractors)
    values = [str(answer), *distractors[: len(OPTION_LABELS) - 1]]
    rng.shuffle(values)
    option_values = {
        str(option_label_for_index(index)): str(value)
        for index, value in enumerate(values)
    }
    correct_letter = next(str(label) for label, value in option_values.items() if str(value) == str(answer))
    return dict(option_values), str(correct_letter)


def _trace_payload(
    *,
    task_identifier: str,
    query_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts,
    sample,
    rendered_scene,
    render_params,
    style: _ResolvedStyle,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    query_params: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble shared scene/render trace fields around task-owned answer and annotation bindings."""

    return {
        "scene_ir": {
            "task_id": str(task_identifier),
            "scene_id": SCENE_ID,
            "scene_kind": "pedigree_chart",
            "entities": list(pedigree_scene_entities(sample, rendered_scene)),
            "relations": {"connectors": list(pedigree_connector_relations(sample, rendered_scene))},
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(rendered_scene.panel_geometry),
            },
        },
        "query_spec": {
            "task_id": str(task_identifier),
            "query_id": str(query_id),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_params),
        },
        "render_spec": {
            "canvas_size": list(rendered_scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "panel_geometry": dict(rendered_scene.panel_geometry),
            "style": {
                "scene_variant": str(style.scene_variant),
                "theme_tone": str(render_params.theme_tone),
                "panel_style_variant": str(render_params.panel_style_variant),
                "background_color_rgb": list(render_params.background_color_rgb),
                "panel_fill_rgb": list(render_params.panel_fill_rgb),
                "panel_border_rgb": list(render_params.panel_border_rgb),
                "title_color_rgb": list(render_params.title_color_rgb),
                "edge_color_rgb": list(render_params.edge_color_rgb),
                "edge_width_px": int(render_params.edge_width_px),
                "node_radius_px": int(render_params.node_radius_px),
                "label_font_size_px": int(render_params.label_font_size_px),
                "resolved_label_font_size_px": int(rendered_scene.resolved_label_font_size_px),
                "label_stroke_width_px": int(rendered_scene.resolved_label_stroke_width_px),
                "background_meta": dict(background_meta),
                "post_image_noise_meta": dict(post_noise_meta),
                "font_family": str(render_params.font_family),
                "font_asset": dict(render_params.font_asset or {}),
                "font_asset_version": str(render_params.font_asset_version),
                "text_legibility": dict(render_params.text_legibility or {}),
            },
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }
