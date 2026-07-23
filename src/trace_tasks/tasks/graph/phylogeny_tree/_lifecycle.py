"""Private lifecycle helpers for phylogeny-tree graph tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from ...shared.variant_sampling import resolve_variant
from ..shared.style import SUPPORTED_NODE_COLOR_NAMES
from ..shared.task_support import resolve_graph_render_params
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from .shared.prompts import TREE_SCENE_PROMPT_KEY, build_phylogeny_prompt_artifacts
from .shared.rendering import phylogeny_scene_entities, render_phylogeny_option_scene, render_phylogeny_tree_scene
from .shared.state import SUPPORTED_PHYLOGENY_SCENE_VARIANTS


SCENE_ID = "phylogeny_tree"


@dataclass(frozen=True)
class PhylogenyDefaults:
    """Stable fallback generation and rendering settings for phylogeny tasks."""

    leaf_count_min: int = 6
    leaf_count_max: int = 12
    target_clade_leaf_count_min: int = 2
    target_clade_leaf_count_max: int = 6
    target_mrca_leaf_count_min: int = 2
    target_mrca_leaf_count_max: int = 8
    option_leaf_count_min: int = 6
    option_leaf_count_max: int = 8
    option_count: int = 4
    canvas_width: int = 920
    canvas_height: int = 800
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    edge_routing_variant: str = "straight"
    node_radius_min_px: int = 18
    node_radius_max_px: int = 24
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 22
    node_color_name: str = "blue"
    background_color_rgb: Tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: Tuple[int, int, int] = (205, 212, 224)
    title_color_rgb: Tuple[int, int, int] = (70, 78, 96)
    edge_color_rgb: Tuple[int, int, int] = (92, 104, 126)
    node_fill_rgb: Tuple[int, int, int] = (92, 124, 250)
    node_border_rgb: Tuple[int, int, int] = (52, 73, 144)
    label_text_rgb: Tuple[int, int, int] = (20, 26, 36)
    label_stroke_rgb: Tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class ResolvedPhylogenyStyle:
    """Resolved nonsemantic scene and node style axes."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    node_color_name: str
    node_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedPhylogenyBundle:
    """Rendered image plus metadata needed by task-owned trace assembly."""

    image: Any
    rendered_scene: Any
    render_params: Any
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


@dataclass(frozen=True)
class SingleTreeCase:
    """Task-owned semantic sample selected before common single-tree rendering."""

    sample: Any
    marked_node_id: str | None = None
    trace_params: Dict[str, Any] | None = None
    semantic_payload: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BoundPhylogenyResult:
    """Task-owned answer, annotation, and trace fields bound after rendering."""

    answer_type: str
    answer_value: Any
    annotation_type: str
    annotation_value: Any
    prompt_slots: Dict[str, Any]
    trace_params: Dict[str, Any]
    scene_relations: Dict[str, Any]
    execution_trace: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    projected_annotation: Dict[str, Any]
    entities: Any | None = None


SingleTreeCaseFactory = Callable[[int, Mapping[str, Any], int], SingleTreeCase]
SingleTreeBinder = Callable[[SingleTreeCase, RenderedPhylogenyBundle, str], BoundPhylogenyResult]


DEFAULTS = PhylogenyDefaults()
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def scene_default_sections(owner_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one public objective."""

    return load_scene_generation_rendering_prompt_defaults("graph", SCENE_ID, task_id=str(owner_id))


def integer_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, ...]:
    """Resolve an inclusive integer support from params, scene config, and defaults."""

    low = int(params.get(f"{key}_min", group_default(gen_defaults, f"{key}_min", int(fallback_min))))
    high = int(params.get(f"{key}_max", group_default(gen_defaults, f"{key}_max", int(fallback_max))))
    if int(high) < int(low):
        raise ValueError(f"{key}_min must be <= {key}_max")
    return tuple(range(int(low), int(high) + 1))


def resolve_integer_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    support: Tuple[int, ...],
    explicit_key: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one task-owned integer target using the repository sampling cursor."""

    if not support:
        raise ValueError(f"empty support for {namespace}")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if int(value) not in set(int(item) for item in support):
            raise ValueError(f"{explicit_key} outside configured support")
        return int(value), dict(uniform_probability_map(support, selected=int(value)))
    value = int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            support,
        )
    )
    return int(value), dict(uniform_probability_map(support))


def resolve_phylogeny_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> ResolvedPhylogenyStyle:
    """Resolve scene-level visual variation without changing the objective."""

    scene_variant, scene_probs = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.scene_variant"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_PHYLOGENY_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    node_color_name, color_probs = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.node_color_name"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_NODE_COLOR_NAMES,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
    )
    return ResolvedPhylogenyStyle(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        node_color_name=str(node_color_name),
        node_color_name_probabilities=dict(color_probs),
    )


def leaf_node_id(sample: Any, label: str) -> str:
    """Return the internal node id for one displayed leaf label."""

    for node in sample.nodes:
        if node.leaf_label == str(label):
            return str(node.node_id)
    raise ValueError(f"missing leaf label: {label}")


def render_single_tree(
    *,
    owner_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    style: ResolvedPhylogenyStyle,
    sample: Any,
    marked_node_id: str | None = None,
) -> RenderedPhylogenyBundle:
    """Render one phylogeny tree and apply post-render visual effects."""

    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(owner_namespace),
        render_defaults=render_defaults,
        fallback_defaults=DEFAULTS,
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
    rendered_scene = render_phylogeny_tree_scene(
        sample=sample,
        render_params=render_params,
        scene_variant=str(style.scene_variant),
        scene_title="Phylogeny",
        layout_seed=int(instance_seed),
        base_image=image,
        marked_node_id=marked_node_id,
    )
    noisy_image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedPhylogenyBundle(
        image=noisy_image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def render_option_trees(
    *,
    owner_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    style: ResolvedPhylogenyStyle,
    option_specs: Any,
) -> RenderedPhylogenyBundle:
    """Render the phylogeny option panels and apply post-render visual effects."""

    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(owner_namespace),
        render_defaults=render_defaults,
        fallback_defaults=DEFAULTS,
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
    rendered_scene = render_phylogeny_option_scene(
        option_specs=option_specs,
        render_params=render_params,
        scene_variant=str(style.scene_variant),
        layout_seed=int(instance_seed),
        base_image=image,
    )
    noisy_image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedPhylogenyBundle(
        image=noisy_image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def rendered_style_metadata(
    *,
    style: ResolvedPhylogenyStyle,
    bundle: RenderedPhylogenyBundle,
) -> Dict[str, Any]:
    """Serialize resolved rendering choices for trace metadata."""

    render_params = bundle.render_params
    rendered_scene = bundle.rendered_scene
    return {
        "scene_variant": str(style.scene_variant),
        "theme_tone": str(render_params.theme_tone),
        "panel_style_variant": str(render_params.panel_style_variant),
        "background_color_rgb": list(render_params.background_color_rgb),
        "panel_fill_rgb": list(render_params.panel_fill_rgb),
        "panel_border_rgb": list(render_params.panel_border_rgb),
        "title_color_rgb": list(render_params.title_color_rgb),
        "edge_color_rgb": list(render_params.edge_color_rgb),
        "edge_width_px": int(render_params.edge_width_px),
        "label_font_size_px": int(render_params.label_font_size_px),
        "resolved_label_font_size_px": int(rendered_scene.resolved_label_font_size_px),
        "label_stroke_width_px": int(rendered_scene.resolved_label_stroke_width_px),
        "background_meta": dict(bundle.background_meta),
        "post_image_noise_meta": dict(bundle.post_noise_meta),
    }


def single_tree_scene_ir(
    *,
    owner_id: str,
    sample: Any,
    bundle: RenderedPhylogenyBundle,
    relations: Mapping[str, Any],
    entities: Any | None = None,
) -> Dict[str, Any]:
    """Build scene-level trace IR for a rendered single cladogram."""

    rendered_scene = bundle.rendered_scene
    return {
        "task_id": str(owner_id),
        "scene_id": SCENE_ID,
        "scene_kind": "phylogeny_tree",
        "entities": list(entities if entities is not None else phylogeny_scene_entities(sample, rendered_scene)),
        "relations": {
            "root_id": str(sample.root_id),
            "leaf_labels": list(sample.leaf_labels),
            "canonical_signature": [list(item) for item in sample.canonical_signature],
            **dict(relations),
        },
        "frames": {
            "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            "panels": dict(rendered_scene.panel_geometry),
        },
    }


def render_spec(bundle: RenderedPhylogenyBundle, *, style: ResolvedPhylogenyStyle) -> Dict[str, Any]:
    """Build render trace metadata for a rendered phylogeny scene."""

    rendered_scene = bundle.rendered_scene
    return {
        "canvas_size": list(rendered_scene.panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "panel_geometry": dict(rendered_scene.panel_geometry),
        "style": rendered_style_metadata(style=style, bundle=bundle),
    }


def query_spec(
    *,
    owner_id: str,
    public_query_id: str,
    prompt_artifacts: PromptTraceArtifacts,
    params: Mapping[str, Any],
    prompt_bundle_id: str,
) -> Dict[str, Any]:
    """Build query trace metadata with prompt variant fields."""

    payload = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params=dict(params),
    )
    payload["task_id"] = str(owner_id)
    payload["template_id"] = str(prompt_bundle_id)
    return payload


def assemble_trace_payload(
    *,
    owner_id: str,
    public_query_id: str,
    prompt_artifacts: PromptTraceArtifacts,
    prompt_bundle_id: str,
    trace_params: Mapping[str, Any],
    scene_ir: Mapping[str, Any],
    rendered: RenderedPhylogenyBundle,
    style: ResolvedPhylogenyStyle,
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Combine task-owned semantic trace sections with common rendered metadata."""

    return {
        "scene_ir": dict(scene_ir),
        "query_spec": query_spec(
            owner_id=str(owner_id),
            public_query_id=str(public_query_id),
            prompt_artifacts=prompt_artifacts,
            params=trace_params,
            prompt_bundle_id=str(prompt_bundle_id),
        ),
        "render_spec": render_spec(rendered, style=style),
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


def finalize_phylogeny_result(
    *,
    prompt_artifacts: PromptTraceArtifacts,
    answer_type: str,
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    image: Any,
    trace_payload: Mapping[str, Any],
    public_query_id: str,
) -> TaskOutput:
    """Return a `TaskOutput` after public code has bound answer and annotation."""

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=TypedValue(type=str(annotation_type), value=annotation_value),
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def run_single_tree_objective(
    *,
    owner_id: str,
    domain: str,
    prompt_key: str,
    prompt_bundle_id: str,
    object_description: str,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    supported_query_ids: Tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_case: SingleTreeCaseFactory,
    bind_rendered: SingleTreeBinder,
) -> TaskOutput:
    """Run common single-tree rendering/output after task hooks bind semantics."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(owner_id),
        namespace=f"{sampling_namespace}.query",
    )
    style = resolve_phylogeny_style(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(sampling_namespace),
    )
    case = prepare_case(int(instance_seed), task_params, int(max_attempts))
    rendered = render_single_tree(
        owner_namespace=str(sampling_namespace),
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        style=style,
        sample=case.sample,
        marked_node_id=case.marked_node_id,
    )
    bound = bind_rendered(case, rendered, str(selected_query))
    prompt_artifacts = build_phylogeny_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_bundle_id),
        scene_key=TREE_SCENE_PROMPT_KEY,
        prompt_key=str(prompt_key),
        dynamic_slots={"object_description": object_description, **dict(bound.prompt_slots)},
        instance_seed=int(instance_seed),
    )
    trace_params: Dict[str, Any] = {
        "query_id_probabilities": dict(query_probabilities),
        "objective": str(prompt_key),
        "scene_variant": str(style.scene_variant),
        "scene_variant_probabilities": dict(style.scene_variant_probabilities),
        "leaf_count": int(case.sample.leaf_count),
        "node_color_name": str(style.node_color_name),
        "node_color_name_probabilities": dict(style.node_color_name_probabilities),
    }
    trace_params.update(dict(case.trace_params or {}))
    trace_params.update(dict(bound.trace_params))
    execution_trace = {
        "task_id": str(owner_id),
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "objective": str(prompt_key),
        **dict(bound.execution_trace),
    }
    trace_payload = assemble_trace_payload(
        owner_id=str(owner_id),
        public_query_id=str(selected_query),
        prompt_artifacts=prompt_artifacts,
        prompt_bundle_id=str(prompt_bundle_id),
        trace_params=trace_params,
        scene_ir=single_tree_scene_ir(
            owner_id=str(owner_id),
            sample=case.sample,
            bundle=rendered,
            entities=bound.entities,
            relations=bound.scene_relations,
        ),
        rendered=rendered,
        style=style,
        execution_trace=execution_trace,
        witness_symbolic=bound.witness_symbolic,
        projected_annotation=bound.projected_annotation,
    )
    return finalize_phylogeny_result(
        prompt_artifacts=prompt_artifacts,
        answer_type=str(bound.answer_type),
        answer_value=bound.answer_value,
        annotation_type=str(bound.annotation_type),
        annotation_value=bound.annotation_value,
        image=rendered.image,
        trace_payload=trace_payload,
        public_query_id=str(selected_query),
    )


__all__ = [
    "assemble_trace_payload",
    "BoundPhylogenyResult",
    "DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "PhylogenyDefaults",
    "RenderedPhylogenyBundle",
    "ResolvedPhylogenyStyle",
    "SingleTreeBinder",
    "SingleTreeCase",
    "SingleTreeCaseFactory",
    "integer_support",
    "finalize_phylogeny_result",
    "leaf_node_id",
    "query_spec",
    "render_option_trees",
    "render_single_tree",
    "render_spec",
    "resolve_integer_axis",
    "resolve_phylogeny_style",
    "run_single_tree_objective",
    "scene_default_sections",
    "single_tree_scene_ir",
]
