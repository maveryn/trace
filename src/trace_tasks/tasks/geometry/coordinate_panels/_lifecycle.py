"""Lifecycle runner for coordinate-panel public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import geometry_probability_map, select_geometry_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.defaults import resolve_label_pool, resolve_int_param
from .shared.output import panel_bboxes_by_label, plot_bboxes_by_label, sorted_panel_entities
from .shared.prompts import build_coordinate_panel_prompt_artifacts

DOMAIN = "geometry"
SCENE_ID = "coordinate_panels"
PANEL_SCENE_ID = SCENE_ID
DEFAULT_PANEL_LABEL_POOL: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
FIXED_PANEL_COUNT = 6
PROMPT_BUNDLE_ID = "geometry_coordinate_quadrilateral_v0"
_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id="coordinate")


@dataclass(frozen=True)
class ResolvedPanelQuery:
    """Task-owned query and answer-label binding used by the shared lifecycle."""

    query_id: str
    kind_value: str
    display_value: str
    query_probabilities: Dict[str, float]
    winner_label: str
    winner_label_probabilities: Dict[str, float]
    label_pool: Tuple[str, ...]
    panel_count: int
    panel_count_probabilities: Dict[str, float]
    extra_axes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatePanelTaskSpec:
    """Identity-free execution wiring supplied by one public task file."""

    public_identifier: str
    query_ids: Tuple[str, ...]
    query_kind_by_id: Mapping[str, str]
    display_by_kind: Mapping[str, str]
    kind_field_name: str
    display_field_name: str
    prompt_object_description_key: str
    prompt_annotation_hint_key: str
    scene_kind: str
    witness_type: str
    answer_type: str
    annotation_type: str
    render_scene: Callable[..., Any]
    panels_trace: Callable[[Any, ResolvedPanelQuery], Mapping[str, Mapping[str, Any]]]
    annotation_value: Callable[[Any, ResolvedPanelQuery], Any]
    render_map_extra: Callable[[Any], Mapping[str, Any]]
    projected_annotation: Callable[[Any, Any], Mapping[str, Any]]
    extra_axes: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] = lambda _params, _defaults: {}


def _split_defaults_for_task(task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=str(task_id),
    )


def _select_winner_label(
    *,
    task_id: str,
    params: Mapping[str, Any],
    instance_seed: int,
    label_pool: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    labels = tuple(str(label) for label in label_pool)
    explicit = params.get("winner_label", params.get("answer_label"))
    if explicit is not None:
        label = str(explicit)
        if label not in set(labels):
            raise ValueError(f"winner_label={label!r} is not in label pool {labels!r}")
        return label, {label: 1.0}
    rng = spawn_rng(int(instance_seed), f"{task_id}.winner_label")
    return str(uniform_choice(rng, labels)), geometry_probability_map(labels)


def _resolve_panel_query(
    *,
    spec: CoordinatePanelTaskSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> ResolvedPanelQuery:
    query_id, query_probabilities = select_geometry_query_id(
        params,
        query_ids=spec.query_ids,
        task_id=str(spec.public_identifier),
        instance_seed=int(instance_seed),
    )
    label_pool = resolve_label_pool(params, generation_defaults, "panel_labels", DEFAULT_PANEL_LABEL_POOL)
    winner_label, winner_probabilities = _select_winner_label(
        task_id=str(spec.public_identifier),
        params=params,
        instance_seed=int(instance_seed),
        label_pool=label_pool,
    )
    kind_value = str(spec.query_kind_by_id[str(query_id)])
    return ResolvedPanelQuery(
        query_id=str(query_id),
        kind_value=str(kind_value),
        display_value=str(spec.display_by_kind[str(kind_value)]),
        query_probabilities=dict(query_probabilities),
        winner_label=str(winner_label),
        winner_label_probabilities=dict(winner_probabilities),
        label_pool=tuple(str(label) for label in label_pool),
        panel_count=FIXED_PANEL_COUNT,
        panel_count_probabilities={str(FIXED_PANEL_COUNT): 1.0},
        extra_axes=dict(spec.extra_axes(params, generation_defaults)),
    )


def fixed_int_axis(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve a fixed integer axis from params/defaults for public task specs."""

    return resolve_int_param(params, defaults, str(key), int(fallback))


def _build_coordinate_panel_trace_payload(
    *,
    query_identifier: str,
    scene_kind: str,
    panels_trace: Mapping[str, Mapping[str, Any]],
    panels_by_label: Mapping[str, Any],
    rendered: Any,
    prompt_artifacts: Any,
    answer_label: str,
    answer_type: str,
    relations: Mapping[str, Any],
    semantic_params: Mapping[str, Any],
    render_map_extra: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    witness_type: str,
    witness_extra: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble query-bearing trace sections inside the private scene lifecycle."""

    return {
        "scene_id": SCENE_ID,
        "query_id": str(query_identifier),
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": sorted_panel_entities(panels_trace),
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(query_identifier),
                **dict(relations),
            },
        },
        "query_spec": {
            "query_id": str(query_identifier),
            "template_id": PROMPT_BUNDLE_ID,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "scene_id": SCENE_ID,
                "query_id": str(query_identifier),
                **dict(semantic_params),
            },
        },
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_count": int(len(panels_by_label)),
            "panel_count_probabilities": dict(rendered.option_count_probabilities),
            "panel_style": dict(rendered.panel_style_meta),
            "marker_style": dict(rendered.marker_meta),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "coord_space": "pixel",
            "panel_bboxes": panel_bboxes_by_label(panels_by_label),
            "plot_bboxes": plot_bboxes_by_label(panels_by_label),
            **dict(render_map_extra),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(query_identifier),
            "answer_type": str(answer_type),
            "answer_value": str(answer_label),
            "panels_by_label": dict(panels_trace),
            "panel_count_probabilities": dict(rendered.option_count_probabilities),
            **dict(execution_extra),
        },
        "witness_symbolic": {
            "type": str(witness_type),
            "answer_label": str(answer_label),
            "panels_by_label": dict(panels_trace),
            **dict(witness_extra),
        },
        "projected_annotation": dict(projected_annotation),
        "prompt": {
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        },
    }


def run_coordinate_panel_task(
    *,
    spec: CoordinatePanelTaskSpec,
    instance_seed: int,
    params: Dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate a coordinate-panel instance after the public file supplies objective wiring."""

    _ = int(max_attempts)
    generation_defaults, rendering_defaults, prompt_defaults_all = _split_defaults_for_task(str(spec.public_identifier))
    query = _resolve_panel_query(
        spec=spec,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    rendered = spec.render_scene(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        query=query,
        noise_defaults=_NOISE_DEFAULTS,
    )
    annotation_value = spec.annotation_value(rendered, query)
    prompt_artifacts = build_coordinate_panel_prompt_artifacts(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=PROMPT_BUNDLE_ID,
        prompt_defaults_all=prompt_defaults_all,
        prompt_query_key=str(query.query_id),
        object_description_key=str(spec.prompt_object_description_key),
        annotation_hint_key=str(spec.prompt_annotation_hint_key),
        annotation_value=annotation_value,
        answer_type=str(spec.answer_type),
        params=params,
        instance_seed=int(instance_seed),
        context=f"prompt defaults for {spec.public_identifier}",
    )
    panels_trace = spec.panels_trace(rendered, query)
    semantic_params = {
        "query_id_probabilities": dict(query.query_probabilities),
        str(spec.kind_field_name): str(query.kind_value),
        str(spec.display_field_name): str(query.display_value),
        "winner_label": str(query.winner_label),
        "winner_label_probabilities": dict(query.winner_label_probabilities),
        "candidate_label_pool": list(rendered.panels_by_label.keys()),
        "panel_count": int(query.panel_count),
        "panel_count_probabilities": dict(query.panel_count_probabilities),
        **dict(query.extra_axes),
    }
    trace_payload = _build_coordinate_panel_trace_payload(
        query_identifier=str(query.query_id),
        scene_kind=str(spec.scene_kind),
        panels_trace=panels_trace,
        panels_by_label=rendered.panels_by_label,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        answer_label=str(query.winner_label),
        answer_type=str(spec.answer_type),
        relations={
            "query_id_probabilities": dict(query.query_probabilities),
            str(spec.kind_field_name): str(query.kind_value),
            "winner_label": str(query.winner_label),
        },
        semantic_params=semantic_params,
        render_map_extra=spec.render_map_extra(rendered),
        execution_extra={
            "query_id_probabilities": dict(query.query_probabilities),
            str(spec.kind_field_name): str(query.kind_value),
            **dict(query.extra_axes),
        },
        witness_type=str(spec.witness_type),
        witness_extra={str(spec.kind_field_name): str(query.kind_value)},
        projected_annotation=spec.projected_annotation(rendered, annotation_value),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(spec.answer_type), value=str(query.winner_label)),
        annotation_gt=TypedValue(type=str(spec.annotation_type), value=annotation_value),
        image=rendered.image,
        image_id=f"{spec.public_identifier}:{int(instance_seed)}",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query.query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "CoordinatePanelTaskSpec",
    "DOMAIN",
    "FIXED_PANEL_COUNT",
    "PANEL_SCENE_ID",
    "PROMPT_BUNDLE_ID",
    "ResolvedPanelQuery",
    "SCENE_ID",
    "fixed_int_axis",
    "run_coordinate_panel_task",
]
