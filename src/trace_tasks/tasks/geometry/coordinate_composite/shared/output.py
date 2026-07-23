"""Output helpers for coordinate-composite candidate-point tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.geometry.shared.vector2d import point_to_list
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .relations import filtered_intersections
from .rendering import render_coordinate_composite_scene
from .prompts import build_coordinate_composite_prompt_artifacts
from .state import PairFilter, SceneObject

_SELECTION_TRACE_KEY = "_".join(("query", "id"))


@dataclass(frozen=True)
class CandidatePointArtifacts:
    """Rendered image, scalar annotation, prompt, and verifier trace payload."""

    rendered: Any
    prompt_artifacts: Any
    annotation_value: list[float]
    trace_payload: dict[str, Any]


def candidate_point_annotation_by_label(rendered: Any, *, selected_label: str) -> list[float]:
    """Return the canonical pixel point for a selected rendered candidate label."""

    point_by_label = {
        str(item["label"]): list(item["point"])
        for item in rendered.render_map["candidate_points_px"]
    }
    return [float(value) for value in point_by_label[str(selected_label)]]


def build_candidate_point_trace_payload(
    *,
    scene_id: str,
    family_code: str,
    selection_key: str,
    selection_probabilities: Mapping[str, float],
    case_id: str,
    case_probabilities: Mapping[str, float],
    transform: str,
    transform_probabilities: Mapping[str, float],
    label_probabilities: Mapping[str, float],
    rendered: Any,
    prompt_artifacts: Any,
    selected_label: str,
    selected_point_graph: Sequence[float],
    annotation_value: Sequence[float],
) -> dict[str, Any]:
    """Build verifier trace metadata for a scalar candidate-point selection."""

    return {
        "scene_id": str(scene_id),
        _SELECTION_TRACE_KEY: str(selection_key),
        "scene_ir": {
            "scene_kind": "geometry_coordinate_composite",
            "scene_id": str(scene_id),
            "objects": [dict(obj) for obj in rendered.object_specs],
            "candidate_points": list(rendered.render_map["candidate_points_graph"]),
        },
        "query_spec": {
            _SELECTION_TRACE_KEY: str(selection_key),
            "query_id_probabilities": dict(selection_probabilities),
            "case_id": str(case_id),
            "case_probabilities": dict(case_probabilities),
            "transform": str(transform),
            "transform_probabilities": dict(transform_probabilities),
            "answer_label_probabilities": dict(label_probabilities),
            "params": {
                _SELECTION_TRACE_KEY: str(selection_key),
                "case_id": str(case_id),
                "transform": str(transform),
            },
        },
        "render_spec": {
            "canvas_width": int(rendered.image.width),
            "canvas_height": int(rendered.image.height),
            "background": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            **dict(rendered.render_spec_extra),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "formula_family": str(family_code),
            _SELECTION_TRACE_KEY: str(selection_key),
            "case_id": str(case_id),
            "answer_label": str(selected_label),
            "answer_point_graph": point_to_list((float(selected_point_graph[0]), float(selected_point_graph[1]))),
            "answer_point_px": list(annotation_value),
        },
        "witness_symbolic": {
            "formula_family": str(family_code),
            "objects_graph": [dict(obj) for obj in rendered.object_specs],
            "candidate_points_graph": list(rendered.render_map["candidate_points_graph"]),
            "answer_label": str(selected_label),
            "answer_point_graph": point_to_list((float(selected_point_graph[0]), float(selected_point_graph[1]))),
        },
        "projected_annotation": {
            "type": "point",
            "point": list(annotation_value),
            "pixel_point": list(annotation_value),
            "source": "answer_candidate_graph_point_projected_to_pixels",
        },
        "prompt": {
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        },
    }


def compose_candidate_point_artifacts(
    *,
    domain: str,
    scene_id: str,
    family_code: str,
    selection_key: str,
    selection_probabilities: Mapping[str, float],
    case_id: str,
    case_probabilities: Mapping[str, float],
    transform: str,
    transform_probabilities: Mapping[str, float],
    label_probabilities: Mapping[str, float],
    selected_label: str,
    selected_point_graph: Sequence[float],
    objects: tuple[SceneObject, ...],
    labeled_points: Sequence[tuple[str, tuple[float, float]]],
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    random_namespace: str,
    prompt_bundle_id: str,
    instance_seed: int,
) -> CandidatePointArtifacts:
    """Compose render, prompt, annotation, and trace primitives for point choices."""

    rendered = render_coordinate_composite_scene(
        instance_seed=int(instance_seed),
        objects=tuple(objects),
        pair_filter=PairFilter.ALL,
        transform=str(transform),
        expected_count=len(filtered_intersections(tuple(objects), PairFilter.ALL)),
        params=params,
        render_defaults=render_defaults,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        random_namespace=str(random_namespace),
        candidate_points=tuple(labeled_points),
    )
    annotation_value = candidate_point_annotation_by_label(rendered, selected_label=str(selected_label))
    prompt_artifacts = build_coordinate_composite_prompt_artifacts(
        domain=str(domain),
        scene_id=str(scene_id),
        prompt_defaults=prompt_defaults,
        params=params,
        instance_seed=int(instance_seed),
        query_key=str(selection_key),
        prompt_bundle_id=str(prompt_bundle_id),
        object_description_key="object_description_candidates",
        annotation_hint_key="annotation_hint_candidate_point",
        answer_hint_key="answer_hint_option_label",
        json_example_key="json_example_candidate_point",
        json_example_answer_only_key="json_example_answer_only_label",
        context=f"prompt defaults for {family_code}",
    )
    trace_payload = build_candidate_point_trace_payload(
        scene_id=str(scene_id),
        family_code=str(family_code),
        selection_key=str(selection_key),
        selection_probabilities=selection_probabilities,
        case_id=str(case_id),
        case_probabilities=case_probabilities,
        transform=str(transform),
        transform_probabilities=transform_probabilities,
        label_probabilities=label_probabilities,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        selected_label=str(selected_label),
        selected_point_graph=selected_point_graph,
        annotation_value=list(annotation_value),
    )
    return CandidatePointArtifacts(
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation_value=list(annotation_value),
        trace_payload=trace_payload,
    )


def compose_resolved_candidate_point_artifacts(
    *,
    domain: str,
    scene_id: str,
    family_code: str,
    resolved: Any,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    random_namespace: str,
    prompt_bundle_id: str,
    instance_seed: int,
) -> CandidatePointArtifacts:
    """Compose artifacts from a task-owned resolved candidate-point problem."""

    case = getattr(resolved, "case")
    return compose_candidate_point_artifacts(
        domain=str(domain),
        scene_id=str(scene_id),
        family_code=str(family_code),
        selection_key=str(getattr(resolved, "selection_key")),
        selection_probabilities=dict(getattr(resolved, "selection_probabilities")),
        case_id=str(getattr(case, "case_id")),
        case_probabilities=dict(getattr(resolved, "case_probabilities")),
        transform=str(getattr(resolved, "transform")),
        transform_probabilities=dict(getattr(resolved, "transform_probabilities")),
        label_probabilities=dict(getattr(resolved, "label_probabilities")),
        selected_label=str(getattr(resolved, "selected_label")),
        selected_point_graph=getattr(resolved, "selected_point_graph"),
        objects=tuple(getattr(resolved, "objects")),
        labeled_points=tuple(getattr(resolved, "labeled_points")),
        params=params,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        random_namespace=str(random_namespace),
        prompt_bundle_id=str(prompt_bundle_id),
        instance_seed=int(instance_seed),
    )


def candidate_point_output_fields(
    *,
    artifacts: CandidatePointArtifacts,
    answer_gt: Any,
    annotation_gt: Any,
    image_id: str,
    scene_id: str,
    selection_key: str,
) -> dict[str, Any]:
    """Return common output fields while public task code calls ``TaskOutput``."""

    return {
        "prompt": str(artifacts.prompt_artifacts.prompt),
        "answer_gt": answer_gt,
        "annotation_gt": annotation_gt,
        "image": artifacts.rendered.image,
        "image_id": str(image_id),
        "trace_payload": dict(artifacts.trace_payload),
        "task_versions": default_task_versions(),
        "scene_id": str(scene_id),
        _SELECTION_TRACE_KEY: str(selection_key),
        "prompt_variants": dict(artifacts.prompt_artifacts.prompt_variants),
    }


__all__ = [
    "CandidatePointArtifacts",
    "build_candidate_point_trace_payload",
    "candidate_point_annotation_by_label",
    "candidate_point_output_fields",
    "compose_candidate_point_artifacts",
    "compose_resolved_candidate_point_artifacts",
]
