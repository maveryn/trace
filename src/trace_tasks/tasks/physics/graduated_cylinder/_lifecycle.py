"""Private lifecycle helpers for graduated-cylinder public objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import projected_bbox, projected_bbox_map
from .shared.output import build_object_witness, build_render_spec
from .shared.prompts import (
    PROMPT_BUNDLE_ID,
    build_graduated_cylinder_prompt_artifacts,
)
from .shared.rendering import (
    render_before_after_cylinder_scene,
    render_single_cylinder_scene,
)
from .shared.sampling import (
    choose_displacement,
    choose_scale,
    choose_volume,
    displacement_support,
    volume_support,
)
from .shared.state import SCENE_ID, SCENE_NAMESPACE


def run_volume_readout_lifecycle(
    *,
    domain: str,
    task_prompt_key: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run the single-cylinder volume readout objective."""

    _ = int(max_attempts)
    scale = choose_scale(int(instance_seed), namespace=SCENE_NAMESPACE)
    raw_volume = task_params.get("volume_ml")
    volume_ml = int(raw_volume) if raw_volume is not None else choose_volume(
        int(instance_seed),
        scale,
        namespace=SCENE_NAMESPACE,
    )
    rendered = render_single_cylinder_scene(
        instance_seed=int(instance_seed),
        params=task_params,
        scale=scale,
        volume_ml=int(volume_ml),
        rendering_defaults=rendering_defaults,
        namespace=SCENE_NAMESPACE,
    )
    prompt_artifacts = build_graduated_cylinder_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        task_key=str(prompt_defaults.get("task_key", task_prompt_key)),
        prompt_query_key="volume_readout",
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    answer_gt = TypedValue(type="integer", value=int(volume_ml))
    annotation_gt = TypedValue(type="bbox", value=list(rendered.annotation_bbox_map["readout"]))
    answer_support = list(volume_support(scale))
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params={
            "query_id": str(selected_branch),
            "operation": "volume_readout",
            "legacy_operation": "single_cylinder_volume_readout",
            "target_answer": int(volume_ml),
            "answer_support": list(answer_support),
            "query_id_probabilities": dict(branch_probabilities),
        },
    )
    trace_payload = {
        "scene_ir": {
            "scene_kind": "physics_graduated_cylinder_volume_readout",
            "entities": list(rendered.scene_entities),
            "relations": {
                "query_id": str(selected_branch),
                "operation": "volume_readout",
                "scale": scale.__dict__,
                "target_answer": int(volume_ml),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": build_render_spec(rendered),
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "query_id": str(selected_branch),
            "operation": "volume_readout",
            "legacy_operation": "single_cylinder_volume_readout",
            "target_answer": int(volume_ml),
            "annotation_entity_ids": ["readout"],
        },
        "witness_symbolic": build_object_witness(ids=["readout"]),
        "projected_annotation": projected_bbox(annotation_gt.value),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
    )


def run_displacement_lifecycle(
    *,
    domain: str,
    task_prompt_key: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run before/after displacement while binding both meniscus-scale roles."""

    _ = int(max_attempts)
    scale = choose_scale(int(instance_seed), namespace=SCENE_NAMESPACE)
    raw_before = task_params.get("before_volume_ml")
    before_ml = int(raw_before) if raw_before is not None else choose_volume(
        int(instance_seed),
        scale,
        min_ml=10,
        max_margin_ml=35,
        namespace=SCENE_NAMESPACE,
    )
    raw_displacement = task_params.get("displacement_ml", task_params.get("target_answer"))
    displacement_ml = (
        int(raw_displacement)
        if raw_displacement is not None
        else choose_displacement(int(instance_seed), scale, before_ml=int(before_ml))
    )
    if int(displacement_ml) not in set(displacement_support(scale, before_ml=int(before_ml))):
        raise ValueError("displacement_ml is not feasible for the selected scale and before volume")
    after_ml = int(before_ml) + int(displacement_ml)
    rendered = render_before_after_cylinder_scene(
        instance_seed=int(instance_seed),
        params=task_params,
        scale=scale,
        before_ml=int(before_ml),
        after_ml=int(after_ml),
        displacement_ml=int(displacement_ml),
        rendering_defaults=rendering_defaults,
        namespace=SCENE_NAMESPACE,
    )
    prompt_artifacts = build_graduated_cylinder_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        task_key=str(prompt_defaults.get("task_key", task_prompt_key)),
        prompt_query_key="displacement_volume",
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    answer_gt = TypedValue(type="integer", value=int(displacement_ml))
    annotation_gt = TypedValue(
        type="bbox_map",
        value={str(key): list(value) for key, value in rendered.annotation_bbox_map.items()},
    )
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params={
            "query_id": str(selected_branch),
            "operation": "displacement_volume",
            "legacy_operation": "before_after_displacement_volume",
            "before_volume_ml": int(before_ml),
            "after_volume_ml": int(after_ml),
            "target_answer": int(displacement_ml),
            "answer_support": list(displacement_support(scale, before_ml=int(before_ml))),
            "query_id_probabilities": dict(branch_probabilities),
        },
    )
    trace_payload = {
        "scene_ir": {
            "scene_kind": "physics_graduated_cylinder_displacement",
            "entities": list(rendered.scene_entities),
            "relations": {
                "query_id": str(selected_branch),
                "operation": "displacement_volume",
                "scale": scale.__dict__,
                "before_volume_ml": int(before_ml),
                "after_volume_ml": int(after_ml),
                "target_answer": int(displacement_ml),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": build_render_spec(rendered),
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "query_id": str(selected_branch),
            "operation": "displacement_volume",
            "legacy_operation": "before_after_displacement_volume",
            "before_volume_ml": int(before_ml),
            "after_volume_ml": int(after_ml),
            "target_answer": int(displacement_ml),
            "annotation_entity_ids": sorted(annotation_gt.value.keys()),
        },
        "witness_symbolic": build_object_witness(ids=sorted(annotation_gt.value.keys())),
        "projected_annotation": projected_bbox_map(annotation_gt.value),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
    )
