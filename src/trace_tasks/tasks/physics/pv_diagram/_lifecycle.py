"""Scene-private render and prompt lifecycle helpers for PV diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.render_variation import resolve_render_int

from .shared.annotations import build_prompt_examples, scalar_bbox_artifacts, single_annotation_bbox
from .shared.output import build_font_trace, object_witness
from .shared.prompts import build_pv_diagram_prompt_artifacts
from .shared.rendering import (
    RENDER_DEFAULT_KEYS,
    render_pv_diagram_scene,
    resolve_pv_layout_placement,
)
from .shared.sampling import (
    pressure_support,
    resolve_sign_choice_axes,
    resolve_work_axes,
    sample_sign_choice_scene_spec,
    sample_work_scene_spec,
    sign_for_work,
    volume_support,
    work_answer_support,
)
from .shared.state import PVDiagramSceneSpec, PVDiagramTaskDefaults, RenderedPVDiagramScene, SCENE_ID
from .shared.state import OPTION_LETTERS


@dataclass(frozen=True)
class NumericWorkLifecyclePlan:
    """Public objective bindings for the numeric PV-work lifecycle."""

    task_identifier: str
    namespace: str
    public_branch_ids: tuple[str, ...]
    default_branch_id: str
    prompt_branch_key: str
    fallback_defaults: PVDiagramTaskDefaults
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]
    prompt_defaults_group: Mapping[str, Any]
    post_noise_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class SignOptionLifecyclePlan:
    """Public objective bindings for the process sign option lifecycle."""

    task_identifier: str
    namespace: str
    public_branch_ids: tuple[str, ...]
    default_branch_id: str
    prompt_branch_key: str
    fallback_defaults: PVDiagramTaskDefaults
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]
    prompt_defaults_group: Mapping[str, Any]
    post_noise_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class PVDiagramRenderedAssets:
    """Rendered image and style metadata reused by PV objective files."""

    rendered_scene: RenderedPVDiagramScene
    image: Image.Image
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    layout_placement_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str


def _scenario_payload(scene_spec: PVDiagramSceneSpec) -> dict[str, Any]:
    scenario = scene_spec.work_scenario
    if scenario is None:
        return {}
    return {
        "work_mode": str(scenario.work_mode),
        "work_value": int(scenario.work_value),
        "work_sign": sign_for_work(int(scenario.work_value)),
        "pressure_kpa": scenario.pressure,
        "volume_start_l": scenario.volume_start,
        "volume_end_l": scenario.volume_end,
        "pressure_low_kpa": scenario.pressure_low,
        "pressure_high_kpa": scenario.pressure_high,
        "volume_left_l": scenario.volume_left,
        "volume_right_l": scenario.volume_right,
        "cycle_direction": scenario.cycle_direction,
    }


def _candidate_payload(scene_spec: PVDiagramSceneSpec) -> list[dict[str, Any]]:
    return [
        {
            "option_letter": str(candidate.letter),
            "sign": str(candidate.sign),
            "pressure_start_kpa": int(candidate.pressure_start),
            "pressure_end_kpa": int(candidate.pressure_end),
            "volume_start_l": int(candidate.volume_start),
            "volume_end_l": int(candidate.volume_end),
            "delta_volume_l": int(candidate.volume_end - candidate.volume_start),
            "is_correct": str(candidate.letter) == str(scene_spec.correct_option_letter),
        }
        for candidate in scene_spec.process_candidates
    ]


def _target_sign_description(target_sign: str) -> str:
    if str(target_sign) == "positive":
        return "positive"
    if str(target_sign) == "negative":
        return "negative"
    return "zero"


def resolve_scene_render_defaults(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback_defaults: PVDiagramTaskDefaults,
    sample_namespace: str,
) -> dict[str, Any]:
    """Resolve integer render defaults for one objective-owned PV scene."""

    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(sample_namespace),
        )
        for key in RENDER_DEFAULT_KEYS
    }


def build_scene_render_assets(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_spec: PVDiagramSceneSpec,
    accent_color_name: str,
    rendering_defaults: Mapping[str, Any],
    fallback_defaults: PVDiagramTaskDefaults,
    sample_namespace: str,
    post_noise_defaults: Mapping[str, Any],
) -> PVDiagramRenderedAssets:
    """Render one PV scene and return final-image projection metadata."""

    render_defaults = resolve_scene_render_defaults(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        fallback_defaults=fallback_defaults,
        sample_namespace=str(sample_namespace),
    )
    render_defaults, layout_placement_meta = resolve_pv_layout_placement(
        render_defaults=render_defaults,
        rendering_defaults=rendering_defaults,
        params=params,
        instance_seed=int(instance_seed),
        scene_spec=scene_spec,
    )
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            scene_id=SCENE_ID,
            canvas_width=int(render_defaults["canvas_width"]),
            canvas_height=int(render_defaults["canvas_height"]),
            instance_seed=int(instance_seed),
            params=params,
            style_profile="graph_paper",
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{sample_namespace}.render.font",
        params=params,
    )
    rendered_scene = render_pv_diagram_scene(
        background=background,
        render_defaults=render_defaults,
        accent_color_name=str(accent_color_name),
        scene_spec=scene_spec,
        diagram_style=diagram_style,
        font_family=str(font_family),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=post_noise_defaults,
    )
    return PVDiagramRenderedAssets(
        rendered_scene=rendered_scene,
        image=image,
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        layout_placement_meta=dict(layout_placement_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


def build_scene_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults_group: Mapping[str, Any],
    prompt_branch_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Load the PV prompt bundle and fill objective-provided dynamic slots."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_group,
        ("bundle_id", "task_key"),
        context="PV diagram prompt defaults",
    )
    return build_pv_diagram_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_defaults["bundle_id"]),
        task_key=str(prompt_defaults["task_key"]),
        prompt_branch_key=str(prompt_branch_key),
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )


def run_numeric_work_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: NumericWorkLifecyclePlan,
) -> TaskOutput:
    """Generate the numeric PV-work objective from sampling through TaskOutput."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=tuple(str(value) for value in plan.public_branch_ids),
        default_query_id=str(plan.default_branch_id),
        task_id=str(plan.task_identifier),
        namespace=f"{plan.namespace}.query",
    )
    axes = resolve_work_axes(
        int(instance_seed),
        params=task_params,
        generation_defaults=plan.generation_defaults,
        namespace=str(plan.namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{plan.namespace}.attempt.{int(attempt_index)}")
        try:
            scene_spec = sample_work_scene_spec(
                attempt_rng,
                axes=axes,
                params=task_params,
                generation_defaults=plan.generation_defaults,
            )
        except ValueError:
            continue

        assets = build_scene_render_assets(
            params=task_params,
            instance_seed=int(instance_seed),
            scene_spec=scene_spec,
            accent_color_name=str(axes.accent_color_name),
            rendering_defaults=plan.rendering_defaults,
            fallback_defaults=plan.fallback_defaults,
            sample_namespace=str(plan.namespace),
            post_noise_defaults=plan.post_noise_defaults,
        )
        rendered_scene = assets.rendered_scene

        json_example, json_example_answer_only = build_prompt_examples(answer_type="integer")
        prompt_artifacts = build_scene_prompt_artifacts(
            domain=str(domain),
            prompt_defaults_group=plan.prompt_defaults_group,
            prompt_branch_key=str(plan.prompt_branch_key),
            dynamic_slots={
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )

        if scene_spec.work_scenario is None:
            raise RuntimeError("missing work scenario after PV scene render")
        answer_value = int(scene_spec.work_scenario.work_value)
        answer_gt = TypedValue(type="integer", value=int(answer_value))
        annotation_bbox = single_annotation_bbox(rendered_scene.annotation_bboxes)
        annotation_artifacts = scalar_bbox_artifacts(annotation_bbox)
        scenario_payload = _scenario_payload(scene_spec)
        query_params = {
            "query_id": str(selected_branch),
            "prompt_branch": str(plan.prompt_branch_key),
            "scene_variant": str(axes.scene_variant),
            "work_mode": str(axes.work_mode),
            "accent_color_name": str(axes.accent_color_name),
            "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
            "query_id_probabilities": dict(branch_probabilities),
            "work_mode_probabilities": dict(axes.work_mode_probabilities),
            "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
            "target_answer": int(answer_value),
            "target_answer_probabilities": dict(axes.target_answer_probabilities),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_pv_diagram_{str(axes.scene_variant)}",
                "entities": [dict(entity) for entity in rendered_scene.scene_entities],
                "relations": {
                    "scene_variant": str(axes.scene_variant),
                    "work_mode": str(axes.work_mode),
                    "accent_color_name": str(axes.accent_color_name),
                    "target_answer": int(answer_value),
                    "answer_type": "integer",
                    "scenario": dict(scenario_payload),
                    "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=query_params,
            ),
            "render_spec": {
                "scene_variant": str(axes.scene_variant),
                "canvas_width": int(assets.image.size[0]),
                "canvas_height": int(assets.image.size[1]),
                "accent_color_name": str(axes.accent_color_name),
                "font": build_font_trace(font_family=str(assets.font_family)),
                "technical_diagram_style": dict(assets.diagram_style_meta),
                "background_style": dict(assets.background_meta),
                "layout_placement": dict(assets.layout_placement_meta),
                "post_image_noise": dict(assets.post_noise_meta),
            },
            "render_map": dict(rendered_scene.render_map),
            "execution_trace": {
                "query_id": str(selected_branch),
                "prompt_branch": str(plan.prompt_branch_key),
                "scene_variant": str(axes.scene_variant),
                "work_mode": str(axes.work_mode),
                "accent_color_name": str(axes.accent_color_name),
                "target_answer": int(answer_value),
                "answer_type": "integer",
                "work_answer_support": list(work_answer_support(task_params, plan.generation_defaults)),
                "pressure_support": list(pressure_support(task_params, plan.generation_defaults)),
                "volume_support": list(volume_support(task_params, plan.generation_defaults)),
                "scenario": dict(scenario_payload),
                "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
            },
            "witness_symbolic": object_witness(rendered_scene.annotation_entity_ids),
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
            "background": dict(assets.background_meta),
            "post_image_noise": dict(assets.post_noise_meta),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=assets.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )

    raise RuntimeError(
        f"{plan.task_identifier} failed to generate a valid scene after {max_attempts} attempts"
    )


def run_sign_option_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: SignOptionLifecyclePlan,
) -> TaskOutput:
    """Generate the PV sign-choice objective from candidates through TaskOutput."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=tuple(str(value) for value in plan.public_branch_ids),
        default_query_id=str(plan.default_branch_id),
        task_id=str(plan.task_identifier),
        namespace=f"{plan.namespace}.query",
    )
    axes = resolve_sign_choice_axes(
        int(instance_seed),
        params=task_params,
        generation_defaults=plan.generation_defaults,
        namespace=str(plan.namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{plan.namespace}.attempt.{int(attempt_index)}")
        scene_spec = sample_sign_choice_scene_spec(attempt_rng, axes=axes)

        assets = build_scene_render_assets(
            params=task_params,
            instance_seed=int(instance_seed),
            scene_spec=scene_spec,
            accent_color_name=str(axes.accent_color_name),
            rendering_defaults=plan.rendering_defaults,
            fallback_defaults=plan.fallback_defaults,
            sample_namespace=str(plan.namespace),
            post_noise_defaults=plan.post_noise_defaults,
        )
        rendered_scene = assets.rendered_scene

        json_example, json_example_answer_only = build_prompt_examples(answer_type="option_letter")
        prompt_artifacts = build_scene_prompt_artifacts(
            domain=str(domain),
            prompt_defaults_group=plan.prompt_defaults_group,
            prompt_branch_key=str(plan.prompt_branch_key),
            dynamic_slots={
                "target_sign": _target_sign_description(str(axes.target_sign)),
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )

        answer_value = str(scene_spec.correct_option_letter)
        answer_gt = TypedValue(type="option_letter", value=str(answer_value))
        annotation_bbox = single_annotation_bbox(rendered_scene.annotation_bboxes)
        annotation_artifacts = scalar_bbox_artifacts(annotation_bbox)
        candidate_payload = _candidate_payload(scene_spec)
        query_params = {
            "query_id": str(selected_branch),
            "prompt_branch": str(plan.prompt_branch_key),
            "scene_variant": str(axes.scene_variant),
            "target_sign": str(axes.target_sign),
            "correct_option_letter": str(axes.correct_option_letter),
            "accent_color_name": str(axes.accent_color_name),
            "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
            "query_id_probabilities": dict(branch_probabilities),
            "target_sign_probabilities": dict(axes.target_sign_probabilities),
            "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
            "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
            "target_answer": str(answer_value),
            "target_answer_probabilities": dict(axes.target_answer_probabilities),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_pv_diagram_{str(axes.scene_variant)}",
                "entities": [dict(entity) for entity in rendered_scene.scene_entities],
                "relations": {
                    "scene_variant": str(axes.scene_variant),
                    "target_sign": str(axes.target_sign),
                    "correct_option_letter": str(axes.correct_option_letter),
                    "accent_color_name": str(axes.accent_color_name),
                    "target_answer": str(answer_value),
                    "answer_type": "option_letter",
                    "process_candidates": list(candidate_payload),
                    "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=query_params,
            ),
            "render_spec": {
                "scene_variant": str(axes.scene_variant),
                "canvas_width": int(assets.image.size[0]),
                "canvas_height": int(assets.image.size[1]),
                "accent_color_name": str(axes.accent_color_name),
                "font": build_font_trace(font_family=str(assets.font_family)),
                "technical_diagram_style": dict(assets.diagram_style_meta),
                "background_style": dict(assets.background_meta),
                "layout_placement": dict(assets.layout_placement_meta),
                "post_image_noise": dict(assets.post_noise_meta),
            },
            "render_map": dict(rendered_scene.render_map),
            "execution_trace": {
                "query_id": str(selected_branch),
                "prompt_branch": str(plan.prompt_branch_key),
                "scene_variant": str(axes.scene_variant),
                "target_sign": str(axes.target_sign),
                "correct_option_letter": str(axes.correct_option_letter),
                "accent_color_name": str(axes.accent_color_name),
                "target_answer": str(answer_value),
                "answer_type": "option_letter",
                "option_letters": list(OPTION_LETTERS),
                "process_candidates": list(candidate_payload),
                "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
            },
            "witness_symbolic": object_witness(rendered_scene.annotation_entity_ids),
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
            "background": dict(assets.background_meta),
            "post_image_noise": dict(assets.post_noise_meta),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=assets.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )

    raise RuntimeError(
        f"{plan.task_identifier} failed to generate a valid scene after {max_attempts} attempts"
    )


__all__ = [
    "NumericWorkLifecyclePlan",
    "PVDiagramRenderedAssets",
    "SignOptionLifecyclePlan",
    "build_scene_prompt_artifacts",
    "build_scene_render_assets",
    "resolve_scene_render_defaults",
    "run_numeric_work_lifecycle",
    "run_sign_option_lifecycle",
]
