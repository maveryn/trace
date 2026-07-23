"""Objective runners for lever mechanics tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.config_defaults import required_group_defaults

from .output import build_lever_trace_payload, prepare_lever_render
from .prompts import PROMPT_BUNDLE_ID, build_lever_prompt_artifacts
from .sampling import (
    distance_support,
    generation_int,
    resolve_missing_weight_axes,
    resolve_side_torque_axes,
    sample_missing_weight_layout,
    sample_side_torque_layout,
    select_lever_public_branch,
)
from .state import LeverTaskDefaults, RenderedLeverScene


@dataclass(frozen=True)
class LeverTaskParts:
    """Generated prompt, answer, annotation, image, and trace before final output binding."""

    prompt: str
    prompt_variants: dict[str, str]
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Any
    trace_payload: dict[str, Any]
    public_branch: str

    def base_fields(self) -> dict[str, Any]:
        """Return fields common to final TaskOutput construction."""

        return {
            "prompt": str(self.prompt),
            "prompt_variants": dict(self.prompt_variants),
            "answer_gt": self.answer_gt,
            "annotation_gt": self.annotation_gt,
            "image": self.image,
            "image_id": "img0",
            "trace_payload": dict(self.trace_payload),
        }


def _annotation_map(rendered_scene: RenderedLeverScene) -> dict[str, list[list[float]]]:
    """Return keyed annotation boxes for the balancing-weight equation."""

    return {
        "known_weights": [list(bbox) for bbox in rendered_scene.known_weight_bboxes],
        "target_weight": [list(bbox) for bbox in rendered_scene.target_weight_bboxes],
    }


def side_torque_value_parts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    domain: str,
    public_name: str,
    namespace: str,
    prompt_key: str,
    prompt_branch: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
) -> LeverTaskParts:
    """Create side-torque generated parts from objective-specific symbolic sampling."""

    selected_branch, branch_probabilities, task_params = select_lever_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        public_name=str(public_name),
        namespace=str(namespace),
    )
    axes = resolve_side_torque_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        public_branch=str(selected_branch),
        public_branch_probabilities=branch_probabilities,
        generation_defaults=generation_defaults,
        defaults=defaults,
        namespace=str(namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{int(attempt_index)}")
        try:
            distances = distance_support(task_params, generation_defaults, defaults)
            placements, layout_metadata = sample_side_torque_layout(
                attempt_rng,
                target_torque=int(axes.target_answer),
                torque_side=str(axes.torque_side),
                distances=distances,
                weight_min=generation_int(task_params, generation_defaults, "weight_value_min", defaults.weight_value_min),
                weight_max=generation_int(task_params, generation_defaults, "weight_value_max", defaults.weight_value_max),
                max_weights=generation_int(task_params, generation_defaults, "max_side_weights", defaults.max_side_weights),
            )
        except ValueError:
            continue

        prepared_render = prepare_lever_render(
            params=task_params,
            instance_seed=int(instance_seed),
            rendering_defaults=rendering_defaults,
            generation_defaults=generation_defaults,
            fallback_defaults=defaults,
            namespace=str(namespace),
            scene_variant=str(axes.scene_variant),
            accent_color_name=str(axes.accent_color_name),
            placements=list(placements),
            support_key="torque_answer_support",
            support_fallback=defaults.torque_answer_support,
        )
        resolved_prompt_defaults = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {public_name}",
        )
        prompt_artifacts = build_lever_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(resolved_prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
            task_key=str(resolved_prompt_defaults.get("task_key", prompt_key)),
            prompt_query_key=str(prompt_branch),
            dynamic_slots={"torque_side": str(axes.torque_side)},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(axes.target_answer))
        annotation_gt = TypedValue(
            type="bbox_set",
            value=[list(bbox) for bbox in prepared_render.rendered_scene.relevant_weight_bboxes],
        )
        trace_payload = build_lever_trace_payload(
            scene_variant=str(axes.scene_variant),
            public_branch=str(axes.public_branch),
            public_branch_probabilities=axes.public_branch_probabilities,
            internal_branch=f"{axes.torque_side}_torque",
            torque_side=str(axes.torque_side),
            accent_color_name=str(axes.accent_color_name),
            target_answer=int(axes.target_answer),
            scene_variant_probabilities=axes.scene_variant_probabilities,
            accent_color_name_probabilities=axes.accent_color_name_probabilities,
            target_answer_probabilities=axes.target_answer_probabilities,
            rendered_scene=prepared_render.rendered_scene,
            prompt_artifacts=prompt_artifacts,
            layout_metadata=layout_metadata,
            layout_placement_meta=prepared_render.layout_placement_meta,
            image_size=prepared_render.image_size,
            background_meta=prepared_render.background_meta,
            diagram_style_meta=prepared_render.diagram_style_meta,
            post_noise_meta=prepared_render.post_noise_meta,
            font_family=prepared_render.font_family,
            target_support=prepared_render.target_support,
            annotation_entity_ids=prepared_render.rendered_scene.relevant_weight_ids,
            projected_annotation_type="bbox_set",
            projected_bbox_set=prepared_render.rendered_scene.relevant_weight_bboxes,
            projected_bbox_set_map=dict(prepared_render.rendered_scene.render_map["annotation_bbox_set_map_px"]),
            extra_prompt_fields={"torque_side_probabilities": dict(axes.torque_side_probabilities)},
        )
        return LeverTaskParts(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=prepared_render.image,
            trace_payload=trace_payload,
            public_branch=str(axes.public_branch),
        )
    raise RuntimeError(f"{public_name} failed to generate a valid scene after {max_attempts} attempts")


def missing_weight_balance_value_parts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    domain: str,
    public_name: str,
    namespace: str,
    prompt_key: str,
    prompt_branch: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
) -> LeverTaskParts:
    """Create missing-weight generated parts from objective-specific symbolic sampling."""

    selected_branch, branch_probabilities, task_params = select_lever_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        public_name=str(public_name),
        namespace=str(namespace),
    )
    axes = resolve_missing_weight_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        public_branch=str(selected_branch),
        public_branch_probabilities=branch_probabilities,
        generation_defaults=generation_defaults,
        defaults=defaults,
        namespace=str(namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{int(attempt_index)}")
        try:
            distances = distance_support(task_params, generation_defaults, defaults)
            placements, layout_metadata = sample_missing_weight_layout(
                attempt_rng,
                target_weight=int(axes.target_answer),
                distances=distances,
                weight_min=generation_int(task_params, generation_defaults, "weight_value_min", defaults.weight_value_min),
                weight_max=generation_int(task_params, generation_defaults, "weight_value_max", defaults.weight_value_max),
                max_side_weights=generation_int(
                    task_params,
                    generation_defaults,
                    "missing_weight_max_side_weights",
                    defaults.missing_weight_max_side_weights,
                ),
            )
        except ValueError:
            continue

        prepared_render = prepare_lever_render(
            params=task_params,
            instance_seed=int(instance_seed),
            rendering_defaults=rendering_defaults,
            generation_defaults=generation_defaults,
            fallback_defaults=defaults,
            namespace=str(namespace),
            scene_variant=str(axes.scene_variant),
            accent_color_name=str(axes.accent_color_name),
            placements=list(placements),
            support_key="missing_weight_support",
            support_fallback=defaults.missing_weight_support,
        )
        resolved_prompt_defaults = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {public_name}",
        )
        prompt_artifacts = build_lever_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(resolved_prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
            task_key=str(resolved_prompt_defaults.get("task_key", prompt_key)),
            prompt_query_key=str(prompt_branch),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        annotation_value = _annotation_map(prepared_render.rendered_scene)
        answer_gt = TypedValue(type="integer", value=int(axes.target_answer))
        annotation_gt = TypedValue(type="bbox_set_map", value=annotation_value)
        trace_payload = build_lever_trace_payload(
            scene_variant=str(axes.scene_variant),
            public_branch=str(axes.public_branch),
            public_branch_probabilities=axes.public_branch_probabilities,
            internal_branch="missing_weight_to_balance",
            torque_side=None,
            accent_color_name=str(axes.accent_color_name),
            target_answer=int(axes.target_answer),
            scene_variant_probabilities=axes.scene_variant_probabilities,
            accent_color_name_probabilities=axes.accent_color_name_probabilities,
            target_answer_probabilities=axes.target_answer_probabilities,
            rendered_scene=prepared_render.rendered_scene,
            prompt_artifacts=prompt_artifacts,
            layout_metadata=layout_metadata,
            layout_placement_meta=prepared_render.layout_placement_meta,
            image_size=prepared_render.image_size,
            background_meta=prepared_render.background_meta,
            diagram_style_meta=prepared_render.diagram_style_meta,
            post_noise_meta=prepared_render.post_noise_meta,
            font_family=prepared_render.font_family,
            target_support=prepared_render.target_support,
            annotation_entity_ids=("known_weights", "target_weight"),
            projected_annotation_type="bbox_set_map",
            projected_bbox_set=prepared_render.rendered_scene.relevant_weight_bboxes,
            projected_bbox_set_map=annotation_value,
        )
        return LeverTaskParts(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=prepared_render.image,
            trace_payload=trace_payload,
            public_branch=str(axes.public_branch),
        )
    raise RuntimeError(f"{public_name} failed to generate a valid scene after {max_attempts} attempts")


__all__ = [
    "LeverTaskParts",
    "missing_weight_balance_value_parts",
    "side_torque_value_parts",
]
