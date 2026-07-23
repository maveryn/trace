"""Objective assembly for spring physics tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import required_group_defaults

from .annotations import spring_missing_value_annotation_map
from .output import build_spring_trace_payload, prepare_spring_render
from .prompts import PROMPT_BUNDLE_ID, build_spring_prompt_artifacts, build_spring_prompt_examples
from .sampling import resolve_spring_axes, sample_spring_scene_spec, scale_factor_support
from .state import SPRING_MODE_DIFFERENCE, SpringTaskDefaults


@dataclass(frozen=True)
class SpringTaskParts:
    """Generated prompt, answer, annotation, image, and trace before final output binding."""

    prompt: str
    prompt_variants: dict[str, str]
    answer_value: int
    annotation_type: str
    annotation_value: Any
    image: Any
    trace_payload: dict[str, Any]
    public_branch: str

    def base_fields(self) -> dict[str, Any]:
        """Return fields common to final TaskOutput construction."""

        return {
            "prompt": str(self.prompt),
            "prompt_variants": dict(self.prompt_variants),
            "image": self.image,
            "image_id": "img0",
            "trace_payload": dict(self.trace_payload),
        }


def spring_task_parts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    domain: str,
    public_name: str,
    namespace: str,
    spring_mode: str,
    public_branch: str,
    internal_branch: str,
    public_branch_probabilities: Mapping[str, float],
    solve_for: str | None,
    prompt_key: str,
    prompt_branch: str,
    prompt_dynamic_slots: Mapping[str, Any],
    target_support_key: str,
    target_support_fallback: Sequence[int],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: SpringTaskDefaults,
) -> SpringTaskParts:
    """Create generated parts from spring symbolic sampling and rendering."""

    axes = resolve_spring_axes(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        defaults=defaults,
        spring_mode=str(spring_mode),
        public_branch=str(public_branch),
        internal_branch=str(internal_branch),
        public_branch_probabilities=public_branch_probabilities,
        solve_for=solve_for,
        target_support_key=str(target_support_key),
        target_support_fallback=tuple(int(value) for value in target_support_fallback),
        namespace=str(namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{int(attempt_index)}")
        try:
            scene_spec = sample_spring_scene_spec(
                attempt_rng,
                scene_variant=str(axes.scene_variant),
                spring_mode=str(axes.spring_mode),
                target_answer=int(axes.target_answer),
                params=params,
                generation_defaults=generation_defaults,
                defaults=defaults,
            )
        except ValueError:
            continue

        prepared_render = prepare_spring_render(
            instance_seed=int(instance_seed),
            params=params,
            rendering_defaults=rendering_defaults,
            generation_defaults=generation_defaults,
            fallback_defaults=defaults,
            namespace=str(namespace),
            scene_variant=str(axes.scene_variant),
            accent_color_name=str(axes.accent_color_name),
            scene_spec=scene_spec,
            support_key=str(target_support_key),
            support_fallback=tuple(int(value) for value in target_support_fallback),
        )
        resolved_prompt_defaults = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {public_name}",
        )
        prompt_artifacts = build_spring_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(resolved_prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
            task_key=str(resolved_prompt_defaults.get("task_key", prompt_key)),
            prompt_query_key=str(prompt_branch),
            dynamic_slots=dict(prompt_dynamic_slots),
            instance_seed=int(instance_seed),
        )
        if str(axes.spring_mode) == SPRING_MODE_DIFFERENCE:
            annotation_value = [
                list(bbox) for bbox in prepared_render.rendered_scene.annotation_bboxes
            ]
            annotation_type = "bbox_set"
            projected_annotation = {
                "type": "bbox_set",
                "bbox_set": list(annotation_value),
                "pixel_bbox_set": list(annotation_value),
            }
        else:
            annotation_value = spring_missing_value_annotation_map(prepared_render.rendered_scene)
            annotation_type = "bbox_map"
            projected_annotation = {
                "type": "bbox_map",
                "bbox_map": dict(annotation_value),
                "pixel_bbox_map": dict(annotation_value),
            }
            prepared_render.rendered_scene.render_map["annotation_bbox_map_px"] = dict(annotation_value)

        trace_payload = build_spring_trace_payload(
            scene_variant=str(axes.scene_variant),
            public_branch=str(axes.public_branch),
            public_branch_probabilities=axes.public_branch_probabilities,
            internal_branch=str(axes.internal_branch),
            solve_for=axes.solve_for,
            accent_color_name=str(axes.accent_color_name),
            target_answer=int(axes.target_answer),
            scale_factor=int(scene_spec.scale_factor),
            scale_factor_support=scale_factor_support(
                params,
                generation_defaults=generation_defaults,
                defaults=defaults,
                spring_mode=str(axes.spring_mode),
            ),
            scene_variant_probabilities=axes.scene_variant_probabilities,
            accent_color_name_probabilities=axes.accent_color_name_probabilities,
            target_answer_probabilities=axes.target_answer_probabilities,
            rendered_scene=prepared_render.rendered_scene,
            prompt_artifacts=prompt_artifacts,
            layout_placement_meta=prepared_render.layout_placement_meta,
            image_size=prepared_render.image_size,
            background_meta=prepared_render.background_meta,
            diagram_style_meta=prepared_render.diagram_style_meta,
            post_noise_meta=prepared_render.post_noise_meta,
            font_family=prepared_render.font_family,
            target_support=prepared_render.target_support,
            projected_annotation=projected_annotation,
        )
        return SpringTaskParts(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_value=int(axes.target_answer),
            annotation_type=str(annotation_type),
            annotation_value=annotation_value,
            image=prepared_render.image,
            trace_payload=trace_payload,
            public_branch=str(axes.public_branch),
        )
    raise RuntimeError(f"{public_name} failed to generate a valid scene after {max_attempts} attempts")


__all__ = ["SpringTaskParts", "build_spring_prompt_examples", "spring_task_parts"]
