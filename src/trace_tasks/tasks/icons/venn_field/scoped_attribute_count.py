"""Count prompt-named procedural icons in Venn-field regions."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.taxonomy import resolve_task_taxonomy
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.procedural_named_icons import (
    procedural_named_icon_display_name,
    sample_procedural_named_icon_fill_style,
)

from .shared.annotations import counted_icon_bbox_set_annotation
from .shared.defaults import DOMAIN, SCENE_ID, VENN_CATEGORIES, VennFieldDefaults
from .shared.output import build_venn_count_trace_payload, venn_scene_summary
from .shared.prompts import render_venn_prompt_artifacts
from .shared.rendering import render_venn_field_with_retries
from .shared.sampling import (
    default_target_mode_support,
    sample_nonmatching_icon,
    sample_venn_count_inputs,
    target_description,
)
from .shared.state import NamedColorEntry, VennIconPlan
from .shared.styles import resolve_venn_render_params

TASK_ID = "task_icons__venn_field__scoped_attribute_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "inside_both_circles_count",
    "inside_either_circle_count",
    "inside_exactly_one_circle_count",
    "outside_both_circles_count",
)
TARGET_ATTRIBUTE_MODES: Tuple[str, ...] = default_target_mode_support()
_QUERY_TO_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "inside_both_circles_count": ("both",),
    "inside_either_circle_count": ("left_only", "right_only", "both"),
    "inside_exactly_one_circle_count": ("left_only", "right_only"),
    "outside_both_circles_count": ("neither",),
}

_DEFAULTS = VennFieldDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _select_query(
    instance_seed: int, params: Mapping[str, Any]
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select one user-facing Venn-region predicate."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SUPPORTED_QUERY_IDS[0],
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _counted_categories(query_id: str) -> Tuple[str, ...]:
    if str(query_id) not in _QUERY_TO_CATEGORIES:
        raise ValueError(f"unsupported query_id: {query_id}")
    return tuple(_QUERY_TO_CATEGORIES[str(query_id)])


def _make_plans(
    rng: Any,
    *,
    counted_categories: Sequence[str],
    mode: str,
    target_count: int,
    object_count: int,
    target_opposite_count: int,
    target_shape_id: str,
    target_color: NamedColorEntry | None,
    shape_ids: Sequence[str],
    colors: Sequence[NamedColorEntry],
    fill_styles: Sequence[str],
    fill_style_weights: Mapping[str, float],
) -> Tuple[VennIconPlan, ...]:
    """Build icon plans; invariant: exactly target_count matching icons land in counted categories."""

    counted = tuple(str(value) for value in counted_categories)
    non_counted = tuple(
        str(value) for value in VENN_CATEGORIES if str(value) not in set(counted)
    )
    if not non_counted:
        raise ValueError(
            "Venn task resolved no non-counted target distractor categories"
        )
    target_color_entry = (
        target_color if str(mode) == "color_shape" else rng.choice(tuple(colors))
    )
    plans: list[VennIconPlan] = []

    def _target_plan(category: str) -> VennIconPlan:
        color = (
            target_color_entry
            if str(mode) == "color_shape"
            else rng.choice(tuple(colors))
        )
        fill_style = sample_procedural_named_icon_fill_style(
            rng, support=fill_styles, probabilities=fill_style_weights
        )
        return VennIconPlan(
            shape_id=str(target_shape_id),
            color_name=str(color.name),
            tint_rgb=tuple(int(channel) for channel in color.rgb),
            fill_style=str(fill_style),
            venn_category=str(category),
            matches_target=True,
        )

    for _ in range(int(target_count)):
        plans.append(_target_plan(str(rng.choice(counted))))
    for _ in range(int(target_opposite_count)):
        plans.append(_target_plan(str(rng.choice(non_counted))))

    while len(plans) < int(object_count):
        shape_id, color, fill_style = sample_nonmatching_icon(
            rng,
            mode=str(mode),
            target_shape_id=str(target_shape_id),
            target_color=target_color,
            shape_ids=shape_ids,
            colors=colors,
            fill_styles=fill_styles,
            fill_style_weights=fill_style_weights,
        )
        plans.append(
            VennIconPlan(
                shape_id=str(shape_id),
                color_name=str(color.name),
                tint_rgb=tuple(int(channel) for channel in color.rgb),
                fill_style=str(fill_style),
                venn_category=str(rng.choice(VENN_CATEGORIES)),
                matches_target=False,
            )
        )
    rng.shuffle(plans)
    return tuple(plans)


@register_task
class IconsVennFieldScopedAttributeCountTask:
    """Count named procedural icons satisfying a visible Venn-region predicate."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Generate one deterministic Venn-field count instance."""

        query_id, query_probabilities, task_params = _select_query(
            int(instance_seed), params
        )
        counted_categories = _counted_categories(str(query_id))
        sample_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sample")

        inputs = sample_venn_count_inputs(
            sample_rng,
            params=task_params,
            defaults=_GEN_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            minimum_extra_icons=4,
            mode_support=TARGET_ATTRIBUTE_MODES,
        )
        target = inputs.target
        counts = inputs.counts
        shape_ids = inputs.shape_ids
        colors = inputs.colors
        fill_styles = inputs.fill_styles
        fill_weights = inputs.fill_style_probabilities
        mode = str(target.mode)
        mode_probabilities = dict(target.mode_probabilities)
        target_shape_id = str(target.shape_id)
        shape_probabilities = dict(target.shape_probabilities)
        target_color = target.color
        color_probabilities = dict(target.color_probabilities)
        target_count = int(counts.target_count)
        target_count_probabilities = dict(counts.target_count_probabilities)
        object_count = int(counts.object_count)
        object_count_probabilities = dict(counts.object_count_probabilities)
        target_opposite_count = int(counts.target_opposite_count)
        plans = _make_plans(
            sample_rng,
            counted_categories=counted_categories,
            mode=mode,
            target_count=target_count,
            object_count=object_count,
            target_opposite_count=target_opposite_count,
            target_shape_id=target_shape_id,
            target_color=target_color,
            shape_ids=shape_ids,
            colors=colors,
            fill_styles=fill_styles,
            fill_style_weights=fill_weights,
        )

        render_params = resolve_venn_render_params(
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        scene = render_venn_field_with_retries(
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            content_namespace=TASK_ID,
            object_count=object_count,
            target_count=target_count,
            plans=plans,
            counted_categories=counted_categories,
            render_params=render_params,
        )

        annotation_bboxes = tuple(
            instance.bbox_xyxy for instance in scene.instances if instance.counted
        )
        if len(annotation_bboxes) != target_count:
            raise RuntimeError("projected Venn annotation did not match target answer")
        annotation_artifacts = counted_icon_bbox_set_annotation(annotation_bboxes)
        target_phrase = target_description(
            mode=mode,
            shape_id=target_shape_id,
            target_color=target_color,
        )
        _prompt_defaults, prompt_artifacts = render_venn_prompt_artifacts(
            instance_seed=int(instance_seed),
            prompt_defaults=_PROMPT_DEFAULTS,
            query_key=str(query_id),
            target_description=str(target_phrase),
        )
        taxonomy = resolve_task_taxonomy(str(self.task_id))
        public_fields = {
            "domain": taxonomy.domain,
            "scene_id": taxonomy.scene_id,
            "task_id": str(self.task_id),
            "query_id": str(query_id),
        }
        taxonomy_fields = {
            "domain": taxonomy.domain,
            "scene_id": taxonomy.scene_id,
            "task_id": str(self.task_id),
            "source_domain": taxonomy.source_domain,
            "source_scene_id": taxonomy.source_scene_id,
            "query_id": str(query_id),
        }
        scene_summary = venn_scene_summary(scene)
        target_metadata = {
            "target_attribute_mode": mode,
            "target_description": str(target_phrase),
            "target_shape_id": target_shape_id,
            "target_shape_name": procedural_named_icon_display_name(target_shape_id),
            "target_color_name": "" if target_color is None else str(target_color.name),
        }
        count_metadata = {
            "target_count": target_count,
            "target_count_probabilities": dict(target_count_probabilities),
            "target_opposite_count": target_opposite_count,
            "object_count_probabilities": dict(object_count_probabilities),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                **target_metadata,
                "target_attribute_mode_probabilities": dict(mode_probabilities),
                **count_metadata,
                "object_count": object_count,
                "shape_id_support": list(shape_ids),
                "shape_probabilities": dict(shape_probabilities),
                "color_probabilities": dict(color_probabilities),
                "fill_style_support": list(fill_styles),
                "fill_style_probabilities": dict(fill_weights),
                "counted_venn_categories": list(counted_categories),
                "venn": dict(scene_summary["venn"]),
            },
        )
        trace_payload = build_venn_count_trace_payload(
            public_fields=public_fields,
            taxonomy_fields=taxonomy_fields,
            scene=scene,
            render_params=render_params,
            query_spec=query_spec,
            scene_kind="icons_named_shape_venn_region_field",
            counting_rule="target_named_icon_membership_in_overlapping_marked_circles",
            scene_variant="single_panel_named_shape_venn_field",
            question_format="count_named_icons_by_venn_region_membership",
            target_metadata=target_metadata,
            count_metadata=count_metadata,
            counted_venn_categories=counted_categories,
            projected_annotation=annotation_artifacts["projected_annotation"],
            extra_execution_fields={
                "query_id_probabilities": dict(query_probabilities),
            },
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(target_count)),
            annotation_gt=TypedValue(
                type=str(annotation_artifacts["annotation_type"]),
                value=list(annotation_artifacts["annotation_value"]),
            ),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants={
                str(key): str(value)
                for key, value in prompt_artifacts.prompt_variants.items()
            },
        )


__all__ = [
    "IconsVennFieldScopedAttributeCountTask",
    "SUPPORTED_QUERY_IDS",
    "TARGET_ATTRIBUTE_MODES",
]
