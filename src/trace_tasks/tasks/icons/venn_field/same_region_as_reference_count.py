"""Count target icons in the same Venn region as a marked reference icon."""

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
    uniform_string_probability_map,
)
from .shared.state import NamedColorEntry, VennIconPlan
from .shared.styles import resolve_venn_render_params

TASK_ID = "task_icons__venn_field__same_region_as_reference_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
TARGET_ATTRIBUTE_MODES: Tuple[str, ...] = default_target_mode_support()

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
    """Select the single same-region count query."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _reference_category(
    rng: Any, params: Mapping[str, Any]
) -> tuple[str, Dict[str, float]]:
    explicit_category = params.get("reference_venn_category")
    if explicit_category is not None:
        category = str(explicit_category)
        if category not in set(VENN_CATEGORIES):
            raise ValueError(
                f"reference_venn_category must be one of {VENN_CATEGORIES}"
            )
        return category, uniform_string_probability_map(
            VENN_CATEGORIES, selected=category
        )
    category = str(rng.choice(VENN_CATEGORIES))
    return category, uniform_string_probability_map(VENN_CATEGORIES)


def _make_reference_region_plans(
    rng: Any,
    *,
    reference_region: str,
    predicate_mode: str,
    answer_count: int,
    icon_total: int,
    off_region_target_count: int,
    requested_shape: str,
    requested_color: NamedColorEntry | None,
    shape_pool: Sequence[str],
    color_pool: Sequence[NamedColorEntry],
    style_pool: Sequence[str],
    style_probabilities: Mapping[str, float],
) -> Tuple[VennIconPlan, ...]:
    """Build plans with one marked non-target reference icon."""

    reference = str(reference_region)
    other_regions = tuple(
        str(value) for value in VENN_CATEGORIES if str(value) != reference
    )
    requested_tint = (
        requested_color
        if str(predicate_mode) == "color_shape"
        else rng.choice(tuple(color_pool))
    )
    icon_plans: list[VennIconPlan] = []

    ref_shape, ref_color, ref_style = sample_nonmatching_icon(
        rng,
        mode=str(predicate_mode),
        target_shape_id=str(requested_shape),
        target_color=requested_color,
        shape_ids=shape_pool,
        colors=color_pool,
        fill_styles=style_pool,
        fill_style_weights=style_probabilities,
    )
    icon_plans.append(
        VennIconPlan(
            shape_id=str(ref_shape),
            color_name=str(ref_color.name),
            tint_rgb=tuple(int(channel) for channel in ref_color.rgb),
            fill_style=str(ref_style),
            venn_category=reference,
            matches_target=False,
            is_reference=True,
        )
    )

    def _matching_icon(region: str) -> VennIconPlan:
        color = (
            requested_tint
            if str(predicate_mode) == "color_shape"
            else rng.choice(tuple(color_pool))
        )
        fill_style = sample_procedural_named_icon_fill_style(
            rng,
            support=style_pool,
            probabilities=style_probabilities,
        )
        return VennIconPlan(
            shape_id=str(requested_shape),
            color_name=str(color.name),
            tint_rgb=tuple(int(channel) for channel in color.rgb),
            fill_style=str(fill_style),
            venn_category=str(region),
            matches_target=True,
        )

    for _ in range(int(answer_count)):
        icon_plans.append(_matching_icon(reference))
    for _ in range(int(off_region_target_count)):
        icon_plans.append(_matching_icon(str(rng.choice(other_regions))))

    while len(icon_plans) < int(icon_total):
        shape_id, color, fill_style = sample_nonmatching_icon(
            rng,
            mode=str(predicate_mode),
            target_shape_id=str(requested_shape),
            target_color=requested_color,
            shape_ids=shape_pool,
            colors=color_pool,
            fill_styles=style_pool,
            fill_style_weights=style_probabilities,
        )
        icon_plans.append(
            VennIconPlan(
                shape_id=str(shape_id),
                color_name=str(color.name),
                tint_rgb=tuple(int(channel) for channel in color.rgb),
                fill_style=str(fill_style),
                venn_category=str(rng.choice(VENN_CATEGORIES)),
                matches_target=False,
            )
        )
    rng.shuffle(icon_plans)
    return tuple(icon_plans)


def _marked_reference_instance(scene: Any) -> Any:
    markers = tuple(instance for instance in scene.instances if instance.is_reference)
    if len(markers) != 1:
        raise RuntimeError(
            "same-region Venn scene must contain exactly one reference icon"
        )
    return markers[0]


def _project_counted_target_boxes(
    scene: Any, *, expected_answer: int
) -> Dict[str, Any]:
    answer_boxes = tuple(
        instance.bbox_xyxy for instance in scene.instances if instance.counted
    )
    if len(answer_boxes) != int(expected_answer):
        raise RuntimeError("projected Venn annotation did not match target answer")
    return counted_icon_bbox_set_annotation(answer_boxes)


def _target_operand_fields(
    *,
    predicate_mode: str,
    requested_shape: str,
    requested_color: NamedColorEntry | None,
    requested_phrase: str,
) -> Dict[str, Any]:
    return {
        "target_attribute_mode": str(predicate_mode),
        "target_description": str(requested_phrase),
        "target_shape_id": str(requested_shape),
        "target_shape_name": procedural_named_icon_display_name(str(requested_shape)),
        "target_color_name": (
            "" if requested_color is None else str(requested_color.name)
        ),
    }


def _reference_scope_fields(
    *,
    reference_icon: Any,
    reference_region: str,
) -> Dict[str, Any]:
    return {
        "reference_instance_id": str(reference_icon.instance_id),
        "reference_venn_category": str(reference_region),
    }


@register_task
class IconsVennFieldSameRegionAsReferenceCountTask:
    """Count target icons in the marked reference icon's Venn region."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Generate one deterministic same-region Venn count instance."""

        branch_id, branch_probabilities, resolved_params = _select_query(
            int(instance_seed), params
        )
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sample")

        input_pack = sample_venn_count_inputs(
            rng,
            params=resolved_params,
            defaults=_GEN_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            minimum_extra_icons=5,
            mode_support=TARGET_ATTRIBUTE_MODES,
        )
        predicate = input_pack.target
        cardinality = input_pack.counts
        shape_pool = input_pack.shape_ids
        color_pool = input_pack.colors
        style_pool = input_pack.fill_styles
        style_probabilities = input_pack.fill_style_probabilities
        reference_region, reference_region_probabilities = _reference_category(
            rng, resolved_params
        )
        predicate_mode = str(predicate.mode)
        predicate_mode_probabilities = dict(predicate.mode_probabilities)
        requested_shape = str(predicate.shape_id)
        requested_shape_probabilities = dict(predicate.shape_probabilities)
        requested_color = predicate.color
        requested_color_probabilities = dict(predicate.color_probabilities)
        answer_count = int(cardinality.target_count)
        answer_probabilities = dict(cardinality.target_count_probabilities)
        icon_total = int(cardinality.object_count)
        icon_total_probabilities = dict(cardinality.object_count_probabilities)
        off_region_target_count = int(cardinality.target_opposite_count)
        plans = _make_reference_region_plans(
            rng,
            reference_region=str(reference_region),
            predicate_mode=predicate_mode,
            answer_count=answer_count,
            icon_total=icon_total,
            off_region_target_count=off_region_target_count,
            requested_shape=requested_shape,
            requested_color=requested_color,
            shape_pool=shape_pool,
            color_pool=color_pool,
            style_pool=style_pool,
            style_probabilities=style_probabilities,
        )

        render_params = resolve_venn_render_params(
            params=resolved_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        scene = render_venn_field_with_retries(
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            content_namespace=TASK_ID,
            object_count=icon_total,
            target_count=answer_count,
            plans=plans,
            counted_categories=(str(reference_region),),
            render_params=render_params,
        )

        reference_icon = _marked_reference_instance(scene)
        annotation_bundle = _project_counted_target_boxes(
            scene, expected_answer=answer_count
        )
        requested_phrase = target_description(
            mode=predicate_mode,
            shape_id=requested_shape,
            target_color=requested_color,
        )
        _prompt_defaults, prompt_result = render_venn_prompt_artifacts(
            instance_seed=int(instance_seed),
            prompt_defaults=_PROMPT_DEFAULTS,
            query_key=str(branch_id),
            target_description=str(requested_phrase),
        )
        tax = resolve_task_taxonomy(str(self.task_id))
        visible_ids = {
            "domain": tax.domain,
            "scene_id": tax.scene_id,
            "task_id": str(self.task_id),
            "query_id": str(branch_id),
        }
        taxonomy_record = {
            "domain": tax.domain,
            "scene_id": tax.scene_id,
            "task_id": str(self.task_id),
            "source_domain": tax.source_domain,
            "source_scene_id": tax.source_scene_id,
            "query_id": str(branch_id),
        }
        rendered_summary = venn_scene_summary(scene)
        predicate_fields = _target_operand_fields(
            predicate_mode=predicate_mode,
            requested_shape=requested_shape,
            requested_color=requested_color,
            requested_phrase=str(requested_phrase),
        )
        answer_fields = {
            "target_count": answer_count,
            "target_count_probabilities": dict(answer_probabilities),
            "target_opposite_count": off_region_target_count,
            "object_count_probabilities": dict(icon_total_probabilities),
        }
        reference_fields = _reference_scope_fields(
            reference_icon=reference_icon,
            reference_region=str(reference_region),
        )
        prompt_query = build_prompt_query_spec(
            prompt_artifacts=prompt_result,
            query_id=str(branch_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(branch_probabilities),
                **predicate_fields,
                "target_attribute_mode_probabilities": dict(
                    predicate_mode_probabilities
                ),
                **answer_fields,
                "object_count": icon_total,
                "shape_id_support": list(shape_pool),
                "shape_probabilities": dict(requested_shape_probabilities),
                "color_probabilities": dict(requested_color_probabilities),
                "fill_style_support": list(style_pool),
                "fill_style_probabilities": dict(style_probabilities),
                **reference_fields,
                "reference_venn_category_probabilities": dict(
                    reference_region_probabilities
                ),
                "counted_venn_categories": [str(reference_region)],
                "venn": dict(rendered_summary["venn"]),
            },
        )
        trace_record = build_venn_count_trace_payload(
            public_fields=visible_ids,
            taxonomy_fields=taxonomy_record,
            scene=scene,
            render_params=render_params,
            query_spec=prompt_query,
            scene_kind="icons_named_shape_venn_reference_region_field",
            counting_rule="target_named_icon_membership_in_marked_reference_venn_region",
            scene_variant="single_panel_named_shape_venn_reference_region_field",
            question_format="count_named_icons_in_same_venn_region_as_reference",
            target_metadata=predicate_fields,
            count_metadata=answer_fields,
            counted_venn_categories=(str(reference_region),),
            projected_annotation=annotation_bundle["projected_annotation"],
            extra_relation_fields=reference_fields,
            extra_render_map_fields={
                **reference_fields,
                "reference_bbox_px": [int(value) for value in reference_icon.bbox_xyxy],
            },
            extra_execution_fields={
                **reference_fields,
                "query_id_probabilities": dict(branch_probabilities),
                "reference_venn_category_probabilities": dict(
                    reference_region_probabilities
                ),
            },
            extra_witness_fields=reference_fields,
        )
        return TaskOutput(
            prompt=str(prompt_result.prompt),
            answer_gt=TypedValue(type="integer", value=answer_count),
            annotation_gt=TypedValue(
                type=str(annotation_bundle["annotation_type"]),
                value=list(annotation_bundle["annotation_value"]),
            ),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_record,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_id),
            prompt_variants={
                str(key): str(value)
                for key, value in prompt_result.prompt_variants.items()
            },
        )


__all__ = [
    "IconsVennFieldSameRegionAsReferenceCountTask",
    "SUPPORTED_QUERY_IDS",
    "TARGET_ATTRIBUTE_MODES",
]
