"""Relatedness-coefficient option task over pedigree charts."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.output_metadata import default_task_versions
from ._lifecycle import (
    OPTION_LABELS,
    PEDIGREE_RELATEDNESS_LABELS,
    PEDIGREE_RELATEDNESS_OPTION_LABELS,
    SCENE_ID,
    PedigreeRelatednessQuerySample,
    _common_slots,
    draw_pedigree_options,
    _person_label,
    _prompt_artifacts,
    _render_sample,
    _resolve_style,
    _sections_for_task,
    _select_relatedness_options,
    _select_variant,
    _trace_payload,
    projected_pedigree_person_bbox_set_annotation,
    sample_pedigree_relatedness,
)


RELATEDNESS_TASK_ID = "task_graph__pedigree_chart__relatedness_coefficient_label"
SCENE_ID = "pedigree_chart"
RELATEDNESS_QUERY_KEY = "relatedness_coefficient_between_two_people"


@register_task
class GraphPedigreeRelatednessCoefficientLabelTask:
    """Select the visual option for the coefficient of relatedness between two people."""

    task_id = RELATEDNESS_TASK_ID
    reasoning_operations = ('topology', 'formula_evaluation')
    domain = "graph"
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Sample one relatedness objective and bind fraction-option answer plus person-box witnesses."""

        requested = params.get("query_id")
        if requested is not None and str(requested) not in {"", "default", SINGLE_QUERY_ID}:
            raise ValueError(f"unsupported query_id for {self.task_id}: {requested}")
        gen_defaults, render_defaults, prompt_defaults = _sections_for_task(self.task_id)
        style = _resolve_style(int(instance_seed), params=params, gen_defaults=gen_defaults, rng_namespace=self.task_id)
        target_relatedness, relatedness_probs = _select_variant(
            int(instance_seed),
            rng_namespace=self.task_id,
            params=params,
            gen_defaults=gen_defaults,
            supported=PEDIGREE_RELATEDNESS_LABELS,
            explicit_key="target_relatedness_label",
            weights_key="relatedness_label_weights",
        )
        sample_query: PedigreeRelatednessQuerySample = sample_pedigree_relatedness(
            int(instance_seed),
            target_relatedness=str(target_relatedness),
            max_attempts=max(120, int(max_attempts)),
        )
        option_values, correct_option = _select_relatedness_options(
            instance_seed=int(instance_seed),
            answer=str(sample_query.answer),
            rng_namespace=self.task_id,
        )
        image, rendered_scene, render_params, background_meta, post_noise_meta = _render_sample(
            render_namespace=self.task_id,
            instance_seed=int(instance_seed),
            params=params,
            render_defaults=render_defaults,
            style=style,
            sample=sample_query.sample,
            highlighted_person_ids=(sample_query.person_a_id, sample_query.person_b_id),
            bottom_reserved_px=88,
        )
        draw_pedigree_options(
            image=image,
            render_params=render_params,
            option_values_by_label=option_values,
        )
        annotation_projection = projected_pedigree_person_bbox_set_annotation(rendered_scene, sample_query.annotation_roles)
        prompt_artifacts = _prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            query_id=RELATEDNESS_QUERY_KEY,
            slots={
                **_common_slots(
                    prompt_defaults,
                    answer_example="C",
                ),
                "person_label_a": _person_label(sample_query.sample, sample_query.person_a_id),
                "person_label_b": _person_label(sample_query.sample, sample_query.person_b_id),
            },
            instance_seed=int(instance_seed),
        )
        query_params = {
            "scene_variant": str(style.scene_variant),
            "scene_variant_probabilities": dict(style.scene_variant_probabilities),
            "relatedness_label_probabilities": dict(relatedness_probs),
            "target_relatedness_label": str(target_relatedness),
            "person_count": int(len(sample_query.sample.people)),
            "template_name": str(sample_query.sample.template_name),
            "node_color_name": str(style.node_color_name),
            "node_color_name_probabilities": dict(style.node_color_name_probabilities),
            "option_count": len(OPTION_LABELS),
            "option_values_by_label": dict(option_values),
            "answer_support": list(PEDIGREE_RELATEDNESS_LABELS),
            "option_value_support": list(PEDIGREE_RELATEDNESS_OPTION_LABELS),
        }
        execution_trace = {
            "task_id": self.task_id,
            "scene_id": SCENE_ID,
            "query_id": RELATEDNESS_QUERY_KEY,
            "answer": str(correct_option),
            "answer_fraction": str(sample_query.answer),
            "option_values_by_label": dict(option_values),
            "person_a_id": str(sample_query.person_a_id),
            "person_a_label": _person_label(sample_query.sample, sample_query.person_a_id),
            "person_b_id": str(sample_query.person_b_id),
            "person_b_label": _person_label(sample_query.sample, sample_query.person_b_id),
            "contributing_paths": list(sample_query.contributing_paths),
            "annotation_role_to_person_id": dict(annotation_projection["role_person_id_map"]),
            "selected_option_label": str(correct_option),
        }
        trace_payload = _trace_payload(
            task_identifier=self.task_id,
            query_id=RELATEDNESS_QUERY_KEY,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            sample=sample_query.sample,
            rendered_scene=rendered_scene,
            render_params=render_params,
            style=style,
            background_meta=background_meta,
            post_noise_meta=post_noise_meta,
            query_params=query_params,
            execution_trace=execution_trace,
            witness_symbolic={
                "type": "pedigree_relatedness_coefficient_label",
                "answer_option": str(correct_option),
                "answer_fraction": str(sample_query.answer),
                "option_values_by_label": dict(option_values),
                "annotation_role_to_person_id": dict(annotation_projection["role_person_id_map"]),
                "contributing_paths": list(sample_query.contributing_paths),
            },
            projected_annotation=annotation_projection,
        )
        answer_gt = TypedValue(type="option_letter", value=str(correct_option))
        annotation_gt = TypedValue(
            type="bbox_set",
            value=list(annotation_projection["bbox_set"]),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=dict(trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=RELATEDNESS_QUERY_KEY,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["GraphPedigreeRelatednessCoefficientLabelTask", "RELATEDNESS_TASK_ID"]
