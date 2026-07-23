"""Read a length from a visible Vernier caliper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, get_font_family_record
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import build_caliper_prompt_artifacts
from .shared.rendering import render_caliper
from .shared.sampling import resolve_caliper_scenario
from .shared.state import OPTION_LETTERS, SCENE_ID, SCENE_NAMESPACE, VERNIER_RESOLUTION_MM


TASK_ID = "task_physics__vernier_caliper__length_readout_value"
TASK_NAMESPACE = "physics_vernier_caliper_length_readout"
TASK_PROMPT_KEY = "vernier_caliper_readout_query"
INTERNAL_QUERY_ID = "main_scale_vernier_mm"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _select_query_id(
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate the public single-query caliper branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_NAMESPACE}.query",
    )


@register_task
class PhysicsVernierCaliperLengthReadoutValueTask:
    """Read a length from a visible Vernier caliper."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one caliper readout instance and bind answer/annotation trace."""

        _ = int(max_attempts)
        raw_params = dict(params or {})
        query_id, query_probabilities, task_params = _select_query_id(
            int(instance_seed),
            raw_params,
        )
        scenario = resolve_caliper_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_caliper(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
            render_defaults=_RENDER_DEFAULTS,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_caliper_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(
            type="option_letter",
            value=str(scenario.correct_option_letter),
        )
        annotation_gt = TypedValue(
            type="bbox",
            value=list(rendered.render_map["correct_option_bbox_px"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "query_id": str(query_id),
                "internal_query_id": INTERNAL_QUERY_ID,
                "answer_support": list(OPTION_LETTERS),
                "main_mm": int(scenario.main_mm),
                "aligned_vernier_tick": int(scenario.aligned_vernier_tick),
                "target_readout_mm": float(scenario.answer_mm),
                "target_answer": str(scenario.correct_option_letter),
                "correct_option_letter": str(scenario.correct_option_letter),
                "option_values_mm": dict(scenario.option_values_mm),
                "query_id_probabilities": dict(query_probabilities),
                "main_mm_probabilities": dict(scenario.main_mm_probabilities),
                "aligned_vernier_tick_probabilities": dict(
                    scenario.aligned_vernier_tick_probabilities
                ),
                "numeric_readout_probabilities": dict(
                    scenario.target_answer_probabilities
                ),
                "target_answer_probabilities": dict(
                    scenario.correct_option_letter_probabilities
                ),
                "correct_option_letter_probabilities": dict(
                    scenario.correct_option_letter_probabilities
                ),
            },
        )
        font_record = get_font_family_record(str(rendered.font_family))
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_vernier_caliper_readout",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(query_id),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "main_mm": int(scenario.main_mm),
                    "aligned_vernier_tick": int(scenario.aligned_vernier_tick),
                    "vernier_resolution_mm": float(VERNIER_RESOLUTION_MM),
                    "answer_mm": float(scenario.answer_mm),
                    "correct_option_letter": str(scenario.correct_option_letter),
                    "option_values_mm": dict(scenario.option_values_mm),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "canvas_width": int(rendered.image.size[0]),
                "canvas_height": int(rendered.image.size[1]),
                "font": {
                    "font_family": str(rendered.font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": "vernier_caliper_diagram",
                },
                "technical_diagram_style": dict(
                    rendered.render_map["technical_diagram_style"]
                ),
                "background_style": dict(rendered.render_map["background_style"]),
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(query_id),
                "internal_query_id": INTERNAL_QUERY_ID,
                "main_mm": int(scenario.main_mm),
                "aligned_vernier_tick": int(scenario.aligned_vernier_tick),
                "vernier_resolution_mm": float(VERNIER_RESOLUTION_MM),
                "target_readout_mm": float(scenario.answer_mm),
                "target_answer": str(scenario.correct_option_letter),
                "correct_option_letter": str(scenario.correct_option_letter),
                "answer_support": list(OPTION_LETTERS),
                "option_values_mm": dict(scenario.option_values_mm),
                "answer_rounding": "one_decimal",
                "annotation_entity_ids": [f"option_{scenario.correct_option_letter}"],
                "readout_witness_entity_ids": sorted(rendered.annotation_point_map.keys()),
                "correct_option_letter_probabilities": dict(
                    scenario.correct_option_letter_probabilities
                ),
            },
            "witness_symbolic": {
                "type": "bbox",
                "entity_id": f"option_{scenario.correct_option_letter}",
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_gt.value),
                "pixel_bbox": list(annotation_gt.value),
            },
            "background": dict(rendered.render_map["background_style"]),
            "post_image_noise": dict(rendered.render_map["post_image_noise"]),
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
            query_id=str(query_id),
        )


__all__ = ["PhysicsVernierCaliperLengthReadoutValueTask"]
