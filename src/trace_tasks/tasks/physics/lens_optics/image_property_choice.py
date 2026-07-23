"""Image-property choice task for the lens-optics scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_generation_rendering_prompt_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ..shared.diagram_style import prepare_physics_diagram_style_and_background
from ..shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from ..shared.visual_defaults import load_physics_noise_defaults
from .shared.rendering import render_lens_optics_scene
from .shared.state import (
    OBJECT_POSITION_CASES,
    OPTION_LETTERS,
    PROPERTY_TEXT,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    LensOpticsVisualScenario,
)


TASK_ID = "task_physics__lens_optics__image_property_choice"
TASK_NAMESPACE = "physics_lens_optics_image_property_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
INTERNAL_QUERY_ID = "converging_lens_image_property_choice"
CASE_TO_PROPERTY: Dict[str, str] = {
    "beyond_2f": "real_inverted_smaller",
    "at_2f": "real_inverted_same_size",
    "between_f_2f": "real_inverted_larger",
    "inside_f": "virtual_upright_larger",
}
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    split_generation_rendering_prompt_defaults(
        _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
        task_id=TASK_ID,
    )
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.5,
)


@dataclass(frozen=True)
class _Axes:
    scene_variant: str
    query_id: str
    object_position_case: str
    correct_option_letter: str
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    query_id_probabilities: Dict[str, float]
    object_position_case_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _Scenario:
    scene_variant: str
    query_id: str
    object_position_case: str
    image_property: str
    correct_option_letter: str
    accent_color_name: str
    option_map: Dict[str, str]
    focal_length_px: float
    object_x_factor: float


def _resolve_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), namespace),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=balance_flag_key,
        explicit_key=explicit_key,
        weights_key=weights_key,
        sampling_namespace=namespace,
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def _resolve_axes(
    instance_seed: int,
    params: Mapping[str, Any],
    *,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> _Axes:
    """Resolve all sampled axes while keeping the public query sentinel fixed."""

    scene_variant, scene_probs = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{TASK_NAMESPACE}.scene_variant",
    )
    position_case, position_probs = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=OBJECT_POSITION_CASES,
        explicit_key="object_position_case",
        weights_key="object_position_case_weights",
        balance_flag_key="balanced_object_position_case_sampling",
        namespace=f"{TASK_NAMESPACE}.object_position_case",
    )
    correct_letter, letter_probs = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        balance_flag_key="balanced_correct_option_letter_sampling",
        namespace=f"{TASK_NAMESPACE}.correct_option_letter",
    )
    accent_color, accent_probs = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        namespace=f"{TASK_NAMESPACE}.accent_color_name",
    )
    return _Axes(
        scene_variant=str(scene_variant),
        query_id=str(selected_query_id),
        object_position_case=str(position_case),
        correct_option_letter=str(correct_letter),
        accent_color_name=str(accent_color),
        scene_variant_probabilities=dict(scene_probs),
        query_id_probabilities={
            str(key): float(value)
            for key, value in dict(query_probabilities).items()
        },
        object_position_case_probabilities=dict(position_probs),
        correct_option_letter_probabilities=dict(letter_probs),
        accent_color_name_probabilities=dict(accent_probs),
    )


def _option_map(
    *,
    instance_seed: int,
    image_property: str,
    correct_option_letter: str,
) -> Dict[str, str]:
    remaining = [
        property_id
        for property_id in PROPERTY_TEXT
        if str(property_id) != str(image_property)
    ]
    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.option_map")
    rng.shuffle(remaining)
    result: Dict[str, str] = {}
    cursor = 0
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            result[str(letter)] = str(image_property)
        else:
            result[str(letter)] = str(remaining[cursor])
            cursor += 1
    return dict(result)


def _make_scenario(
    instance_seed: int,
    axes: _Axes,
    params: Mapping[str, Any],
) -> _Scenario:
    focal_length = float(
        params.get("focal_length_px", group_default(_RENDER_DEFAULTS, "focal_length_px", 116))
    )
    factor_by_case = {
        "beyond_2f": 2.55,
        "at_2f": 2.0,
        "between_f_2f": 1.55,
        "inside_f": 0.62,
    }
    image_property = str(CASE_TO_PROPERTY[str(axes.object_position_case)])
    return _Scenario(
        scene_variant=str(axes.scene_variant),
        query_id=str(axes.query_id),
        object_position_case=str(axes.object_position_case),
        image_property=str(image_property),
        correct_option_letter=str(axes.correct_option_letter),
        accent_color_name=str(axes.accent_color_name),
        option_map=_option_map(
            instance_seed=int(instance_seed),
            image_property=str(image_property),
            correct_option_letter=str(axes.correct_option_letter),
        ),
        focal_length_px=float(focal_length),
        object_x_factor=float(factor_by_case[str(axes.object_position_case)]),
    )


def _visual_scenario(scenario: _Scenario) -> LensOpticsVisualScenario:
    """Convert task-owned answer state into renderer-facing visual state."""

    option_text_map = {
        str(letter): PROPERTY_TEXT[str(property_id)]
        for letter, property_id in scenario.option_map.items()
    }
    return LensOpticsVisualScenario(
        scene_variant=str(scenario.scene_variant),
        object_position_case=str(scenario.object_position_case),
        image_property=str(scenario.image_property),
        correct_option_letter=str(scenario.correct_option_letter),
        accent_color_name=str(scenario.accent_color_name),
        option_text_map=option_text_map,
        focal_length_px=float(scenario.focal_length_px),
        object_x_factor=float(scenario.object_x_factor),
    )


@register_task
class PhysicsLensOpticsImagePropertyChoiceTask:
    """Choose the image property implied by a converging-lens object position."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Build the task-owned scenario, bind answer/annotation, and return output."""

        del max_attempts
        params = dict(params or {})
        selected_query, query_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        axes = _resolve_axes(
            int(instance_seed),
            task_params,
            selected_query_id=str(selected_query),
            query_probabilities=query_probs,
        )
        scenario = _make_scenario(int(instance_seed), axes, task_params)
        canvas_width = int(
            task_params.get(
                "canvas_width",
                group_default(_RENDER_DEFAULTS, "canvas_width", 1120),
            )
        )
        canvas_height = int(
            task_params.get(
                "canvas_height",
                group_default(_RENDER_DEFAULTS, "canvas_height", 720),
            )
        )
        background, background_meta, diagram_style, diagram_style_meta = (
            prepare_physics_diagram_style_and_background(
                instance_seed=int(instance_seed),
                params=task_params,
                scene_id=SCENE_ID,
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                require_grid=True,
            )
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}.font",
            params=task_params,
        )
        font_record = get_font_family_record(str(font_family))
        rendered = render_lens_optics_scene(
            image=background,
            scenario=_visual_scenario(scenario),
            font_family=str(font_family),
            style=diagram_style,
            render_defaults=_RENDER_DEFAULTS,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key="lens_optics_diagram",
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(axes.query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_option_letter))
        annotation_gt = TypedValue(
            type="bbox_map",
            value={
                str(key): list(value)
                for key, value in rendered.annotation_bbox_map.items()
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_lens_optics_{scenario.scene_variant}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(axes.query_id),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "lens_type": "converging",
                    "object_position_case": str(scenario.object_position_case),
                    "image_property": str(scenario.image_property),
                    "correct_option_letter": str(scenario.correct_option_letter),
                    "accent_color_name": str(scenario.accent_color_name),
                    "option_map": dict(scenario.option_map),
                },
            },
            "query_spec": {
                "query_id": str(axes.query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(axes.query_id),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "answer_support": list(OPTION_LETTERS),
                    "object_position_case": str(scenario.object_position_case),
                    "image_property": str(scenario.image_property),
                    "correct_option_letter": str(scenario.correct_option_letter),
                    "accent_color_name": str(axes.accent_color_name),
                    "target_answer": str(scenario.correct_option_letter),
                    "scene_variant": str(scenario.scene_variant),
                    "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                    "query_id_probabilities": dict(axes.query_id_probabilities),
                    "object_position_case_probabilities": (
                        dict(axes.object_position_case_probabilities)
                    ),
                    "correct_option_letter_probabilities": (
                        dict(axes.correct_option_letter_probabilities)
                    ),
                    "accent_color_name_probabilities": dict(
                        axes.accent_color_name_probabilities
                    ),
                },
            },
            "render_spec": {
                "canvas_width": int(image.size[0]),
                "canvas_height": int(image.size[1]),
                "font": {
                    "font_family": str(font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": "lens_optics_diagram",
                    "selection_policy": {
                        "pool": "global_approved_font_pool",
                        "include_tags": [],
                        "exclude_tags": [],
                        "exclusion_reason": "",
                    },
                },
                "technical_diagram_style": dict(diagram_style_meta),
                "background_style": background_meta,
                "post_image_noise": post_noise_meta,
            },
            "render_map": {
                **dict(rendered.render_map),
                "option_map": dict(scenario.option_map),
            },
            "execution_trace": {
                "query_id": str(axes.query_id),
                "internal_query_id": INTERNAL_QUERY_ID,
                "lens_type": "converging",
                "object_position_case": str(scenario.object_position_case),
                "image_property": str(scenario.image_property),
                "option_map": dict(scenario.option_map),
                "correct_option_letter": str(scenario.correct_option_letter),
                "accent_color_name": str(scenario.accent_color_name),
                "focal_length_px": float(scenario.focal_length_px),
                "object_x_factor": float(scenario.object_x_factor),
                "annotation_entity_ids": sorted(annotation_gt.value.keys()),
            },
            "witness_symbolic": {
                "type": "object_map",
                "ids": sorted(annotation_gt.value.keys()),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": {
                    str(key): list(value)
                    for key, value in annotation_gt.value.items()
                },
                "pixel_bbox_map": {
                    str(key): list(value)
                    for key, value in annotation_gt.value.items()
                },
            },
            "background": background_meta,
            "post_image_noise": post_noise_meta,
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(axes.query_id),
            scene_id=SCENE_ID,
        )


__all__ = [
    "CASE_TO_PROPERTY",
    "OBJECT_POSITION_CASES",
    "OPTION_LETTERS",
    "PROPERTY_TEXT",
    "PhysicsLensOpticsImagePropertyChoiceTask",
]
