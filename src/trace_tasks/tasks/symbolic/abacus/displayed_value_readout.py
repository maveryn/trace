"""Read the integer represented by a three-column soroban-style abacus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.common import get_int_range as _get_range
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.sampling import choose_value_options, resolve_six_option_labels
from .shared.rendering import render_abacus_single_board_scene
from .shared.rules import (
    ABACUS_ANNOTATION_KEYS,
    ABACUS_COLUMN_ROLES,
    SUPPORTED_ABACUS_SCENE_VARIANTS,
    digits_for_abacus_value,
)
from .shared.state import AbacusColumnSpec, AbacusReadoutOptionSpec
from .shared.styles import resolve_readout_render_params


DOMAIN = "symbolic"
SCENE_ID = "abacus"
TASK_ID = "task_symbolic__abacus__displayed_value_readout"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
INTERNAL_QUERY_KEY = "displayed_value_readout"
QUESTION_FORMAT = "displayed_value_readout"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _Dataset:
    scene_variant: str
    answer_value: int
    target_answer_support: tuple[int, int]
    columns: tuple[AbacusColumnSpec, ...]
    digits_by_role: dict[str, int]
    place_values_by_role: dict[str, int]
    option_labels: tuple[str, ...]
    correct_label: str
    option_values_by_label: dict[str, int]
    scene_variant_probabilities: dict[str, float]
    correct_label_probabilities: dict[str, float]


def _resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_ABACUS_SCENE_VARIANTS,
        task_id=TASK_ID,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _resolve_correct_label(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    option_labels: tuple[str, ...],
) -> Tuple[str, Dict[str, float]]:
    axis_params = dict(params)
    if "correct_label" in params and "answer_label" not in params:
        axis_params["answer_label"] = params["correct_label"]
    selected, probabilities = resolve_symbolic_axis_variant(
        params=axis_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=option_labels,
        task_id=TASK_ID,
        explicit_key="answer_label",
        weights_key="correct_option_label_weights",
        balance_flag_key="balanced_correct_option_label_sampling",
        axis_namespace="correct_option_label",
    )
    return str(selected), dict(probabilities)


def _build_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> _Dataset:
    """Sample the displayed value and construct the three place-value columns."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    option_labels = resolve_six_option_labels(params, gen_defaults)
    correct_label, correct_label_probabilities = _resolve_correct_label(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        option_labels=option_labels,
    )
    answer_min, answer_max = _get_range(
        params,
        gen_defaults,
        min_key="target_answer_min",
        max_key="target_answer_max",
        fallback_min=0,
        fallback_max=999,
    )
    if int(answer_min) < 0 or int(answer_max) > 999:
        raise ValueError("abacus displayed-value answer support must stay within 0..999")
    if "answer_value" in params:
        answer_value = int(params["answer_value"])
    elif "displayed_value" in params:
        answer_value = int(params["displayed_value"])
    else:
        answer_value = int(rng.randint(int(answer_min), int(answer_max)))
    if not int(answer_min) <= int(answer_value) <= int(answer_max):
        raise ValueError("answer_value is outside configured abacus answer support")

    digits = digits_for_abacus_value(int(answer_value))
    place_labels = ("100", "10", "1")
    place_values = (100, 10, 1)
    columns = tuple(
        AbacusColumnSpec(
            item_id=f"column_{role}",
            role=str(role),
            place_label=str(place_label),
            place_value=int(place_value),
            digit=int(digit),
        )
        for role, place_label, place_value, digit in zip(ABACUS_COLUMN_ROLES, place_labels, place_values, digits)
    )
    digits_by_role = {str(column.role): int(column.digit) for column in columns}
    place_values_by_role = {str(column.role): int(column.place_value) for column in columns}
    option_values_by_label = choose_value_options(
        instance_seed=int(instance_seed),
        seed_namespace=TASK_ID,
        target_value=int(answer_value),
        option_labels=option_labels,
        correct_label=str(correct_label),
        min_value=int(answer_min),
        max_value=int(answer_max),
    )
    return _Dataset(
        scene_variant=str(scene_variant),
        answer_value=int(answer_value),
        target_answer_support=(int(answer_min), int(answer_max)),
        columns=tuple(columns),
        digits_by_role=dict(digits_by_role),
        place_values_by_role=dict(place_values_by_role),
        option_labels=tuple(str(label) for label in option_labels),
        correct_label=str(correct_label),
        option_values_by_label=dict(option_values_by_label),
        scene_variant_probabilities={},
        correct_label_probabilities=dict(correct_label_probabilities),
    )


def _build_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any], Any]:
    """Render the external prompt bundle for the abacus displayed-value task."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        f"object_description_displayed_value_readout_{scene_variant}",
        "question_text",
        "annotation_hint",
        "answer_hint",
        "json_example",
        "json_example_answer_only",
    )
    prompt_values = required_group_defaults(prompt_defaults, required_keys, context=f"prompt defaults for {TASK_ID}")
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[f"object_description_displayed_value_readout_{scene_variant}"]),
            "question_text": str(prompt_values["question_text"]),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_values["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_values["annotation_hint"]),
            "answer_hint": str(prompt_values["answer_hint"]),
            "json_example": str(prompt_values["json_example"]),
            "json_example_answer_only": str(prompt_values["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(prompt_artifacts.prompt), dict(prompt_artifacts.prompt_variants), {
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
        "bundle_id": str(prompt_values["bundle_id"]),
    }, prompt_artifacts


@register_task
class SymbolicAbacusDisplayedValueReadoutTask:
    """Read the integer represented by a three-column soroban-style abacus."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one readout instance with task-owned answer and annotation binding."""

        scene_variant, scene_variant_probabilities = _resolve_scene_variant(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        last_error: Exception | None = None
        dataset: _Dataset | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                dataset = _build_dataset(
                    instance_seed=int(instance_seed) + int(attempt_index),
                    scene_variant=str(scene_variant),
                    params=params,
                    gen_defaults=_GEN_DEFAULTS,
                )
                dataset = _Dataset(
                    scene_variant=str(dataset.scene_variant),
                    answer_value=int(dataset.answer_value),
                    target_answer_support=tuple(dataset.target_answer_support),
                    columns=tuple(dataset.columns),
                    digits_by_role=dict(dataset.digits_by_role),
                    place_values_by_role=dict(dataset.place_values_by_role),
                    option_labels=tuple(str(label) for label in dataset.option_labels),
                    correct_label=str(dataset.correct_label),
                    option_values_by_label=dict(dataset.option_values_by_label),
                    scene_variant_probabilities=dict(scene_variant_probabilities),
                    correct_label_probabilities=dict(dataset.correct_label_probabilities),
                )
                break
            except Exception as exc:
                last_error = exc
        if dataset is None:
            raise RuntimeError(f"failed to generate abacus readout instance for {TASK_ID}") from last_error

        render_params = resolve_readout_render_params(_RENDER_DEFAULTS)
        scene_style, scene_style_meta = resolve_symbolic_scene_style(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.background",
        )
        background, background_meta = make_symbolic_scene_background(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            style=scene_style,
        )
        rendered_scene = render_abacus_single_board_scene(
            background,
            columns=dataset.columns,
            params=render_params,
            scene_variant=str(dataset.scene_variant),
            style=scene_style,
            options=tuple(
                AbacusReadoutOptionSpec(
                    label=str(label),
                    text=str(dataset.option_values_by_label[str(label)]),
                    value=int(dataset.option_values_by_label[str(label)]),
                    is_correct=bool(str(label) == str(dataset.correct_label)),
                )
                for label in dataset.option_labels
            ),
            correct_label=str(dataset.correct_label),
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt, prompt_variants, prompt_meta, prompt_artifacts = _build_prompt(
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_variant=str(dataset.scene_variant),
            instance_seed=int(instance_seed),
        )

        if rendered_scene.selected_option_card_bbox is None:
            raise RuntimeError("abacus displayed-value option rendering did not return a selected option bbox")
        annotation_payload = bbox_annotation_artifacts(rendered_scene.selected_option_card_bbox)
        annotation_bbox = list(annotation_payload.value)
        keyed_points = {
            str(key): [list(point) for point in rendered_scene.active_bead_points_by_column.get(str(key), [])]
            for key in ABACUS_ANNOTATION_KEYS
        }
        annotation_gt = annotation_payload.annotation_gt
        answer_gt = TypedValue(type="option_letter", value=str(dataset.correct_label))
        projected_annotation = dict(annotation_payload.projected_annotation)
        query_params = {
            "query_id": SINGLE_QUERY_ID,
            "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
            "internal_query_id": INTERNAL_QUERY_KEY,
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "target_answer_support": [int(dataset.target_answer_support[0]), int(dataset.target_answer_support[1])],
            "option_labels": [str(label) for label in dataset.option_labels],
            "correct_label_probabilities": dict(dataset.correct_label_probabilities),
            "column_roles": [str(role) for role in ABACUS_COLUMN_ROLES],
            "annotation_keys": [str(key) for key in ABACUS_ANNOTATION_KEYS],
            "question_format": QUESTION_FORMAT,
        }
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=SINGLE_QUERY_ID,
            params=query_params,
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_abacus_single_board",
                "entities": [dict(entity) for entity in rendered_scene.entities],
                "relations": {
                    "query_id": SINGLE_QUERY_ID,
                    "internal_query_id": INTERNAL_QUERY_KEY,
                    "scene_id": SCENE_ID,
                    "scene_variant": str(dataset.scene_variant),
                    "answer_value": int(dataset.answer_value),
                    "correct_label": str(dataset.correct_label),
                    "option_values_by_label": dict(dataset.option_values_by_label),
                    "digits_by_role": dict(dataset.digits_by_role),
                    "place_values_by_role": dict(dataset.place_values_by_role),
                },
            },
            "query_spec": {
                **dict(prompt_query_spec),
                "template_id": str(prompt_meta["bundle_id"]),
            },
            "render_spec": {
                "scene_id": SCENE_ID,
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "coord_space": "pixel",
                "scene_variant": str(dataset.scene_variant),
                "scene_style": dict(scene_style_meta),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "abacus_style": dict(rendered_scene.style_metadata),
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "item_bboxes_px": dict(rendered_scene.item_bboxes),
                "bead_bboxes_px": dict(rendered_scene.bead_bboxes),
                "option_card_bboxes_px": dict(rendered_scene.option_card_bboxes),
                "selected_option_card_bbox_px": list(annotation_bbox),
                "correct_label": str(dataset.correct_label),
                "active_bead_bboxes_by_column_px": dict(rendered_scene.active_bead_bboxes_by_column),
                "active_bead_points_by_column_px": dict(keyed_points),
                "active_bead_ids_by_column": dict(rendered_scene.active_bead_ids_by_column),
                "column_bboxes_px": dict(rendered_scene.column_bboxes),
                "label_bboxes_px": dict(rendered_scene.label_bboxes),
                "annotation_source": "selected_option_card_bbox_px",
            },
            "execution_trace": {
                **dict(query_params),
                "answer_value": int(dataset.answer_value),
                "answer_type": "option_letter",
                "correct_label": str(dataset.correct_label),
                "option_values_by_label": dict(dataset.option_values_by_label),
                "digits_by_role": dict(dataset.digits_by_role),
                "place_values_by_role": dict(dataset.place_values_by_role),
                "columns": [
                    {
                        "item_id": str(column.item_id),
                        "role": str(column.role),
                        "place_label": str(column.place_label),
                        "place_value": int(column.place_value),
                        "digit": int(column.digit),
                        "active_bead_ids": [str(item) for item in rendered_scene.active_bead_ids_by_column[str(column.role)]],
                    }
                    for column in dataset.columns
                ],
                "supporting_bbox_roles": ["selected_option_card"],
                "active_bead_points_by_column_px": dict(keyed_points),
            },
            "witness_symbolic": {
                "type": "bbox",
                "value": list(annotation_bbox),
            },
            "projected_annotation": dict(projected_annotation),
            "answer_gt": answer_gt.to_dict(),
            "annotation_gt": annotation_gt.to_dict(),
        }
        return TaskOutput(
            prompt=str(prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=SINGLE_QUERY_ID,
            prompt_variants=dict(prompt_variants),
        )


__all__ = [
    "SymbolicAbacusDisplayedValueReadoutTask",
    "TASK_ID",
]
