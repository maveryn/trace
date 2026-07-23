"""Scene-private lifecycle helpers for symbolic abacus readout panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
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
from .shared.sampling import choose_digit_options, resolve_six_option_labels
from .shared.rendering import render_abacus_single_board_scene
from .shared.rules import (
    ABACUS_COLUMN_ROLES,
    SUPPORTED_ABACUS_SCENE_VARIANTS,
    digits_for_abacus_value,
)
from .shared.state import AbacusColumnSpec
from .shared.state import AbacusReadoutOptionSpec
from .shared.styles import resolve_readout_render_params


DOMAIN = "symbolic"
SCENE_ID = "abacus"
_PLACE_LABELS_BY_ROLE = {"hundreds": "100", "tens": "10", "ones": "1"}
_PLACE_VALUES_BY_ROLE = {"hundreds": 100, "tens": 10, "ones": 1}


@dataclass(frozen=True)
class AbacusColumnReadoutBinding:
    """Task-owned prompt and trace binding for a queried abacus column."""

    public_task_id: str
    internal_query_key: str
    question_format: str
    object_description_prefix: str
    question_text_key: str
    annotation_hint_key: str
    answer_hint_key: str
    json_example_key: str
    json_example_answer_only_key: str
    failure_message: str


@dataclass(frozen=True)
class _ColumnReadoutDataset:
    scene_variant: str
    displayed_value: int
    displayed_value_support: tuple[int, int]
    columns: tuple[AbacusColumnSpec, ...]
    digits_by_role: dict[str, int]
    place_values_by_role: dict[str, int]
    target_column_role: str
    target_place_label: str
    target_place_value: int
    answer_digit: int
    annotation_key: str
    option_labels: tuple[str, ...]
    correct_label: str
    option_values_by_label: dict[str, int]
    scene_variant_probabilities: dict[str, float]
    target_column_role_probabilities: dict[str, float]
    correct_label_probabilities: dict[str, float]


def load_abacus_readout_defaults(
    public_task_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load scene config defaults for one abacus readout objective."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_task_id),
    )


def _resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_ABACUS_SCENE_VARIANTS,
        task_id=str(public_task_id),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _resolve_target_column_role(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
) -> Tuple[str, Dict[str, float]]:
    axis_params = dict(params)
    if "column_role" in params and "target_column_role" not in params:
        axis_params["target_column_role"] = params["column_role"]
    return resolve_symbolic_axis_variant(
        params=axis_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=ABACUS_COLUMN_ROLES,
        task_id=str(public_task_id),
        explicit_key="target_column_role",
        weights_key="target_column_role_weights",
        balance_flag_key="balanced_target_column_role_sampling",
        axis_namespace="target_column_role",
    )


def _resolve_correct_option_label(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
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
        task_id=str(public_task_id),
        explicit_key="answer_label",
        weights_key="correct_option_label_weights",
        balance_flag_key="balanced_correct_option_label_sampling",
        axis_namespace="correct_option_label",
    )
    return str(selected), dict(probabilities)


def _build_column_readout_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    binding: AbacusColumnReadoutBinding,
) -> _ColumnReadoutDataset:
    """Sample one displayed value and one queried place-value column."""

    scene_variant, scene_variant_probabilities = _resolve_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(binding.public_task_id),
    )
    target_column_role, target_column_role_probabilities = _resolve_target_column_role(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(binding.public_task_id),
    )
    option_labels = resolve_six_option_labels(params, gen_defaults)
    correct_label, correct_label_probabilities = _resolve_correct_option_label(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(binding.public_task_id),
        option_labels=option_labels,
    )
    value_min, value_max = _get_range(
        params,
        gen_defaults,
        min_key="target_value_min",
        max_key="target_value_max",
        fallback_min=0,
        fallback_max=999,
    )
    if int(value_min) < 0 or int(value_max) > 999:
        raise ValueError("abacus displayed value support must stay within 0..999")
    rng = spawn_rng(int(instance_seed), f"{binding.public_task_id}.dataset")
    if "displayed_value" in params:
        displayed_value = int(params["displayed_value"])
    elif "answer_value" in params:
        displayed_value = int(params["answer_value"])
    else:
        displayed_value = int(rng.randint(int(value_min), int(value_max)))
    if not int(value_min) <= int(displayed_value) <= int(value_max):
        raise ValueError("displayed_value is outside configured abacus support")

    digits = digits_for_abacus_value(int(displayed_value))
    columns = tuple(
        AbacusColumnSpec(
            item_id=f"column_{role}",
            role=str(role),
            place_label=str(_PLACE_LABELS_BY_ROLE[str(role)]),
            place_value=int(_PLACE_VALUES_BY_ROLE[str(role)]),
            digit=int(digit),
        )
        for role, digit in zip(ABACUS_COLUMN_ROLES, digits)
    )
    digits_by_role = {str(column.role): int(column.digit) for column in columns}
    place_values_by_role = {
        str(column.role): int(column.place_value) for column in columns
    }
    target_role = str(target_column_role)
    answer_digit = int(digits_by_role[target_role])
    option_values_by_label = choose_digit_options(
        instance_seed=int(instance_seed),
        seed_namespace=str(binding.public_task_id),
        target_digit=int(answer_digit),
        option_labels=option_labels,
        correct_label=str(correct_label),
    )
    return _ColumnReadoutDataset(
        scene_variant=str(scene_variant),
        displayed_value=int(displayed_value),
        displayed_value_support=(int(value_min), int(value_max)),
        columns=tuple(columns),
        digits_by_role=dict(digits_by_role),
        place_values_by_role=dict(place_values_by_role),
        target_column_role=str(target_role),
        target_place_label=str(_PLACE_LABELS_BY_ROLE[target_role]),
        target_place_value=int(_PLACE_VALUES_BY_ROLE[target_role]),
        answer_digit=int(answer_digit),
        annotation_key=f"{target_role}_active_beads",
        option_labels=tuple(str(label) for label in option_labels),
        correct_label=str(correct_label),
        option_values_by_label=dict(option_values_by_label),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        target_column_role_probabilities=dict(target_column_role_probabilities),
        correct_label_probabilities=dict(correct_label_probabilities),
    )


def _build_column_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    target_place_label: str,
    instance_seed: int,
    binding: AbacusColumnReadoutBinding,
) -> tuple[str, dict[str, str], dict[str, Any], Any]:
    """Render the external prompt bundle for the queried-column digit task."""

    object_description_key = f"{binding.object_description_prefix}_{scene_variant}"
    required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        object_description_key,
        str(binding.question_text_key),
        str(binding.annotation_hint_key),
        str(binding.answer_hint_key),
        str(binding.json_example_key),
        str(binding.json_example_answer_only_key),
    )
    prompt_values = required_group_defaults(
        prompt_defaults,
        required_keys,
        context=f"prompt defaults for {binding.public_task_id}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[object_description_key]),
            "question_text": str(prompt_values[str(binding.question_text_key)]).format(
                target_place_label=str(target_place_label)
            ),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(
                prompt_values["json_output_contract_answer_only"]
            ),
            "annotation_hint": str(prompt_values[str(binding.annotation_hint_key)]),
            "answer_hint": str(prompt_values[str(binding.answer_hint_key)]),
            "json_example": str(prompt_values[str(binding.json_example_key)]),
            "json_example_answer_only": str(
                prompt_values[str(binding.json_example_answer_only_key)]
            ),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return (
        str(prompt_artifacts.prompt),
        dict(prompt_artifacts.prompt_variants),
        {
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
            "bundle_id": str(prompt_values["bundle_id"]),
        },
        prompt_artifacts,
    )


def run_abacus_column_readout_instance(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    binding: AbacusColumnReadoutBinding,
) -> TaskOutput:
    """Run the common abacus board lifecycle for a task-bound column query."""

    last_error: Exception | None = None
    dataset: _ColumnReadoutDataset | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = _build_column_readout_dataset(
                instance_seed=int(instance_seed) + int(attempt_index),
                params=params,
                gen_defaults=gen_defaults,
                binding=binding,
            )
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(str(binding.failure_message)) from last_error

    render_params = resolve_readout_render_params(render_defaults)
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{binding.public_task_id}.background",
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
    prompt, prompt_variants, prompt_meta, prompt_artifacts = _build_column_prompt(
        prompt_defaults=prompt_defaults,
        scene_variant=str(dataset.scene_variant),
        target_place_label=str(dataset.target_place_label),
        instance_seed=int(instance_seed),
        binding=binding,
    )

    if rendered_scene.selected_option_card_bbox is None:
        raise RuntimeError("abacus place-digit option rendering did not return a selected option bbox")
    annotation_artifacts = bbox_annotation_artifacts(rendered_scene.selected_option_card_bbox)
    annotation_bbox = list(annotation_artifacts.value)
    target_active_points = [
        list(point)
        for point in rendered_scene.active_bead_points_by_column.get(
            str(dataset.annotation_key),
            [],
        )
    ]
    answer_gt = TypedValue(type="option_letter", value=str(dataset.correct_label))
    query_params = {
        "query_id": SINGLE_QUERY_ID,
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "internal_query_id": str(binding.internal_query_key),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
        "question_format": str(binding.question_format),
        "target_column_role": str(dataset.target_column_role),
        "target_column_role_probabilities": dict(
            dataset.target_column_role_probabilities
        ),
        "target_place_label": str(dataset.target_place_label),
        "target_place_value": int(dataset.target_place_value),
        "target_answer_support": [str(label) for label in dataset.option_labels],
        "option_labels": [str(label) for label in dataset.option_labels],
        "correct_label_probabilities": dict(dataset.correct_label_probabilities),
        "displayed_value_support": [
            int(dataset.displayed_value_support[0]),
            int(dataset.displayed_value_support[1]),
        ],
        "annotation_key": str(dataset.annotation_key),
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
                "internal_query_id": str(binding.internal_query_key),
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "displayed_value": int(dataset.displayed_value),
                "answer_digit": int(dataset.answer_digit),
                "correct_label": str(dataset.correct_label),
                "option_values_by_label": dict(dataset.option_values_by_label),
                "target_column_role": str(dataset.target_column_role),
                "target_place_label": str(dataset.target_place_label),
                "target_place_value": int(dataset.target_place_value),
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
            "active_bead_points_by_column_px": dict(
                rendered_scene.active_bead_points_by_column
            ),
            "active_bead_ids_by_column": dict(rendered_scene.active_bead_ids_by_column),
            "column_bboxes_px": dict(rendered_scene.column_bboxes),
            "label_bboxes_px": dict(rendered_scene.label_bboxes),
            "target_column_bbox_px": list(
                rendered_scene.column_bboxes[str(dataset.target_column_role)]
            ),
            "target_active_bead_points_px": [list(point) for point in target_active_points],
            "annotation_source": "selected_option_card_bbox_px",
        },
        "execution_trace": {
            **dict(query_params),
            "displayed_value": int(dataset.displayed_value),
            "answer_value": int(dataset.answer_digit),
            "answer_digit": int(dataset.answer_digit),
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
                    "active_bead_ids": [
                        str(item)
                        for item in rendered_scene.active_bead_ids_by_column[
                            str(column.role)
                        ]
                    ],
                }
                for column in dataset.columns
            ],
            "supporting_bbox_roles": ["selected_option_card"],
            "target_active_bead_points_px": [list(point) for point in target_active_points],
        },
        "witness_symbolic": {
            "type": "bbox",
            "value": list(annotation_bbox),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
    }
    return TaskOutput(
        prompt=str(prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=SINGLE_QUERY_ID,
        prompt_variants=dict(prompt_variants),
    )


__all__ = [
    "AbacusColumnReadoutBinding",
    "load_abacus_readout_defaults",
    "run_abacus_column_readout_instance",
]
