"""Choose the visual option that completes an icon-rotation progression."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from .shared.output import build_completion_trace_payload, render_completion_artifacts
from .shared.prompts import render_sequence_strip_prompt_artifacts
from .shared.rendering import validate_sequence_cell_box_bounds
from .shared.sampling import (
    CyclicProgressionSample,
    SequenceCompletionDefaults,
    SequenceCompletionPlan,
    resolve_completion_render_params,
    resolve_cyclic_progression_sample,
    sample_sequence_icon_id,
    sample_sequence_tint,
    sequence_missing_index,
    single_icon_completion_cells,
)


TASK_ID = "task_icons__sequence_strip__rotation_progression_completion_label"
DOMAIN = "icons"
SCENE_ID = "sequence_strip"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "rotation_progression_completion_label"
SCENE_KIND = "icons_sequence_rotation_completion"
QUESTION_FORMAT = "choose_rotation_progression_completion_option"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
COMMON_IDS = {"domain": DOMAIN, "scene_id": SCENE_ID, "task_id": TASK_ID, "query_id": QUERY_ID}
TASK_VERSIONS = default_task_versions()
_DEFAULTS = SequenceCompletionDefaults(pool_manifest="non_symmetry.txt", scene_icon_size_min_px=62, scene_icon_size_max_px=72)
_ROTATION_VALUE_SUPPORT = (0, 45, 90, 135, 180, 225, 270, 315)
_ROTATION_STEP_SUPPORT = (45, 90, 270, 315)


def _resolve_rotation_progression(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    missing_index: int,
) -> CyclicProgressionSample:
    """Resolve the hidden rotation in a four-cell constant-step sequence."""

    return resolve_cyclic_progression_sample(
        params=params,
        defaults=generation_defaults,
        instance_seed=int(instance_seed),
        missing_index=int(missing_index),
        value_key="rotation_value_candidates_degrees",
        step_key="rotation_step_candidates_degrees",
        answer_key="answer_rotation_degrees",
        start_key="start_rotation_degrees",
        fallback_values=_ROTATION_VALUE_SUPPORT,
        fallback_steps=_ROTATION_STEP_SUPPORT,
        selection_namespace="rotation_progression",
        probability_value_key="rotation_values",
        probability_step_key="rotation_steps",
    )


def _rotation_cells(
    *,
    instance_seed: int,
    progression: CyclicProgressionSample,
    option_map: Mapping[str, int],
    missing_index: int,
    icon_id: str,
    tint_rgb: Tuple[int, int, int],
    nominal_size: int,
    render_params: Mapping[str, Any],
):
    """Render rotation values as same-size icons turned by the candidate angle."""

    return single_icon_completion_cells(
        instance_seed=int(instance_seed),
        noise_stem="rotation_progression",
        sequence_values=progression.sequence_values,
        option_values_by_label={label: int(value) for label, value in option_map.items()},
        missing_index=int(missing_index),
        icon_id=str(icon_id),
        render_params=render_params,
        tint_for_value=lambda _value: tint_rgb,
        size_for_value=lambda _value: int(nominal_size),
        rotation_for_value=lambda value: int(value),
    )


def _rotation_trace_extra(*, progression: CyclicProgressionSample, nominal_size: int) -> Dict[str, Any]:
    """Serialize rotation-specific symbolic support."""

    return {
        "start_rotation_degrees": int(progression.start_value),
        "rotation_step_degrees": int(progression.step_value),
        "rotation_value_support_degrees": [int(value) for value in progression.value_support],
        "rotation_icon_size_px": int(nominal_size),
    }


def _validate_rotation_option_angles(*, option_map: Mapping[str, int], progression: CyclicProgressionSample) -> None:
    """Ensure every visual rotation option is distinct and on the configured compass grid."""

    angle_by_label = {str(label): int(value) % 360 for label, value in option_map.items()}
    if set(angle_by_label) != {"A", "B", "C", "D"}:
        raise ValueError("rotation completion requires fixed A-D option labels")
    if len(set(angle_by_label.values())) != len(angle_by_label):
        raise ValueError("rotation completion options must use distinct angles")
    unsupported = sorted(set(angle_by_label.values()).difference(set(progression.value_support)))
    if unsupported:
        raise ValueError(f"rotation option angle is outside support: {unsupported[0]}")
    for left_label, left_angle in angle_by_label.items():
        for right_label, right_angle in angle_by_label.items():
            if left_label >= right_label:
                continue
            clockwise_gap = (int(right_angle) - int(left_angle)) % 360
            nearest_gap = min(clockwise_gap, 360 - clockwise_gap)
            if nearest_gap < 45:
                raise ValueError("rotation completion options must be separated by at least 45 degrees")


def _rotation_option_values(rng, *, progression: CyclicProgressionSample) -> Tuple[str, Dict[str, int]]:
    """Place the correct rotation option in a stable balanced A-D slot."""

    labels = ("A", "B", "C", "D")
    answer_value = int(progression.answer_value)
    support = tuple(int(value) for value in progression.value_support)
    if answer_value not in set(support):
        raise ValueError("rotation answer must be in the configured value support")
    correct_label = str(uniform_choice(rng, labels))
    distractors = [int(value) for value in support if int(value) != answer_value]
    if len(distractors) < 3:
        raise ValueError("rotation completion requires at least three distractor values")
    rng.shuffle(distractors)
    option_map: Dict[str, int] = {str(correct_label): int(answer_value)}
    distractor_iter = iter(distractors)
    for label in labels:
        if str(label) == str(correct_label):
            continue
        option_map[str(label)] = int(next(distractor_iter))
    return str(correct_label), option_map


def _build_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    fallback_defaults: SequenceCompletionDefaults,
) -> SequenceCompletionPlan:
    """Build rotation-specific visual cells and symbolic option bindings."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:plan")
    missing_index = sequence_missing_index(params=params, generation_defaults=generation_defaults, instance_seed=int(instance_seed))
    progression = _resolve_rotation_progression(
        params=params,
        instance_seed=int(instance_seed),
        missing_index=int(missing_index),
        generation_defaults=generation_defaults,
    )
    icon_id = sample_sequence_icon_id(rng, params=params, generation_defaults=generation_defaults, fallback=fallback_defaults.pool_manifest)
    tint_rgb, palette = sample_sequence_tint(rng, render_params=render_params)
    correct_label, option_map = _rotation_option_values(rng, progression=progression)
    _validate_rotation_option_angles(
        option_map={label: int(value) for label, value in option_map.items()},
        progression=progression,
    )
    nominal_size = int(params.get("rotation_icon_size_px", group_default(generation_defaults, "rotation_icon_size_px", 66)))
    sequence_cells, option_cells = _rotation_cells(
        instance_seed=int(instance_seed),
        progression=progression,
        option_map={label: int(value) for label, value in option_map.items()},
        missing_index=int(missing_index),
        icon_id=str(icon_id),
        tint_rgb=tuple(int(channel) for channel in tint_rgb),
        nominal_size=int(nominal_size),
        render_params=render_params,
    )

    return SequenceCompletionPlan(
        attribute_id="rotation",
        sequence_rule="constant_rotation_step",
        sequence_icon_id=str(icon_id),
        full_sequence_values=tuple(int(value) for value in progression.sequence_values),
        missing_index=int(missing_index),
        correct_option_label=str(correct_label),
        correct_option_value=int(progression.answer_value),
        option_values_by_label={label: int(value) for label, value in option_map.items()},
        sequence_cells=sequence_cells,
        option_cells=option_cells,
        sampled_palette_rgb=palette,
        support_probabilities=dict(progression.support_probabilities),
        extra_trace=_rotation_trace_extra(progression=progression, nominal_size=int(nominal_size)),
    )


@register_task
class IconsSequenceStripRotationProgressionCompletionTask:
    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate rotation-completion output while binding answer and annotation locally."""

        generation_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
            DOMAIN,
            SCENE_ID,
            task_id=TASK_ID,
        )
        render_params = resolve_completion_render_params(
            params=params,
            render_defaults=render_defaults,
            fallback_defaults=_DEFAULTS,
            instance_seed=instance_seed,
        )
        validate_sequence_cell_box_bounds(render_params)
        plan = _build_plan(instance_seed, params, generation_defaults, render_params, _DEFAULTS)
        rendered = render_completion_artifacts(plan=plan, render_params=render_params)
        annotation_payload = bbox_annotation_artifacts(rendered.correct_option_bbox)
        prompt_defaults, prompt_artifacts = render_sequence_strip_prompt_artifacts(
            instance_seed=instance_seed,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
        )
        trace_payload = build_completion_trace_payload(
            common_ids=COMMON_IDS,
            scene_kind=SCENE_KIND,
            question_format=QUESTION_FORMAT,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            plan=plan,
            rendered=rendered,
            annotation_payload=annotation_payload,
            render_params=render_params,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue("string", str(plan.correct_option_label)),
            annotation_gt=annotation_payload.annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=TASK_VERSIONS,
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsSequenceStripRotationProgressionCompletionTask"]
