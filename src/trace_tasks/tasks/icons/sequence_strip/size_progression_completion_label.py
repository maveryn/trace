"""Choose the visual option that completes an icon-size progression."""

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
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.output_metadata import default_task_versions
from .shared.output import build_completion_trace_payload, render_completion_artifacts
from .shared.prompts import render_sequence_strip_prompt_artifacts
from .shared.rendering import validate_sequence_cell_box_bounds
from .shared.sampling import (
    SequenceCompletionDefaults,
    SequenceCompletionPlan,
    int_support_from_bounds,
    option_values,
    resolve_completion_render_params,
    sample_sequence_icon_id,
    sample_sequence_tint,
    sequence_missing_index,
    single_icon_completion_cells,
)


TASK_ID = "task_icons__sequence_strip__size_progression_completion_label"
DOMAIN = "icons"
SCENE_ID = "sequence_strip"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "size_progression_completion_label"
SCENE_KIND = "icons_sequence_size_completion"
QUESTION_FORMAT = "choose_size_progression_completion_option"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
COMMON_IDS = {"domain": DOMAIN, "scene_id": SCENE_ID, "task_id": TASK_ID, "query_id": QUERY_ID}
TASK_VERSIONS = default_task_versions()
_DEFAULTS = SequenceCompletionDefaults(scene_icon_size_min_px=32, scene_icon_size_max_px=92)


def _step_candidates(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("size_step_candidates_px", group_default(generation_defaults, "size_step_candidates_px", (-16, -12, 12, 16)))
    values = tuple(int(value) for value in raw)
    values = tuple(value for value in dict.fromkeys(values) if int(value) != 0)
    if not values:
        raise ValueError("size_step_candidates_px must include at least one non-zero step")
    return values


def _validate_size_values(*, sequence: Tuple[int, ...], option_map: Mapping[str, int]) -> None:
    """Ensure size progression candidates stay visually separable."""

    if len(set(int(value) for value in sequence)) != 4:
        raise ValueError("size progression sequence must contain four distinct sizes")
    for label, size_px in option_map.items():
        if int(size_px) <= 0:
            raise ValueError(f"size option {label} must be positive")
    if len(set(int(value) for value in option_map.values())) != len(option_map):
        raise ValueError("size completion options must be unique")
    for left, right in zip(sequence, sequence[1:]):
        if abs(int(right) - int(left)) < 12:
            raise ValueError("size progression steps must differ by at least 12 px")


def _build_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    fallback_defaults: SequenceCompletionDefaults,
) -> SequenceCompletionPlan:
    """Build size-specific visual cells and symbolic option bindings."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:plan")
    missing_index = sequence_missing_index(params=params, generation_defaults=generation_defaults, instance_seed=int(instance_seed))
    size_support = int_support_from_bounds(
        params=params,
        defaults=generation_defaults,
        min_key="size_value_min_px",
        max_key="size_value_max_px",
        fallback_min=36,
        fallback_max=88,
    )
    step_support = _step_candidates(params, generation_defaults)
    explicit_answer = params.get("answer_size_px")
    explicit_step = params.get("size_step_px")
    feasible: list[tuple[int, int, Tuple[int, ...]]] = []
    support_set = set(int(value) for value in size_support)
    for answer_size in size_support:
        if explicit_answer is not None and int(answer_size) != int(explicit_answer):
            continue
        for step in step_support:
            if explicit_step is not None and int(step) != int(explicit_step):
                continue
            sequence = tuple(int(answer_size) + ((index - int(missing_index)) * int(step)) for index in range(4))
            if len(set(sequence)) == 4 and all(int(value) in support_set for value in sequence):
                feasible.append((int(answer_size), int(step), sequence))
    if not feasible:
        raise ValueError("no feasible size progression for requested parameters")
    answer_size, step, sequence = uniform_choice(
        spawn_rng(int(instance_seed), f"{TASK_ID}:size_progression"),
        tuple(feasible),
    )
    icon_id = sample_sequence_icon_id(rng, params=params, generation_defaults=generation_defaults, fallback=fallback_defaults.pool_manifest)
    tint_rgb, palette = sample_sequence_tint(rng, render_params=render_params)
    distractors = [value for value in size_support if int(value) != int(answer_size) and abs(int(value) - int(answer_size)) >= 12]
    correct_label, option_map = option_values(rng, correct_value=int(answer_size), distractor_values=distractors)
    _validate_size_values(sequence=sequence, option_map={label: int(value) for label, value in option_map.items()})
    sequence_cells, option_cells = single_icon_completion_cells(
        instance_seed=int(instance_seed),
        noise_stem="size_progression",
        sequence_values=sequence,
        option_values_by_label={label: int(value) for label, value in option_map.items()},
        missing_index=int(missing_index),
        icon_id=str(icon_id),
        render_params=render_params,
        tint_for_value=lambda _value: tint_rgb,
        size_for_value=lambda value: int(value),
    )

    return SequenceCompletionPlan(
        attribute_id="size",
        sequence_rule="constant_size_step",
        sequence_icon_id=str(icon_id),
        full_sequence_values=tuple(int(value) for value in sequence),
        missing_index=int(missing_index),
        correct_option_label=str(correct_label),
        correct_option_value=int(answer_size),
        option_values_by_label={label: int(value) for label, value in option_map.items()},
        sequence_cells=sequence_cells,
        option_cells=option_cells,
        sampled_palette_rgb=palette,
        support_probabilities={
            "size_values_px": uniform_probability_map(size_support, selected=int(answer_size) if explicit_answer is not None else None),
            "size_steps_px": uniform_probability_map(step_support, selected=int(step) if explicit_step is not None else None),
        },
        extra_trace={
            "size_step_px": int(step),
            "size_value_support_px": [int(value) for value in size_support],
        },
    )


@register_task
class IconsSequenceStripSizeProgressionCompletionTask:
    task_id = TASK_ID
    reasoning_operations = ('transformation', 'formula_evaluation', 'matching')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate size-completion output while binding answer and annotation locally."""

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


__all__ = ["IconsSequenceStripSizeProgressionCompletionTask"]
