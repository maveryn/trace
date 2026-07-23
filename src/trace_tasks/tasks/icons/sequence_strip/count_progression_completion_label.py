"""Choose the visual option that completes an icon-count progression."""

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
from .shared.rendering import IconSequenceCellSpec, validate_sequence_cell_box_bounds
from .shared.sampling import (
    SequenceCompletionDefaults,
    SequenceCompletionPlan,
    icon_instance,
    int_support_from_bounds,
    option_values,
    resolve_completion_render_params,
    sample_sequence_icon_id,
    sample_sequence_tint,
    sequence_missing_index,
)


TASK_ID = "task_icons__sequence_strip__count_progression_completion_label"
DOMAIN = "icons"
SCENE_ID = "sequence_strip"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "count_progression_completion_label"
SCENE_KIND = "icons_sequence_count_completion"
QUESTION_FORMAT = "choose_count_progression_completion_option"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
COMMON_IDS = {"domain": DOMAIN, "scene_id": SCENE_ID, "task_id": TASK_ID, "query_id": QUERY_ID}
TASK_VERSIONS = default_task_versions()
_DEFAULTS = SequenceCompletionDefaults(scene_icon_size_min_px=24, scene_icon_size_max_px=42)


def _step_candidates(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("count_step_candidates", group_default(generation_defaults, "count_step_candidates", (-2, -1, 1, 2)))
    values = tuple(int(value) for value in raw)
    values = tuple(value for value in dict.fromkeys(values) if int(value) != 0)
    if not values:
        raise ValueError("count_step_candidates must include at least one non-zero step")
    return values


def _resolve_counts(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    missing_index: int,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Dict[str, Any]]:
    """Resolve the arithmetic count sequence with the hidden value as answer."""

    count_support = int_support_from_bounds(
        params=params,
        defaults=generation_defaults,
        min_key="count_value_min",
        max_key="count_value_max",
        fallback_min=1,
        fallback_max=9,
    )
    step_support = _step_candidates(params, generation_defaults)
    explicit_answer = params.get("answer_count")
    explicit_step = params.get("count_step")
    feasible: list[tuple[int, int, Tuple[int, ...]]] = []
    for answer_count in count_support:
        if explicit_answer is not None and int(answer_count) != int(explicit_answer):
            continue
        for step in step_support:
            if explicit_step is not None and int(step) != int(explicit_step):
                continue
            sequence = tuple(int(answer_count) + ((index - int(missing_index)) * int(step)) for index in range(4))
            if all(int(value) in set(count_support) for value in sequence):
                feasible.append((int(answer_count), int(step), sequence))
    if not feasible:
        raise ValueError("no feasible count progression for requested parameters")
    answer_count, step, sequence = uniform_choice(
        spawn_rng(int(instance_seed), f"{TASK_ID}:count_progression"),
        tuple(feasible),
    )
    probabilities = {
        "count_values": uniform_probability_map(count_support, selected=int(answer_count) if explicit_answer is not None else None),
        "count_steps": uniform_probability_map(step_support, selected=int(step) if explicit_step is not None else None),
    }
    return int(answer_count), int(step), tuple(int(value) for value in sequence), count_support, probabilities


def _build_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    fallback_defaults: SequenceCompletionDefaults,
) -> SequenceCompletionPlan:
    """Build count-specific visual cells and symbolic answer bindings."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:plan")
    missing_index = sequence_missing_index(params=params, generation_defaults=generation_defaults, instance_seed=int(instance_seed))
    answer_count, step, sequence, count_support, probabilities = _resolve_counts(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        missing_index=int(missing_index),
    )
    icon_id = sample_sequence_icon_id(rng, params=params, generation_defaults=generation_defaults, fallback=fallback_defaults.pool_manifest)
    tint_rgb, palette = sample_sequence_tint(rng, render_params=render_params)
    distractors = [value for value in count_support if int(value) != int(answer_count)]
    correct_label, option_map = option_values(rng, correct_value=int(answer_count), distractor_values=distractors)

    def cell_for_count(count: int, *, namespace: str, label: str | None = None, missing: bool = False) -> IconSequenceCellSpec:
        if missing:
            return IconSequenceCellSpec(is_missing=True)
        icons = tuple(
            icon_instance(
                instance_seed=int(instance_seed),
                noise_namespace=f"{TASK_ID}:{namespace}_icon_{icon_index}",
                icon_id=icon_id,
                tint_rgb=tint_rgb,
                render_params=render_params,
            )
            for icon_index in range(int(count))
        )
        return IconSequenceCellSpec(icon_instances=icons, cell_label_text=label)

    sequence_cells = tuple(
        cell_for_count(
            int(count),
            namespace=f"sequence_{index}",
            missing=int(index) == int(missing_index),
        )
        for index, count in enumerate(sequence)
    )
    option_cells = tuple(
        cell_for_count(int(option_map[label]), namespace=f"option_{label}", label=str(label))
        for label in ("A", "B", "C", "D")
    )
    return SequenceCompletionPlan(
        attribute_id="count",
        sequence_rule="arithmetic_count_progression",
        sequence_icon_id=str(icon_id),
        full_sequence_values=tuple(int(value) for value in sequence),
        missing_index=int(missing_index),
        correct_option_label=str(correct_label),
        correct_option_value=int(answer_count),
        option_values_by_label={label: int(value) for label, value in option_map.items()},
        sequence_cells=sequence_cells,
        option_cells=option_cells,
        sampled_palette_rgb=palette,
        support_probabilities=probabilities,
        extra_trace={
            "count_step": int(step),
            "count_value_support": [int(value) for value in count_support],
        },
    )


@register_task
class IconsSequenceStripCountProgressionCompletionTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation', 'matching')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate count-completion output while binding answer and annotation locally."""

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


__all__ = ["IconsSequenceStripCountProgressionCompletionTask"]
