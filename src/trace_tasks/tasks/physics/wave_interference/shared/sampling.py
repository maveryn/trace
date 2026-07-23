"""Sampling and symbolic construction for wave-interference scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    is_uniform_probability_map,
    resolve_variant,
)

from .state import (
    OPTION_LETTERS,
    SOURCE_SEPARATION_STEPS,
    SUPPORTED_PHASE_RELATIONS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_TARGET_CONDITIONS,
    ChoiceAxes,
    ChoiceScenario,
    CommonAxes,
    CandidatePoint,
    PathDifferenceAxes,
    PathDifferenceScenario,
    PointTemplate,
    SceneSpec,
    WaveInterferenceDefaults,
)


def uniform_string_probability_map(
    values: Sequence[str],
    *,
    selected: str | None = None,
) -> Dict[str, float]:
    """Return a deterministic uniform probability map over string support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        selected_text = str(selected)
        return {
            value: (1.0 if value == selected_text else 0.0)
            for value in support
        }
    probability = 1.0 / float(len(support))
    return {value: float(probability) for value in support}


def path_difference_support(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback_defaults: WaveInterferenceDefaults,
) -> Tuple[int, ...]:
    """Return configured half-wavelength path-difference support."""

    support = resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="path_difference_step_support",
        fallback=fallback_defaults.path_difference_step_support,
    )
    resolved = tuple(int(value) for value in support if int(value) > 0)
    if not resolved:
        raise ValueError("path_difference_value requires positive path-difference support")
    return resolved


def resolve_common_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> CommonAxes:
    """Resolve the scene/render axes shared by both public tasks."""

    scene_rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    scene_variant, scene_probs = resolve_variant(
        scene_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    scene_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(scene_variant),
        variant_probabilities=scene_probs,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )

    phase_rng = spawn_rng(int(instance_seed), f"{namespace}.phase_relation")
    phase_relation, phase_probs = resolve_variant(
        phase_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_PHASE_RELATIONS,
        explicit_key="phase_relation",
        weights_key="phase_relation_weights",
    )
    phase_relation = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(phase_relation),
        variant_probabilities=phase_probs,
        supported_variants=SUPPORTED_PHASE_RELATIONS,
        balance_flag_key="balanced_phase_relation_sampling",
        explicit_key="phase_relation",
        weights_key="phase_relation_weights",
        sampling_namespace=f"{namespace}.phase_relation",
    )

    accent_rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
    accent_name, accent_probs = resolve_variant(
        accent_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(accent_name),
        variant_probabilities=accent_probs,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )

    return CommonAxes(
        scene_variant=str(scene_variant),
        phase_relation=str(phase_relation),
        accent_color_name=str(accent_name),
        scene_variant_probabilities={
            str(key): float(value)
            for key, value in sorted(scene_probs.items())
        },
        phase_relation_probabilities={
            str(key): float(value)
            for key, value in sorted(phase_probs.items())
        },
        accent_color_name_probabilities={
            str(key): float(value)
            for key, value in sorted(accent_probs.items())
        },
    )


def resolve_choice_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> ChoiceAxes:
    """Resolve condition and answer-letter axes for point-choice tasks."""

    condition_rng = spawn_rng(int(instance_seed), f"{namespace}.target_condition")
    target_condition, condition_probs = resolve_variant(
        condition_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_TARGET_CONDITIONS,
        explicit_key="target_condition",
        weights_key="target_condition_weights",
    )
    target_condition = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(target_condition),
        variant_probabilities=condition_probs,
        supported_variants=SUPPORTED_TARGET_CONDITIONS,
        balance_flag_key="balanced_target_condition_sampling",
        explicit_key="target_condition",
        weights_key="target_condition_weights",
        sampling_namespace=f"{namespace}.target_condition",
    )

    adjusted_params = dict(params)
    if (
        adjusted_params.get("correct_option_letter") is None
        and adjusted_params.get("target_answer") is not None
    ):
        adjusted_params["correct_option_letter"] = str(
            adjusted_params["target_answer"]
        ).strip().upper()
    option_rng = spawn_rng(int(instance_seed), f"{namespace}.correct_option_letter")
    correct_letter, option_probs = resolve_variant(
        option_rng,
        params=adjusted_params,
        gen_defaults=generation_defaults,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="option_letter_weights",
    )
    balanced_enabled = bool(
        adjusted_params.get(
            "balanced_option_letter_sampling",
            group_default(generation_defaults, "balanced_option_letter_sampling", True),
        )
    )
    has_override = any(
        adjusted_params.get(str(key)) is not None
        for key in ("correct_option_letter", "option_letter_weights")
    )
    if bool(balanced_enabled) and not bool(has_override) and is_uniform_probability_map(option_probs):
        correct_letter = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=adjusted_params,
            gen_defaults=generation_defaults,
            selected_variant=str(correct_letter),
            variant_probabilities=option_probs,
            supported_variants=OPTION_LETTERS,
            balance_flag_key="balanced_option_letter_sampling",
            explicit_key="correct_option_letter",
            weights_key="option_letter_weights",
            sampling_namespace=f"{namespace}.correct_option_letter",
        )

    return ChoiceAxes(
        target_condition=str(target_condition),
        correct_option_letter=str(correct_letter),
        target_condition_probabilities={
            str(key): float(value)
            for key, value in sorted(condition_probs.items())
        },
        correct_option_letter_probabilities={
            str(key): float(value)
            for key, value in sorted(option_probs.items())
        },
    )


def resolve_path_difference_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_defaults: WaveInterferenceDefaults,
    namespace: str,
) -> PathDifferenceAxes:
    """Resolve the integer path difference in half-wavelength steps."""

    support = path_difference_support(
        params,
        generation_defaults=generation_defaults,
        fallback_defaults=fallback_defaults,
    )
    explicit = params.get("target_answer", params.get("path_difference_steps"))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported path_difference target_answer: {selected}")
        return PathDifferenceAxes(
            path_difference_steps=int(selected),
            path_difference_steps_probabilities=uniform_probability_map(
                support,
                selected=int(selected),
            ),
        )

    balanced_enabled = bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(generation_defaults, "balanced_target_answer_sampling", True),
        )
    )
    if balanced_enabled:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.path_difference_value")
        selected = int(rng.choice(support))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.path_difference_steps")
        selected = int(support[int(rng.randrange(len(support)))])
    return PathDifferenceAxes(
        path_difference_steps=int(selected),
        path_difference_steps_probabilities=uniform_probability_map(support),
    )


def point_templates() -> Tuple[PointTemplate, ...]:
    """Return feasible candidate points with exact half-wavelength distances."""

    templates: list[PointTemplate] = []
    source_separation = float(SOURCE_SEPARATION_STEPS)
    for r1_steps in range(3, 11):
        for r2_steps in range(3, 11):
            x_from_s1 = (
                (r1_steps * r1_steps)
                - (r2_steps * r2_steps)
                + (SOURCE_SEPARATION_STEPS * SOURCE_SEPARATION_STEPS)
            ) / (2.0 * source_separation)
            y_squared = float((r1_steps * r1_steps) - (x_from_s1 * x_from_s1))
            if y_squared <= 1e-6:
                continue
            y_steps = math.sqrt(y_squared)
            x_steps = float(x_from_s1 - (source_separation / 2.0))
            if abs(x_steps) > 5.9 or y_steps > 4.9:
                continue
            for sign_y in (1, -1):
                templates.append(
                    PointTemplate(
                        r1_steps=int(r1_steps),
                        r2_steps=int(r2_steps),
                        sign_y=int(sign_y),
                        x_steps=round(float(x_steps), 6),
                        y_steps=round(float(sign_y) * float(y_steps), 6),
                    )
                )
    return tuple(templates)


def classify_condition(
    *,
    r1_steps: int,
    r2_steps: int,
    phase_relation: str,
) -> str:
    """Classify a point as constructive or destructive from phase parity."""

    source_2_offset = 0 if str(phase_relation) == "in_phase" else 1
    phase_1 = int(r1_steps) % 2
    phase_2 = (int(r2_steps) + int(source_2_offset)) % 2
    return "constructive" if phase_1 == phase_2 else "destructive"


def _template_distance(a: PointTemplate, b: PointTemplate) -> float:
    """Return distance between two templates in half-wavelength coordinates."""

    return math.hypot(float(a.x_steps) - float(b.x_steps), float(a.y_steps) - float(b.y_steps))


def select_spaced_templates(
    rng,
    *,
    pool: Sequence[PointTemplate],
    count: int,
    existing: Sequence[PointTemplate],
) -> Tuple[PointTemplate, ...]:
    """Select visually separated candidate templates."""

    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected: list[PointTemplate] = []
    for template in shuffled:
        if any(
            _template_distance(template, other) < 1.0
            for other in tuple(existing) + tuple(selected)
        ):
            continue
        selected.append(template)
        if len(selected) >= int(count):
            return tuple(selected)
    for template in shuffled:
        if template in selected or template in existing:
            continue
        selected.append(template)
        if len(selected) >= int(count):
            return tuple(selected)
    if len(selected) < int(count):
        raise ValueError("not enough spaced wave-interference candidate points")
    return tuple(selected)


def sample_choice_scenario(
    rng,
    *,
    common_axes: CommonAxes,
    choice_axes: ChoiceAxes,
) -> ChoiceScenario:
    """Build an interference point-choice scenario with one correct candidate."""

    templates = point_templates()
    target_pool = [
        template
        for template in templates
        if classify_condition(
            r1_steps=template.r1_steps,
            r2_steps=template.r2_steps,
            phase_relation=str(common_axes.phase_relation),
        )
        == str(choice_axes.target_condition)
    ]
    distractor_pool = [
        template
        for template in templates
        if classify_condition(
            r1_steps=template.r1_steps,
            r2_steps=template.r2_steps,
            phase_relation=str(common_axes.phase_relation),
        )
        != str(choice_axes.target_condition)
    ]
    if not target_pool or len(distractor_pool) < len(OPTION_LETTERS) - 1:
        raise ValueError("not enough feasible wave-interference candidates")
    correct_template = target_pool[int(rng.randrange(len(target_pool)))]
    distractors = select_spaced_templates(
        rng,
        pool=distractor_pool,
        count=len(OPTION_LETTERS) - 1,
        existing=(correct_template,),
    )
    distractor_iter = iter(distractors)
    candidates: list[CandidatePoint] = []
    for letter in OPTION_LETTERS:
        template = (
            correct_template
            if str(letter) == str(choice_axes.correct_option_letter)
            else next(distractor_iter)
        )
        condition = classify_condition(
            r1_steps=int(template.r1_steps),
            r2_steps=int(template.r2_steps),
            phase_relation=str(common_axes.phase_relation),
        )
        candidates.append(
            CandidatePoint(
                letter=str(letter),
                x_steps=float(template.x_steps),
                y_steps=float(template.y_steps),
                r1_steps=int(template.r1_steps),
                r2_steps=int(template.r2_steps),
                condition=str(condition),
                is_correct=str(letter) == str(choice_axes.correct_option_letter),
            )
        )
    return ChoiceScenario(
        phase_relation=str(common_axes.phase_relation),
        target_condition=str(choice_axes.target_condition),
        candidates=tuple(candidates),
        correct_option_letter=str(choice_axes.correct_option_letter),
    )


def sample_path_scenario(
    rng,
    *,
    common_axes: CommonAxes,
    path_axes: PathDifferenceAxes,
) -> PathDifferenceScenario:
    """Build a path-difference value scenario."""

    templates = [
        template
        for template in point_templates()
        if abs(int(template.r1_steps) - int(template.r2_steps))
        == int(path_axes.path_difference_steps)
    ]
    if not templates:
        raise ValueError(
            f"no wave-interference path scenario for answer {path_axes.path_difference_steps}"
        )
    template = templates[int(rng.randrange(len(templates)))]
    return PathDifferenceScenario(
        phase_relation=str(common_axes.phase_relation),
        point_x_steps=float(template.x_steps),
        point_y_steps=float(template.y_steps),
        r1_steps=int(template.r1_steps),
        r2_steps=int(template.r2_steps),
        path_difference_steps=abs(int(template.r1_steps) - int(template.r2_steps)),
    )


def make_choice_scene_spec(
    *,
    common_axes: CommonAxes,
    scenario: ChoiceScenario,
) -> SceneSpec:
    """Bind one sampled choice scenario into a renderable scene spec."""

    return SceneSpec(
        scene_variant=str(common_axes.scene_variant),
        phase_relation=str(common_axes.phase_relation),
        choice_scenario=scenario,
        path_scenario=None,
        target_answer=str(scenario.correct_option_letter),
        annotation_entity_ids=(f"candidate_{str(scenario.correct_option_letter)}",),
    )


def make_path_scene_spec(
    *,
    common_axes: CommonAxes,
    scenario: PathDifferenceScenario,
) -> SceneSpec:
    """Bind one sampled path-difference scenario into a renderable scene spec."""

    return SceneSpec(
        scene_variant=str(common_axes.scene_variant),
        phase_relation=str(common_axes.phase_relation),
        choice_scenario=None,
        path_scenario=scenario,
        target_answer=int(scenario.path_difference_steps),
        annotation_entity_ids=("path_S1P", "path_S2P"),
    )


__all__ = [
    "classify_condition",
    "make_choice_scene_spec",
    "make_path_scene_spec",
    "path_difference_support",
    "point_templates",
    "resolve_choice_axes",
    "resolve_common_axes",
    "resolve_path_difference_axes",
    "sample_choice_scenario",
    "sample_path_scenario",
    "uniform_string_probability_map",
]
