"""Sampling helpers for thermal-mixing scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .formulas import integer_average
from .state import (
    CUP_COUNTS,
    FINAL_TEMPERATURE_SUPPORT,
    OFFSET_PATTERNS,
    SCENE_NAMESPACE,
    ThermalMixingDefaults,
    ThermalMixingScenario,
)


def resolve_cup_count(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Resolve how many equal-amount liquid cups are visible."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.cup_count"),
        params={
            **dict(params),
            "cup_count": str(params["cup_count"]) if params.get("cup_count") is not None else None,
        },
        gen_defaults=generation_defaults,
        supported_variants=CUP_COUNTS,
        explicit_key="cup_count",
        weights_key="cup_count_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=CUP_COUNTS,
        balance_flag_key="balanced_cup_count_sampling",
        explicit_key="cup_count",
        weights_key="cup_count_weights",
        sampling_namespace=f"{SCENE_NAMESPACE}.cup_count",
    )
    return int(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_final_temperature(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Resolve the integer final equilibrium temperature."""

    support = [
        int(value)
        for value in params.get(
            "final_temperature_support",
            group_default(
                generation_defaults,
                "final_temperature_support",
                list(FINAL_TEMPERATURE_SUPPORT),
            ),
        )
    ]
    if not support:
        raise ValueError("final_temperature_support must not be empty")
    explicit = params.get("target_answer", params.get("final_temperature_c"))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported final_temperature_c: {selected}")
        return selected, {str(value): (1.0 if int(value) == selected else 0.0) for value in support}
    if bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(generation_defaults, "balanced_target_answer_sampling", True),
        )
    ):
        selected = int(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.final_temperature").choice(support))
    else:
        selected = int(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.final_temperature").choice(support))
    probability = 1.0 / float(len(support))
    return int(selected), {str(value): float(probability) for value in support}


def make_thermal_mixing_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> ThermalMixingScenario:
    """Build a scenario whose visible cup temperatures average to the answer."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scenario")
    cup_count, cup_probs = resolve_cup_count(
        int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    final_temperature, final_probs = resolve_final_temperature(
        int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    patterns = list(OFFSET_PATTERNS[int(cup_count)])
    rng.shuffle(patterns)
    selected_offsets: Tuple[int, ...] | None = None
    for offsets in patterns:
        temperatures = [int(final_temperature) + int(offset) for offset in offsets]
        if all(0 <= temp <= 100 for temp in temperatures):
            selected_offsets = tuple(int(offset) for offset in offsets)
            break
    if selected_offsets is None:
        raise ValueError(
            f"could not find bounded temperature offsets for final temperature {final_temperature}"
        )
    temperatures = [int(final_temperature) + int(offset) for offset in selected_offsets]
    rng.shuffle(temperatures)
    computed_final = integer_average([int(temp) for temp in temperatures])
    if computed_final != int(final_temperature):
        raise AssertionError("thermal mixing construction drifted from target final temperature")
    return ThermalMixingScenario(
        cup_count=int(cup_count),
        initial_temperatures_c=tuple(int(temp) for temp in temperatures),
        final_temperature_c=int(final_temperature),
        cup_count_probabilities=dict(cup_probs),
        final_temperature_probabilities=dict(final_probs),
    )


def resolve_thermal_mixing_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    defaults: ThermalMixingDefaults,
) -> Dict[str, int]:
    """Resolve numeric rendering controls."""

    keys = (
        "panel_left_px",
        "panel_top_px",
        "panel_right_margin_px",
        "panel_bottom_margin_px",
        "cup_width_px",
        "cup_height_px",
        "cup_top_px",
        "cup_gap_px",
        "mixer_width_px",
        "mixer_height_px",
        "mixer_top_px",
        "title_font_size_px",
        "label_font_size_px",
        "temp_font_size_px",
        "note_font_size_px",
        "label_stroke_width_px",
    )
    return {
        str(key): resolve_render_int(
            params,
            rendering_defaults,
            str(key),
            int(getattr(defaults, str(key))),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        )
        for key in keys
    }


__all__ = [
    "make_thermal_mixing_scenario",
    "resolve_cup_count",
    "resolve_final_temperature",
    "resolve_thermal_mixing_render_defaults",
]
