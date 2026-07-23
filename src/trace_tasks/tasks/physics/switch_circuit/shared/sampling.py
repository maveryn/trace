"""Sampling helpers for switch-circuit scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_choice
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .circuitry import enumerate_switch_state_solutions, lit_bulbs_from_edges, make_edges
from .state import (
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    SWITCH_LABELS,
    TARGET_SUPPORT,
    SwitchCircuitDefaults,
    SwitchCircuitScenario,
)


def probability_map(values: Tuple[str, ...], selected: str | None = None) -> Dict[str, float]:
    """Return a uniform or one-hot probability map for trace metadata."""

    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(value): float(probability) for value in values}


def parse_switch_value(raw: Any, *, label: str) -> bool:
    """Parse a switch value from task params."""

    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) in {0, 1}:
            return bool(int(raw))
    if isinstance(raw, str):
        text = str(raw).strip().lower()
        if text in {"closed", "close", "on", "true", "1", "yes", "y"}:
            return True
        if text in {"open", "off", "false", "0", "no", "n"}:
            return False
    raise ValueError(f"unsupported switch state for {label}: {raw!r}")


def parse_switch_states(params: Mapping[str, Any]) -> Dict[str, bool] | None:
    """Parse explicit switch states if supplied."""

    raw = params.get("switch_states")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("switch_states must be a mapping from S1..S5 to open/closed values")
    missing = [label for label in SWITCH_LABELS if label not in raw]
    if missing:
        raise ValueError(f"switch_states is missing switch labels: {missing}")
    return {
        str(label): parse_switch_value(raw[str(label)], label=str(label))
        for label in SWITCH_LABELS
    }


def resolve_scene_variant(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the switch-circuit scene variant."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scene_variant"),
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{SCENE_NAMESPACE}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_target_answer(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Resolve the target lit-bulb count."""

    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=TARGET_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )


def resolve_accent_color(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the circuit accent color."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.accent_color_name"),
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{SCENE_NAMESPACE}.accent_color_name",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_switch_states(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_answer: int,
) -> Tuple[Dict[str, bool], Tuple[str, ...]]:
    """Resolve switch states and the corresponding lit bulbs."""

    explicit_states = parse_switch_states(params)
    if explicit_states is not None:
        edges = make_edges(explicit_states)
        lit_bulbs = lit_bulbs_from_edges(edges)
        if params.get("target_answer") is not None and len(lit_bulbs) != int(target_answer):
            raise ValueError("target_answer must match the lit-bulb count implied by switch_states")
        return dict(explicit_states), tuple(lit_bulbs)

    solutions = enumerate_switch_state_solutions(int(target_answer))
    if not solutions:
        raise ValueError(f"no switch-state assignment can realize target_answer={target_answer}")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.switch_states.{int(target_answer)}")
    selected = solutions[int(rng.randrange(len(solutions)))]
    lit_bulbs = lit_bulbs_from_edges(make_edges(selected))
    return dict(selected), tuple(lit_bulbs)


def make_switch_circuit_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    public_query_id: str,
    public_query_probabilities: Mapping[str, float],
    generation_defaults: Mapping[str, Any],
) -> SwitchCircuitScenario:
    """Build one switch-circuit scenario for a public task file."""

    scene_variant, scene_probs = resolve_scene_variant(
        int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    target_answer, answer_probs = resolve_target_answer(
        int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    accent_color, accent_probs = resolve_accent_color(
        int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
    )
    switch_states, lit_bulbs = resolve_switch_states(
        instance_seed=int(instance_seed),
        params=params,
        target_answer=int(target_answer),
    )
    realized_answer = len(lit_bulbs)
    if params.get("target_answer") is not None and int(params["target_answer"]) != int(realized_answer):
        raise ValueError("target_answer must match the generated lit-bulb count")
    return SwitchCircuitScenario(
        public_query_id=str(public_query_id),
        scene_variant=str(scene_variant),
        target_answer=int(realized_answer),
        accent_color_name=str(accent_color),
        switch_states=dict(switch_states),
        edges=make_edges(switch_states),
        lit_bulbs=tuple(lit_bulbs),
        query_id_probabilities={
            str(key): float(value)
            for key, value in public_query_probabilities.items()
        },
        scene_variant_probabilities=dict(scene_probs),
        target_answer_probabilities={str(key): float(value) for key, value in answer_probs.items()},
        accent_color_name_probabilities=dict(accent_probs),
    )


def resolve_switch_circuit_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    defaults: SwitchCircuitDefaults,
) -> Dict[str, int]:
    """Resolve numeric rendering controls."""

    keys = (
        "panel_left_px",
        "panel_top_px",
        "panel_right_px",
        "panel_bottom_px",
        "wire_width_px",
        "bulb_radius_px",
        "switch_width_px",
        "switch_height_px",
        "component_label_font_size_px",
        "title_font_size_px",
        "label_stroke_width_px",
    )
    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(defaults, key)),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        )
        for key in keys
    }


__all__ = [
    "make_switch_circuit_scenario",
    "parse_switch_states",
    "parse_switch_value",
    "probability_map",
    "resolve_accent_color",
    "resolve_scene_variant",
    "resolve_switch_circuit_render_defaults",
    "resolve_switch_states",
    "resolve_target_answer",
]
