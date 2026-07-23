"""Sampling helpers for waveform-panel diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import (
    PANEL_COUNT_SUPPORT,
    PANEL_LABELS,
    SUPPORTED_SCENE_VARIANTS,
    WaveformAxes,
    WaveformPanelSemanticSpec,
)


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map for a finite support."""

    supported = tuple(str(value) for value in values if str(value))
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in supported}
    probability = 1.0 / float(len(supported)) if supported else 0.0
    return {value: float(probability) for value in supported}


def integer_probability_map(values: Sequence[int], selected: int | None = None) -> Dict[str, float]:
    """Return an integer-support probability map with string keys."""

    supported = tuple(int(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if int(value) == int(selected) else 0.0) for value in supported}
    probability = 1.0 / float(len(supported)) if supported else 0.0
    return {str(value): float(probability) for value in supported}


def panel_count_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve the allowed visible panel counts."""

    raw_support = params.get("panel_count_support", group_default(defaults, "panel_count_support", None))
    if isinstance(raw_support, Sequence) and not isinstance(raw_support, (str, bytes)):
        support = tuple(int(value) for value in raw_support)
    else:
        raw_weights = params.get("panel_count_weights", group_default(defaults, "panel_count_weights", None))
        if isinstance(raw_weights, Mapping):
            support = tuple(int(key) for key, value in raw_weights.items() if float(value) > 0.0)
        else:
            support = PANEL_COUNT_SUPPORT
    support = tuple(int(value) for value in support if int(value) in set(PANEL_COUNT_SUPPORT))
    if not support:
        raise ValueError("waveform-panel panel count support must include at least one value in 4..6")
    return tuple(sorted(set(support)))


def resolve_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the waveform panel visual variant."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_panel_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve the number of visible waveform panels."""

    support = panel_count_support(params, defaults)
    explicit = params.get("panel_count")
    if explicit is not None:
        selected = int(explicit)
        if selected not in support:
            raise ValueError(f"unsupported waveform-panel panel_count: {selected}")
        return int(selected), integer_probability_map(support, selected=int(selected))

    if bool(params.get("balanced_panel_count_sampling", group_default(defaults, "balanced_panel_count_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.panel_count")
        selected = int(rng.choice(support))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.panel_count")
        selected = int(rng.choice(support))
    return int(selected), integer_probability_map(support)


def resolve_target_label(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    panel_count: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the selected panel label that will satisfy the query."""

    labels = tuple(PANEL_LABELS[: int(panel_count)])
    raw = params.get("target_label", params.get("correct_option_letter", params.get("target_answer")))
    if raw is not None:
        selected = str(raw).strip().upper()
        if selected not in labels:
            raise ValueError(f"target label {selected!r} is not visible for panel_count={panel_count}")
        return selected, probability_map(labels, selected=selected)

    if bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label.{int(panel_count)}")
        selected = str(rng.choice(labels))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label.{int(panel_count)}")
        selected = str(rng.choice(labels))
    return selected, probability_map(labels)


def resolve_waveform_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> WaveformAxes:
    """Resolve scene and answer axes that are independent of the query operation."""

    scene_variant, scene_probs = resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    panel_count, panel_probs = resolve_panel_count(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    correct_label, target_probs = resolve_target_label(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        panel_count=int(panel_count),
        namespace=str(namespace),
    )
    return WaveformAxes(
        scene_variant=str(scene_variant),
        panel_count=int(panel_count),
        correct_option_letter=str(correct_label),
        scene_variant_probabilities=dict(scene_probs),
        panel_count_probabilities=dict(panel_probs),
        target_answer_probabilities=dict(target_probs),
    )


def _shuffled_values(values: Sequence[int], *, instance_seed: int, namespace: str) -> List[int]:
    """Return a deterministic shuffled copy of integer values."""

    out = [int(value) for value in values]
    rng = spawn_rng(int(instance_seed), str(namespace))
    rng.shuffle(out)
    return out


def _rank_assignment(
    *,
    count: int,
    target_index: int | None,
    target_mode: str | None,
    instance_seed: int,
    namespace: str,
) -> List[int]:
    """Assign unique ranks while optionally forcing the target to an extremum."""

    support = list(range(1, int(count) + 1))
    if target_index is None or target_mode not in {"high", "low"}:
        return _shuffled_values(support, instance_seed=int(instance_seed), namespace=str(namespace))

    target_value = int(count) if target_mode == "high" else 1
    remaining_values = [value for value in support if int(value) != int(target_value)]
    remaining_values = _shuffled_values(
        remaining_values,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.remaining",
    )
    out: List[int] = []
    cursor = 0
    for index in range(int(count)):
        if int(index) == int(target_index):
            out.append(int(target_value))
        else:
            out.append(int(remaining_values[cursor]))
            cursor += 1
    return out


def build_panel_semantic_specs(
    *,
    axes: WaveformAxes,
    amplitude_mode: str | None,
    cycle_mode: str | None,
    instance_seed: int,
    namespace: str,
) -> Tuple[WaveformPanelSemanticSpec, ...]:
    """Build unique waveform property ranks for the visible panels."""

    labels = list(PANEL_LABELS[: int(axes.panel_count)])
    target_index = labels.index(str(axes.correct_option_letter))
    amplitude_ranks = _rank_assignment(
        count=int(axes.panel_count),
        target_index=target_index if amplitude_mode else None,
        target_mode=amplitude_mode,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.amplitude_ranks.{amplitude_mode or 'free'}",
    )
    cycle_counts = _rank_assignment(
        count=int(axes.panel_count),
        target_index=target_index if cycle_mode else None,
        target_mode=cycle_mode,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.cycle_counts.{cycle_mode or 'free'}",
    )
    return tuple(
        WaveformPanelSemanticSpec(
            label=str(label),
            amplitude_rank=int(amplitude_ranks[index]),
            cycle_count=int(cycle_counts[index]),
            is_correct=str(label) == str(axes.correct_option_letter),
        )
        for index, label in enumerate(labels)
    )


__all__ = [
    "build_panel_semantic_specs",
    "integer_probability_map",
    "panel_count_support",
    "probability_map",
    "resolve_panel_count",
    "resolve_scene_variant",
    "resolve_target_label",
    "resolve_waveform_axes",
]
