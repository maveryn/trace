"""Sampling and symbolic spectrum construction for signal-transform scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int

from .state import (
    OPTION_LABELS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    SignalScenario,
    SignalTransformAxes,
    SignalTransformTaskDefaults,
    SpectrumSpec,
)


def bbox(values: Sequence[float]) -> List[float]:
    """Return a rounded pixel bbox."""

    return [round(float(value), 3) for value in values]


def uniform_probability(values: Sequence[str]) -> Dict[str, float]:
    """Return a uniform probability map over string values."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {value: float(probability) for value in support}


def weighted_support(
    generation_defaults: Mapping[str, Any],
    defaults_key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve positive-weight support from generation defaults."""

    weights = generation_defaults.get(defaults_key, {})
    if isinstance(weights, Mapping):
        supported = tuple(str(value) for value, weight in weights.items() if float(weight) > 0.0)
        if supported:
            return supported
    return tuple(str(value) for value in fallback)


def resolve_canvas_size(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: SignalTransformTaskDefaults,
) -> Tuple[int, int]:
    """Resolve canvas dimensions."""

    return (
        int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", fallback_defaults.canvas_width))),
        int(
            params.get(
                "canvas_height",
                group_default(rendering_defaults, "canvas_height", fallback_defaults.canvas_height),
            )
        ),
    )


def resolve_signal_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: SignalTransformTaskDefaults,
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve render defaults with deterministic variation hooks."""

    keys = (
        "sheet_left_px",
        "sheet_top_px",
        "sheet_right_margin_px",
        "sheet_bottom_margin_px",
        "input_left_px",
        "input_top_px",
        "input_width_px",
        "input_height_px",
        "options_left_px",
        "options_top_px",
        "option_width_px",
        "option_height_px",
        "option_gap_x_px",
        "option_gap_y_px",
        "title_font_size_px",
        "label_font_size_px",
        "axis_font_size_px",
        "waveform_line_width_px",
        "spectrum_line_width_px",
        "grid_line_width_px",
    )
    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in keys
    }


def resolve_scene_variant(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visual scene variant."""

    support = weighted_support(generation_defaults, "scene_variant_weights", SUPPORTED_SCENE_VARIANTS)
    explicit = str(params.get("scene_variant", "") or "").strip()
    if explicit:
        if explicit not in support:
            raise ValueError(f"unsupported signal-transform scene_variant: {explicit}")
        return explicit, {explicit: 1.0}
    rng = spawn_rng(int(instance_seed), f"physics_{SCENE_ID}.scene_variant")
    return str(rng.choice(support)), uniform_probability(support)


def resolve_waveform_family(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    supported_waveform_families: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the waveform family for the public task."""

    support = weighted_support(
        generation_defaults,
        "waveform_family_weights",
        tuple(str(value) for value in supported_waveform_families),
    )
    supported = tuple(value for value in support if value in set(str(item) for item in supported_waveform_families))
    if not supported:
        supported = tuple(str(value) for value in supported_waveform_families)
    explicit = str(params.get("waveform_family", params.get("waveform", "")) or "").strip()
    if explicit:
        if explicit not in supported:
            raise ValueError(f"waveform_family {explicit!r} is not supported for this signal-transform task")
        return explicit, {explicit: 1.0}
    rng = spawn_rng(int(instance_seed), f"physics_{SCENE_ID}.waveform_family")
    return str(rng.choice(supported)), uniform_probability(supported)


def resolve_correct_option_letter(
    instance_seed: int,
    params: Mapping[str, Any],
    *,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the correct spectrum option letter."""

    raw = params.get("correct_option_letter", params.get("target_label", params.get("target_answer")))
    if raw is not None:
        label = str(raw).strip().upper()
        if label not in OPTION_LABELS:
            raise ValueError(f"unsupported signal-transform option label: {label}")
        return label, {label: 1.0}
    rng = spawn_rng(int(instance_seed), f"{namespace}.correct_option_letter")
    return str(rng.choice(OPTION_LABELS)), uniform_probability(OPTION_LABELS)


def resolve_signal_transform_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    supported_waveform_families: Sequence[str],
    namespace: str,
) -> SignalTransformAxes:
    """Resolve scene/sample axes for a public signal-transform task."""

    scene_variant, scene_probs = resolve_scene_variant(
        int(instance_seed),
        params,
        generation_defaults,
    )
    waveform_family, waveform_probs = resolve_waveform_family(
        int(instance_seed),
        params,
        generation_defaults,
        supported_waveform_families=supported_waveform_families,
    )
    correct_label, target_probs = resolve_correct_option_letter(
        int(instance_seed),
        params,
        namespace=str(namespace),
    )
    return SignalTransformAxes(
        scene_variant=str(scene_variant),
        waveform_family=str(waveform_family),
        correct_option_letter=str(correct_label),
        scene_variant_probabilities=dict(scene_probs),
        waveform_family_probabilities=dict(waveform_probs),
        target_answer_probabilities=dict(target_probs),
    )


def spikes(
    signature: str,
    bins: Sequence[int],
    amplitudes: Sequence[float],
    *,
    decay: str = "custom",
) -> SpectrumSpec:
    """Create a spike-spectrum spec."""

    return SpectrumSpec(
        signature=str(signature),
        kind="spikes",
        bins=tuple(int(value) for value in bins),
        amplitudes=tuple(float(value) for value in amplitudes),
        lobe_width=0.0,
        decay=str(decay),
    )


def sinc(signature: str, *, lobe_width: float) -> SpectrumSpec:
    """Create a sinc-envelope spectrum spec."""

    return SpectrumSpec(
        signature=str(signature),
        kind="sinc",
        bins=(),
        amplitudes=(),
        lobe_width=float(lobe_width),
        decay="sinc_envelope",
    )


def bell(signature: str) -> SpectrumSpec:
    """Create a bell-shaped spectrum distractor."""

    return SpectrumSpec(
        signature=str(signature),
        kind="bell",
        bins=(),
        amplitudes=(),
        lobe_width=1.0,
        decay="smooth_bell",
    )


def spectrum_payload(spec: SpectrumSpec) -> Dict[str, Any]:
    """Serialize a spectrum spec into verifier metadata."""

    return {
        "signature": str(spec.signature),
        "kind": str(spec.kind),
        "bins": list(spec.bins),
        "amplitudes": [round(float(value), 4) for value in spec.amplitudes],
        "lobe_width": round(float(spec.lobe_width), 4),
        "decay": str(spec.decay),
    }


def unique_specs(specs: Sequence[SpectrumSpec], *, exclude_signature: str) -> List[SpectrumSpec]:
    """Keep spectrum specs with unique signatures."""

    out: List[SpectrumSpec] = []
    seen = {str(exclude_signature)}
    for spec in specs:
        if str(spec.signature) in seen:
            continue
        seen.add(str(spec.signature))
        out.append(spec)
    return out


def scenario_for_family(
    family: str,
    instance_seed: int,
    *,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], float, SpectrumSpec, List[SpectrumSpec]]:
    """Build a correct spectrum and distractors for one waveform family."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scenario.{family}")
    if family == "square_wave":
        correct = spikes("odd_harmonics_slow", [1, 3, 5, 7], [1.0, 0.35, 0.22, 0.16], decay="odd_slow")
        distractors = [
            spikes("all_harmonics_slow", [1, 2, 3, 4, 5, 6], [1.0, 0.5, 0.33, 0.25, 0.2, 0.17], decay="all_slow"),
            spikes("odd_harmonics_fast", [1, 3, 5, 7], [1.0, 0.18, 0.09, 0.05], decay="odd_fast"),
            spikes("even_harmonics_slow", [2, 4, 6, 8], [1.0, 0.5, 0.33, 0.25], decay="even_slow"),
            spikes("odd_harmonics_flat", [1, 3, 5, 7], [1.0, 0.9, 0.82, 0.74], decay="odd_flat"),
            sinc("wide_sinc_envelope", lobe_width=2.9),
        ]
        return 3, (1, 3, 5, 7), 0.0, correct, distractors
    if family == "triangle_wave":
        correct = spikes("odd_harmonics_fast", [1, 3, 5, 7], [1.0, 0.18, 0.09, 0.05], decay="odd_fast")
        distractors = [
            spikes("odd_harmonics_slow", [1, 3, 5, 7], [1.0, 0.35, 0.22, 0.16], decay="odd_slow"),
            spikes("all_harmonics_fast", [1, 2, 3, 4, 5, 6], [1.0, 0.25, 0.12, 0.07, 0.05, 0.04], decay="all_fast"),
            spikes("even_harmonics_slow", [2, 4, 6, 8], [1.0, 0.5, 0.33, 0.25], decay="even_slow"),
            spikes("single_spike_f3", [3], [1.0]),
            sinc("narrow_sinc_envelope", lobe_width=1.45),
        ]
        return 3, (1, 3, 5, 7), 0.0, correct, distractors
    if family == "sawtooth_wave":
        correct = spikes("all_harmonics_slow", [1, 2, 3, 4, 5, 6], [1.0, 0.5, 0.33, 0.25, 0.2, 0.17], decay="all_slow")
        distractors = [
            spikes("odd_harmonics_slow", [1, 3, 5, 7], [1.0, 0.35, 0.22, 0.16], decay="odd_slow"),
            spikes("all_harmonics_flat", [1, 2, 3, 4, 5, 6], [1.0, 0.92, 0.84, 0.76, 0.68, 0.6], decay="all_flat"),
            spikes("even_harmonics_slow", [2, 4, 6, 8], [1.0, 0.5, 0.33, 0.25], decay="even_slow"),
            spikes("two_spikes_f2_5", [2, 5], [1.0, 0.65]),
            bell("smooth_bell_spectrum"),
        ]
        return 3, (1, 2, 3, 4, 5, 6), 0.0, correct, distractors
    raise ValueError(f"unsupported waveform family: {family}")


def build_signal_scenario(
    axes: SignalTransformAxes,
    instance_seed: int,
    *,
    namespace: str,
) -> SignalScenario:
    """Build the displayed waveform and option spectra."""

    time_cycles, tone_bins, pulse_width, correct, distractors = scenario_for_family(
        str(axes.waveform_family),
        int(instance_seed),
        namespace=str(namespace),
    )
    distractor_specs = unique_specs(distractors, exclude_signature=str(correct.signature))
    rng = spawn_rng(int(instance_seed), f"{namespace}.option_order.{axes.waveform_family}")
    rng.shuffle(distractor_specs)
    option_specs: Dict[str, SpectrumSpec] = {}
    cursor = 0
    for label in OPTION_LABELS:
        if str(label) == str(axes.correct_option_letter):
            option_specs[str(label)] = correct
        else:
            option_specs[str(label)] = distractor_specs[cursor]
            cursor += 1
    return SignalScenario(
        waveform_family=str(axes.waveform_family),
        time_cycles=int(time_cycles),
        tone_bins=tuple(int(value) for value in tone_bins),
        pulse_width=float(pulse_width),
        correct_spectrum=correct,
        option_specs=dict(option_specs),
    )
