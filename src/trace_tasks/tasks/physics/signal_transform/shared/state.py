"""State and constants for the signal-transform scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image


SCENE_ID = "signal_transform"
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("clean_match", "grid_match", "lab_sheet")

PERIODIC_WAVEFORM_FAMILIES: Tuple[str, ...] = (
    "square_wave",
    "triangle_wave",
    "sawtooth_wave",
)


@dataclass(frozen=True)
class SignalTransformTaskDefaults:
    """Fallback render defaults for signal-transform diagrams."""

    canvas_width: int = 1280
    canvas_height: int = 860
    sheet_left_px: int = 54
    sheet_top_px: int = 46
    sheet_right_margin_px: int = 54
    sheet_bottom_margin_px: int = 46
    input_left_px: int = 92
    input_top_px: int = 112
    input_width_px: int = 1096
    input_height_px: int = 194
    options_left_px: int = 92
    options_top_px: int = 346
    option_width_px: int = 530
    option_height_px: int = 198
    option_gap_x_px: int = 35
    option_gap_y_px: int = 34
    title_font_size_px: int = 28
    label_font_size_px: int = 24
    axis_font_size_px: int = 19
    waveform_line_width_px: int = 4
    spectrum_line_width_px: int = 4
    grid_line_width_px: int = 1


@dataclass(frozen=True)
class SignalTransformAxes:
    """Resolved sampled axes for one signal-transform instance."""

    scene_variant: str
    waveform_family: str
    correct_option_letter: str
    scene_variant_probabilities: Dict[str, float]
    waveform_family_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SpectrumSpec:
    """Symbolic one-sided magnitude spectrum specification."""

    signature: str
    kind: str
    bins: Tuple[int, ...]
    amplitudes: Tuple[float, ...]
    lobe_width: float
    decay: str


@dataclass(frozen=True)
class SignalScenario:
    """One time-domain waveform and its candidate spectrum options."""

    waveform_family: str
    time_cycles: int
    tone_bins: Tuple[int, ...]
    pulse_width: float
    correct_spectrum: SpectrumSpec
    option_specs: Dict[str, SpectrumSpec]


@dataclass(frozen=True)
class RenderedSignalTransformScene:
    """Rendered signal-transform image and verifier-facing geometry."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, object]]
    render_map: Dict[str, object]
